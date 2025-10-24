# graph/retriever.py  
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re, os, pickle, hashlib
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import minmax_scale


def _chunk_text(text: str, size: int = 900, overlap: int = 120) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paras:
        paras = [text]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= size:
            buf += ("\n" if buf else "") + p
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    if not chunks:
        return []
    out = []
    for i, ch in enumerate(chunks):
        out.append(ch)
        if i < len(chunks) - 1 and overlap > 0:
            tail = ch[-overlap:]
            head = chunks[i + 1][: max(0, size - len(tail))]
            out[-1] = (tail + head) if head else ch
    return out

def _norm_path(p: str) -> str:
    return Path(p).as_posix()

def _fp(files: List[Path], model_name: str, code_rev: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode()); h.update(code_rev.encode())
    for f in files:
        h.update(str(f).encode())
        try:
            h.update(str(f.stat().st_mtime_ns).encode())
        except:
            pass
    return h.hexdigest()[:16]


class ChunkedSemanticRetriever:
    _MODEL = None  # 单例模型

    def __init__(
        self,
        kb_dir: str = "kb",
        encoding: str = "utf-8",
        chunk_size: int = 900,
        overlap: int = 120,
        model_name: str = None,
        min_score: float = None,
        topic_boost: float = 1.3,        # 匹配主题上调倍率
        off_topic_penalty: float = 0.6,  # 非匹配主题下调倍率
        cache_path: str = ".kb_semantic_cache.pkl",
        # Hybrid 相关
        enable_hybrid: bool = True,
        tfidf_max_df: float = 0.95,
        tfidf_ngram: tuple = (2, 4),
        embed_weight: float = None,
        tfidf_weight: float = None,
    ):
        # 环境变量可覆盖
        env_model = os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3"
        self.model_name = model_name or env_model
        self.min_score = float(os.getenv("MIN_SCORE", "0.15")) if min_score is None else min_score
        self.embed_weight = float(os.getenv("HYBRID_EMBED_WEIGHT", "0.7")) if embed_weight is None else embed_weight
        self.tfidf_weight = float(os.getenv("HYBRID_TFIDF_WEIGHT", "0.3")) if tfidf_weight is None else tfidf_weight

        self.kb_dir = kb_dir
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.topic_boost = topic_boost
        self.off_topic_penalty = off_topic_penalty
        self.cache_path = cache_path
        self.enable_hybrid = enable_hybrid
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_ngram = tfidf_ngram

        # 单例加载模型
        if ChunkedSemanticRetriever._MODEL is None:
            ChunkedSemanticRetriever._MODEL = SentenceTransformer(self.model_name)
        self.model: SentenceTransformer = ChunkedSemanticRetriever._MODEL

        kb_path = Path(kb_dir)
        files = sorted(kb_path.glob("*.md"))
        CODE_REV = "hybrid-v2-gating-no_prefix"   # 改这里可强制重建缓存
        self._fingerprint = _fp(files, self.model_name, CODE_REV)

        # 缓存加载 / 构建
        if Path(cache_path).exists():
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                if data.get("fingerprint") == self._fingerprint:
                    self.chunks = data["chunks"]
                    self.embed_mat = data["embed_mat"]
                    self.tfidf_vec = data.get("tfidf_vec")
                    self.tfidf_mat = data.get("tfidf_mat")
                else:
                    self._build_index()
            except Exception:
                self._build_index()
        else:
            self._build_index()

    def _topic_of(self, name: str) -> str:
        n = name.lower()
        if ("skin" in n) or ("care" in n): return "skincare"
        if ("exercise" in n) or ("train" in n): return "exercise"
        if ("diet" in n) or ("nutri" in n): return "diet"
        if "sleep" in n: return "sleep"
        if "psychology" in n: return "psychology"
        return "general"

    def _build_index(self):
        kb_path = Path(self.kb_dir)
        files = sorted(kb_path.glob("*.md"))

        self.chunks: List[Tuple[str, Dict]] = []
        if not files:
            builtin = [
                ("基础饮食：全食物、蛋白1.6–2.2g/kg，TDEE±300kcal 调整。", {"path": "builtin/diet", "chunk_id": "diet-1", "topic": "diet"}),
                ("基础训练：抗阻每周3–5次，心肺150–300分钟。", {"path": "builtin/exercise", "chunk_id": "ex-1", "topic": "exercise"}),
                ("基础护肤：清洁+保湿+防晒；活性按耐受。", {"path": "builtin/skincare", "chunk_id": "sk-1", "topic": "skincare"}),
            ]
            self.chunks = builtin
        else:
            for f in files:
                text = f.read_text(encoding=self.encoding)
                parts = _chunk_text(text, size=self.chunk_size, overlap=self.overlap)
                topic = self._topic_of(f.name)
                for i, p in enumerate(parts, 1):
                    cite = f"{f.name} §{i}"
                    self.chunks.append((p, {"path": _norm_path(str(f)), "chunk_id": cite, "topic": topic}))

        # —— 关键：两套视图 —— #
        passages_embed = [f"passage: {t}" for (t, _) in self.chunks]  # Embedding 用
        passages_lex   = [t for (t, _) in self.chunks]                # TF-IDF 用（纯正文）

        # Embedding 向量
        self.embed_mat = self.model.encode(
            passages_embed, batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)

        # TF-IDF 向量（字符 n-gram 适配中文）
        if self.enable_hybrid:
            self.tfidf_vec = TfidfVectorizer(
                analyzer="char", ngram_range=self.tfidf_ngram,
                max_df=self.tfidf_max_df, sublinear_tf=True, lowercase=False
            )
            self.tfidf_mat = self.tfidf_vec.fit_transform(passages_lex)
        else:
            self.tfidf_vec = None
            self.tfidf_mat = None

        # 写缓存
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump({
                    "fingerprint": self._fingerprint,
                    "chunks": self.chunks,
                    "embed_mat": self.embed_mat,
                    "tfidf_vec": self.tfidf_vec,
                    "tfidf_mat": self.tfidf_mat,
                }, f)
        except Exception:
            pass

    def _infer_topic(self, query: str) -> Optional[str]:
        q = query.lower()
        if any(k in q for k in ["护肤","痘","闭口","美白","a醇","果酸","vc","防晒","敏感肌","水杨酸","脱皮","干燥","起皮"]): return "skincare"
        if any(k in q for k in ["训练","力量","肌肉","有氧","hiit","跑步","运动","拉伸","恢复"]): return "exercise"
        if any(k in q for k in ["饮食","热量","tdee","蛋白","碳水","脂肪","减脂","增肌","营养"]): return "diet"
        if any(k in q for k in ["睡眠","失眠","作息","生物钟"]): return "sleep"
        if any(k in q for k in ["情绪","压力","焦虑","自律","心理"]): return "psychology"
        return None

    def search(self, query: str, k: int = 3) -> List[Tuple[str, dict, float, str]]:
        # Embedding 查询（带前缀）
        qv = self.model.encode([f"query: {query}"], normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)[0]
        sims_embed = (self.embed_mat @ qv)

        # TF-IDF 查询（不用前缀）
        if self.enable_hybrid and self.tfidf_vec is not None:
            q_tfidf = self.tfidf_vec.transform([query])
            sims_tfidf = (self.tfidf_mat @ q_tfidf.T).toarray().ravel()
            sims_tfidf = minmax_scale(sims_tfidf, feature_range=(0.0, 1.0), copy=False)
        else:
            sims_tfidf = np.zeros_like(sims_embed)

        # —— 主题门控：匹配↑1.3，不匹配↓0.6 —— #
        hint = self._infer_topic(query)
        if hint:
            topics = np.array([meta["topic"] for _, meta in self.chunks])
            match = (topics == hint)
            sims_embed *= np.where(match, self.topic_boost, self.off_topic_penalty)
            sims_tfidf *= np.where(match, self.topic_boost, self.off_topic_penalty)

        # 钳制到 [0,1] 防止后续显示 >1.0
        sims_embed  = np.clip(sims_embed,  0.0, 1.0)
        sims_tfidf  = np.clip(sims_tfidf,  0.0, 1.0)

        # 融合
        sims_hybrid = self.embed_weight * sims_embed + self.tfidf_weight * sims_tfidf
        sims_hybrid = np.clip(sims_hybrid, 0.0, 1.0)

        # 阈值 + 排序
        cand = np.where(sims_hybrid >= self.min_score)[0]
        idxs = cand[np.argsort(-sims_hybrid[cand])[:k]] if cand.size else np.argsort(-sims_hybrid)[:k]

        results = []
        for rank, i in enumerate(idxs, 1):
            text, meta = self.chunks[i]
            meta = dict(meta)
            meta["path"] = _norm_path(meta["path"])
            meta["score_embed"] = float(sims_embed[i])
            meta["score_tfidf"] = float(sims_tfidf[i])
            meta["score_hybrid"] = float(sims_hybrid[i])
            results.append((text, meta, meta["score_hybrid"], str(rank)))
        return results

# 兼容旧名
ChunkedTfidfRetriever = ChunkedSemanticRetriever

