# scripts/eval_retrieval.py
import os, csv
from dotenv import load_dotenv
load_dotenv()

from graph.retriever import ChunkedTfidfRetriever

QUERIES = [
    "我脸上总脱皮怎么办",
    "怎么安排每周的训练？",
    "如何分配蛋白碳水脂肪？",
    "晚上难以入睡，白天犯困怎么办",
    "有闭口和痘印，成分怎么搭配",
]

def main():
    r = ChunkedTfidfRetriever(kb_dir="kb", enable_hybrid=True)
    rows = []
    for q in QUERIES:
        docs = r.search(q, k=3)
        for _, meta, score, rank in docs:
            rows.append({
                "query": q,
                "rank": rank,
                "file": os.path.basename(meta["path"]),
                "score_hybrid": f"{meta['score_hybrid']:.3f}",
                "score_embed": f"{meta['score_embed']:.3f}",
                "score_tfidf": f"{meta['score_tfidf']:.3f}",
            })

    from pathlib import Path
    # 保存到 scripts/output/，并带时间戳
    out_dir = Path("scripts") / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "retrieval_eval.csv"

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("✅ Done. Saved to:", out.resolve())

if __name__ == "__main__":
    main()
