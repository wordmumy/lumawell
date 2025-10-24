# graph/nodes.py
import os, re, json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import random
from dotenv import load_dotenv
load_dotenv(override=True)
from pydantic import BaseModel
from openai import OpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage

# ---- 统一时区与当前时间 ----
from datetime import datetime
import pytz

from tools.core import (
    bmi_tool, tdee_tool, skincare_tool,
    mood_to_workout_tool, uv_aqi_advice_tool,
    get_weather_realtime, get_weather_forecast, 
)


# 用 WeatherAPI 的 tz_id/localtime_epoch 构造“城市本地时间” =====
def dt_from_weather(localtime_epoch: Optional[int], tz_id: Optional[str]) -> datetime:
    """
    返回“城市当前本地时间”。优先用本机 UTC -> 城市时区的实时时间；
    仅当 WeatherAPI 的 localtime_epoch 与当前时间相差不超过2分钟时，才采用 API 时间。
    这样可消除 WeatherAPI 刷新/缓存带来的几分钟偏差。
    """
    try:
        tz = pytz.timezone(tz_id or "Australia/Sydney")
    except Exception:
        tz = pytz.timezone("Australia/Sydney")

    # 以本机UTC时间为准，转换到城市时区（实时时钟）
    now_tz = datetime.now(pytz.utc).astimezone(tz)

    # 如果 API 给了 localtime_epoch，且与 now_tz 相差不超过 120 秒，则采用 API 时间
    if isinstance(localtime_epoch, (int, float)):
        try:
            api_dt = datetime.fromtimestamp(localtime_epoch, tz)
            if abs((now_tz - api_dt).total_seconds()) <= 120:
                return api_dt
        except Exception:
            pass

    # 默认返回实时换算的时刻
    return now_tz

# ===== OpenAI(兼容 DashScope) =====
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
)
MODEL = os.getenv("MODEL_NAME", "qwen-plus")
CHAT = client.with_options(timeout=120.0, max_retries=2) 

# ===== Profile 持久化与检索器 =====
from graph.retriever import ChunkedSemanticRetriever as ChunkedTfidfRetriever
from memory.store import ProfileStore

# ===== AgentState =====
class AgentState(MessagesState):
    route: Optional[str]
    sub_intent: Optional[str]
    retrieved: Optional[List[dict]]
    tool_outputs: Optional[List[dict]]
    need_clarify: Optional[Dict]
    profile: Optional[Dict]
    days_requested: Optional[int] = None
    start_offset: Optional[int] = None


PROFILE = ProfileStore("memory/profile.json")

def _last_text(state: AgentState) -> str:
    msgs = state.get("messages") or []
    if not msgs: return ""
    m = msgs[-1]
    if isinstance(m, dict): return m.get("content", "") or ""
    if hasattr(m, "content"): return m.content or ""
    return str(m)

# ===== 路由 =====
EMERGENCY = ["胸痛","昏厥","失去意识","大出血","猝死","处方","诊断"]
GREET_WORDS = ["hi","hello","hey","你好","您好","嗨","在吗","早上好","下午好","晚上好"]

# 时间/日期/对比关键词
TIME_KWS = ["时间","几点","日期","今天","今日","现在","今天几号","现在的时间","现在几点","now","time","date","today"]
ADVICE_BLOCKERS = ["建议","出行","穿衣","装备","适不适合","能不能","安排","计划","是否适合","推荐"]  # NEW
# 用户是否需要“建议/出行/推荐”等（用于控制是否追加分析）
ADVICE_KWS = [
    "建议","出行","穿衣","装备","适不适合","能不能","安排","计划","是否适合","推荐",
    "advice","advise","suggest","suggestion","plan","itinerary"
]

def _wants_advice(text: str) -> bool:
    t = (text or "").lower()
    return any(k in text for k in ADVICE_KWS) or ("advice" in t) or ("suggest" in t)

def _is_blank(text: str) -> bool:
    return not (text or "").strip()

NOW_TODAY_KWS_ZH = ["现在","此刻","今天","今日","当前"]
NOW_TODAY_KWS_EN = ["now","today","current"]


# 预报意图关键词
FORECAST_KWS = [
    "预报","预測","预测","未来","接下来","这周","本周","下周","周末","一周","1周","一星期","一礼拜",
    "明天","明日","翌日","次日","后天","后日","两周","十四天","14天","14 日",
    "七天","7天","10天","长期","多日","趋势",
    "forecast","next","coming","this week","next week","tomorrow","weekend","two weeks","14-day","14d","7-day","10-day","one week"
]

def _is_time_only_query(text: str) -> bool:
    t = text.lower()
    hit_time = any(k in text for k in TIME_KWS)
    hit_advice = any(k in text for k in ADVICE_BLOCKERS) or ("advice" in t) or ("suggest" in t)
    return hit_time and not hit_advice  # 只有纯时间才走 time_only

COMPARE_KWS = ["对比","比较","比一比","compare","vs","对照"]

TOOLS_KEYWORDS = {
    "bmi":"bmi","tdee":"tdee","热量":"tdee","卡路里":"tdee",
    "护肤":"skin","a醇":"skin","维c":"skin","vc":"skin","烟酰胺":"skin","果酸":"skin","水杨酸":"skin","敏感肌":"skin"
}

# 加入时间/日期/对比等关键词
INTENT_MAP = {
    "environment": ["天气","紫外线","uv","空气质量","户外","公园","步道","海边","海滩","beach","自然疗愈","森林", *TIME_KWS, *COMPARE_KWS],  # CHANGED
    "fitness": ["训练","健身","力量","hiit","跑步","拉伸","增肌","减脂","运动"],
    "nutrition": ["饮食","食谱","高蛋白","低gi","抗炎","买菜","woolworths","coles","营养"],
    "mind": ["焦虑","低落","情绪","冥想","呼吸","拖延","压力","正念","睡不好","失眠","情感"],
    "medical": ["体检","化验","血糖","胆固醇","bmi指标","gp","理疗","康复","疼痛","医学解释"],
}

def _detect_mood(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["焦虑","anxious","紧张","心慌"]): return "anxious"
    if any(k in t for k in ["低落","沮丧","没劲","无力","down"]): return "low"
    if any(k in t for k in ["兴奋","嗨","激动","excited"]): return "excited"
    return "neutral"

# ===== 城市识别=====
CITY_RULES = [  # NEW
    ({"en": [r"\bsydney\b"], "zh": ["悉尼"]}, "Sydney,AU"),
    ({"en": [r"\bmelbourne\b"], "zh": ["墨尔本","墨市"]}, "Melbourne,AU"),
    ({"en": [r"\bbrisbane\b"], "zh": ["布里斯班"]}, "Brisbane,AU"),
    ({"en": [r"\bcanberra\b"], "zh": ["堪培拉"]}, "Canberra,AU"),
    ({"en": [r"\bperth\b"], "zh": ["珀斯"]}, "Perth,AU"),
    ({"en": [r"\badelaide\b"], "zh": ["阿德莱德","阿德"]}, "Adelaide,AU"),
    ({"en": [r"\bhobart\b"], "zh": ["霍巴特"]}, "Hobart,AU"),
    ({"en": [r"\bgold\s*coast\b"], "zh": ["黄金海岸"]}, "Gold Coast,AU"),
    ({"en": [r"\bnewcastle\b"], "zh": ["纽卡斯尔","纽卡"]}, "Newcastle,AU"),
    ({"en": [r"\bdarwin\b"], "zh": ["达尔文"]}, "Darwin,AU"),
    ({"en": [r"\bcairns\b"], "zh": ["凯恩斯"]}, "Cairns,AU"),
    ({"en": [r"\bsunshine\s*coast\b"], "zh": ["阳光海岸"]}, "Sunshine Coast,AU"),
    ({"en": [r"\bgeelong\b"], "zh": ["吉朗"]}, "Geelong,AU"),
]

def _detect_au_cities(q: str) -> List[str]:
    """
   支持识别多个澳洲城市（包括“悉尼和墨尔本对比天气”）
    - 自动分割“和、与、及、、、and”等连接词
    - 保留多个匹配结果并去重
    """
    text_en = (q or "").lower()
    text_zh = q or ""
    out = []

    # 尝试先分割句子，提取出各部分 
    # 比如 “悉尼和墨尔本今天对比一下天气”变成 ["悉尼", "墨尔本今天对比一下天气"]
    parts = re.split(r"[和、与及,，\s]+|and", text_zh)
    parts = [p.strip() for p in parts if p.strip()]

    # 针对每一部分执行城市匹配 
    for part in parts:
        for rule, city in CITY_RULES:
            matched_en = any(re.search(pat, part.lower()) for pat in rule.get("en", []))
            matched_zh = any(word in part for word in rule.get("zh", []))
            if matched_en or matched_zh:
                out.append(city)

    # 去重但保留顺序
    out = list(dict.fromkeys(out))

    # 若没识别到任何城市，兜底为悉尼
    if not out:
        raw = text_zh.strip()
        out = [raw] if raw else ["Sydney,AU"]

    return out


# NLP + LLM 时间跨度解析 
def _parse_time_span_nlp(text: str) -> int:
    """
    NLP层：解析自然语言中的时间范围 → 返回天数（0~14）
    """
    t = text.lower()

    # 明确关键词优先匹配
    if "今天" in t or "今日" in t or "today" in t:
        return 0
    if any(k in t for k in ["明天", "明日", "翌日", "次日", "tomorrow"]):
        return 1
    if any(k in t for k in ["后天", "后日", "day after tomorrow"]):
        return 2
    if "大后天" in t:
        return 3
    if any(k in t for k in ["这周","本周","下周","this week","next week","七天","7天"]):
        return 7
    if any(k in t for k in ["两周","14天","十四天","two weeks","14 days"]):
        return 14
    if any(k in t for k in ["一周","1周","一星期","一礼拜","one week"]):
        return 7

    # 任意数字 + “天/日/days”
    m = re.search(r"(\d+)\s*(天|日|days?)", t)
    if m:
        d = int(m.group(1))
        return min(max(d, 1), 14)

    # 模糊表达
    if "未来" in t or "接下来" in t:
        return 3

    return 3



def _parse_time_span_llm(text: str) -> int:
    """
    LLM层：当NLP无法解析或用户表述模糊时调用小模型语义判断
    返回天数（1/3/7/14）
    """
    try:
        mini = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"))
        resp = mini.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": "你是一个时间解析助手，请根据句子判断用户想查询几天的天气，返回纯数字（0,1,2,3,7,14）"},
                {"role": "user", "content": text}
            ],
            temperature=0
        )
        ans = re.findall(r"\d+", resp.choices[0].message.content)
        if ans:
            days = int(ans[0])
            return min(max(days, 0), 14)
        return _parse_time_span_nlp(text)
    except Exception:
        return _parse_time_span_nlp(text)

def _explicit_day_offset(text: str) -> int:
    """返回起始偏移（0=今天,1=明天/明日,2=后天/后日,3=大后天）"""
    t = (text or "").lower()
    if any(k in t for k in ["明天", "明日", "翌日", "次日", "tomorrow"]):
        return 1
    if any(k in t for k in ["后天", "后日", "day after tomorrow"]):
        return 2
    if "大后天" in t:
        return 3
    if any(k in t for k in ["今天", "今日", "today"]):
        return 0
    return 0

def _offset_to_next_week_start(tz_id: str = "Australia/Sydney") -> int:
    """
    计算从“今天”到“下周一”的天数偏移（今天是周一则返回 7）。
    仅用于“下周”语义的起始日定位。
    """
    try:
        tz = pytz.timezone(tz_id)
    except Exception:
        tz = pytz.timezone("Australia/Sydney")
    now = datetime.now(tz)
    # Python: Monday=0 ... Sunday=6
    days_until_next_monday = (7 - now.weekday()) % 7
    return days_until_next_monday or 7

def router_node(state: AgentState):
    """
    智能路由节点：
    - 优先识别：时间查询、城市对比、紧急/工具类
    - 其次识别：各领域意图（fitness / nutrition / environment / mind / medical）
    - 对 environment 再细分：today / forecast_n / compare / time_only
    """
    text = _last_text(state)
    low = text.lower()
    state["profile"] = PROFILE.load()

    # ---------- 空输入：进入 idle，不产生任何回复 ----------
    if _is_blank(text):
        state["route"] = "idle"
        return state

    # ---------- 通用预处理 ----------
    cities = _detect_au_cities(text)
    state["sub_intent"] = None

    # ---------- 问候优先：进入 greet（自我介绍+能做什么） ----------
    if any(w in low for w in GREET_WORDS):
        state["route"] = "greet"
        return state
    
    # ---------- 跨领域高优先级 ----------
    # 先判断两城对比
    if any(k in text for k in COMPARE_KWS) and len(cities) >= 2:
        # 如果同时包含“未来/明天/后天/下周/next week/周末”等 → 走多城 forecast 对比
        if any(k in low for k in FORECAST_KWS) or any(w in text for w in ["明天","明日","翌日","次日","后天","后日","大后天","未来","周末","weekend","days"]):
            days = _parse_time_span_nlp(text)
            if days not in [0,1,2,3,7,14]:
                days = _parse_time_span_llm(text)

            # “下周/next week”→ 从下周一开始的 7 天窗口
            if ("下周" in text) or ("next week" in low):
                days = 7
                # 也可用第一个城市的 tz 来更精细：tz_id = get_weather_realtime(cities[0]).get("tz_id") or "Australia/Sydney"
                offset = _offset_to_next_week_start("Australia/Sydney")
            else:
                offset = _explicit_day_offset(text)

            state["route"] = "environment"
            state["days_requested"] = days
            state["start_offset"] = offset

            if days == 0:
                state["sub_intent"] = "today"
            elif days <= 1:
                state["sub_intent"] = "forecast_1"
            elif days <= 3:
                state["sub_intent"] = "forecast_3"
            elif days <= 7:
                state["sub_intent"] = "forecast_7"
            else:
                state["sub_intent"] = "forecast_14"
            return state

        # 否则：纯“今日实况对比”
        state["route"] = "environment"
        state["sub_intent"] = "compare"
        return state

    # 只问时间（若命中 compare 或多城则不走 time_only）
    if _is_time_only_query(text) and not any(k in text for k in COMPARE_KWS) and len(cities) < 2:
        state["route"] = "environment"
        state["sub_intent"] = "time_only"
        return state

    #实时天气（含“现在”“今天/今日/today”）
    if any(w in text for w in ["现在","此刻","today","今天","今日","当前"]):
        state["route"] = "environment"
        state["sub_intent"] = "today"
        return state

    # 工具 / 安全 / 问候 
    if any(w in text for w in EMERGENCY):
        state["route"] = "safety"
        return state
    for k, v in TOOLS_KEYWORDS.items():
        if k in low:
            state["route"] = v
            return state

    # 三级：意图识别 
    for intent, kws in INTENT_MAP.items():
        if any(k in low for k in kws):
            state["route"] = intent

            # ---------- 环境领域细分 ----------
            if intent == "environment":
                # 未来预报（动态天数）
                if any(k in low for k in FORECAST_KWS) or any(w in text for w in ["明天","明日","翌日","次日","后天","后日","大后天","未来","days"]):
                    days = _parse_time_span_nlp(text)
                    if days not in [0,1,2,3,7,14]:
                        days = _parse_time_span_llm(text)
                    # 新增：显式偏移（明天/后天/大后天）
                    offset = _explicit_day_offset(text)
                    state["days_requested"] = days
                    state["start_offset"] = offset     

                    if days == 0:
                        state["sub_intent"] = "today"
                    elif days <= 1:
                        state["sub_intent"] = "forecast_1"
                    elif days <= 3:
                        state["sub_intent"] = "forecast_3"
                    elif days <= 7:
                        state["sub_intent"] = "forecast_7"
                    else:
                        state["sub_intent"] = "forecast_14"
                    return state

                # 其它天气询问（未提 forecast）就是 today 默认
                state["sub_intent"] = "today"
                return state
            
            return state
        
    # 都没命中时回退到 RAG
    state["route"] = "rag"
    return state

# ====== Pydantic 输入（BMI/TDEE/Skin） ======
class BMIInput(BaseModel):
    height_cm: float
    weight_kg: float

class TDEEInput(BaseModel):
    sex: str
    age: int
    height_cm: float
    weight_kg: float
    activity_level: str

class SkinInput(BaseModel):
    skin_type: str
    concerns: str
    actives_in_use: str

# ====== 解析 + 工具节点（BMI/TDEE/Skin）======
def _parse_numbers(text: str) -> List[float]:
    return list(map(float, re.findall(r"\d+\.?\d*", text)))

def _parse_bmi(text: str, profile: Dict) -> Tuple[Optional[BMIInput], List[str]]:
    nums = _parse_numbers(text)
    height = weight = None
    if len(nums) >= 2:
        h_m = re.search(r"(\d+\.?\d*)\s*cm", text.lower())
        w_m = re.search(r"(\d+\.?\d*)\s*kg", text.lower())
        height = float(h_m.group(1)) if h_m else float(nums[0])
        weight = float(w_m.group(1)) if w_m else float(nums[1])
    else:
        height = profile.get("height_cm")
        weight = profile.get("weight_kg")
    missing = []
    if height is None: missing.append("身高(cm)")
    if weight is None: missing.append("体重(kg)")
    if missing: return None, missing
    return BMIInput(height_cm=height, weight_kg=weight), []

def _parse_tdee(text: str, profile: Dict) -> Tuple[Optional[TDEEInput], List[str]]:
    tlow = text.lower()

    # 性别解析
    sex = None
    if any(x in text for x in ["女","女生","女性"]) or "female" in tlow or re.search(r"\bf\b", tlow):
        sex = "female"
    elif any(x in text for x in ["男","男生","男性"]) or "male" in tlow or re.search(r"\bm\b", tlow):
        sex = "male"

    # 全部数字抓取（不含单位）
    nums = _parse_numbers(text)  # 例如 "男，38岁，身高170，体重52" -> [38,170,52]
    nums_remain = nums[:]        # 后面会逐步剔除已用到的数字

    # 先用显式正则抓“有标签/有单位”的值
    age = None
    height = None
    weight = None

    age_m = re.search(r"(\d+)\s*岁", text)
    if age_m:
        age = int(age_m.group(1))
        # 从剩余数字里剔除这个年龄
        try:
            nums_remain.remove(float(age))
        except Exception:
            pass

    # 带单位
    h_cm = re.search(r"(\d+\.?\d*)\s*cm", tlow)
    w_kg = re.search(r"(\d+\.?\d*)\s*kg", tlow)
    if h_cm:
        height = float(h_cm.group(1))
        try:
            nums_remain.remove(float(height))
        except Exception:
            pass
    if w_kg:
        weight = float(w_kg.group(1))
        try:
            nums_remain.remove(float(weight))
        except Exception:
            pass

    # 没写单位时：按中文标签抓取
    if height is None:
        h2 = re.search(r"(身高|高)\s*(\d+\.?\d*)", text)
        if h2:
            height = float(h2.group(2))
            try:
                nums_remain.remove(float(height))
            except Exception:
                pass
    if weight is None:
        w2 = re.search(r"(体重|重)\s*(\d+\.?\d*)", text)
        if w2:
            weight = float(w2.group(2))
            try:
                nums_remain.remove(float(weight))
            except Exception:
                pass

    # 兜底：还缺就从“剩余数字”里按常识规则填
    def pick_height_weight_from_remaining(arr):
        h, w = None, None
        # 先按范围猜
        for v in list(arr):
            if h is None and 120 <= v <= 230:
                h = float(v); arr.remove(v); break
        for v in list(arr):
            if w is None and 30 <= v <= 200:
                w = float(v); arr.remove(v); break
        return h, w

    if height is None or weight is None:
        h_guess, w_guess = pick_height_weight_from_remaining(nums_remain)
        if height is None and h_guess is not None:
            height = h_guess
        if weight is None and w_guess is not None:
            weight = w_guess

    # 若仍有空，最后按“文本出现顺序”兜底
    if height is None and nums_remain:
        height = float(nums_remain.pop(0))
    if weight is None and nums_remain:
        weight = float(nums_remain.pop(0))

    # 常识性纠错：若 height<100 且 weight>100，极可能写反了 → 交换
    if (height is not None and weight is not None) and (height < 100 and weight > 100):
        height, weight = weight, height

    # 活动水平
    level_map = {
        "久坐":"sedentary","不怎么动":"sedentary","sedentary":"sedentary",
        "轻度":"light","light":"light","中等":"moderate","适中":"moderate","moderate":"moderate",
        "活跃":"active","运动较多":"active","active":"active","运动员":"athlete","高强度":"athlete","athlete":"athlete"
    }
    activity_level = None
    for k,v in level_map.items():
        if k in tlow:
            activity_level = v; break
    if not activity_level:
        activity_level = profile.get("activity_level")

    # 画像兜底
    if sex is None:   sex   = profile.get("sex")
    if age is None:   age   = profile.get("age")
    if height is None:height= profile.get("height_cm")
    if weight is None:weight= profile.get("weight_kg")

    # 缺失检查
    missing = []
    if not sex: missing.append("性别")
    if age is None: missing.append("年龄")
    if height is None: missing.append("身高(cm)")
    if weight is None: missing.append("体重(kg)")
    if not activity_level: missing.append("活动水平(久坐/轻度/中等/活跃/运动员)")
    if missing:
        return None, missing

    return TDEEInput(
        sex=sex,
        age=int(age),
        height_cm=float(height),
        weight_kg=float(weight),
        activity_level=activity_level
    ), []

def _parse_skin(text: str, profile: Dict) -> Tuple[Optional[SkinInput], List[str]]:
    tlow = text.lower()
    skin_map = {"大干皮":"dry","干":"dry","油":"oily","出油":"oily","混合":"oily","敏":"sensitive","敏感":"sensitive"}
    skin_type = None
    for k,v in skin_map.items():
        if k in tlow: skin_type = v; break
    if not skin_type: skin_type = profile.get("skin_type")
    active_alias = {
        "a醇":"retinol","视黄醇":"retinol",
        "果酸":"aha_bha","水杨酸":"aha_bha","aha":"aha_bha","bha":"aha_bha",
        "vc":"vitc","维c":"vitc","烟酰胺":"niacinamide",
        "壬二酸":"aza","过氧化苯甲酰":"benzoyl_peroxide","bpo":"benzoyl_peroxide"
    }
    found = set()
    for k,v in active_alias.items():
        if k in tlow: found.add(v)
    actives = list(found) or (profile.get("actives_in_use", "").split(",") if profile.get("actives_in_use") else [])
    actives = [a.strip() for a in actives if a and a.strip()]
    concerns = []
    for kw in ["痘","粉刺","闭口","美白","淡斑","抗老","修护","脱皮","干燥"]:
        if kw in text: concerns.append(kw)
    concerns = "、".join(concerns) or profile.get("concerns", "基础护理")
    missing = []
    if not skin_type: missing.append("肤质")
    if missing: return None, missing
    return SkinInput(skin_type=skin_type, concerns=concerns, actives_in_use=", ".join(actives)), []

def _tdee_advice_fallback(parsed: "TDEEInput", res: Dict) -> str:
    h_m = parsed.height_cm / 100.0
    bmi = round(parsed.weight_kg / (h_m * h_m), 1)
    tdee = int(res.get("tdee", 0))
    goal = "减脂" if bmi >= 27 else ("体态重塑" if 23 <= bmi < 27 else "增肌/力量提升")
    # 热量建议
    if goal == "减脂":
        kcal = max(tdee - 300, tdee - int(0.15 * tdee))
        kcal_line = f"每日热量建议：约 **{kcal} kcal**（较TDEE -300 ~ -15%）"
    elif goal == "体态重塑":
        kcal_line = f"每日热量建议：≈ **{tdee} kcal**（围绕TDEE微调 ±100）"
    else:
        kcal_line = f"每日热量建议：≈ **{tdee + 300} kcal**（较TDEE +300）"

    # 周计划（按久坐起步）
    plan = (
        "**一周训练模板（久坐起步）**\n"
        "- 力量 x3：全身A/B交替（深蹲/硬拉/卧推或俯卧撑/划船/推举），每练习 3–4 组 × 6–12 次，RPE 7–8。\n"
        "- 有氧 x2：LISS 30–40 分钟（快走/椭圆/骑行），若体能好可加入 1 次 10×(1′快+1′慢) 的间歇。\n"
        "- NEAT：**日步数 7000–9000**，分散到白天（番茄钟每 50 分钟起身走 3–5 分钟）。\n"
    )

    # 进阶与恢复
    prog = (
        "**进阶与恢复**\n"
        "- 渐进超负荷：当某动作 3 组都能轻松完成上限次数，下次加 **2.5–5kg** 或增加 1–2 次重复。\n"
        "- 灵活性：训练前 5 分钟动态热身（髋/肩/踝），训练后 5 分钟静态拉伸（腘绳肌、小腿、胸背）。\n"
        "- 睡眠：**7–9 小时**；蛋白 1.6–2.2 g/kg；水：**体重×35ml**/日，训练日略增。\n"
        "- 伤痛预防：疼痛>24h未缓解或加重→减量/就医；核心稳定与髋周激活每次训练 5–8 分钟。\n"
    )

    # 示例日程
    sched = (
        "**参考日程**\n"
        "- 周一：力量A（深蹲/卧推/划船）+ 核心 8′\n"
        "- 周二：LISS 35′ + 伸展 5′\n"
        "- 周三：休息/步行 30′\n"
        "- 周四：力量B（硬拉/推举/引体或下拉）+ 核心 8′\n"
        "- 周五：LISS 30′（可做10×(1′快+1′慢)）\n"
        "- 周六：力量A（降低 5–10% 负重，练技术）\n"
        "- 周日：休息/户外轻松走 30–60′\n"
    )

    return (
        f"### 个性化训练与营养建议\n"
        f"- 目标倾向：**{goal}**（基于 BMI {bmi}）\n"
        f"- {kcal_line}\n\n"
        f"{plan}\n{prog}\n{sched}"
        "若最近完全不运动，可从 **力量 x2 + LISS x2** 起步，2–4 周后再加到上述周频次。"
    )


def tool_node(state: AgentState):
    """
    工具节点：解析用户输入 → 调用对应工具函数 → 生成结构化输出
    适用：BMI / TDEE / Skin 三类
    """
    text = _last_text(state)
    prof = state.get("profile") or {}
    out = []

    # ==== ① BMI 计算 ====
    if state.get("route") == "bmi":
        parsed, missing = _parse_bmi(text, prof)
        if missing:
            state["need_clarify"] = {
                "tool": "bmi",
                "missing": missing,
                "hint": "示例：身高165cm 体重55kg"
            }
            state["tool_outputs"] = []
            return state

        res = bmi_tool(**parsed.model_dump())
        out.append({
            "tool": "bmi",
            "input": parsed.model_dump(),
            "result": res,
            "assumption": "从文本/画像解析"
        })
        PROFILE.update(height_cm=parsed.height_cm, weight_kg=parsed.weight_kg)
        content = (
            f"**BMI 计算结果**\n"
            f"- 身高：{parsed.height_cm} cm\n"
            f"- 体重：{parsed.weight_kg} kg\n"
            f"- BMI：{res['bmi']}\n"
            f"- 体重分类：{res['category']}\n\n"
            f"👉 正常范围为 18.5–24。若想优化体型，可结合 TDEE 计算每日能量消耗。"
        )
        wants = _wants_advice(text)

        final_text = content  # 先默认只输出计算结果
        if wants:
            # 计算健康体重区间（便于给出落地目标）
            w_min, w_max = _bmi_weight_range(parsed.height_cm)
            tool_ctx = [{
                "tool": "bmi",
                "input": parsed.model_dump(),
                "result": {**res, "healthy_weight_range": [w_min, w_max]}
            }]
            ctx, srcs = _ctx_and_sources(state, tool_ctx)

            prompt = (
                "基于以下 BMI 结果，为用户生成“可直接照做”的运动与饮食建议：\n"
                f"- 身高：{parsed.height_cm} cm；体重：{parsed.weight_kg} kg；BMI：{res.get('bmi')}\n"
                f"- 体重分类：{res.get('category')}；健康体重区间（BMI 18.5–24）：约 {w_min}–{w_max} kg\n\n"
                "输出要求：\n"
                "1) 目标体重与节奏（每周0.25–0.75 kg更安全），并说明热量收支原则\n"
                "2) 一周训练模板：力量（部位/动作/组次/RPE）+ 有氧（频次/时长/强度）+ 拉伸\n"
                "3) 饮食结构：蛋白/碳水/脂肪建议，简单可执行的餐例（超市可买到）\n"
                "4) 生活方式：睡眠、步数（NEAT）、补水、电解质与风险提示\n"
                "5) 要点式中文输出，数值明确，可直接执行\n"
            )

            fallback_advice = _bmi_advice_fallback(parsed, res, (w_min, w_max))
            advice_text = _safe_llm_answer(prompt, ctx, srcs, fallback_advice)

            final_text = content + "\n\n" + advice_text  # 需要建议时再拼接
        state["messages"].append(AIMessage(content=final_text))

        state["tool_outputs"] = out
        state["need_clarify"] = None
        return state
    
    # ==== ② TDEE 计算（合并为一条消息发送） ====
    elif state.get("route") == "tdee":
        parsed, missing = _parse_tdee(text, prof)
        if missing:
            state["need_clarify"] = {
                "tool": "tdee",
                "missing": missing,
                "hint": "示例：女 23岁 165cm 55kg 中等活动"
            }
            state["tool_outputs"] = []
            return state

        res = tdee_tool(**parsed.model_dump())
        out.append({
            "tool": "tdee",
            "input": parsed.model_dump(),
            "result": res,
            "assumption": "从文本/画像解析"
        })
        PROFILE.update(**parsed.model_dump())

        content = (
            f"**TDEE 计算结果**\n"
            f"- 性别：{parsed.sex}\n"
            f"- 年龄：{parsed.age} 岁\n"
            f"- 身高：{parsed.height_cm} cm\n"
            f"- 体重：{parsed.weight_kg} kg\n"
            f"- 活动水平：{parsed.activity_level}\n\n"
            f"**结果：**\n"
            f"- 基础代谢 (BMR)：{res['bmr']} kcal\n"
            f"- 每日总消耗 (TDEE)：{res['tdee']} kcal\n\n"
            f"👉 若目标为减脂：每日摄入比 TDEE 低约 300 kcal。\n"
            f"👉 若目标为增肌：每日摄入比 TDEE 高约 300 kcal。"
        )

        source_note = "（身高/体重来自个人画像，如需更新请告诉我）"
        content = content + "\n" + source_note

        wants = _wants_advice(text)

        advice_text = ""
        if wants:
            tool_ctx = [{"tool": "tdee", "input": parsed.model_dump(), "result": res}]
            ctx, srcs = _ctx_and_sources(state, tool_ctx)
            prompt = (
                "请基于以下个人资料与能量消耗，生成“**可直接照做**”的训练建议：\n"
                f"- 性别：{parsed.sex}；年龄：{parsed.age}；身高：{parsed.height_cm}cm；体重：{parsed.weight_kg}kg；活动水平：{parsed.activity_level}\n"
                f"- TDEE：{int(res.get('tdee',0))} kcal；BMR：{int(res.get('bmr',0))} kcal\n\n"
                "输出要求：\n"
                "1) 目标建议（减脂/体态重塑/增肌）与每日热量建议（结合TDEE给出数值或区间）\n"
                "2) 一周训练模板（力量/有氧频次、动作组合、组次/重复、RPE、休息）\n"
                "3) 渐进超负荷规则、NEAT（日步数）与灵活性/拉伸安排\n"
                "4) 恢复与伤痛预防要点（睡眠/营养/何时减量或就医）\n"
                "5) 用中文要点式输出，段落清晰，数值明确\n"
            )
            advice_text = _safe_llm_answer(prompt, ctx, srcs, _tdee_advice_fallback(parsed, res))

        final_text = content if not wants else (content + "\n\n" + advice_text)
        state["messages"].append(AIMessage(content=final_text))

        state["tool_outputs"] = out
        state["need_clarify"] = None
        return state

    # ==== 护肤建议 ====
    elif state.get("route") == "skin":
        parsed, missing = _parse_skin(text, prof)
        if missing:
            state["need_clarify"] = {
                "tool": "skincare",
                "missing": missing,
                "hint": "示例：敏感肌；或：大干皮/混油；可写：晚上A醇、白天VC（可不写）"
            }
            state["tool_outputs"] = []
            return state

        # 工具计算（结构化规则）
        res = skincare_tool(**parsed.model_dump())
        out.append({
            "tool": "skincare",
            "input": parsed.model_dump(),
            "result": res,
            "assumption": "从文本/画像解析"
        })
        PROFILE.update(
            skin_type=parsed.skin_type,
            actives_in_use=parsed.actives_in_use,
            concerns=parsed.concerns
        )

        # 构造 LLM Prompt（让模型生成自然语言解释）
        ctx_json = json.dumps(res, ensure_ascii=False, indent=2)
        user_prompt = (
            f"用户肤质：{parsed.skin_type}\n"
            f"在用活性：{parsed.actives_in_use}\n"
            f"主要问题：{parsed.concerns}\n\n"
            f"系统规则结果：\n{ctx_json}\n\n"
            "请以澳洲皮肤管理专家的身份回答：\n"
            "1️⃣ 说明这种肤质的典型特征与风险；\n"
            "2️⃣ 结合澳洲气候（UV强、干燥）给出早晚护肤建议；\n"
            "3️⃣ 若有配伍冲突，请解释原理并提供替代搭配；\n"
            "4️⃣ 最后给出一段鼓励性结语（自然温暖语气）。\n"
            "输出语言风格应自然、有温度、有层次。"
        )

        # 调用 LLM + 兜底输出
        ctx, srcs = _ctx_and_sources(state, out)
        fallback = (
            f"**肤质识别**：{parsed.skin_type}\n"
            f"**核心思路**：{res.get('general', '基础保湿防晒')}\n"
            "（模型暂不可用，已提供规则建议）"
        )
        content = _safe_llm_answer(user_prompt, ctx, srcs, fallback)

        # 存入消息
        state["messages"].append(AIMessage(content=content))
        state["tool_outputs"] = out
        state["need_clarify"] = None
        return state

    # ==== 默认 ====
    state["tool_outputs"] = out
    state["need_clarify"] = None
    return state

# ===== RAG（Hybrid）=====
retriever = ChunkedTfidfRetriever(kb_dir="kb", enable_hybrid=True)

def rag_gather(text: str) -> List[dict]:
    docs = retriever.search(text, k=5)
    out = []
    for (t, m, _, cid) in docs:
        out.append({
            "path": m["path"], "snippet": t, "cite": cid,
            "score": m.get("score_hybrid", 0.0),
            "score_embed": m.get("score_embed", 0.0),
            "score_tfidf": m.get("score_tfidf", 0.0),
        })
    return out

def _ctx_and_sources(state: AgentState, extra_tools: List[dict]) -> Tuple[str,List[str]]:
    user = _last_text(state)
    retrieved = rag_gather(user)
    state["retrieved"] = retrieved
    lines, srcs = [], []
    for d in retrieved:
        lines.append(f"[{d['cite']}] {d['snippet']}")
        name = Path(d["path"]).name.replace(".md","")
        srcs.append(f"[{d['cite']}] {name} (hyb={d['score']:.3f}, emb={d['score_embed']:.3f}, tfidf={d['score_tfidf']:.3f})")
    if extra_tools:
        lines.append("\n[Tools]\n" + json.dumps(extra_tools, ensure_ascii=False, indent=2, default=str))
    return "\n".join(lines), srcs

# ===== LLM 共用 =====
SYSTEM = ("You are a helpful Health & Wellness advisor. "
          "You can discuss nutrition, fitness, and skincare. "
          "Always be cautious: you are not a doctor; avoid diagnosis or prescriptions.")

def _llm_answer(user: str, ctx: str, sources_lines: List[str]) -> str:
    prompt = (f"### Instruction\n{user}\n\n"
              f"### Context (RAG/Tools)\n{ctx}\n"
              "### Requirements\n"
              "- Use bullet points and short paragraphs.\n"
              "- Offer 3–6 actionable steps.\n"
              "- If context is used, cite as [1], [2].\n"
              "- Be local to Australia when relevant.\n"
              "- Answer in Chinese if user spoke Chinese.\n")
    try:
        resp = CHAT.chat.completions.create( 
            model=MODEL,
            messages=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
            temperature=0.4,
        )
        content = resp.choices[0].message.content
        if sources_lines:
            content += "\n\nSources:\n" + "\n".join(sources_lines)
        return content
    except Exception as e:
        return f"抱歉，当前模型不可用。\n[debug] {type(e).__name__}: {e}"

# 安全 LLM
def _safe_llm_answer(user: str, ctx: str, sources_lines: List[str], fallback: str) -> str:
    try:
        txt = _llm_answer(user, ctx, sources_lines)
        if txt.strip().startswith("抱歉，当前模型不可用"):
            return fallback + "\n（系统提示：模型响应超时，请稍后重试）"
        return txt
    except Exception as e:
        return fallback + f"\n（系统提示：{type(e).__name__}，请稍后重试）"

def _append_once(state: AgentState, *chunks: str) -> None:
    """将多个文本片段用空行拼好，只 append 一次。"""
    text = "\n\n".join([c for c in chunks if c and c.strip()])
    state["messages"].append(AIMessage(content=text))

# 将“实况卡片”渲染为确定性文本（不走 LLM）
def _render_observation_card(city_label: str, tz_id: str, dt: datetime,
                             cond: str, temp_c: float, humidity: Optional[int],
                             uv_index: float, aqi_val: int) -> str:  # NEW
    WEEKDAY_ZH = ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"]
    line_time = f"{dt.year}年{dt.month:02d}月{dt.day:02d}日，{WEEKDAY_ZH[dt.weekday()] }，{dt:%H:%M}（{tz_id}）"
    hum = "-" if humidity is None else f"{humidity}%"
    return (
        f"**{city_label} 实况**\n"
        f"- 本地时间：{line_time}\n"
        f"- 天气：{cond}\n"
        f"- 温度：{temp_c}°C\n"
        f"- 湿度：{hum}\n"
        f"- UV：{uv_index}\n"
        f"- AQI：{aqi_val}\n"
    )

def fitness_agent_node(state: AgentState):
    """
    心理/情绪 → 运动匹配：
    - 先用 mood_to_workout_tool 给出结构化方案
    - 再请 LLM 生成自然语言建议
    - 若 LLM 不可用，用兜底模板直接输出
    """
    text = _last_text(state)
    mood = _detect_mood(text)

    # 1) 工具结果（结构化）
    plan = mood_to_workout_tool(mood)
    tool_out = [{"tool": "mood→workout", "input": {"mood": mood}, "result": plan}]
    ctx, srcs = _ctx_and_sources(state, tool_out)

    # 2) 让 LLM 生成“可直接照做”的训练建议
    prompt = (
        "用户的当前情绪与状态如下（从文本解析）：\n"
        f"- mood: {mood}\n\n"
        "下面是根据情绪匹配得到的结构化运动方案（来自工具）：\n"
        f"{json.dumps(plan, ensure_ascii=False)}\n\n"
        "请据此给出可执行训练建议，要求：\n"
        "1) 训练类型与强度（含 RPE 或心率区间）；时长/组次/休息；\n"
        "2) 热身与冷却流程，加入 2–3 分钟的呼吸练习脚本；\n"
        "3) 安全注意与恢复（睡眠/补水/量控/疼痛处理）；\n"
        "4) 语气同理、鼓励；中文要点式输出。"
    )

    # 3) 兜底文本（不依赖 LLM，按情绪直接给）
    if mood == "anxious":
        fallback = (
            "**情绪识别**：焦虑/亢奋感偏高；\n"
            "**建议训练**：低—中强度有氧（快走/骑行/椭圆 20–30 分钟，RPE 4–6），配合全身舒缓拉伸 8–10 分钟；\n"
            "**呼吸脚本**：4-4-4-4（吸4秒-停4秒-呼4秒-停4秒）× 8 轮；步行时尽量鼻吸鼻呼；\n"
            "**注意**：避免高强度冲刺与过量咖啡因；若心悸/胸闷持续或加重，停止运动并就医；\n"
            "**恢复**：温水淋浴、补水 300–500ml、睡前 5 分钟腹式呼吸。"
        )
    elif mood == "excited":
        fallback = (
            "**情绪识别**：兴奋/动机高；\n"
            "**建议训练**：高质量力量为主（全身 4–6 个复合动作，3–4 组×6–10 次，RPE ≤8），\n"
            "可加 8–12 分钟间歇有氧（如 8×(30s快+60s慢)）；\n"
            "**热身/冷却**：关节动态热身 5 分钟；结束后小腿/股四头/腘绳/背部静态拉伸各 30–45s；\n"
            "**量控**：避免“爆量”；本次总时长 45–60 分钟；\n"
            "**恢复**：训练后 30–60 分钟内补水电解质与优质蛋白；晚间 7–9 小时睡眠。"
        )
    elif mood == "low":
        fallback = (
            "**情绪识别**：低落/能量低；\n"
            "**建议训练**：节律维持为主：全身轻量力量 20–30 分钟（每动作 2–3 组×10–12 次，RPE 5），\n"
            "或 20 分钟轻松步行/骑行；\n"
            "**呼吸**：鼻吸 4 秒-口呼 6 秒，持续 3–5 分钟；\n"
            "**小目标**：只要求出门/换装/完成前 10 分钟即可，完成即可算赢；\n"
            "**恢复**：温和拉伸、补水 300ml、记录一次“完成 ✔”。"
        )
    else:  # neutral / 其它
        fallback = (
            "**情绪识别**：平稳/一般；\n"
            "**建议训练**：常规力量 + 中等强度有氧：\n"
            "- 力量：全身 5 个动作（蹲/推/拉/髋伸/核心），3×8–12 次，RPE 6–7；\n"
            "- 有氧：20–30 分钟，能流畅对话略微喘；\n"
            "**热身/冷却**：动态热身 5′ + 收操拉伸 8′；\n"
            "**恢复**：蛋白 1.6 g/kg·d、日步数 7k–10k、睡眠 7–9h。"
        )

    # 用“安全 LLM”生成，有问题就落回 fallback
    content = _safe_llm_answer(prompt, ctx, srcs, fallback)
    state["messages"].append(AIMessage(content=content))
    return state


def nutrition_agent_node(state: AgentState):
    text = _last_text(state)
    ctx, srcs = _ctx_and_sources(state, [])
    content = _llm_answer(text + "\n\n请给出：①餐次与宏量建议 ②示例食谱（含食材及克重）③可在Coles/Woolworths购买的通用食材清单。", ctx, srcs)
    state["messages"].append(AIMessage(content=content))
    return state

def mind_agent_node(state: AgentState):
    """
    纯心理/情绪抚慰通道（不做运动匹配）：
    - 调 RAG（偏向 psychology kb）
    - 用 _safe_llm_answer 生成同理+可执行情绪调节方案
    - LLM 不可用时走心理抚慰兜底模板（不输出任何运动建议）
    """
    text = _last_text(state)

    extra_tools = [{
        "tool": "3min_breathing",
        "input": {},
        "result": {"guide": "鼻吸4秒-停4秒-口呼4秒-停4秒 × 8–10轮；配合肩颈放松与正念标签：看见→听见→感觉。"}
    }]

    # 检索 psychology 等 KB（_ctx_and_sources 会把检索结果和 tools 一起放入上下文）
    ctx, srcs = _ctx_and_sources(state, extra_tools)

    # 明确告诉 LLM：这是“心理抚慰”而非“运动匹配”
    prompt = (
        "请用同理、温和且不评判的语气，针对用户的心理/情绪困扰提供支持与抚慰（不要给运动/训练建议）。\n"
        "目标：帮助对方先稳住身心，再给出可执行的小步骤。\n\n"
        "请按下面结构输出：\n"
        "1) 先**命名情绪**并验证（反映式聆听，1–2 句即可）。\n"
        "2) **3 分钟稳态练习脚本**：\n"
        "   - 30–60 秒地面着力觉（脚掌/坐骨的触地感）\n"
        "   - 2–3 分钟节律呼吸（如 4-4-4-4 盒式呼吸或 4-6 呼吸）\n"
        "   - 30 秒感官锚定（看见/听见/触感各 1–2 个）\n"
        "3) **当下可以做的 3 个小步骤**（<5 分钟即可完成，清单式，含“触发-行动-奖励”）。\n"
        "4) **自我关怀与边界**（睡眠/信息摄入/与人联系的建议，避免自责用语）。\n"
        "5) 若出现**风险信号**（如持续失眠>2周、强烈绝望/自伤想法等），请温柔提醒寻求专业支持与热线信息（如 Beyond Blue：1300 22 4636；Lifeline：13 11 14）。\n"
        "6) 全文不涉及卡路里、训练、跑步、HIIT 等任何运动建议。\n"
        "如用到知识库内容请以 [1]、[2] 标注。\n"
        f"\n用户消息：{text}\n"
    )

    # 兜底文本
    fallback = (
        "你现在承受着不少压力，这很不容易。我听见你在担心、也在努力撑着。先不急着解决所有问题，"
        "我们先把身心稳住：\n\n"
        "**3 分钟稳态练习**\n"
        "- 30 秒着地：感受脚掌或坐骨的触地点，注意身体与地面的接触。\n"
        "- 2 分钟节律呼吸：吸气 4 秒 → 停 4 秒 → 呼气 4 秒 → 停 4 秒，重复 8–10 轮；呼气时在心里默念“松”。\n"
        "- 30 秒感官锚定：环顾四周找 3 个你能看见的事物、2 个能听到的声音、1 个触感。\n\n"
        "**马上能做的小步骤（各<5分钟）**\n"
        "1) 触发：倒一杯温水 → 行动：慢慢喝 5–10 口，感受温度 → 奖励：给自己一个“我在照顾自己”的打勾标记。\n"
        "2) 触发：手机闹钟 → 行动：给信任的人发一句“我现在有点难，能听我说 3 分钟吗？” → 奖励：呼吸 4-6 两轮。\n"
        "3) 触发：桌面便签 → 行动：写下明天想完成的**最小任务**（可在 10–15 分钟内完成）→ 奖励：播放一首轻音乐。\n\n"
        "**自我关怀与边界**\n"
        "- 今晚尽量提前上床 30 分钟，睡前做 3 分钟呼吸；\n"
        "- 给自己一个“信息止损点”：睡前 1 小时不刷工作相关信息；\n"
        "- 若连续两周情绪低落、睡不好，或出现自伤/无望的念头，请尽快联系专业支持：\n"
        "  • Beyond Blue：1300 22 4636    • Lifeline：13 11 14\n"
        "你已经在努力了，允许自己慢下来，一次只迈一小步就好。"
    )

    content = _safe_llm_answer(prompt, ctx, srcs, fallback)
    state["messages"].append(AIMessage(content=content))
    return state

def greet_node(state: AgentState):
    """
    用 LLM 生成自然语言问候与自我介绍，替代固定模板。
    不改路由/检索/工具，仅此节点行为变更。
    """
    user_text = _last_text(state)
    profile = PROFILE.load() if "PROFILE" in globals() else {}
    # 用画像里可能的时区（没有就默认悉尼）
    tz = profile.get("tz_id") or "Australia/Sydney"
    try:
        now_tz = datetime.now(pytz.timezone(tz))
    except Exception:
        now_tz = datetime.now(pytz.timezone("Australia/Sydney"))

    example_pool = [
        "悉尼今天天气（或现在悉尼时间）",
        "珀斯和阿德莱德今天对比一下天气",
        "墨尔本未来7天天气及出行建议",
        "帮我算下TDEE：女 26岁 165cm 55kg 中等活动",
        "敏感肌晚上用A醇，白天怎么护肤？",
    ]
    examples = ", ".join(random.sample(example_pool, k=2))

    ctx, srcs = _ctx_and_sources(state, [])

    prompt = f"""
你是 LumaWell，一位“健康与出行”助手。请根据场景，用**自然语言**和用户打招呼并自我介绍，避免模板化、避免列表/标题。
要求：
- 先一句简短问候，并按当地时间段（早/午/晚）选择合适措辞。
- 用 1–2 句概括你能做的事（实时天气/两城对比/1~14天预报 + 户外建议；BMI/TDEE；护肤；饮食与情绪练习），语气轻松亲切。
- 给 1–2 个贴近用户的示例问法（可从“示例库”挑选或改写，不要逐条罗列）。
- 全文 2–4 句，不要使用项目符号或标题。

现在当地时间：{now_tz:%Y-%m-%d %H:%M}（{tz}）
用户刚刚说：“{user_text}”
示例库（供你参考与改写，不要逐字照搬）：{examples}
请用中文输出。
""".strip()

    fallback = "嗨！我是你的健康与出行助手 LumaWell。可以帮你看当地天气与户外时段、做两城对比，或计算 BMI/TDEE、给护肤与饮食建议。想从哪一项开始？"
    text = _safe_llm_answer(prompt, ctx, srcs, fallback)

    state["messages"].append(AIMessage(content=text))
    return state


def idle_node(state: AgentState):
    """
    空输入（只回车/空白）时不回复任何内容。
    注意：不要往 state["messages"] 里 append 消息。
    """
    return state


def medical_agent_node(state: AgentState):
    text = _last_text(state)
    ctx, srcs = _ctx_and_sources(state, [])
    content = _llm_answer(text + "\n\n请做：指标解释（通俗类比）+ 生活方式干预 + 何时就医/复查提示（非诊断）。", ctx, srcs)
    state["messages"].append(AIMessage(content=content))
    return state

def environment_agent_node(state: AgentState):  # CHANGED
    text = _last_text(state)
    cities = _detect_au_cities(text)
    sub = state.get("sub_intent")
    wants = _wants_advice(text)   # 是否需要建议

     # (A)城市对比动态未来预报
    if sub in ["forecast_1","forecast_3","forecast_7","forecast_14"] and len(cities) >= 2:
        c1, c2 = cities[:2]
        days_map = {"forecast_1": 1, "forecast_3": 3, "forecast_7": 7, "forecast_14": 14}
        days = int(state.get("days_requested", days_map[sub]))
        offset = int(state.get("start_offset", 0))
        fetch_days = min(max(days + offset, days, 1), 14)
        fc1 = get_weather_forecast(c1, days=fetch_days)
        fc2 = get_weather_forecast(c2, days=fetch_days)

        days1 = fc1.get("forecast_days", [])
        days2 = fc2.get("forecast_days", [])
        start = min(max(offset, 0), max(len(days1) - 1, 0), max(len(days2) - 1, 0))
        end = min(start + max(days, 1), len(days1), len(days2))
        disp1 = days1[start:end]
        disp2 = days2[start:end]

        # 任一失败就回退各自实况卡（不写建议）
        if "error" in fc1 or "error" in fc2:
            out = []
            for city in [c1, c2]:
                w = get_weather_realtime(city)
                if "error" in w:
                    continue
                dt = dt_from_weather(w.get("localtime_epoch"), w.get("tz_id"))
                out.append(_render_observation_card(
                    f"{w.get('name')}, {w.get('region')}",
                    w.get("tz_id") or "Australia/Sydney",
                    dt,
                    w.get("condition","N/A"),
                    w.get("temp_c") or 22.0,
                    w.get("humidity"),
                    w.get("uv_index") or 6.0,
                    w.get("aqi") or 40
                ))
            if out:
                state["messages"].append(AIMessage(content="\n\n".join(out)))
            return state

        name1 = f"{fc1.get('name')}, {fc1.get('region')}"
        name2 = f"{fc2.get('name')}, {fc2.get('region')}"

        # —— 标题（去掉时区串联）——
        title = f"### {name1} vs {name2} — 未来{len(disp1)}天逐日对比（从{['今天','明天','后天','大后天'][offset] if offset<4 else f'第{offset}天起'}）"

        # —— 每行把“同一日期的两城”放一行，列更少更清晰 —— 
        headers = ["日期", name1, name2]
        rows = []

        for d1, d2 in zip(disp1, disp2):
            def pack(d):
                cond = _clean_cond(d.get("condition"))
                t    = _fmt_tpair(d.get("maxtemp_c"), d.get("mintemp_c"))
                avgT = f"{d.get('avgtemp_c','-')}°C" if d.get('avgtemp_c') is not None else "-"
                hum  = f"{int(round(float(d.get('avghumidity',0))))}%" if d.get('avghumidity') is not None else "-"
                r    = _fmt_rain(d.get("daily_chance_of_rain", 0))
                uv   = _fmt_uv(d.get("uv"))
                w    = _fmt_kph(d.get("maxwind_kph"))
                vis  = f"{d.get('avgvis_km','-')}km" if d.get('avgvis_km') is not None else "-"
                pres = (str(d.get("pressure_mb")) + "mb") if d.get("pressure_mb") is not None else "-"
                # 打包成一列文字（便于两城并排对比）
                return f"{cond} · {t} · 平均{avgT} · 湿度 {hum} · Rain {r} · Vis {vis} · Wind {w} · UV {uv} · P {pres}"
            rows.append([d1["date"], pack(d1), pack(d2)])

        table_text = title + "\n" + _mk_table(headers, rows, align=["<","<","<"])  # ← 去掉 max_col_widths
        if wants:
            ctx, srcs = _ctx_and_sources(state, [{
                "tool": "forecast_compare",
                "input": {"c1": name1, "c2": name2, "days": days, "offset": offset},
                "result": {"rows": len(disp1), "c1_days": disp1, "c2_days": disp2}
            }])

            payload = json.dumps({
                "city1": name1, "days1": disp1,
                "city2": name2, "days2": disp2
            }, ensure_ascii=False)

            compare_prompt = (
                f"以下是两个城市未来{len(disp1)}天逐日数据（JSON）：\n{payload}\n\n"
                "请完成：\n"
                "1) 为每座城市标出**更适合户外**的日期与建议时段（早/晚），给出理由（降雨概率/风/温度/UV）。\n"
                "2) 标出**需要规避**的日期（强风/强降雨/极端温度/高UV），列出触发阈值与原因。\n"
                "3) 给出两城**差异化**的防晒/补水/装备建议（例如 SPF 数值、每小时补水量、是否带风壳/雨具/防晒衣）。\n"
                "用中文要点式输出。"
            )
            llm_text = _safe_llm_answer(compare_prompt, ctx, srcs, table_text + "\n\n（说明：模型暂未响应，已提供对比表）")
            _append_once(state, table_text, llm_text)
        else:
            _append_once(state, table_text)
        return state

        # ----------(B) 动态未来预报 ----------
    if sub in ["forecast_1", "forecast_3", "forecast_7", "forecast_14"]:
        city_query = cities[0]
        days_map = {"forecast_1": 1, "forecast_3": 3, "forecast_7": 7, "forecast_14": 14}

        req_days = int(state.get("days_requested", days_map[sub]))
        offset = int(state.get("start_offset", 0))  # 新增：从 offset 开始展示

        # 为了拿得到 offset 之后的窗口，需要多取一些天数
        fetch_days = min(max(req_days + offset, req_days, 1), 14)
        fc = get_weather_forecast(city_query, days=fetch_days)

        if "error" in fc:
            # 退回到实时卡 + 提示
            weather = get_weather_realtime(city_query)
            tz_id = (weather.get("tz_id") if "error" not in weather else "Australia/Sydney")
            dt = dt_from_weather(weather.get("localtime_epoch"), tz_id)
            city_label = city_query if "error" in weather else f"{weather.get('name')}, {weather.get('region')}"
            obs = _render_observation_card(city_label, tz_id, dt,
                                        weather.get("condition","N/A"),
                                        weather.get("temp_c") or 22.0,
                                        weather.get("humidity"),
                                        weather.get("uv_index") or 6.0,
                                        weather.get("aqi") or 40)
            note = f"（未能获取{req_days}天预报：{fc['error']}）"
            state["messages"].append(AIMessage(content=obs + "\n" + note))
            return state

        # ---------- 输出阶段1：纯数据表格 ----------
        city_label = f"{fc.get('name')}, {fc.get('region')}"
        tz_id = fc.get("tz_id") or "Australia/Sydney"

        all_days = fc.get("forecast_days", [])  # 天气服务通常 day[0] = 今天
        if not all_days:
            state["messages"].append(AIMessage(content=f"暂时没有 {city_label} 的预报数据。"))
            return state

        # 依据 offset 截取：从 offset 开始，取 req_days 天（防溢出）
        start = min(max(offset, 0), max(len(all_days) - 1, 0))
        end = min(start + max(req_days, 1), len(all_days))
        display_days = all_days[start:end]
        title_days = len(display_days)

        # 标题后缀（可读性增强）
        if offset == 1:
            title_suffix = "（从明天起）"
        elif offset == 2:
            title_suffix = "（从后天起）" if title_days > 1 else "（仅后天）"
        elif offset == 3:
            title_suffix = "（从大后天起）" if title_days > 1 else "（仅大后天）"
        else:
            title_suffix = ""

        title = f"### {city_label} — 未来{title_days}天预报{title_suffix}（本地时区）"
        headers = ["日期", "天气", "高/低", "平均气温", "湿度", "降雨", "降水", "能见度", "风速", "UV", "气压"]
        rows = []
        for d in display_days:
            rows.append([
                d["date"],
                _clean_cond(d.get("condition")),
                _fmt_tpair(d.get("maxtemp_c"), d.get("mintemp_c")),
                f"{d.get('avgtemp_c','-')}°C" if d.get('avgtemp_c') is not None else "-",
                f"{int(round(float(d.get('avghumidity',0))))}%" if d.get('avghumidity') is not None else "-",
                _fmt_rain(d.get("daily_chance_of_rain", 0)),
                _fmt_mm(d.get("totalprecip_mm")),
                f"{d.get('avgvis_km','-')}km" if d.get('avgvis_km') is not None else "-",
                _fmt_kph(d.get("maxwind_kph")),
                _fmt_uv(d.get("uv")),
                (str(d.get("pressure_mb")) + "mb") if d.get("pressure_mb") is not None else "-",
            ])

        table_text = title + "\n" + _mk_table(headers, rows, align=["<","<",">",">",">",">",">",">",">",">",">"])
        state["messages"].append(AIMessage(content=table_text))
            
        # ---------- 输出阶段2：LLM建议（原逻辑保留） ----------
        if _wants_advice(text):
            ctx, srcs = _ctx_and_sources(state, [{
                "tool": "forecast",
                "input": {"city": city_query, "days": req_days, "offset": offset},
                "result": {"records": len(display_days)}
            }])
            summary_prompt = (
                f"以下是{city_label}未来{title_days}天（offset={offset}）的天气数据，请分析：\n"
                f"{json.dumps(display_days, ensure_ascii=False)}\n\n"
                "请输出：\n"
                "1) 天气趋势（温度、降雨、UV变化）\n"
                "2) 适合户外活动的日期与时段\n"
                "3) 需规避的天气与原因\n"
                "4) 防晒与装备建议。\n"
                "输出格式应清晰分层，用要点式中文说明。"
            )
            fallback = table_text + "\n\n（说明：模型生成暂时不可用，已提供逐日表格供参考）"
            advice_text = _safe_llm_answer(summary_prompt, ctx, srcs, fallback)
            state["messages"][-1] = AIMessage(content=(table_text + "\n\n" + advice_text).strip())
        return state


       # ---------- (C) 两城对比（确定性 + LLM增强） ----------
    if sub == "compare" and len(cities) >= 2:
        c1, c2 = cities[:2]
        w1 = get_weather_realtime(c1)
        w2 = get_weather_realtime(c2)

        def _norm(w, default_city):
            if "error" in w:
                return {
                    "city": default_city,
                    "tz_id": "Australia/Sydney",
                    "dt": dt_from_weather(None, "Australia/Sydney"),
                    "temp_c": 22.0, "uv": 6.0, "aqi": 40,
                    "cond": f"N/A（{w['error']}）",
                    "humidity": None,
                }
            dt = dt_from_weather(w.get("localtime_epoch"), w.get("tz_id"))
            return {
                "city": f"{w.get('name')}, {w.get('region')}（{w.get('tz_id')}）",
                "tz_id": w.get("tz_id"),
                "dt": dt,
                "temp_c": w.get("temp_c"),
                "uv": w.get("uv_index"),
                "aqi": w.get("aqi"),
                "cond": w.get("condition", "N/A"),
                "humidity": w.get("humidity"),
            }

        a = _norm(w1, c1)
        b = _norm(w2, c2)

        title = f"### 今日天气对比"
        headers = ["指标", a["city"], b["city"]]
        rows = [
            ["本地时间", f"{a['dt']:%Y-%m-%d %H:%M}", f"{b['dt']:%Y-%m-%d %H:%M}"],
            ["天气",      _clean_cond(a["cond"]),       _clean_cond(b["cond"])],
            ["气温",      f"{a['temp_c']}°C",           f"{b['temp_c']}°C"],
            ["湿度",      "-" if a['humidity'] is None else f"{a['humidity']}%",
                        "-" if b['humidity'] is None else f"{b['humidity']}%"],
            ["UV",        _fmt_uv(a["uv"]),             _fmt_uv(b["uv"])], 
            ["AQI",       a["aqi"],                     b["aqi"]],
        ]

        table_text = title + "\n" + _mk_table(headers, rows, align=["<","<","<"])  # 去掉 max_col_widths
        # 为了标题更干净，这里把 a["city"] 中的 “（tz_id）” 去掉，仅保留 “城市, 州/省”
        a_label = a["city"].split("（")[0]
        b_label = b["city"].split("（")[0]

        full_a = "" if "error" in w1 else _render_full_realtime_table(
            a_label,
            w1,
            show_header=True,
            dedupe_basics=False,   # 显示“全部指标”：基础项 + 扩展项
            dt=a["dt"],            # 表头前加“本地时间：YYYY-MM-DD HH:MM（tz）”
            tz_id=a["tz_id"],
            show_time=True
        )
        full_b = "" if "error" in w2 else _render_full_realtime_table(
            b_label,
            w2,
            show_header=True,
            dedupe_basics=False,
            dt=b["dt"],
            tz_id=b["tz_id"],
            show_time=True
        )
        full_block = "\n\n".join([full_a, full_b]).strip()

        if not wants:
            _append_once(state, table_text, full_block)
            return state

        # 工具建议
        advice_a = uv_aqi_advice_tool(a["temp_c"], a["uv"], a["aqi"])
        advice_b = uv_aqi_advice_tool(b["temp_c"], b["uv"], b["aqi"])
        advice_text = (
            f"**{a['city']}**\n"
            f"- 今日最佳户外时段：{advice_a['best_window']}\n"
            f"- 注意事项：{'；'.join(advice_a['notes'])}\n\n"
            f"**{b['city']}**\n"
            f"- 今日最佳户外时段：{advice_b['best_window']}\n"
            f"- 注意事项：{'；'.join(advice_b['notes'])}\n"
        )

        # LLM 综合分析
        ctx, srcs = _ctx_and_sources(state, [{
            "tool": "compare_weather",
            "input": {"city1": a["city"], "city2": b["city"]},
            "result": {"a": a, "b": b}
        }])
        compare_prompt = (
            f"以下是两个城市的天气数据：\n{table_text}\n\n"
            "请分析：\n"
            "1) 哪个城市今日更适合户外活动？\n"
            "2) 各自需要注意哪些风险（UV、空气质量、温度等）？\n"
            "3) 若用户计划通勤/出行，请给出差异化建议。\n"
            "请用中文分条说明。"
        )
        fallback = table_text + "\n\n" + advice_text + "\n（说明：模型暂未响应，已提供工具对比结果。）"
        llm_text = _safe_llm_answer(compare_prompt, ctx, srcs, fallback)

        # 只发一条消息：表格 + 工具建议 + LLM 分析
        _append_once(state, table_text, full_block, advice_text, llm_text)
        return state

    # ---------- (D) 单城（含“只问时间/日期”） ----------
    city_query = cities[0]
    weather = get_weather_realtime(city_query)

    if "error" in weather:
        temp_c, uv_index, aqi_val = 22.0, 6.0, 40
        tz_id, local_epoch = "Australia/Sydney", None
        city_label = city_query
        cond = "N/A"
        humidity = None
    else:
        temp_c  = weather.get("temp_c")   if weather.get("temp_c")   is not None else 22.0
        uv_index = weather.get("uv_index") if weather.get("uv_index") is not None else 6.0
        aqi_val = weather.get("aqi")      if weather.get("aqi")      is not None else 40
        tz_id = weather.get("tz_id") or "Australia/Sydney"
        local_epoch = weather.get("localtime_epoch")
        city_label = f"{weather.get('name')}, {weather.get('region')}"
        cond = weather.get("condition","N/A")
        humidity = weather.get("humidity")

    local_dt = dt_from_weather(local_epoch, tz_id)
    fc_today = get_weather_forecast(city_query, days=1)
    today_block = ""
    if "error" not in fc_today and fc_today.get("forecast_days"):
        today_block = _render_full_forecast_day_table(
            f"{fc_today.get('name')}, {fc_today.get('region')}",
            fc_today["forecast_days"][0]
        )

    # 实况全指标（显示本地时间；并包含基础项=“全部指标”）
    full_table = ""
    if "error" not in weather:
        full_table = _render_full_realtime_table(
            city_label,
            weather,
            show_header=True,
            dedupe_basics=False,     # 包含基础项：天气/气温/湿度/UV/AQI
            dt=local_dt,             # 显示本地时间
            tz_id=tz_id,
            show_time=True
        )

    # 组合输出（先“日级完整”，再“实时全指标”）
    state["messages"].append(AIMessage(content="\n\n".join([b for b in [today_block, full_table] if b]).strip()))
    if not wants:
        return state
    obs_table = today_block or full_table


    # === LLM 增值 ===
    advice = uv_aqi_advice_tool(temp_c, uv_index, aqi_val)
    tool_out = [{
        "tool": "weatherapi→outdoor",
        "input": {"city": city_query, "temp_c": temp_c, "uv": uv_index, "aqi": aqi_val},
        "result": advice
    }]
    ctx, srcs = _ctx_and_sources(state, tool_out)

    SHOW_FALLBACK_NOTE = os.getenv("SHOW_FALLBACK_NOTE", "1")  
    fallback_note = "" if SHOW_FALLBACK_NOTE == "0" else "\n（说明：模型生成暂时不可用，已用工具规则直接给到可执行建议）"  # NEW

    fallback = (
        obs_table + "\n\n" + full_table + "\n"
        f"**今日最佳户外时段**：{advice['best_window']}\n"
        f"**注意事项**：{'；'.join(advice['notes'])}\n"
        f"{fallback_note}"
    )


    prompt_user = (
    "请基于下面两张表给出**可直接照做**的建议。\n"
    "— 表1：实况概览（人类可读）\n"
    f"{obs_table}\n\n"
    "— 表2：实况全指标（机器可读、字段更全）\n"
    f"{full_table}\n\n"
    f"核心数值提示：温度 {temp_c}°C，UV {uv_index}，AQI {aqi_val}。\n\n"
    "请输出：\n"
    "1) 今日最佳户外时段（给到具体钟点范围并简述理由：温度/风/UV/降水）\n"
    "2) 防晒/补水/装备清单（SPF/UPF、每小时补水量、是否带风壳/雨具/墨镜/遮阳帽等）\n"
    "3) 适合的自然疗愈活动（举1–3个，说明为何适合当前条件）\n"
    "4) 若有风险（高UV、强风、降水、空气质量偏差等），给出规避方法或替代方案。\n"
    "要求：引用上述具体数值；用要点式中文输出；如匹配到户外手册/装备/防晒等知识库，请以 [1]、[2]… 标注。"
)

    content = _safe_llm_answer(prompt_user, ctx, srcs, fallback)

    NEEDLES = ["最佳户外时段", "注意事项", "防晒", "补水", "装备", "自然疗愈"] 
    if not any(n in content for n in NEEDLES):  
        extra = (
            f"\n\n**今日最佳户外时段**：{advice['best_window']}\n"
            f"**注意事项**：{'；'.join(advice['notes'])}\n"
        )
        content = content.strip() + extra  

    state["messages"].append(AIMessage(content=content))
    return state

# ===== 通用 RAG =====
def rag_node(state: AgentState):
    text = _last_text(state)
    ctx, srcs = _ctx_and_sources(state, [])
    content = _llm_answer(text, ctx, srcs)
    state["messages"].append(AIMessage(content=content))
    return state

# ===== 安全节点 =====
def safety_node(state: AgentState):
    msg = ("你描述的情况可能涉及医疗紧急/诊断范畴。此对话仅供健康建议学习，"
           "不替代专业医疗。若出现急性症状或不适，请立即联系当地急救/全科医生。")
    state["messages"].append(AIMessage(content=msg))
    return state

def _bmi_weight_range(height_cm: float) -> Tuple[float, float]:
    """按 BMI 18.5–24 计算对应健康体重区间（kg，保留1位小数）"""
    h = max(height_cm / 100.0, 0.5)  # 防御
    w_min = round(18.5 * h * h, 1)
    w_max = round(24.0 * h * h, 1)
    return w_min, w_max


def _bmi_advice_fallback(parsed: "BMIInput", res: Dict, wrange: Tuple[float, float]) -> str:
    """当 LLM 不可用时的稳妥建议（基于 BMI 分类给出可执行方案）"""
    bmi = res.get("bmi")
    cat = (res.get("category") or "").strip()
    w_min, w_max = wrange
    h = parsed.height_cm
    w = parsed.weight_kg

    # 基于分类给出热量方向与侧重
    if "偏低" in cat or "过低" in cat or "Under" in cat:
        goal = f"增重到 {w_min}–{w_max} kg 区间内"
        kcal_tip = "每日较维持热量 +300~400 kcal，优先高蛋白+高能量密度（坚果、全脂奶、橄榄油）"
        train_tip = "优先力量训练（每周3次，推/拉/腿或全身），RPE 7–8，渐进加重；有氧轻到中等每周1–2次维持心肺"
    elif "正常" in cat or "Normal" in cat:
        goal = f"维持在 {w_min}–{w_max} kg，体态重塑（增肌减脂）"
        kcal_tip = "围绕维持热量±100~150 kcal 微调；蛋白 ≥1.6 g/kg·d"
        train_tip = "力量训练每周3–4次（全身或上下肢分化），复合动作为主；有氧每周2–3次（30–40min 中等强度）"
    else:
        goal = f"逐步减重至 {w_max} kg 附近（或医嘱目标）"
        kcal_tip = "每日较维持热量 -300~500 kcal，蛋白 1.6–2.0 g/kg·d，优先高纤全谷"
        train_tip = "力量训练每周3次维持肌肉（深蹲/硬拉/卧推/划船/推举）；有氧每周3–5次（快走或骑行 30–45min）"

    text = (
        f"### BMI 建议（fallback）\n"
        f"- 身高：{h} cm；体重：{w} kg；BMI：{bmi}；分类：{cat}\n"
        f"- 健康体重区间（BMI 18.5–24）：约 **{w_min}–{w_max} kg**\n\n"
        f"**目标**：{goal}\n"
        f"**热量原则**：{kcal_tip}\n\n"
        f"**一周训练模板**：\n"
        f"- 力量（3次）：全身或推-拉-腿；每次 5–6 个动作 × 3–4 组 × 6–12 次，RPE 7–8；组间休息 60–120s\n"
        f"- 有氧（2–4次）：快走/椭圆/骑行 30–45min，中等强度（说话略喘）\n"
        f"- 机动：每天 10–15min 关节灵活性与拉伸（髋、踝、肩），久坐人群优先髋屈伸与胸椎伸展\n\n"
        f"**饮食结构**：\n"
        f"- 蛋白：≥1.6 g/kg·d（鸡胸/鱼/瘦牛/鸡蛋/希腊酸奶/豆制品）\n"
        f"- 碳水：以全谷和高纤为主（燕麦、糙米、全麦面包、土豆/红薯）\n"
        f"- 脂肪：坚果、牛油果、橄榄油；减少深加工与反式脂肪\n"
        f"- 简单餐例：\n"
        f"  • 早：燕麦+牛奶+酸奶+浆果；\n"
        f"  • 午：鸡胸/三文鱼+糙米+蔬菜沙拉；\n"
        f"  • 晚：瘦牛/豆腐+全麦意面/土豆+时蔬；\n"
        f"  • 加餐：香蕉/坚果/蛋白酸奶\n\n"
        f"**生活方式**：\n"
        f"- NEAT：日步数 7k–10k；每坐 45–60min 起身活动 2–3min\n"
        f"- 睡眠：7–9h，固定就寝与起床时间；水分：体重×30–40 ml/d，运动天补电解质\n"
        f"- 监测：体重每周波动正常，建议以 4 周移动平均看趋势；训练记录重量/次数做渐进\n"
    )
    return text

# ===== 通用数值与表格渲染工具（统一所有表格的对齐风格） =====
def _fmt_uv(x):
    # 改：不再取整，原样输出（含小数）
    return "-" if x is None else str(x)

def _fmt_kph(x):       return "-" if x is None else f"{int(round(float(x)))}kph"
def _fmt_mm(x):        return "-" if x is None else (f"{float(x):.1f}mm" if isinstance(x,(int,float,str)) else "-")
def _fmt_rain(p):      return "-" if p is None else f"{int(round(float(p)))}%"
def _fmt_tpair(hi, lo):return f"{int(round(float(hi)))}°/{int(round(float(lo)))}°"
def _clean_cond(s):    return (s or "").replace("|", "/").strip()

def _ellipsis(s, width):
    s = "-" if s is None else str(s)
    return s if len(s) <= width else (s[:max(1,width-1)] + "…")

def _mk_table(headers, rows, align=None, max_col_widths=None):
    """
    生成等宽 ASCII 表格，并用代码块包裹。
    - 默认不做任何截断；只有显式提供 max_col_widths 时，才对相应列做省略号。
    """
    headers = [str(h) for h in headers]
    n = len(headers)
    align = align or ["<"] * n
    max_col_widths = max_col_widths or [None] * n

    # 1) 按内容计算真正的列宽（不设上限）
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            w = len(str(cell))
            if w > widths[i]:
                widths[i] = w

    # 2) 渲染时，仅当该列设置了上限，才进行省略
    def fmt_cell_raw(s, i):
        a = align[i]
        return f"{s:{a}{widths[i]}}"

    def fmt_cell(s, i):
        s = "-" if s is None else str(s)
        if max_col_widths[i] is not None:
            s = _ellipsis(s, max_col_widths[i])
        return fmt_cell_raw(s, i)

    head = " | ".join(fmt_cell(h, i) for i, h in enumerate(headers))
    sep  = "-+-".join("-" * widths[i] for i in range(n))
    body = "\n".join(" | ".join(fmt_cell(str(c), i) for i, c in enumerate(r)) for r in rows)
    return "```\n" + head + "\n" + sep + "\n" + body + "\n```"

# ====== 实况全指标（尽量列出 WeatherAPI 实时里可能出现的关键字段；缺失则跳过） ======
def _fmt(s):  # 通用字符串化
    return "-" if s is None else str(s)

def _join_nonempty(*parts):
    return " ".join([p for p in parts if p and str(p).strip() and p != "-"])

def _render_full_realtime_table(
    city_label: str,
    w: Dict,
    show_header: bool = True,
    dedupe_basics: bool = True,
    basics: Tuple[str, ...] = ("天气", "气温", "湿度", "UV", "AQI"),
    dt: Optional[datetime] = None,
    tz_id: Optional[str] = None,
    show_time: bool = False,
) -> str:
    # ---- 取值 ----
    temp_c      = w.get("temp_c")
    feelslike_c = w.get("feelslike_c") or w.get("feels_like_c") or w.get("apparent_temp_c")
    cond        = w.get("condition") or w.get("text") or w.get("weather")
    humidity    = w.get("humidity")
    uv          = w.get("uv_index") if w.get("uv_index") is not None else w.get("uv")
    aqi         = w.get("aqi")
    wind_kph    = w.get("wind_kph") or w.get("wind_speed_kph")
    wind_dir    = w.get("wind_dir") or w.get("wind_direction")
    gust_kph    = w.get("gust_kph") or w.get("gust_speed_kph")
    pressure_mb = w.get("pressure_mb") or w.get("pressure") or w.get("pressure_hpa")
    cloud       = w.get("cloud")
    vis_km      = w.get("vis_km") or w.get("visibility_km")
    dew_c       = w.get("dewpoint_c") or w.get("dew_point_c")
    precip_mm   = w.get("precip_mm") or w.get("precip_1hr_mm") or w.get("rain_1h_mm")

    # ---- 基础项（只定义一次，不要重复 append）----
    base_rows = [
        ["天气",  _fmt(_clean_cond(cond))],
        ["气温",  _join_nonempty(_fmt(temp_c) + "°C", "(体感 " + _fmt(feelslike_c) + "°C)" if feelslike_c is not None else "")],
        ["湿度",  (_fmt(humidity) + "%") if humidity is not None else "-"],
        ["UV",    _fmt_uv(uv)],
        ["AQI",   _fmt(aqi)],
    ]

    # ---- 扩展项 ----
    extra_rows = []
    if wind_kph is not None:     extra_rows.append(["风速/向", _join_nonempty(_fmt(int(round(float(wind_kph)))) + "kph", _fmt(wind_dir))])
    if gust_kph is not None:     extra_rows.append(["阵风",    _fmt(int(round(float(gust_kph)))) + "kph"])
    if pressure_mb is not None:  extra_rows.append(["气压",    _fmt(pressure_mb) + "mb"])
    if cloud is not None:        extra_rows.append(["云量",    _fmt(cloud) + "%"])
    if vis_km is not None:       extra_rows.append(["能见度",  _fmt(vis_km) + "km"])
    if dew_c is not None:        extra_rows.append(["露点",    _fmt(dew_c) + "°C"])
    if precip_mm is not None:    extra_rows.append(["近降水",  _fmt(precip_mm) + "mm"])

    rows = extra_rows if dedupe_basics else (base_rows + extra_rows)
    if not rows:
        return ""

    time_line = f"本地时间：{dt:%Y-%m-%d %H:%M}（{tz_id or ''}）\n" if (show_time and isinstance(dt, datetime)) else ""
    head = f"#### {city_label} —— 实况全指标" if show_header else ""
    block = _mk_table(["指标","数值"], rows, align=["<","<"])
    return (time_line + (head + "\n" if head else "") + block).strip()

# ====== 单天“预报-日级”完整指标表（用于今天 or 未来的每天） ======
def _render_full_forecast_day_table(city_label: str, d: Dict) -> str:
    """
    用 forecast 返回的单日记录 d，渲染尽量全的“日级指标”表。
    兼容 key 不存在的情况（用 '-'）。
    """
    def g(k, default="-"):
        v = d.get(k)
        return "-" if v is None else v

    rows = [
        ["天气",         _clean_cond(g("condition"))],
        ["高/低",        _fmt_tpair(g("maxtemp_c"), g("mintemp_c"))],
        ["平均气温",     f"{g('avgtemp_c')}°C" if g('avgtemp_c') != "-" else "-"],
        ["相对湿度(均值)", f"{int(round(float(g('avghumidity'))))}%" if g('avghumidity') not in (None,"-") else "-"],
        ["降雨概率",     _fmt_rain(g("daily_chance_of_rain", 0))],
        ["降水量(总)",   _fmt_mm(g("totalprecip_mm"))],
        ["能见度(均值)", f"{g('avgvis_km')}km" if g('avgvis_km') not in (None,"-") else "-"],
        ["最大风速",     _fmt_kph(g("maxwind_kph"))],
        ["UV",           _fmt_uv(g("uv"))],
        # 下面几个字段不同数据源可能没有，尽量取到就显示
        ["气压(均值)",   f"{g('pressure_mb')}mb" if g('pressure_mb') not in (None,"-") else "-"],
        ["云量(均值)",   f"{int(round(float(g('cloud'))))}%" if g('cloud') not in (None,"-") else "-"],
        ["露点(均值)",   f"{g('dewpoint_c')}°C" if g('dewpoint_c') not in (None,"-") else "-"],
    ]
    head = f"#### {city_label} —— 日级完整指标（{d.get('date','未知日期')}）"
    return head + "\n" + _mk_table(["指标","数值"], rows, align=["<","<"])
