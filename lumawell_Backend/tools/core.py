# tools/core.py
from typing import List, Literal, Dict
import os, requests

# BMI 工具
def bmi_tool(height_cm: float, weight_kg: float) -> Dict:
    h = height_cm / 100.0
    bmi = round(weight_kg / (h * h), 2)
    if bmi < 18.5:
        cat = "偏瘦"
    elif bmi < 24:
        cat = "正常"
    elif bmi < 28:
        cat = "超重"
    else:
        cat = "肥胖"
    return {"bmi": bmi, "category": cat}

# TDEE 工具（Mifflin-St Jeor）
def tdee_tool(
    sex: Literal["male", "female"],
    age: int,
    height_cm: float,
    weight_kg: float,
    activity_level: Literal["sedentary","light","moderate","active","athlete"]
) -> Dict:
    # BMR
    if sex == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    factors = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "athlete": 1.9,
    }
    tdee = round(bmr * factors[activity_level])
    return {"bmr": round(bmr), "tdee": tdee, "activity_factor": factors[activity_level]}

# 护肤配伍规则
RULES = [
    ("retinol", "strong_acid", "A醇与强酸同用易刺激，建议分时段或隔天用"),
    ("vitc", "niacinamide", "部分配方/高浓度条件下 VC×烟酰胺可能致刺激，分时段更稳妥"),
    ("aha_bha", "benzoyl_peroxide", "酸类与过氧化苯甲酰同用易过度刺激，建议分开"),
]

SAFE_SUGGEST = {
    "dry": "优先补水保湿，温和清洁；酸类频次从低",
    "oily": "轻薄保湿、控油，防晒要充足；酸类逐步耐受",
    "sensitive": "避高刺激（强酸、高浓度A醇），先修护屏障再上活性",
}

def skincare_tool(skin_type: str, concerns: str, actives_in_use: str) -> Dict:
    actives = {a.strip().lower() for a in actives_in_use.split(',') if a.strip()}
    conflicts = []
    for a, b, tip in RULES:
        if a in actives and b in actives:
            conflicts.append({"a": a, "b": b, "tip": tip})
    hint = SAFE_SUGGEST.get(skin_type.lower(), "日常温和清洁+保湿+防晒是底线")
    return {"conflicts": conflicts, "general": hint, "concerns": concerns}

# ====== 心情 → 运动匹配 ======
def mood_to_workout_tool(mood: Literal["anxious", "low", "neutral", "excited"]) -> Dict:
    """
    根据语气/心情匹配训练类型与强度建议。
    """
    mapping = {
        "anxious": {
            "primary": "低到中强度有氧 + 伸展/瑜伽",
            "why": "降低交感兴奋，稳定呼吸与心率",
            "session": "快走20–30min + 10min box breathing/4-4-4-4"
        },
        "low": {
            "primary": "阳光暴露 + 轻力量循环",
            "why": "提升多巴胺/去甲肾上腺素水平与自我效能感",
            "session": "全身循环(徒手深蹲/俯卧撑/髋桥/划船) 各12次 × 3轮"
        },
        "neutral": {
            "primary": "常规力量 + 中等有氧",
            "why": "维持训练节律，刺激肌肥大与心肺",
            "session": "上/下肢分化 45–60min；RPE 7–8"
        },
        "excited": {
            "primary": "高质量力量/HIIT（注意量控）",
            "why": "利用动机高峰推进强度；避免过度训练",
            "session": "复合举（硬拉/深蹲/卧推/划船）+ 8×20s 间歇跑"
        }
    }
    return mapping[mood]

# 获取实时天气
def get_weather_realtime(city_query: str = "Sydney,AU") -> Dict:
    """
    仅使用 WeatherAPI 的 current.json（Pro/Pro+ 均可）
    返回尽可能多的“实时”字段，便于前端/渲染使用。
    """
    key = os.getenv("WEATHERAPI_KEY")
    if not key:
        return {"error": "missing WEATHERAPI_KEY"}

    try:
        url = "https://api.weatherapi.com/v1/current.json"
        params = {
            "key": key,
            "q": city_query,
            "aqi": "yes",   # 必须：空气质量字段
            "lang": "zh"    # 可选：中文返回
        }
        r = requests.get(url, params=params, timeout=12)
        j = r.json()
        if "error" in j:
            return {"error": f"weatherapi: {j['error'].get('message', 'unknown')}"}

        loc = j.get("location", {}) or {}
        cur = j.get("current", {}) or {}
        air = cur.get("air_quality", {}) or {}

        # 继续保留你之前的“指数→粗略数值”兼容，用于 uv_aqi_advice_tool 的阈值
        us_index = air.get("us-epa-index")  # 1~6
        aqi_for_app = int((us_index or 2) * 30 + 20)  # ≈ 50~200 粗映射

        return {
            # ——— 位置/时间 ———
            "name": loc.get("name"),
            "region": loc.get("region"),
            "country": loc.get("country"),
            "tz_id": loc.get("tz_id"),
            "localtime": loc.get("localtime"),
            "localtime_epoch": loc.get("localtime_epoch"),

            # ——— 实况基础 ———
            "condition": (cur.get("condition") or {}).get("text", "N/A"),
            "temp_c": cur.get("temp_c"),
            "feelslike_c": cur.get("feelslike_c"),
            "humidity": cur.get("humidity"),
            "uv_index": cur.get("uv"),

            # ——— 风/能见度/气压/云/降水/露点（Pro 可返）———
            "wind_kph": cur.get("wind_kph"),
            "wind_dir": cur.get("wind_dir"),
            "gust_kph": cur.get("gust_kph"),
            "vis_km": cur.get("vis_km"),
            "pressure_mb": cur.get("pressure_mb"),
            "cloud": cur.get("cloud"),
            "precip_mm": cur.get("precip_mm"),
            "dewpoint_c": cur.get("dewpoint_c"),  # 有些套餐/地区才返回，没就 None

            # ——— 空气质量（两套：兼容用 aqi + 原始明细）———
            "aqi": aqi_for_app,                 # 供你现在的建议工具用（80/150 阈值）
            "aqi_us_epa_index": us_index,       # 原始 EPA 1~6 指数（保留）
            "air_quality": {                    # 全量污染物，Pro/Pro+ 提供
                "pm2_5": air.get("pm2_5"),
                "pm10": air.get("pm10"),
                "o3": air.get("o3"),
                "no2": air.get("no2"),
                "so2": air.get("so2"),
                "co": air.get("co"),
                "us-epa-index": us_index,
                "gb-defra-index": air.get("gb-defra-index"),
            },
        }
    except Exception as e:
        return {"error": f"request failed: {type(e).__name__}: {e}"}


# ====== 14天预报 ======
def get_weather_forecast(city_query: str, days: int = 14) -> Dict:
    """
    WeatherAPI forecast.json（支持逐小时）
    将 hour 里的字段做“日均/最大”聚合，补齐 day 缺失的字段：
      - avgvis_km / pressure_mb / cloud / dewpoint_c（均值）
    """
    key = os.getenv("WEATHERAPI_KEY")
    if not key:
        return {"error": "missing WEATHERAPI_KEY"}

    def _safe_mean(nums):
        vals = [float(x) for x in nums if isinstance(x, (int, float)) or (isinstance(x, str) and str(x).strip() != "")]
        if not vals:
            return None
        return sum(vals) / len(vals)

    try:
        url = "https://api.weatherapi.com/v1/forecast.json"
        params = {
            "key": key,
            "q": city_query,
            "days": max(1, min(int(days), 14)),
            "aqi": "yes",
            "alerts": "no",
            "lang": "zh"
        }
        r = requests.get(url, params=params, timeout=15)
        j = r.json()
        if "error" in j:
            return {"error": f"weatherapi: {j['error'].get('message', 'unknown')}"}

        loc = j.get("location", {}) or {}
        fc = j.get("forecast", {}) or {}
        days_raw = fc.get("forecastday", []) or []
        out_days: List[Dict] = []

        for d in days_raw:
            day = (d.get("day") or {})
            astro = (d.get("astro") or {})
            hours = d.get("hour") or []  # 逐小时

            # ——— 从 hour 聚合“日均/日最大” ———
            avg_vis_km    = _safe_mean([h.get("vis_km") for h in hours])
            avg_pressure  = _safe_mean([h.get("pressure_mb") for h in hours])
            avg_cloud     = _safe_mean([h.get("cloud") for h in hours])
            avg_dewpoint  = _safe_mean([h.get("dewpoint_c") for h in hours])

            out_days.append({
                "date": d.get("date"),
                "maxtemp_c": day.get("maxtemp_c"),
                "mintemp_c": day.get("mintemp_c"),
                "avgtemp_c": day.get("avgtemp_c"),
                "uv": day.get("uv"),
                "daily_chance_of_rain": (day.get("daily_chance_of_rain") or 0),
                "totalprecip_mm": day.get("totalprecip_mm"),
                "maxwind_kph": day.get("maxwind_kph"),
                "condition": (day.get("condition") or {}).get("text", "N/A"),
                "sunrise": astro.get("sunrise"),
                "sunset": astro.get("sunset"),

                # —— 补齐给“日级完整指标表”用的字段（来自逐小时的均值）——
                "avgvis_km": None if avg_vis_km is None else round(avg_vis_km, 1),
                "pressure_mb": None if avg_pressure is None else round(avg_pressure, 0),
                "cloud": None if avg_cloud is None else round(avg_cloud, 0),
                "dewpoint_c": None if avg_dewpoint is None else round(avg_dewpoint, 1),
            })
        return {
            "name": loc.get("name"),
            "region": loc.get("region"),
            "country": loc.get("country"),
            "tz_id": loc.get("tz_id"),
            "forecast_days": out_days,
        }
    except Exception as e:
        return {"error": f"request failed: {type(e).__name__}: {e}"}


# ====== UV / 气温 / 空气质量 → 户外建议 ======
def uv_aqi_advice_tool(temp_c: float, uv_index: float, aqi: int) -> Dict:
    """
    纯逻辑工具：根据已知天气要素给出户外窗口 & 防晒/呼吸道建议。
    """
    advice = []
    # 温度舒适度
    if 18 <= temp_c <= 26:
        advice.append("体感舒适，适合户外中等强度运动")
    elif temp_c < 10:
        advice.append("较冷，注意分层与热身延长")
    elif temp_c > 32:
        advice.append("偏热，缩短时长并补水；避免正午时段")

    # UV
    if uv_index >= 8:
        advice.append("UV非常强：11:00–15:00 尽量避免暴晒，SPF50+、帽檐/墨镜")
    elif uv_index >= 3:
        advice.append("UV中等：出门需SPF50+和物理遮阳")
    else:
        advice.append("UV较低：基础防晒仍建议")

    # AQI
    if aqi >= 150:
        advice.append("空气质量较差：减少户外高强度运动，必要时佩戴口罩")
    elif aqi >= 80:
        advice.append("空气质量一般：缩短高强度时段，关注呼吸道不适")
    else:
        advice.append("空气质量良好：可正常户外活动")

    # 最佳时段（简单规则）
    best = "清晨或傍晚（避开正午高UV）"
    return {"best_window": best, "notes": advice}
