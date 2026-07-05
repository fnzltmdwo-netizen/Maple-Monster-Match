from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import re
import hashlib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CSV_PATH = "monsters_ai.csv"
df = pd.read_csv(CSV_PATH).fillna("")


class MatchRequest(BaseModel):
    image_base64: str


COMMON_MONSTER_PENALTY = [
    "단지",
    "달팽이",
    "스포아",
    "슬라임",
    "주황버섯",
    "파란버섯",
    "리본 돼지",
    "울트라 코-크 달팽이",
    "코-크 달팽이",
]

NON_FACE_MONSTERS = [
    "도라지",
    "네펜데스",
    "모래난쟁이",
    "스텀프",
    "고스텀프",
    "엑스텀프",
    "나무",
    "꽃",
    "풀",
    "화분",
    "버섯집",
    "돌",
    "바위",
    "상자",
    "식물",
]

FACE_SHAPES = ["round", "square", "sharp", "long", "small", "wide"]
VIBES = [
    "cute",
    "calm",
    "dark",
    "mysterious",
    "playful",
    "cool",
    "strong",
    "sleepy",
    "bright",
    "elegant",
]


def clean_base64(image_base64: str):
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    return image_base64.strip().replace("\n", "").replace("\r", "").replace(" ", "")


def make_image_hash(image_base64: str):
    cleaned = clean_base64(image_base64)
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()


def stable_tiebreaker(image_hash: str, monster_name: str):
    key = f"{image_hash}-{monster_name}"
    value = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(value[:6], 16) / 0xFFFFFF


def safe_json(text: str):
    text = text.strip()
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        text = match.group(0)
    return json.loads(text)


def get_value(row, possible_cols, default=""):
    for col in possible_cols:
        if col in row.index and str(row[col]).strip():
            return str(row[col]).strip()
    return default


def get_float(row, possible_cols, default=5):
    for col in possible_cols:
        if col in row.index:
            try:
                return float(row[col])
            except:
                pass
    return default


def normalize_choice(value, allowed, default):
    value = str(value).strip().lower()

    if value in allowed:
        return value

    for item in allowed:
        if item in value or value in item:
            return item

    return default


def is_non_face_monster(name: str):
    return any(word in name for word in NON_FACE_MONSTERS)


def is_common_monster(name: str):
    return any(word in name for word in COMMON_MONSTER_PENALTY)


def analyze_user_image(image_base64: str):
    prompt = """
너는 실제 사람 얼굴 사진을 보고 메이플스토리 몬스터 닮은꼴 매칭용 특징만 뽑는 분석기야.

중요:
- 신원/이름/성별/나이 추정 금지
- 외모 비하 금지
- 아래 JSON만 답하기
- 같은 사진이면 최대한 같은 분석이 나오게 일관적으로 판단하기
- 실제 얼굴형 그대로가 아니라, 몬스터와 매칭하기 위한 '캐릭터식 인상'으로 판단하기

face_shape 후보:
round, square, sharp, long, small, wide

vibe 후보:
cute, calm, dark, mysterious, playful, cool, strong, sleepy, bright, elegant

JSON 형식:
{
  "face_shape": "round",
  "vibe": "calm",
  "scores": {
    "cute_level": 1~10,
    "dark_level": 1~10,
    "power_level": 1~10
  },
  "tags": ["round", "calm"],
  "description": "짧은 분위기 설명"
}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
    )

    analysis = safe_json(response.choices[0].message.content)

    face_shape = normalize_choice(
        analysis.get("face_shape", "round"),
        FACE_SHAPES,
        "round"
    )

    vibe = normalize_choice(
        analysis.get("vibe", "calm"),
        VIBES,
        "calm"
    )

    scores = analysis.get("scores", {})

    user = {
        "face_shape": face_shape,
        "vibe": vibe,
        "cute_level": float(scores.get("cute_level", 5)),
        "dark_level": float(scores.get("dark_level", 3)),
        "power_level": float(scores.get("power_level", 4)),
        "description": analysis.get("description", ""),
        "tags": [face_shape, vibe],
    }

    return user


def score_monster(user, row, image_hash):
    name = get_value(row, ["name"], "이름 없음")

    monster_face_shape = normalize_choice(
        get_value(row, ["face_shape"], "round"),
        FACE_SHAPES,
        "round"
    )

    monster_vibe = normalize_choice(
        get_value(row, ["vibe"], "calm"),
        VIBES,
        "calm"
    )

    monster_cute = get_float(row, ["cute_level"], 5)
    monster_dark = get_float(row, ["dark_level"], 3)
    monster_power = get_float(row, ["power_level"], 4)

    score = 0

    # 얼굴형 매칭
    if user["face_shape"] == monster_face_shape:
        score += 38
    elif user["face_shape"] == "round" and monster_face_shape in ["small", "wide"]:
        score += 22
    elif user["face_shape"] in ["small", "wide"] and monster_face_shape == "round":
        score += 22
    elif user["face_shape"] == "sharp" and monster_face_shape == "long":
        score += 18
    else:
        score += 8

    # 분위기 매칭
    if user["vibe"] == monster_vibe:
        score += 32
    elif {user["vibe"], monster_vibe} <= {"calm", "cute", "sleepy", "bright"}:
        score += 18
    elif {user["vibe"], monster_vibe} <= {"dark", "mysterious", "cool", "strong"}:
        score += 18
    else:
        score += 6

    # 숫자 점수 매칭
    score += max(0, 10 - abs(user["cute_level"] - monster_cute)) * 2.3
    score += max(0, 10 - abs(user["dark_level"] - monster_dark)) * 1.7
    score += max(0, 10 - abs(user["power_level"] - monster_power)) * 1.7

    # 과출현 기본몹 약한 감점
    if is_common_monster(name):
        score -= 8

    # 사람 얼굴과 잘 안 맞는 식물/사물형 강한 감점
    if is_non_face_monster(name):
        score -= 35

    # 같은 사진이면 항상 같은 미세 보정
    score += stable_tiebreaker(image_hash, name) * 2

    return round(score, 4)


def add_percent(results):
    if not results:
        return results

    max_score = results[0]["score"] or 1

    for index, r in enumerate(results):
        base_percent = int((r["score"] / max_score) * 96)

        if index == 0:
            r["percent"] = max(90, min(base_percent, 98))
        elif index == 1:
            r["percent"] = max(80, min(base_percent, 89))
        else:
            r["percent"] = max(70, min(base_percent, 79))

    return results


def generate_reason(monster_name, user):
    try:
        prompt = f"""
사용자 분위기:
- 얼굴형 느낌: {user["face_shape"]}
- 분위기: {user["vibe"]}
- 귀여움: {user["cute_level"]}
- 어두움: {user["dark_level"]}
- 포스: {user["power_level"]}
- 설명: {user["description"]}

매칭 몬스터: {monster_name}

조건:
- 2문장
- 귀엽고 재밌게
- 메이플 닮은꼴 테스트 결과처럼 작성
- 외모 비하 금지
- 신원/성별/나이 추정 금지
- 80자 이내
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return f"{monster_name}와 전체 분위기가 비슷해요!"


@app.get("/")
def home():
    return {
        "message": "Maple Monster Match API is running!",
        "monster_count": len(df),
        "mode": "A4 csv-native matching",
        "columns": list(df.columns),
    }


@app.post("/match")
def match_monster(req: MatchRequest):
    try:
        image_base64 = clean_base64(req.image_base64)
        image_hash = make_image_hash(image_base64)

        user = analyze_user_image(image_base64)

        results = []

        for _, row in df.iterrows():
            name = get_value(row, ["name"], "이름 없음")
            image_url = get_value(row, ["image_url"], "")

            score = score_monster(user, row, image_hash)

            results.append({
                "name": name,
                "image_url": image_url,
                "score": score,
                "tags": [
                    get_value(row, ["face_shape"], ""),
                    get_value(row, ["vibe"], ""),
                ],
                "reason": "",
            })

        results = sorted(results, key=lambda x: (-x["score"], x["name"]))

        unique = []
        seen = set()

        for r in results:
            if r["name"] in seen:
                continue

            seen.add(r["name"])
            unique.append(r)

            if len(unique) >= 3:
                break

        unique = add_percent(unique)

        for r in unique:
            r["reason"] = generate_reason(r["name"], user)

        return {
            "image_hash": image_hash,
            "analysis": {
                "tags": user["tags"],
                "vibe": user["description"],
                "scores": {
                    "cute_level": user["cute_level"],
                    "dark_level": user["dark_level"],
                    "power_level": user["power_level"],
                },
                "face_shape": user["face_shape"],
                "main_vibe": user["vibe"],
            },
            "results": unique,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
