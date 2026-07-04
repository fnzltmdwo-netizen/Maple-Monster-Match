from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import pandas as pd
import os
import json
import re
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

df = pd.read_csv("monsters_ai.csv").fillna("")
print("Loaded monsters:", len(df))


class MatchRequest(BaseModel):
    image_base64: str | None = None
    image: str | None = None


def safe_str(value, default=""):
    if value is None:
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
    except Exception:
        pass
    return str(value)


def safe_int(value, default=5):
    try:
        if value is None or value == "":
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(float(value))
    except Exception:
        return default


def clean_base64(image_base64: str) -> str:
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    return image_base64.strip()


def extract_json(text: str):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("JSON을 찾을 수 없습니다.")

    return json.loads(match.group(0))


def analyze_person(image_base64: str):
    image_base64 = clean_base64(image_base64)

    prompt = """
너는 사람 사진을 보고 메이플스토리 몬스터 닮은꼴 매칭용 특징을 뽑는 분석기야.

반드시 JSON만 출력해.
설명 문장 금지.

가능한 face_shape:
round, oval, long, square, triangle

가능한 vibe:
cute, dark, strong, calm, mysterious

가능한 eye_shape:
round, sharp, sleepy, calm, intense

가능한 jawline:
soft, sharp, square, narrow, round

가능한 animal_type:
puppy, cat, fox, bear, rabbit, turtle, bird, dragon, unknown

cute_level, dark_level, power_level은 0~10 정수.
softness, sharpness, mature_level도 0~10 정수.

옷, 배경, 포즈보다 얼굴형/눈매/표정/전체 인상을 우선해.

출력 형식:
{
  "face_shape": "oval",
  "vibe": "calm",
  "eye_shape": "calm",
  "jawline": "soft",
  "animal_type": "cat",
  "cute_level": 6,
  "dark_level": 1,
  "power_level": 3,
  "softness": 7,
  "sharpness": 4,
  "mature_level": 5
}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=300,
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

    text = response.choices[0].message.content
    return extract_json(text)


def score_monster(features, row):
    score = 0

    user_face = safe_str(features.get("face_shape"))
    user_vibe = safe_str(features.get("vibe"))
    user_eye = safe_str(features.get("eye_shape"))
    user_jaw = safe_str(features.get("jawline"))
    user_animal = safe_str(features.get("animal_type"))

    monster_face = safe_str(row.get("face_shape"))
    monster_vibe = safe_str(row.get("vibe"))
    monster_eye = safe_str(row.get("eye_shape"))
    monster_jaw = safe_str(row.get("jawline"))
    monster_animal = safe_str(row.get("animal_type"))

    user_cute = safe_int(features.get("cute_level"))
    user_dark = safe_int(features.get("dark_level"))
    user_power = safe_int(features.get("power_level"))
    user_soft = safe_int(features.get("softness"))
    user_sharp = safe_int(features.get("sharpness"))
    user_mature = safe_int(features.get("mature_level"))

    monster_cute = safe_int(row.get("cute_level"))
    monster_dark = safe_int(row.get("dark_level"))
    monster_power = safe_int(row.get("power_level"))
    monster_soft = safe_int(row.get("softness"))
    monster_sharp = safe_int(row.get("sharpness"))
    monster_mature = safe_int(row.get("mature_level"))

    if user_face == monster_face:
        score += 25
    if user_vibe == monster_vibe:
        score += 25
    if user_eye and user_eye == monster_eye:
        score += 12
    if user_jaw and user_jaw == monster_jaw:
        score += 8
    if user_animal and user_animal == monster_animal:
        score += 10

    score += max(0, 15 - abs(user_cute - monster_cute) * 2)
    score += max(0, 12 - abs(user_dark - monster_dark) * 2)
    score += max(0, 12 - abs(user_power - monster_power) * 2)
    score += max(0, 10 - abs(user_soft - monster_soft) * 2)
    score += max(0, 10 - abs(user_sharp - monster_sharp) * 2)
    score += max(0, 10 - abs(user_mature - monster_mature) * 2)

    return round(score, 2)


def score_to_percent(score, rank_index):
    percent = int(70 + min(score, 100) * 0.28)
    percent -= rank_index * 3
    return max(70, min(99, percent))


def make_reason(features, monster):
    vibe = safe_str(monster.get("vibe"))
    face = safe_str(monster.get("face_shape"))
    cute = safe_int(monster.get("cute_level"))
    dark = safe_int(monster.get("dark_level"))
    power = safe_int(monster.get("power_level"))

    name = safe_str(monster.get("name"))

    face_text = {
        "round": "둥글고 부드러운 인상",
        "oval": "깔끔한 세로형 인상",
        "long": "길쭉하고 차분한 인상",
        "square": "단단하고 안정적인 인상",
        "triangle": "날렵하고 개성 있는 인상",
    }.get(face, "전체적인 얼굴형")

    vibe_text = {
        "cute": "귀여운 분위기",
        "calm": "차분한 분위기",
        "strong": "강한 존재감",
        "dark": "어둡고 신비로운 느낌",
        "mysterious": "묘하고 독특한 분위기",
    }.get(vibe, "전체적인 분위기")

    extra = []

    if cute >= 7:
        extra.append("귀여운 느낌이 강해요")
    if dark >= 6:
        extra.append("살짝 어두운 매력이 있어요")
    if power >= 7:
        extra.append("존재감이 또렷해요")
    if not extra:
        extra.append("부담스럽지 않고 자연스러운 인상이 있어요")

    return f"{name}은 {face_text}과 {vibe_text}가 잘 살아나는 몬스터예요. {extra[0]}."


def find_top3(features):
    results = []

    for _, row in df.iterrows():
        score = score_monster(features, row)

        monster = {
            "name": safe_str(row.get("name")),
            "image_url": safe_str(row.get("image_url")),
            "score": score,
            "face_shape": safe_str(row.get("face_shape")),
            "vibe": safe_str(row.get("vibe")),
            "cute_level": safe_int(row.get("cute_level")),
            "dark_level": safe_int(row.get("dark_level")),
            "power_level": safe_int(row.get("power_level")),
            "description": safe_str(row.get("description")),
        }

        monster["reason"] = make_reason(features, monster)
        results.append(monster)

    results = sorted(results, key=lambda x: (-x["score"], x["name"]))[:3]

    for i, item in enumerate(results):
        item["match_percent"] = score_to_percent(item["score"], i)

    return results


@app.get("/")
def home():
    return {
        "message": "Maple Monster Match API is running!",
        "monster_count": len(df),
        "mode": "image_base64 stable python top3"
    }


@app.post("/match")
def match_monster(req: MatchRequest):
    image_data = req.image_base64 or req.image

    if not image_data:
        raise HTTPException(status_code=422, detail="image_base64 is required")

    try:
        features = analyze_person(image_data)
        top3 = find_top3(features)

        return {
            "person_features": features,
            "features": features,
            "top3": top3
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
