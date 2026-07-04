from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
from openai import OpenAI
import pandas as pd
import base64
import json
import os
import math

app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("monsters_ai.csv")
print("Loaded monsters:", len(df))


class MatchRequest(BaseModel):
    face_shape: str
    vibe: str
    cute_level: int
    dark_level: int
    power_level: int


def safe_int(value, default=5):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_str(value, default=""):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return str(value)
    except Exception:
        return default


def extract_json(text: str):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end <= 0:
        raise ValueError("JSON을 찾을 수 없습니다.")

    return json.loads(text[start:end])


def score_monster(features, row):
    score = 0

    user_face = safe_str(features.get("face_shape"))
    user_vibe = safe_str(features.get("vibe"))

    monster_face = safe_str(row.get("face_shape"))
    monster_vibe = safe_str(row.get("vibe"))

    user_cute = safe_int(features.get("cute_level"))
    user_dark = safe_int(features.get("dark_level"))
    user_power = safe_int(features.get("power_level"))
    user_softness = safe_int(features.get("softness"), 5)
    user_sharpness = safe_int(features.get("sharpness"), 5)
    user_mature = safe_int(features.get("mature_level"), 5)

    monster_cute = safe_int(row.get("cute_level"))
    monster_dark = safe_int(row.get("dark_level"))
    monster_power = safe_int(row.get("power_level"))

    if user_face == monster_face:
        score += 30

    if user_vibe == monster_vibe:
        score += 25

    score += max(0, 20 - abs(user_cute - monster_cute) * 3)
    score += max(0, 15 - abs(user_dark - monster_dark) * 2)
    score += max(0, 15 - abs(user_power - monster_power) * 2)

    # 추가 feature 보정
    if user_softness >= 7 and monster_cute >= 7:
        score += 8

    if user_sharpness >= 7 and monster_power >= 7:
        score += 8

    if user_mature >= 7 and monster_vibe in ["calm", "mysterious"]:
        score += 6

    return score


def make_reason(features, monster):
    vibe = monster["vibe"]
    face = monster["face_shape"]

    vibe_text = {
        "cute": "귀엽고 부드러운 분위기",
        "calm": "차분하고 안정적인 분위기",
        "strong": "강하고 또렷한 분위기",
        "dark": "어둡고 신비로운 분위기",
        "mysterious": "묘하고 독특한 분위기",
    }.get(vibe, "전체적인 분위기")

    face_text = {
        "round": "둥근 인상",
        "oval": "부드러운 세로형 인상",
        "long": "길쭉한 인상",
        "square": "단단한 인상",
        "triangle": "날카로운 인상",
    }.get(face, "얼굴형")

    return f"{face_text}과 {vibe_text}가 사진 속 인물의 인상과 잘 어울립니다."


def find_top3(features):
    results = []

    for _, row in df.iterrows():
        score = score_monster(features, row)
        match_percent = min(98, max(70, int(score)))

        monster = {
            "name": safe_str(row.get("name")),
            "score": score,
            "match_percent": match_percent,
            "face_shape": safe_str(row.get("face_shape")),
            "vibe": safe_str(row.get("vibe")),
            "cute_level": safe_int(row.get("cute_level")),
            "dark_level": safe_int(row.get("dark_level")),
            "power_level": safe_int(row.get("power_level")),
            "description": safe_str(row.get("description")),
            "image_url": safe_str(row.get("image_url")),
        }

        monster["reason"] = make_reason(features, monster)
        results.append(monster)

    return sorted(results, key=lambda x: x["score"], reverse=True)[:3]


@app.get("/")
def home():
    return {
        "message": "Maple Monster Match API is running!",
        "monster_count": len(df),
        "mode": "stable python ranking + detailed face features"
    }


@app.post("/match")
def match_monster(req: MatchRequest):
    features = req.dict()
    return {
        "features": features,
        "top3": find_top3(features)
    }


@app.post("/match-image")
async def match_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"error": "이미지 파일을 읽을 수 없습니다."}

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    feature_prompt = """
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

판단 기준:
- face_shape: 얼굴 전체 형태
- vibe: 전체 인상
- eye_shape: 눈매 느낌
- jawline: 턱선 느낌
- animal_type: 닮은 동물상
- softness: 부드럽고 순한 정도
- sharpness: 날카롭고 또렷한 정도
- mature_level: 성숙하고 차분한 정도

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

    try:
        feature_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": feature_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file.content_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=250,
            temperature=0,
        )

        feature_text = feature_response.choices[0].message.content.strip()
        features = extract_json(feature_text)

    except Exception as e:
        return {
            "error": "GPT 특징 분석 실패",
            "detail": str(e)
        }

    top3 = find_top3(features)

    return {
        "features": features,
        "top3": top3
    }
