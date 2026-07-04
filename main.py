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
import re

app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("monsters.csv")


class MatchRequest(BaseModel):
    face_shape: str
    vibe: str
    cute_level: int
    dark_level: int
    power_level: int


def extract_json(text: str):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text)
        text = re.sub(r"^```", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == 0:
        raise ValueError("JSON을 찾을 수 없습니다.")

    return json.loads(text[start:end])


def find_top3(features):
    results = []

    for _, row in df.iterrows():
        score = 0

        if str(row["face_shape"]) != features["face_shape"]:
            score += 3

        if str(row["vibe"]) != features["vibe"]:
            score += 3

        score += abs(int(row["cute_level"]) - int(features["cute_level"]))
        score += abs(int(row["dark_level"]) - int(features["dark_level"]))
        score += abs(int(row["power_level"]) - int(features["power_level"]))

        match_percent = max(0, 100 - score * 5)

        results.append({
            "name": row["name"],
            "score": score,
            "match_percent": match_percent,
            "reason": f"{row['vibe']} 분위기, 귀여움 {row['cute_level']}, 어둠 {row['dark_level']}, 포스 {row['power_level']}"
        })

    return sorted(results, key=lambda x: x["score"])[:3]


def find_candidates(features, limit=10):
    results = []

    for _, row in df.iterrows():
        score = 0

        if str(row["face_shape"]) != features["face_shape"]:
            score += 3

        if str(row["vibe"]) != features["vibe"]:
            score += 3

        score += abs(int(row["cute_level"]) - int(features["cute_level"]))
        score += abs(int(row["dark_level"]) - int(features["dark_level"]))
        score += abs(int(row["power_level"]) - int(features["power_level"]))

        description = ""
        if "description" in df.columns and pd.notna(row.get("description", "")):
            description = str(row["description"])

        results.append({
            "name": row["name"],
            "score": score,
            "face_shape": row["face_shape"],
            "vibe": row["vibe"],
            "cute_level": int(row["cute_level"]),
            "dark_level": int(row["dark_level"]),
            "power_level": int(row["power_level"]),
            "description": description,
        })

    return sorted(results, key=lambda x: x["score"])[:limit]


def make_candidate_text(candidates):
    text = ""

    for i, monster in enumerate(candidates, start=1):
        text += (
            f"{i}. {monster['name']} "
            f"(face_shape={monster['face_shape']}, vibe={monster['vibe']}, "
            f"cute={monster['cute_level']}, dark={monster['dark_level']}, "
            f"power={monster['power_level']})"
        )

        if monster.get("description"):
            text += f" - {monster['description']}"

        text += "\n"

    return text


@app.get("/")
def home():
    return {
        "message": "Maple Monster Match API is running!",
        "monster_count": len(df),
        "mode": "A2 GPT final judge"
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

반드시 아래 JSON 형식만 출력해.
설명 문장 금지.

가능한 face_shape:
round, oval, long, square, triangle

가능한 vibe:
cute, dark, strong, calm, mysterious

cute_level, dark_level, power_level은 0~10 정수.

출력 예시:
{
  "face_shape": "round",
  "vibe": "cute",
  "cute_level": 8,
  "dark_level": 2,
  "power_level": 4
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
            max_tokens=200
        )

        feature_text = feature_response.choices[0].message.content.strip()
        features = extract_json(feature_text)

    except Exception as e:
        return {
            "error": "GPT 특징 분석 실패",
            "detail": str(e)
        }

    candidates = find_candidates(features, limit=10)
    candidate_text = make_candidate_text(candidates)

    judge_prompt = f"""
너는 메이플스토리 몬스터 닮은꼴 최종 심사위원이야.

사진 속 인물과 아래 후보 몬스터 10마리를 비교해서
가장 닮은 Top 3를 골라.

판단 기준:
- 얼굴형
- 눈매
- 표정
- 전체 분위기
- 귀여움/차분함/강함/어두움/신비로움
- 사진 속 인물의 인상과 몬스터 설명의 유사성

후보 몬스터:
{candidate_text}

반드시 JSON만 출력해.
후보 목록에 있는 name만 사용해.
match_percent는 70~98 사이 정수로 줘.

출력 형식:
{
  "top3": [
    {
      "name": "몬스터명",
      "match_percent": 92,
      "reason": "닮은 이유를 자연스럽게 한 문장으로 설명"
    },
    {
      "name": "몬스터명",
      "match_percent": 88,
      "reason": "닮은 이유를 자연스럽게 한 문장으로 설명"
    },
    {
      "name": "몬스터명",
      "match_percent": 84,
      "reason": "닮은 이유를 자연스럽게 한 문장으로 설명"
    }
  ]
}
"""

    try:
        judge_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": judge_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file.content_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        judge_text = judge_response.choices[0].message.content.strip()
        final_result = extract_json(judge_text)

    except Exception as e:
        return {
            "error": "GPT 최종 심사 실패",
            "features": features,
            "candidates": candidates,
            "detail": str(e)
        }

    return {
        "features": features,
        "candidates": candidates,
        "top3": final_result["top3"]
    }
