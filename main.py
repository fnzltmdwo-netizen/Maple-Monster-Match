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


def extract_json(text: str):
    text = text.strip()
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end <= 0:
        raise ValueError("JSON을 찾을 수 없습니다.")

    return json.loads(text[start:end])


def score_monster(features, row):
    score = 0

    user_face = str(features["face_shape"])
    user_vibe = str(features["vibe"])

    monster_face = str(row["face_shape"])
    monster_vibe = str(row["vibe"])

    user_cute = int(features["cute_level"])
    user_dark = int(features["dark_level"])
    user_power = int(features["power_level"])

    monster_cute = int(row["cute_level"])
    monster_dark = int(row["dark_level"])
    monster_power = int(row["power_level"])

    if user_face == monster_face:
        score += 35

    if user_vibe == monster_vibe:
        score += 25

    score += max(0, 20 - abs(user_cute - monster_cute) * 3)
    score += max(0, 10 - abs(user_dark - monster_dark) * 2)
    score += max(0, 10 - abs(user_power - monster_power) * 2)

    return score


def find_top3(features):
    results = []

    for _, row in df.iterrows():
        score = score_monster(features, row)
        match_percent = min(98, max(70, int(score)))
        name = str(row["name"])

        results.append({
            "name": name,
            "score": score,
            "match_percent": match_percent,
            "reason": str(row["description"]),
            "image_url": str(row.get("image_url", "")),
            "face_shape": str(row["face_shape"]),
            "vibe": str(row["vibe"]),
            "cute_level": int(row["cute_level"]),
            "dark_level": int(row["dark_level"]),
            "power_level": int(row["power_level"]),
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:3]


def find_candidates(features, limit=20):
    results = []

    for _, row in df.iterrows():
        score = score_monster(features, row)
        match_percent = min(98, max(70, int(score)))
        name = str(row["name"])

        results.append({
            "name": name,
            "score": score,
            "match_percent": match_percent,
            "face_shape": str(row["face_shape"]),
            "vibe": str(row["vibe"]),
            "cute_level": int(row["cute_level"]),
            "dark_level": int(row["dark_level"]),
            "power_level": int(row["power_level"]),
            "description": str(row["description"]),
            "image_url": str(row.get("image_url", "")),
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]


def make_candidate_text(candidates):
    lines = []

    for i, monster in enumerate(candidates, start=1):
        line = (
            f"{i}. {monster['name']} "
            f"(face_shape={monster['face_shape']}, "
            f"vibe={monster['vibe']}, "
            f"cute={monster['cute_level']}, "
            f"dark={monster['dark_level']}, "
            f"power={monster['power_level']}, "
            f"score={monster['score']})"
        )

        if monster.get("description"):
            line += f" - {monster['description']}"

        lines.append(line)

    return "\n".join(lines)


@app.get("/")
def home():
    return {
        "message": "Maple Monster Match API is running!",
        "monster_count": len(df),
        "mode": "FULL AI monster DB + A2 GPT final judge"
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

cute_level, dark_level, power_level은 0~10 정수.

정면 얼굴 기준으로 판단해.
옷, 배경, 포즈보다 얼굴형/표정/눈매/전체 인상을 우선해.

출력 형식:
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

    candidates = find_candidates(features, limit=20)
    candidate_text = make_candidate_text(candidates)

    judge_prompt = f"""
너는 메이플스토리 몬스터 닮은꼴 최종 심사위원이야.

사진 속 인물과 아래 후보 몬스터 20마리를 비교해서 가장 닮은 Top 3를 골라.

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
{{
  "top3": [
    {{
      "name": "몬스터명",
      "match_percent": 92,
      "reason": "닮은 이유를 자연스럽게 한 문장으로 설명"
    }},
    {{
      "name": "몬스터명",
      "match_percent": 88,
      "reason": "닮은 이유를 자연스럽게 한 문장으로 설명"
    }},
    {{
      "name": "몬스터명",
      "match_percent": 84,
      "reason": "닮은 이유를 자연스럽게 한 문장으로 설명"
    }}
  ]
}}
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

    for monster in final_result.get("top3", []):
        matched = next(
            (c for c in candidates if c["name"] == monster["name"]),
            None
        )

        if matched:
            monster["image_url"] = matched.get("image_url", "")
            monster["score"] = matched.get("score", 0)
            monster["face_shape"] = matched.get("face_shape", "")
            monster["vibe"] = matched.get("vibe", "")
            monster["cute_level"] = matched.get("cute_level", 0)
            monster["dark_level"] = matched.get("dark_level", 0)
            monster["power_level"] = matched.get("power_level", 0)
        else:
            monster["image_url"] = ""

    return {
        "features": features,
        "candidates": candidates,
        "top3": final_result.get("top3", [])
    }
