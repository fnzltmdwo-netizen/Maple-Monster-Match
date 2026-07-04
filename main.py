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

df = pd.read_csv("monsters.csv")


class MatchRequest(BaseModel):
    face_shape: str
    vibe: str
    cute_level: int
    dark_level: int
    power_level: int


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

        results.append({
            "name": row["name"],
            "score": score,
            "reason": f"{row['vibe']} 분위기, 귀여움 {row['cute_level']}, 어둠 {row['dark_level']}, 포스 {row['power_level']}"
        })

    return sorted(results, key=lambda x: x["score"])[:3]


@app.get("/")
def home():
    return {
        "message": "Maple Monster Match API is running!",
        "monster_count": len(df)
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

    prompt = """
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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

    text = response.choices[0].message.content.strip()
    features = json.loads(text)

    return {
        "features": features,
        "top3": find_top3(features)
    }
