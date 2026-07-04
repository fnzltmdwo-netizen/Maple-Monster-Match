from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from fastapi import UploadFile, File
from PIL import Image
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_PATH = "monsters.csv"
df = pd.read_csv(CSV_PATH)


class MatchRequest(BaseModel):
    face_shape: str
    vibe: str
    cute_level: int
    dark_level: int
    power_level: int


@app.get("/")
def home():
    return {
        "message": "Maple Monster Match API is running!",
        "monster_count": len(df)
    }


@app.post("/match")
def match_monster(req: MatchRequest):
    results = []

    for _, row in df.iterrows():
        score = 0

        if str(row["face_shape"]) != req.face_shape:
            score += 3

        if str(row["vibe"]) != req.vibe:
            score += 3

        score += abs(int(row["cute_level"]) - req.cute_level)
        score += abs(int(row["dark_level"]) - req.dark_level)
        score += abs(int(row["power_level"]) - req.power_level)

        results.append({
            "name": row["name"],
            "score": score,
            "face_shape": row["face_shape"],
            "vibe": row["vibe"],
            "cute_level": int(row["cute_level"]),
            "dark_level": int(row["dark_level"]),
            "power_level": int(row["power_level"]),
        })

    results = sorted(results, key=lambda x: x["score"])

    return {
        "top3": results[:3]
    }
@app.post("/match-image")
async def match_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {
            "error": "이미지 파일을 읽을 수 없습니다."
        }

    width, height = image.size

    return {
        "message": "이미지 업로드 성공!",
        "filename": file.filename,
        "width": width,
        "height": height
    }
