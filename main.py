from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
from io import BytesIO

import os
import base64
import pickle
import requests
import pandas as pd
import torch
import open_clip


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_PATH = "monsters_ai.csv"
EMBEDDING_PATH = "monster_embeddings.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
model = model.to(device)
model.eval()

MONSTER_DB = []


class MatchRequest(BaseModel):
    image_base64: str


def load_monster_db():
    global MONSTER_DB

    if not os.path.exists(EMBEDDING_PATH):
        MONSTER_DB = []
        return

    with open(EMBEDDING_PATH, "rb") as f:
        MONSTER_DB = pickle.load(f)


def get_value(row, possible_cols, default=""):
    for col in possible_cols:
        if col in row.index and str(row[col]).strip():
            return str(row[col]).strip()
    return default


def clean_base64(image_base64: str):
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    return image_base64.strip().replace("\n", "").replace("\r", "").replace(" ", "")


def decode_image(image_base64: str):
    try:
        cleaned = clean_base64(image_base64)
        image_bytes = base64.b64decode(cleaned)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패")


def encode_image(image: Image.Image):
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(image_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.cpu()


def build_monster_embeddings():
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=500, detail="monsters_ai.csv 파일이 없습니다.")

    df = pd.read_csv(CSV_PATH).fillna("")
    embeddings = []

    for index, row in df.iterrows():
        name = get_value(row, ["name", "monster_name", "몬스터명"], "")
        image_url = get_value(row, ["image_url", "img_url", "url", "image"], "")

        if not name or not image_url:
            continue

        try:
            response = requests.get(image_url, timeout=20)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content)).convert("RGB")
            embedding = encode_image(image)

            embeddings.append({
                "name": name,
                "image_url": image_url,
                "embedding": embedding,
            })

            print(f"done {index + 1}/{len(df)}: {name}")

        except Exception as e:
            print(f"skip {name}: {e}")

    with open(EMBEDDING_PATH, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings


def similarity_to_percent(score: float, rank: int):
    raw = int(score * 100)

    if rank == 0:
        return max(90, min(raw + 25, 98))
    elif rank == 1:
        return max(80, min(raw + 22, 89))
    else:
        return max(70, min(raw + 20, 79))


def make_reason(name: str, percent: int, rank: int):
    if rank == 0:
        return f"전체적인 실루엣과 분위기가 가장 비슷해서 {name}와 닮은꼴로 매칭됐어요!"
    elif rank == 1:
        return f"이미지의 느낌과 색감 밸런스가 {name}와 꽤 잘 맞아요."
    else:
        return f"세부 인상은 다르지만 전체 분위기가 {name}와 은근히 닮았어요."


def match_by_clip(user_embedding):
    if not MONSTER_DB:
        raise HTTPException(
            status_code=500,
            detail="monster_embeddings.pkl이 없습니다. 먼저 /build-embeddings 를 실행해주세요."
        )

    results = []

    for monster in MONSTER_DB:
        monster_embedding = monster["embedding"]

        if isinstance(monster_embedding, torch.Tensor):
            monster_embedding = monster_embedding.cpu()

        score = torch.cosine_similarity(
            user_embedding,
            monster_embedding
        ).item()

        results.append({
            "name": monster["name"],
            "image_url": monster["image_url"],
            "score": round(score, 6),
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    top3 = results[:3]

    for index, r in enumerate(top3):
        r["percent"] = similarity_to_percent(r["score"], index)
        r["reason"] = make_reason(r["name"], r["percent"], index)
        r["tags"] = ["clip", "image-match"]

    return top3


@app.on_event("startup")
def startup_event():
    load_monster_db()


@app.get("/")
def home():
    return {
        "message": "Maple Monster Match API is running!",
        "mode": "CLIP image matching",
        "device": device,
        "embedding_loaded": len(MONSTER_DB),
        "need_build": len(MONSTER_DB) == 0
    }


@app.get("/build-embeddings")
def build_embeddings_api():
    embeddings = build_monster_embeddings()

    global MONSTER_DB
    MONSTER_DB = embeddings

    return {
        "message": "monster embeddings built",
        "count": len(MONSTER_DB),
        "path": EMBEDDING_PATH
    }


@app.post("/match")
def match_monster(req: MatchRequest):
    try:
        image = decode_image(req.image_base64)
        user_embedding = encode_image(image)

        results = match_by_clip(user_embedding)

        return {
            "analysis": {
                "vibe": "CLIP 이미지 유사도 기반으로 몬스터 이미지와 직접 비교했어요.",
                "tags": ["clip", "image-match"],
                "scores": {}
            },
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
