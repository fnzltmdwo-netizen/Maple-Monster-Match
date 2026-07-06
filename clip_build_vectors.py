# clip_build_vectors.py

import os
import pickle
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image

import torch
import open_clip


CSV_PATH = "monsters_ai_v3.csv"
OUT_PATH = "monster_clip_vectors.pkl"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(url):
    res = requests.get(url, timeout=20)
    res.raise_for_status()
    return Image.open(BytesIO(res.content)).convert("RGB")


def main():
    print("DEVICE:", DEVICE)
    print("CSV:", CSV_PATH)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("monsters_ai_v3.csv 파일이 없습니다.")

    df = pd.read_csv(CSV_PATH).fillna("")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=DEVICE,
    )
    model.eval()

    vectors = []

    for idx, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        image_url = str(row.get("image_url", "")).strip()

        if not name or not image_url:
            print(f"[SKIP] {idx+1}: name/image_url 없음")
            continue

        try:
            print(f"[{idx+1}/{len(df)}] {name}")

            img = load_image(image_url)
            image_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = model.encode_image(image_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            vector = emb.cpu().numpy()[0].astype(np.float32)

            vectors.append({
                "name": name,
                "mob_id": str(row.get("mob_id", "")).strip(),
                "image_url": image_url,
                "source_url": str(row.get("source_url", "")).strip(),
                "vector": vector,
            })

        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(vectors, f)

    print("\n완료!")
    print("저장:", OUT_PATH)
    print("개수:", len(vectors))


if __name__ == "__main__":
    main()
