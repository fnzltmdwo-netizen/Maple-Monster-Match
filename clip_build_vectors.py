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


BLOCKED_NAME_KEYWORDS = [
    "포장마차",
    "가로등",
    "상자",
    "문",
    "간판",
    "보물상자",
    "나무",
    "스텀프",
    "테이블",
    "의자",
    "책상",
    "기계",
    "장난감",
    "석상",
    "바위",
    "돌",
    "화분",
    "버섯집",
    "오브젝트",
    "코-크",
    "코크",
]


def to_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def should_skip(row):
    name = str(row.get("name", "")).strip()

    if any(keyword in name for keyword in BLOCKED_NAME_KEYWORDS):
        return True, "blocked name"

    object_like = to_int(row.get("object_like", 0))
    plant_like = to_int(row.get("plant_like", 0))
    human_like = to_int(row.get("human_like", 0))
    animal_like = to_int(row.get("animal_like", 0))
    blob_like = to_int(row.get("blob_like", 0))
    ghost_like = to_int(row.get("ghost_like", 0))

    species_tag = str(row.get("species_tag", "")).strip()
    body_type = str(row.get("body_type", "")).strip()

    # 오브젝트/식물형 강한 애들 제외
    if object_like >= 8:
        return True, "object_like"
    if plant_like >= 8:
        return True, "plant_like"

    # species 자체가 물건/식물인 경우 제외
    if species_tag in ["object", "plant"]:
        return True, "species_tag"

    # 사람/동물/슬라임/유령 느낌이 거의 없는 애들 제외
    living_score = human_like + animal_like + blob_like + ghost_like
    if living_score < 4:
        return True, "low living score"

    # body_type이 명확히 object/plant면 제외
    if body_type in ["object", "plant"]:
        return True, "body_type"

    return False, ""


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
    skipped = 0

    for idx, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        image_url = str(row.get("image_url", "")).strip()

        skip, reason = should_skip(row)
        if skip:
            skipped += 1
            print(f"[SKIP] {idx+1}/{len(df)} {name} - {reason}")
            continue

        if not name or not image_url:
            skipped += 1
            print(f"[SKIP] {idx+1}/{len(df)} name/image_url 없음")
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
                "species_tag": str(row.get("species_tag", "")).strip(),
                "body_type": str(row.get("body_type", "")).strip(),
            })

        except Exception as e:
            skipped += 1
            print(f"[FAIL] {name}: {e}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(vectors, f)

    print("\n완료!")
    print("저장:", OUT_PATH)
    print("벡터 개수:", len(vectors))
    print("스킵 개수:", skipped)


if __name__ == "__main__":
    main()
