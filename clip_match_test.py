# clip_match_test.py

import os
import pickle
import base64
from io import BytesIO

import numpy as np
from PIL import Image

import torch
import open_clip


VECTOR_PATH = "monster_clip_vectors.pkl"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_local_image(path):
    return Image.open(path).convert("RGB")


def image_to_vector(image, model, preprocess):
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model.encode_image(image_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().numpy()[0].astype(np.float32)


def cosine_similarity(a, b):
    return float(np.dot(a, b))


def main():
    print("DEVICE:", DEVICE)

    if not os.path.exists(VECTOR_PATH):
        raise FileNotFoundError("monster_clip_vectors.pkl 파일이 없습니다. 먼저 python clip_build_vectors.py 실행!")

    image_path = input("테스트할 사람 사진 파일 경로를 입력하세요: ").strip()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"사진 파일을 찾을 수 없습니다: {image_path}")

    with open(VECTOR_PATH, "rb") as f:
        monsters = pickle.load(f)

    print("몬스터 벡터 개수:", len(monsters))

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=DEVICE,
    )
    model.eval()

    user_img = load_local_image(image_path)
    user_vec = image_to_vector(user_img, model, preprocess)

    scored = []

    for m in monsters:
        score = cosine_similarity(user_vec, m["vector"])
        scored.append({
            "name": m["name"],
            "mob_id": m.get("mob_id", ""),
            "image_url": m.get("image_url", ""),
            "score": score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    print("\n===== CLIP TOP 10 =====")
    for i, r in enumerate(scored[:10], start=1):
        percent = round((r["score"] + 1) / 2 * 100, 2)
        print(f"{i}. {r['name']} / score={r['score']:.4f} / approx={percent}%")
        print(f"   {r['image_url']}")


if __name__ == "__main__":
    main()
