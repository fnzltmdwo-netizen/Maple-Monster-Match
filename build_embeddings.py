import pandas as pd
import torch
import open_clip
import requests
import pickle
from PIL import Image
from io import BytesIO

CSV_PATH = "monsters_ai.csv"
OUTPUT_PATH = "monster_embeddings.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
model = model.to(device)
model.eval()

df = pd.read_csv(CSV_PATH).fillna("")
embeddings = []

for index, row in df.iterrows():
    name = str(row.get("name", row.get("monster_name", ""))).strip()
    image_url = str(row.get("image_url", row.get("img_url", row.get("url", "")))).strip()

    if not name or not image_url:
        print("skip empty:", index)
        continue

    try:
        response = requests.get(image_url, timeout=20)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")
        image_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model.encode_image(image_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        embeddings.append({
            "name": name,
            "image_url": image_url,
            "embedding": feat.cpu()
        })

        print(f"done {index + 1}/{len(df)}: {name}")

    except Exception as e:
        print(f"skip {name}: {e}")

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(embeddings, f)

print("saved:", OUTPUT_PATH, "count:", len(embeddings))
