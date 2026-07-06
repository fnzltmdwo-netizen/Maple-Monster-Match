from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from pathlib import Path
import os, json, uuid, base64, pickle, requests
import numpy as np
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

CSV_PATH = "monsters_ai_v3.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = "monsters_ai_v2.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = "monsters_ai.csv"

df = pd.read_csv(CSV_PATH).fillna("")

RESULT_DIR = Path("saved_results")
RESULT_DIR.mkdir(exist_ok=True)

BASE_URL = os.getenv("BASE_URL", "https://maple-monster-match-v2.onrender.com")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://maple-monster-frontend.onrender.com")

CLIP_VECTOR_PATH = "monster_clip_vectors.pkl"
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = None
clip_preprocess = None
clip_monsters = None

BLOCKED_NAME_KEYWORDS = ["코-크", "코크", "Coke", "coke"]

class MatchRequest(BaseModel):
    image_base64: str

class SaveResultRequest(BaseModel):
    user_name: str = "친구"
    analysis: dict
    results: list

def clean_base64(image_base64: str):
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    return image_base64.strip().replace("\n", "").replace("\r", "").replace(" ", "")

def base64_to_pil(image_base64: str):
    image_bytes = base64.b64decode(clean_base64(image_base64))
    return Image.open(BytesIO(image_bytes)).convert("RGB")

def escape_html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def load_result(result_id: str):
    path = RESULT_DIR / f"{result_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_clip_engine():
    global clip_model, clip_preprocess, clip_monsters
    if not os.path.exists(CLIP_VECTOR_PATH):
        raise FileNotFoundError("monster_clip_vectors.pkl 파일이 없습니다. 먼저 python clip_build_vectors.py 실행 후 GitHub에 업로드하세요.")
    if clip_model is None or clip_preprocess is None:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
            device=CLIP_DEVICE,
        )
        clip_model.eval()
    if clip_monsters is None:
        with open(CLIP_VECTOR_PATH, "rb") as f:
            clip_monsters = pickle.load(f)
    return clip_model, clip_preprocess, clip_monsters

def clip_image_vector(image: Image.Image):
    model, preprocess, _ = load_clip_engine()
    tensor = preprocess(image).unsqueeze(0).to(CLIP_DEVICE)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].astype(np.float32)

def score_to_percent(score, top_score, rank):
    if top_score > 0:
        base = int(72 + (score / top_score) * 24)
    else:
        base = int((score + 1) / 2 * 100)
    if rank == 0:
        return max(90, min(98, base))
    if rank == 1:
        return max(82, min(91, base))
    return max(74, min(86, base))

def make_reason(name, percent):
    if percent >= 94:
        return f"전체 실루엣과 분위기가 {name}와 꽤 잘 맞아요!"
    if percent >= 86:
        return f"이미지 분위기와 캐릭터성이 {name} 쪽에 가깝게 나왔어요."
    return f"AI가 시각적 느낌을 비교했을 때 {name}와 비슷한 편이에요."

def run_clip_match(image_base64: str, top_k=10):
    _, _, monsters = load_clip_engine()
    user_vec = clip_image_vector(base64_to_pil(image_base64))
    scored = []
    for m in monsters:
        name = str(m.get("name", "")).strip()
        if not name:
            continue
        if any(keyword in name for keyword in BLOCKED_NAME_KEYWORDS):
            continue
        vector = m.get("vector")
        if vector is None:
            continue
        score = float(np.dot(user_vec, vector))
        scored.append({
            "name": name,
            "mob_id": str(m.get("mob_id", "")).strip(),
            "image_url": str(m.get("image_url", "")).strip(),
            "source_url": str(m.get("source_url", "")).strip(),
            "score": round(score, 5),
            "reason": "",
            "common_tags": [],
            "tags": [],
            "species_tag": "",
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    if not scored:
        return []
    top_score = scored[0]["score"]
    results = []
    for idx, r in enumerate(scored[:top_k]):
        copied = dict(r)
        copied["percent"] = score_to_percent(copied["score"], top_score, idx)
        copied["reason"] = make_reason(copied["name"], copied["percent"])
        results.append(copied)
    return results

def get_font(size: int, bold=False):
    candidates = [
        "fonts/NotoSansKR-Bold.ttf" if bold else "fonts/NotoSansKR-Regular.ttf",
        "fonts/Pretendard-Bold.otf" if bold else "fonts/Pretendard-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

def fetch_monster_image(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        img = Image.open(BytesIO(res.content)).convert("RGBA")
        img.thumbnail((210, 210))
        return img
    except Exception:
        return None

def draw_center_text(draw, box, text, font, fill):
    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), str(text), font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x1 + ((x2 - x1) - w) / 2
    y = y1 + ((y2 - y1) - h) / 2
    draw.text((x, y), str(text), font=font, fill=fill)

def draw_wrapped_text(draw, text, x, y, max_width, font, fill, line_gap=8, max_lines=3):
    lines, current = [], ""
    for ch in list(str(text)):
        test = current + ch
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = ch
            if len(lines) >= max_lines:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    for line in lines[:max_lines]:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap

def make_og_image(data):
    name = data.get("user_name", "친구")
    results = data.get("results", [])[:3]
    W, H = 1080, 1080
    img = Image.new("RGB", (W, H), "#f7f8ff")
    draw = ImageDraw.Draw(img)
    draw.ellipse((790, -120, 1220, 310), fill="#dbe6ff")
    draw.ellipse((-150, 760, 260, 1180), fill="#fff0fb")
    draw.rounded_rectangle((60, 60, 1020, 1020), radius=56, fill="white", outline="#e9ecff", width=6)
    title_font = get_font(54, True)
    sub_font = get_font(30, True)
    badge_font = get_font(25, True)
    rank_font = get_font(28, True)
    name_font = get_font(30, True)
    percent_font = get_font(24, True)
    small_font = get_font(19, False)
    draw.rounded_rectangle((350, 105, 730, 155), radius=25, fill="#6370ff")
    draw_center_text(draw, (350, 105, 730, 155), "AI Maple Monster Match", badge_font, "white")
    draw_center_text(draw, (80, 185, 1000, 255), f"{name}님의 결과입니다", title_font, "#25243a")
    draw_center_text(draw, (80, 260, 1000, 310), "나의 메이플 몬스터 타입 TOP 3", sub_font, "#6c6a83")
    card_w, card_h, gap, start_x, y = 292, 560, 24, 92, 350
    for i, m in enumerate(results):
        x = start_x + i * (card_w + gap)
        border = "#ffd45a" if i == 0 else "#e5e9ff"
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=38, fill="#f8f9ff", outline=border, width=6 if i == 0 else 3)
        rank_color = "#ff9f3d" if i == 0 else "#6370ff"
        draw.ellipse((x + 110, y + 24, x + 182, y + 96), fill=rank_color)
        draw_center_text(draw, (x + 110, y + 24, x + 182, y + 96), f"{i+1}위", rank_font, "white")
        monster_img = fetch_monster_image(m.get("image_url", ""))
        if monster_img:
            bx = x + (card_w - monster_img.width) // 2
            by = y + 125
            img.paste(monster_img, (bx, by), monster_img)
        draw_wrapped_text(draw, m.get("name", "이름 없음"), x + 28, y + 330, card_w - 56, name_font, "#25243a", 5, 2)
        draw_center_text(draw, (x + 20, y + 425, x + card_w - 20, y + 465), f"매칭도 {m.get('percent', 90)}%", percent_font, "#6370ff")
        reason = str(m.get("reason", "전체 분위기가 비슷해요!"))[:45]
        draw_wrapped_text(draw, reason, x + 28, y + 485, card_w - 56, small_font, "#5e5b76", 5, 3)
    draw_center_text(draw, (80, 940, 1000, 990), "메이플 몬스터 타입 테스트 ✨", sub_font, "#25243a")
    out = BytesIO()
    img.save(out, format="PNG")
    out.seek(0)
    return out

@app.get("/")
def home():
    return {
        "message": "Maple Monster Match API is running!",
        "monster_count": len(df),
        "mode": "CLIP image vector matching v2",
        "csv_path": CSV_PATH,
        "clip_vector_path": CLIP_VECTOR_PATH,
        "clip_vector_exists": os.path.exists(CLIP_VECTOR_PATH),
        "clip_device": CLIP_DEVICE,
        "columns": list(df.columns),
    }

@app.get("/download-v3")
def download_v3():
    if not os.path.exists("monsters_ai_v3.csv"):
        raise HTTPException(status_code=404, detail="monsters_ai_v3.csv 파일이 아직 없습니다.")
    return FileResponse("monsters_ai_v3.csv", media_type="text/csv", filename="monsters_ai_v3.csv")

@app.post("/clip-match")
def clip_match(req: MatchRequest):
    try:
        results = run_clip_match(req.image_base64, top_k=10)
        return {"mode": "CLIP image vector matching v2", "device": CLIP_DEVICE, "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match")
def match_monster(req: MatchRequest):
    try:
        results = run_clip_match(req.image_base64, top_k=3)
        return {
            "analysis": {
                "vibe": "CLIP 이미지 벡터로 사람 사진과 몬스터 이미지를 직접 비교했어요.",
                "tags": ["clip", "image-vector", "visual-match"],
                "scores": {},
            },
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-result")
def save_result(req: SaveResultRequest):
    result_id = uuid.uuid4().hex[:10]
    safe_name = str(req.user_name).strip()[:12] or "친구"
    data = {"id": result_id, "user_name": safe_name, "analysis": req.analysis, "results": req.results}
    path = RESULT_DIR / f"{result_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"id": result_id, "share_url": f"{BASE_URL}/result/{result_id}", "og_image": f"{BASE_URL}/og/{result_id}.png"}

@app.get("/og/{result_id}.png")
def og_image(result_id: str):
    data = load_result(result_id)
    image_io = make_og_image(data)
    output_path = RESULT_DIR / f"{result_id}.png"
    with open(output_path, "wb") as f:
        f.write(image_io.getvalue())
    return FileResponse(output_path, media_type="image/png", filename=f"{result_id}.png")

@app.get("/result/{result_id}", response_class=HTMLResponse)
def result_page(result_id: str):
    data = load_result(result_id)
    user_name = escape_html(data.get("user_name", "친구"))
    results = data.get("results", [])
    top1 = escape_html(results[0].get("name", "메이플 몬스터")) if len(results) > 0 else "메이플 몬스터"
    top2 = escape_html(results[1].get("name", "")) if len(results) > 1 else ""
    top3 = escape_html(results[2].get("name", "")) if len(results) > 2 else ""
    title = f"{user_name}님의 메이플 몬스터 타입 결과가 도착했어요!"
    desc = f"🥇 {top1} · 🥈 {top2} · 🥉 {top3}"
    image_url = f"{BASE_URL}/og/{result_id}.png"
    page_url = f"{BASE_URL}/result/{result_id}"
    cards = ""
    for idx, m in enumerate(results[:3]):
        cards += f'''
        <div class="card {'top' if idx == 0 else ''}">
          <div class="rank">{idx + 1}위</div>
          <img src="{escape_html(m.get('image_url', ''))}" />
          <h2>{escape_html(m.get('name', '이름 없음'))}</h2>
          <b>매칭도 {m.get('percent', 90)}%</b>
          <p>{escape_html(m.get('reason', '전체 분위기가 비슷해요!'))}</p>
        </div>
        '''
    html = f'''
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title}</title>
  <meta property="og:type" content="website" />
  <meta property="og:title" content="{title}" />
  <meta property="og:description" content="{desc}" />
  <meta property="og:image" content="{image_url}" />
  <meta property="og:image:width" content="1080" />
  <meta property="og:image:height" content="1080" />
  <meta property="og:url" content="{page_url}" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="{title}" />
  <meta name="twitter:description" content="{desc}" />
  <meta name="twitter:image" content="{image_url}" />
  <style>
    body {{ margin:0; min-height:100vh; background:linear-gradient(135deg,#f7f8ff,#fff4fb); font-family:Arial,sans-serif; color:#25243a; padding:30px 16px; }}
    .wrap {{ max-width:760px; margin:0 auto; background:white; border-radius:30px; padding:26px; box-shadow:0 20px 60px rgba(74,86,160,.18); text-align:center; }}
    h1 {{ font-size:30px; margin:12px 0 8px; }}
    .desc {{ color:#6c6a83; margin-bottom:24px; line-height:1.6; }}
    .card {{ border:2px solid #e7eaff; border-radius:24px; padding:22px; margin:16px 0; background:#fafbff; }}
    .card.top {{ border-color:#ffd45a; box-shadow:0 12px 32px rgba(255,191,38,.22); }}
    .rank {{ display:inline-block; background:linear-gradient(135deg,#6370ff,#9b5cff); color:white; padding:8px 16px; border-radius:999px; font-weight:900; }}
    .top .rank {{ background:linear-gradient(135deg,#ffbd2f,#ff8f3d); }}
    img {{ width:170px; height:170px; object-fit:contain; display:block; margin:16px auto; background:white; border-radius:20px; }}
    p {{ color:#5e5b76; line-height:1.55; }}
    a {{ display:inline-block; margin-top:20px; padding:14px 20px; border-radius:16px; background:linear-gradient(135deg,#6370ff,#9b5cff); color:white; text-decoration:none; font-weight:900; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{title} 💌</h1>
    <p class="desc">{desc}</p>
    {cards}
    <a href="{FRONTEND_URL}">나도 테스트하기</a>
  </div>
</body>
</html>
'''
    return HTMLResponse(content=html)
