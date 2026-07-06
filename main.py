from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from pathlib import Path

import os
import json
import re
import uuid
import hashlib
import requests
import pandas as pd


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CSV_PATH = "monsters_ai_v2.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = "monsters_ai.csv"

df = pd.read_csv(CSV_PATH).fillna("")

RESULT_DIR = Path("saved_results")
RESULT_DIR.mkdir(exist_ok=True)

BASE_URL = os.getenv("BASE_URL", "https://maple-monster-match-v2.onrender.com")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://maple-monster-frontend.onrender.com")


class MatchRequest(BaseModel):
    image_base64: str


class SaveResultRequest(BaseModel):
    user_name: str = "친구"
    analysis: dict
    results: list


FACE_SHAPES = ["round", "square", "sharp", "long", "small", "wide"]

VIBES = [
    "cute",
    "calm",
    "dark",
    "mysterious",
    "playful",
    "cool",
    "strong",
    "sleepy",
    "bright",
    "elegant",
]

EYE_STYLES = [
    "big",
    "small",
    "sharp",
    "sleepy",
    "angry",
    "none",
    "simple",
]

BODY_TYPES = [
    "blob",
    "animal",
    "humanoid",
    "object",
    "plant",
    "fish",
    "insect",
    "ghost",
]

BLOCKED_NAME_KEYWORDS = [
    "코-크",
    "코크",
    "Coke",
    "coke",
]


def clean_base64(image_base64: str):
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    return image_base64.strip().replace("\n", "").replace("\r", "").replace(" ", "")


def make_image_hash(image_base64: str):
    return hashlib.md5(clean_base64(image_base64).encode("utf-8")).hexdigest()


def stable_tiebreaker(image_hash: str, monster_name: str):
    key = f"{image_hash}-{monster_name}"
    value = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(value[:6], 16) / 0xFFFFFF


def safe_json(text: str):
    text = text.strip()
    text = re.sub(r"```json|```", "", text).strip()

    match = re.search(r"\{.*\}", text, re.S)
    if match:
        text = match.group(0)

    return json.loads(text)


def get_value(row, cols, default=""):
    for col in cols:
        if col in row.index and str(row[col]).strip():
            return str(row[col]).strip()
    return default


def get_float(row, cols, default=5):
    for col in cols:
        if col in row.index:
            try:
                return float(row[col])
            except:
                pass
    return default


def normalize_choice(value, allowed, default):
    value = str(value).strip().lower()

    if value in allowed:
        return value

    for item in allowed:
        if item in value or value in item:
            return item

    return default


def analyze_user_image(image_base64: str):
    prompt = """
너는 실제 사람 얼굴 사진을 보고 메이플스토리 몬스터 닮은꼴 매칭용 특징만 뽑는 분석기야.

중요:
- 신원/이름/성별/나이 추정 금지
- 외모 비하 금지
- 아래 JSON만 답하기
- 같은 사진이면 최대한 같은 분석이 나오게 일관적으로 판단하기
- 실제 얼굴형 그대로가 아니라, 몬스터와 매칭하기 위한 '캐릭터식 인상'으로 판단하기
- 사람 얼굴을 무조건 humanoid로만 판단하지 말고, 분위기상 blob/animal/ghost 느낌도 가능하면 반영하기
- 눈 크기는 실제 눈 크기보다 캐릭터식 인상으로 판단하기

face_shape 후보:
round, square, sharp, long, small, wide

vibe 후보:
cute, calm, dark, mysterious, playful, cool, strong, sleepy, bright, elegant

eye_style 후보:
big, small, sharp, sleepy, angry, none, simple

body_type 후보:
blob, animal, humanoid, object, plant, fish, insect, ghost

JSON 형식:
{
  "face_shape": "round",
  "vibe": "calm",
  "eye_style": "small",
  "body_type": "humanoid",
  "scores": {
    "cute_level": 1~10,
    "dark_level": 1~10,
    "power_level": 1~10
  },
  "tags": ["round", "calm", "small-eyes"],
  "description": "짧은 분위기 설명"
}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
    )

    analysis = safe_json(response.choices[0].message.content)
    scores = analysis.get("scores", {})

    return {
        "face_shape": normalize_choice(analysis.get("face_shape", "round"), FACE_SHAPES, "round"),
        "vibe": normalize_choice(analysis.get("vibe", "calm"), VIBES, "calm"),
        "eye_style": normalize_choice(analysis.get("eye_style", "simple"), EYE_STYLES, "simple"),
        "body_type": normalize_choice(analysis.get("body_type", "humanoid"), BODY_TYPES, "humanoid"),
        "cute_level": float(scores.get("cute_level", 5)),
        "dark_level": float(scores.get("dark_level", 3)),
        "power_level": float(scores.get("power_level", 4)),
        "description": analysis.get("description", ""),
        "tags": analysis.get("tags", []),
    }


def shape_score(user_shape, monster_shape):
    if user_shape == monster_shape:
        return 30

    similar_groups = [
        {"round", "small", "wide"},
        {"sharp", "long"},
        {"square", "wide"},
    ]

    for group in similar_groups:
        if user_shape in group and monster_shape in group:
            return 18

    return 4


def vibe_score(user_vibe, monster_vibe):
    if user_vibe == monster_vibe:
        return 30

    soft_group = {"cute", "calm", "sleepy", "bright", "playful"}
    dark_group = {"dark", "mysterious", "cool", "strong", "elegant"}

    if user_vibe in soft_group and monster_vibe in soft_group:
        return 12

    if user_vibe in dark_group and monster_vibe in dark_group:
        return 12

    return 2


def eye_score(user_eye, monster_eye):
    if user_eye == monster_eye:
        return 24

    if {user_eye, monster_eye} <= {"small", "simple", "sleepy"}:
        return 12

    if {user_eye, monster_eye} <= {"sharp", "angry"}:
        return 12

    if {user_eye, monster_eye} <= {"big", "simple"}:
        return 8

    if user_eye == "none" or monster_eye == "none":
        return -10

    return 2


def body_score(user_body, monster_body):
    if user_body == monster_body:
        return 18

    if monster_body == "humanoid":
        return 8

    if monster_body in ["animal", "blob", "ghost"]:
        return 10

    if monster_body in ["object", "plant", "fish", "insect"]:
        return -8

    return 0


def numeric_similarity(user_value, monster_value, weight):
    diff = abs(user_value - monster_value)
    return max(0, 10 - diff) * weight


def score_monster(user, row, image_hash):
    name = get_value(row, ["name"], "이름 없음")

    monster_face = normalize_choice(get_value(row, ["face_shape"], "round"), FACE_SHAPES, "round")
    monster_vibe = normalize_choice(get_value(row, ["vibe"], "calm"), VIBES, "calm")
    monster_eye = normalize_choice(get_value(row, ["eye_style"], "simple"), EYE_STYLES, "simple")
    monster_body = normalize_choice(get_value(row, ["body_type"], "object"), BODY_TYPES, "object")

    monster_cute = get_float(row, ["cute_level"], 5)
    monster_dark = get_float(row, ["dark_level"], 3)
    monster_power = get_float(row, ["power_level"], 4)

    human_match = get_float(row, ["human_match_score"], 5)
    face_visibility = get_float(row, ["face_visibility"], 5)
    object_like = get_float(row, ["object_like"], 1)
    plant_like = get_float(row, ["plant_like"], 1)

    score = 0

    score += shape_score(user["face_shape"], monster_face)
    score += vibe_score(user["vibe"], monster_vibe)
    score += eye_score(user["eye_style"], monster_eye)
    score += body_score(user["body_type"], monster_body)

    score += numeric_similarity(user["cute_level"], monster_cute, 2.8)
    score += numeric_similarity(user["dark_level"], monster_dark, 2.0)
    score += numeric_similarity(user["power_level"], monster_power, 2.0)

    # 너무 강했던 기본 보너스 약화
    score += human_match * 1.2
    score += face_visibility * 0.8

    # 너무 물건/식물스러운 몬스터는 약간 감점
    score -= object_like * 2.0
    score -= plant_like * 2.2

    # 실제 얼굴 매칭에서 너무 멀어지는 유형만 약한 감점
    if monster_body in ["object", "plant", "fish", "insect"]:
        score -= 8

    # 이벤트/렌더링 오류 계열 약한 감점은 이름 차단에서 처리
    score += stable_tiebreaker(image_hash, name) * 2

    return round(score, 4)


def add_percent(results):
    if not results:
        return results

    max_score = results[0]["score"] or 1

    for index, r in enumerate(results):
        base_percent = int((r["score"] / max_score) * 96)

        if index == 0:
            r["percent"] = max(90, min(base_percent, 98))
        elif index == 1:
            r["percent"] = max(80, min(base_percent, 89))
        else:
            r["percent"] = max(70, min(base_percent, 79))

    return results


def pick_diverse_top3(results):
    if not results:
        return []

    # 1위는 진짜 점수 1등 유지
    selected = [results[0]]
    used_names = {results[0]["name"]}
    body_counts = {results[0].get("body_type", ""): 1}
    vibe_counts = {results[0].get("vibe", ""): 1}

    # 2~3위만 다양성 적용
    for r in results[1:]:
        name = r["name"]
        body = r.get("body_type", "")
        vibe = r.get("vibe", "")

        if name in used_names:
            continue

        if body_counts.get(body, 0) >= 1:
            continue

        if vibe_counts.get(vibe, 0) >= 2:
            continue

        selected.append(r)
        used_names.add(name)
        body_counts[body] = body_counts.get(body, 0) + 1
        vibe_counts[vibe] = vibe_counts.get(vibe, 0) + 1

        if len(selected) >= 3:
            return selected

    # 부족하면 body 2개까지 허용
    for r in results[1:]:
        name = r["name"]
        body = r.get("body_type", "")

        if name in used_names:
            continue

        if body_counts.get(body, 0) >= 2:
            continue

        selected.append(r)
        used_names.add(name)
        body_counts[body] = body_counts.get(body, 0) + 1

        if len(selected) >= 3:
            return selected

    # 그래도 부족하면 점수순 채움
    for r in results[1:]:
        name = r["name"]

        if name in used_names:
            continue

        selected.append(r)
        used_names.add(name)

        if len(selected) >= 3:
            return selected

    return selected


def generate_reason(monster_name, user):
    try:
        prompt = f"""
사용자 분위기:
- 얼굴형 느낌: {user["face_shape"]}
- 분위기: {user["vibe"]}
- 눈매 느낌: {user["eye_style"]}
- 귀여움: {user["cute_level"]}
- 어두움: {user["dark_level"]}
- 포스: {user["power_level"]}
- 설명: {user["description"]}

매칭 몬스터: {monster_name}

조건:
- 2문장
- 귀엽고 재밌게
- 메이플 닮은꼴 테스트 결과처럼 작성
- 외모 비하 금지
- 신원/성별/나이 추정 금지
- 80자 이내
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return f"{monster_name}와 전체 분위기가 비슷해요!"


def escape_html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def load_result(result_id: str):
    path = RESULT_DIR / f"{result_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_font(size: int, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for path in candidates:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size)

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
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x1 + ((x2 - x1) - w) / 2
    y = y1 + ((y2 - y1) - h) / 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_wrapped_text(draw, text, x, y, max_width, font, fill, line_gap=8, max_lines=3):
    chars = list(str(text))
    lines = []
    current = ""

    for ch in chars:
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

    draw.rounded_rectangle((370, 105, 710, 155), radius=25, fill="#6370ff")
    draw_center_text(draw, (370, 105, 710, 155), "AI Maple Monster Match", badge_font, "white")

    draw_center_text(draw, (80, 185, 1000, 255), f"{name}님의 결과입니다", title_font, "#25243a")
    draw_center_text(draw, (80, 260, 1000, 310), "닮은 메이플 몬스터 TOP 3", sub_font, "#6c6a83")

    card_w = 292
    card_h = 560
    gap = 24
    start_x = 92
    y = 350

    for i, m in enumerate(results):
        x = start_x + i * (card_w + gap)
        border = "#ffd45a" if i == 0 else "#e5e9ff"

        draw.rounded_rectangle(
            (x, y, x + card_w, y + card_h),
            radius=38,
            fill="#f8f9ff",
            outline=border,
            width=6 if i == 0 else 3,
        )

        rank_color = "#ff9f3d" if i == 0 else "#6370ff"
        draw.ellipse((x + 110, y + 24, x + 182, y + 96), fill=rank_color)
        draw_center_text(draw, (x + 110, y + 24, x + 182, y + 96), f"{i+1}위", rank_font, "white")

        monster_img = fetch_monster_image(m.get("image_url", ""))
        if monster_img:
            bx = x + (card_w - monster_img.width) // 2
            by = y + 125
            img.paste(monster_img, (bx, by), monster_img)
        else:
            draw.rounded_rectangle((x + 58, y + 130, x + 234, y + 306), radius=28, fill="white")

        draw_wrapped_text(
            draw,
            m.get("name", "이름 없음"),
            x + 28,
            y + 330,
            card_w - 56,
            name_font,
            "#25243a",
            line_gap=5,
            max_lines=2,
        )

        draw_center_text(
            draw,
            (x + 20, y + 425, x + card_w - 20, y + 465),
            f"닮은 정도 {m.get('percent', 90)}%",
            percent_font,
            "#6370ff",
        )

        reason = str(m.get("reason", "전체 분위기가 비슷해요!"))[:45]
        draw_wrapped_text(
            draw,
            reason,
            x + 28,
            y + 485,
            card_w - 56,
            small_font,
            "#5e5b76",
            line_gap=5,
            max_lines=3,
        )

    draw_center_text(draw, (80, 940, 1000, 990), "메이플 몬스터 닮은꼴 테스트 ✨", sub_font, "#25243a")

    out = BytesIO()
    img.save(out, format="PNG")
    out.seek(0)
    return out


@app.get("/")
def home():
    return {
        "message": "Maple Monster Match API is running!",
        "monster_count": len(df),
        "mode": "A8 balanced matching",
        "csv_path": CSV_PATH,
        "columns": list(df.columns),
    }


@app.get("/download-v2")
def download_v2():
    if not os.path.exists("monsters_ai_v2.csv"):
        raise HTTPException(status_code=404, detail="monsters_ai_v2.csv 파일이 아직 없습니다.")

    return FileResponse(
        "monsters_ai_v2.csv",
        media_type="text/csv",
        filename="monsters_ai_v2.csv",
    )

@app.get("/download-v3")
def download_v3():
    if not os.path.exists("monsters_ai_v3.csv"):
        raise HTTPException(status_code=404, detail="monsters_ai_v3.csv 파일이 아직 없습니다.")

    return FileResponse(
        "monsters_ai_v3.csv",
        media_type="text/csv",
        filename="monsters_ai_v3.csv"
    )

@app.post("/match")
def match_monster(req: MatchRequest):
    try:
        image_base64 = clean_base64(req.image_base64)
        image_hash = make_image_hash(image_base64)

        user = analyze_user_image(image_base64)

        results = []

        for _, row in df.iterrows():
            name = get_value(row, ["name"], "이름 없음")

            if any(keyword in name for keyword in BLOCKED_NAME_KEYWORDS):
                continue

            image_url = get_value(row, ["image_url"], "")
            monster_vibe = get_value(row, ["vibe"], "")
            monster_body = get_value(row, ["body_type"], "")

            score = score_monster(user, row, image_hash)

            results.append({
                "name": name,
                "image_url": image_url,
                "score": score,
                "vibe": monster_vibe,
                "body_type": monster_body,
                "tags": [
                    get_value(row, ["face_shape"], ""),
                    monster_vibe,
                    get_value(row, ["eye_style"], ""),
                    monster_body,
                ],
                "reason": "",
            })

        results = sorted(results, key=lambda x: (-x["score"], x["name"]))

        unique = []
        seen = set()
        for r in results:
            if r["name"] in seen:
                continue
            seen.add(r["name"])
            unique.append(r)

        unique = pick_diverse_top3(unique)
        unique = add_percent(unique)

        for r in unique:
            r["reason"] = generate_reason(r["name"], user)

        return {
            "image_hash": image_hash,
            "analysis": {
                "tags": user["tags"],
                "vibe": user["description"],
                "scores": {
                    "cute_level": user["cute_level"],
                    "dark_level": user["dark_level"],
                    "power_level": user["power_level"],
                },
                "face_shape": user["face_shape"],
                "main_vibe": user["vibe"],
                "eye_style": user["eye_style"],
                "body_type": user["body_type"],
            },
            "results": unique,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save-result")
def save_result(req: SaveResultRequest):
    result_id = uuid.uuid4().hex[:10]
    safe_name = str(req.user_name).strip()[:12] or "친구"

    data = {
        "id": result_id,
        "user_name": safe_name,
        "analysis": req.analysis,
        "results": req.results,
    }

    path = RESULT_DIR / f"{result_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "id": result_id,
        "share_url": f"{BASE_URL}/result/{result_id}",
        "og_image": f"{BASE_URL}/og/{result_id}.png",
    }


@app.get("/og/{result_id}.png")
def og_image(result_id: str):
    data = load_result(result_id)
    image_io = make_og_image(data)

    output_path = RESULT_DIR / f"{result_id}.png"
    with open(output_path, "wb") as f:
        f.write(image_io.getvalue())

    return FileResponse(
        output_path,
        media_type="image/png",
        filename=f"{result_id}.png",
    )


@app.get("/result/{result_id}", response_class=HTMLResponse)
def result_page(result_id: str):
    data = load_result(result_id)

    user_name = escape_html(data.get("user_name", "친구"))
    results = data.get("results", [])

    top1 = escape_html(results[0].get("name", "메이플 몬스터")) if len(results) > 0 else "메이플 몬스터"
    top2 = escape_html(results[1].get("name", "")) if len(results) > 1 else ""
    top3 = escape_html(results[2].get("name", "")) if len(results) > 2 else ""

    title = f"{user_name}님의 메이플 몬스터 닮은꼴 결과가 도착했어요!"
    desc = f"🥇 {top1} · 🥈 {top2} · 🥉 {top3}"
    image_url = f"{BASE_URL}/og/{result_id}.png"
    page_url = f"{BASE_URL}/result/{result_id}"

    cards = ""
    for idx, m in enumerate(results[:3]):
        cards += f"""
        <div class="card {'top' if idx == 0 else ''}">
          <div class="rank">{idx + 1}위</div>
          <img src="{escape_html(m.get('image_url', ''))}" />
          <h2>{escape_html(m.get('name', '이름 없음'))}</h2>
          <b>닮은 정도 {m.get('percent', 90)}%</b>
          <p>{escape_html(m.get('reason', '전체 분위기가 비슷해요!'))}</p>
        </div>
        """

    html = f"""
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
    body {{
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(135deg,#f7f8ff,#fff4fb);
      font-family: Arial, sans-serif;
      color: #25243a;
      padding: 30px 16px;
    }}
    .wrap {{
      max-width: 760px;
      margin: 0 auto;
      background: white;
      border-radius: 30px;
      padding: 26px;
      box-shadow: 0 20px 60px rgba(74,86,160,.18);
      text-align: center;
    }}
    h1 {{
      font-size: 30px;
      margin: 12px 0 8px;
    }}
    .desc {{
      color: #6c6a83;
      margin-bottom: 24px;
      line-height: 1.6;
    }}
    .card {{
      border: 2px solid #e7eaff;
      border-radius: 24px;
      padding: 22px;
      margin: 16px 0;
      background: #fafbff;
    }}
    .card.top {{
      border-color: #ffd45a;
      box-shadow: 0 12px 32px rgba(255,191,38,.22);
    }}
    .rank {{
      display: inline-block;
      background: linear-gradient(135deg,#6370ff,#9b5cff);
      color: white;
      padding: 8px 16px;
      border-radius: 999px;
      font-weight: 900;
    }}
    .top .rank {{
      background: linear-gradient(135deg,#ffbd2f,#ff8f3d);
    }}
    img {{
      width: 170px;
      height: 170px;
      object-fit: contain;
      display: block;
      margin: 16px auto;
      background: white;
      border-radius: 20px;
    }}
    h2 {{
      margin: 8px 0;
    }}
    p {{
      color: #5e5b76;
      line-height: 1.55;
    }}
    a {{
      display: inline-block;
      margin-top: 20px;
      padding: 14px 20px;
      border-radius: 16px;
      background: linear-gradient(135deg,#6370ff,#9b5cff);
      color: white;
      text-decoration: none;
      font-weight: 900;
    }}
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
"""
    return HTMLResponse(content=html)
