from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from pathlib import Path
import os, json, re, uuid, hashlib, math, requests
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


class MatchRequest(BaseModel):
    image_base64: str


class SaveResultRequest(BaseModel):
    user_name: str = "친구"
    analysis: dict
    results: list


TAG_CATEGORIES = {
    "shape_tag": ["round", "square", "sharp", "long", "small-face", "wide-face", "no-face"],
    "eye_tag": ["big-eye", "small-eye", "sharp-eye", "sleepy-eye", "simple-eye", "angry-eye", "no-eye"],
    "expression_tag": ["happy", "gentle", "blank", "angry", "playful", "mysterious", "scary", "sleepy"],
    "species_tag": ["human", "animal", "blob", "ghost", "object", "plant", "fish", "insect", "mixed-body"],
    "mood_tag": ["cute", "cool", "dark", "bright", "soft", "hard", "wild", "calm", "elegant", "magic", "strong", "weak"],
    "color_tag": ["green", "blue", "red", "yellow", "purple", "brown", "white", "black", "gray", "pink", "orange", "mixed-color"],
    "size_tag": ["tiny", "small", "medium", "large", "huge", "chubby", "thin", "tall", "short"],
    "feature_tag": ["fluffy", "jelly", "wood", "stone", "metal", "fire", "ice", "water", "poison", "shadow", "baby-like", "monster-like", "warrior", "mage", "forest", "robot", "undead", "pet-like", "boss-like", "npc-like", "render-risk"],
}

DNA_KEYS = [
    "human_like",
    "animal_like",
    "blob_like",
    "ghost_like",
    "object_like",
    "plant_like",
    "cute_level",
    "dark_level",
    "power_level",
    "energy",
]

BLOCKED_NAME_KEYWORDS = ["코-크", "코크", "Coke", "coke"]


def clean_base64(image_base64: str):
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    return image_base64.strip().replace("\n", "").replace("\r", "").replace(" ", "")


def make_image_hash(image_base64: str):
    return hashlib.md5(clean_base64(image_base64).encode("utf-8")).hexdigest()


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


def get_float(row, cols, default=0):
    for col in cols:
        if col in row.index:
            try:
                return float(row[col])
            except Exception:
                pass
    return default


def normalize_score(value, default=5):
    try:
        num = int(float(value))
        return max(0, min(num, 10))
    except Exception:
        return default


def normalize_category_tag(value, category, default):
    value = str(value).strip().lower()
    allowed = TAG_CATEGORIES.get(category, [])
    if value in allowed:
        return value
    return default


def tags_from_row(row):
    tags = []
    for col in [
        "shape_tag",
        "eye_tag",
        "expression_tag",
        "species_tag",
        "mood_tag",
        "color_tag",
        "size_tag",
        "feature_tag",
    ]:
        value = get_value(row, [col], "")
        if value:
            tags.append(value)

    if not tags:
        raw = get_value(row, ["match_tags"], "")
        tags = [x.strip() for x in raw.split(",") if x.strip()]

    return tags


def vector_from_row(row):
    return [get_float(row, [key], 0) for key in DNA_KEYS]


def vector_from_analysis(a):
    return [float(a.get(key, 0)) for key in DNA_KEYS]


def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0
    return dot / (n1 * n2)


def tag_score(user_tags, monster_tags):
    user_set = set(user_tags)
    monster_set = set(monster_tags)
    common = list(user_set & monster_set)
    return len(common), common


def stable_tiebreaker(image_hash: str, monster_name: str):
    key = f"{image_hash}-{monster_name}"
    value = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(value[:6], 16) / 0xFFFFFF


def analyze_user_image(image_base64: str):
    tag_vocab_text = json.dumps(TAG_CATEGORIES, ensure_ascii=False, indent=2)

    prompt = f"""
너는 실제 사람 얼굴 사진을 보고, 메이플스토리 몬스터와 닮은꼴 매칭하기 위한 Human DNA를 만드는 분석기야.

중요:
- 신원/이름/성별/나이 추정 금지
- 외모 비하 금지
- JSON만 출력
- 모든 수치 점수는 0~10 정수
- 태그는 반드시 태그 사전에서만 선택
- 각 태그 카테고리마다 정확히 1개씩 선택
- 사람 얼굴을 현실적으로만 보지 말고, 메이플 몬스터로 치환했을 때의 캐릭터 느낌으로 판단

태그 사전:
{tag_vocab_text}

출력 JSON:
{{
  "human_like": 8,
  "animal_like": 1,
  "blob_like": 2,
  "ghost_like": 0,
  "object_like": 0,
  "plant_like": 0,

  "cute_level": 7,
  "dark_level": 2,
  "power_level": 4,
  "energy": 5,

  "shape_tag": "round",
  "eye_tag": "small-eye",
  "expression_tag": "gentle",
  "species_tag": "human",
  "mood_tag": "calm",
  "color_tag": "mixed-color",
  "size_tag": "medium",
  "feature_tag": "npc-like",

  "description": "차분하고 부드러운 인상의 캐릭터 느낌"
}}
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
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ],
    )

    raw = safe_json(response.choices[0].message.content)

    result = {
        "human_like": normalize_score(raw.get("human_like"), 5),
        "animal_like": normalize_score(raw.get("animal_like"), 0),
        "blob_like": normalize_score(raw.get("blob_like"), 0),
        "ghost_like": normalize_score(raw.get("ghost_like"), 0),
        "object_like": normalize_score(raw.get("object_like"), 0),
        "plant_like": normalize_score(raw.get("plant_like"), 0),
        "cute_level": normalize_score(raw.get("cute_level"), 5),
        "dark_level": normalize_score(raw.get("dark_level"), 3),
        "power_level": normalize_score(raw.get("power_level"), 4),
        "energy": normalize_score(raw.get("energy"), 5),
        "shape_tag": normalize_category_tag(raw.get("shape_tag"), "shape_tag", "round"),
        "eye_tag": normalize_category_tag(raw.get("eye_tag"), "eye_tag", "simple-eye"),
        "expression_tag": normalize_category_tag(raw.get("expression_tag"), "expression_tag", "blank"),
        "species_tag": normalize_category_tag(raw.get("species_tag"), "species_tag", "human"),
        "mood_tag": normalize_category_tag(raw.get("mood_tag"), "mood_tag", "calm"),
        "color_tag": normalize_category_tag(raw.get("color_tag"), "color_tag", "mixed-color"),
        "size_tag": normalize_category_tag(raw.get("size_tag"), "size_tag", "medium"),
        "feature_tag": normalize_category_tag(raw.get("feature_tag"), "feature_tag", "npc-like"),
        "description": str(raw.get("description", "사진의 캐릭터 분위기를 분석했어요.")).strip(),
    }

    result["match_tags"] = [
        result["shape_tag"],
        result["eye_tag"],
        result["expression_tag"],
        result["species_tag"],
        result["mood_tag"],
        result["color_tag"],
        result["size_tag"],
        result["feature_tag"],
    ]

    return result


def score_monster(user, row, image_hash):
    name = get_value(row, ["name"], "이름 없음")

    user_vec = vector_from_analysis(user)
    monster_vec = vector_from_row(row)
    dna_sim = cosine_similarity(user_vec, monster_vec)

    user_tags = user.get("match_tags", [])
    monster_tags = tags_from_row(row)
    common_count, common_tags = tag_score(user_tags, monster_tags)

    score = 0
    score += dna_sim * 70
    score += common_count * 6

    if user.get("species_tag") == get_value(row, ["species_tag"], ""):
        score += 8
    if user.get("expression_tag") == get_value(row, ["expression_tag"], ""):
        score += 5
    if user.get("eye_tag") == get_value(row, ["eye_tag"], ""):
        score += 5
    if user.get("mood_tag") == get_value(row, ["mood_tag"], ""):
        score += 5

    object_like = get_float(row, ["object_like"], 0)
    plant_like = get_float(row, ["plant_like"], 0)
    species_tag = get_value(row, ["species_tag"], "")

    if species_tag in ["object", "plant", "fish", "insect"] and user.get("species_tag") == "human":
        score -= 8

    if object_like >= 8:
        score -= 5

    if plant_like >= 8 and user.get("plant_like", 0) < 5:
        score -= 5

    if "render-risk" in monster_tags:
        score -= 20

    score += stable_tiebreaker(image_hash, name) * 1.5

    return round(score, 4), round(dna_sim, 4), common_tags, monster_tags


def add_percent(results):
    if not results:
        return results

    max_score = results[0]["score"] or 1

    for index, r in enumerate(results):
        base = int((r["score"] / max_score) * 96)

        if index == 0:
            r["percent"] = max(90, min(base, 98))
        elif index == 1:
            r["percent"] = max(80, min(base, 89))
        else:
            r["percent"] = max(70, min(base, 79))

    return results


def pick_diverse_top3(results):
    if not results:
        return []

    selected = [results[0]]
    used = {results[0]["name"]}
    species_count = {results[0].get("species_tag", ""): 1}

    for r in results[1:]:
        if r["name"] in used:
            continue

        species = r.get("species_tag", "")

        if species_count.get(species, 0) >= 2:
            continue

        selected.append(r)
        used.add(r["name"])
        species_count[species] = species_count.get(species, 0) + 1

        if len(selected) >= 3:
            return selected

    for r in results[1:]:
        if r["name"] in used:
            continue

        selected.append(r)
        used.add(r["name"])

        if len(selected) >= 3:
            return selected

    return selected


def generate_reason(monster_name, user, common_tags):
    tag_text = ", ".join(common_tags[:5]) if common_tags else "전체 분위기"

    try:
        prompt = f"""
메이플 몬스터 닮은꼴 테스트 결과 설명을 써줘.

몬스터: {monster_name}
사용자 분석: {user.get("description")}
공통 특징 태그: {tag_text}

조건:
- 2문장
- 90자 이내
- 귀엽고 재밌게
- 외모 비하 금지
- 신원/성별/나이 추정 금지
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return f"{monster_name}와 {tag_text} 느낌이 비슷해서 매칭됐어요!"


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

        draw_wrapped_text(draw, m.get("name", "이름 없음"), x + 28, y + 330, card_w - 56, name_font, "#25243a", 5, 2)

        draw_center_text(
            draw,
            (x + 20, y + 425, x + card_w - 20, y + 465),
            f"닮은 정도 {m.get('percent', 90)}%",
            percent_font,
            "#6370ff",
        )

        reason = str(m.get("reason", "전체 분위기가 비슷해요!"))[:45]
        draw_wrapped_text(draw, reason, x + 28, y + 485, card_w - 56, small_font, "#5e5b76", 5, 3)

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
        "mode": "A9 DNA vector matching",
        "csv_path": CSV_PATH,
        "columns": list(df.columns),
    }


@app.get("/download-v3")
def download_v3():
    if not os.path.exists("monsters_ai_v3.csv"):
        raise HTTPException(status_code=404, detail="monsters_ai_v3.csv 파일이 아직 없습니다.")

    return FileResponse(
        "monsters_ai_v3.csv",
        media_type="text/csv",
        filename="monsters_ai_v3.csv",
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

            score, dna_sim, common_tags, monster_tags = score_monster(user, row, image_hash)

            results.append({
                "name": name,
                "image_url": get_value(row, ["image_url"], ""),
                "score": score,
                "dna_similarity": dna_sim,
                "common_tags": common_tags,
                "tags": monster_tags,
                "species_tag": get_value(row, ["species_tag"], ""),
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

        picked = pick_diverse_top3(unique)
        picked = add_percent(picked)

        for r in picked:
            r["reason"] = generate_reason(r["name"], user, r.get("common_tags", []))

        return {
            "image_hash": image_hash,
            "analysis": {
                "tags": user["match_tags"],
                "vibe": user["description"],
                "scores": {key: user.get(key) for key in DNA_KEYS},
                "shape_tag": user["shape_tag"],
                "eye_tag": user["eye_tag"],
                "expression_tag": user["expression_tag"],
                "species_tag": user["species_tag"],
                "mood_tag": user["mood_tag"],
                "color_tag": user["color_tag"],
                "size_tag": user["size_tag"],
                "feature_tag": user["feature_tag"],
            },
            "results": picked,
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

    return FileResponse(output_path, media_type="image/png", filename=f"{result_id}.png")


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
        common = ", ".join(m.get("common_tags", [])[:4])

        cards += f"""
        <div class="card {'top' if idx == 0 else ''}">
          <div class="rank">{idx + 1}위</div>
          <img src="{escape_html(m.get('image_url', ''))}" />
          <h2>{escape_html(m.get('name', '이름 없음'))}</h2>
          <b>닮은 정도 {m.get('percent', 90)}%</b>
          <p>{escape_html(m.get('reason', '전체 분위기가 비슷해요!'))}</p>
          <small>공통 특징: {escape_html(common)}</small>
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
    h1 {{ font-size: 30px; margin: 12px 0 8px; }}
    .desc {{ color: #6c6a83; margin-bottom: 24px; line-height: 1.6; }}
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
    p {{ color: #5e5b76; line-height: 1.55; }}
    small {{ color: #6370ff; font-weight: 800; }}
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
