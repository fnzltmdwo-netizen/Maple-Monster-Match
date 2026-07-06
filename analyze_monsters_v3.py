import os
import json
import time
import re
from io import BytesIO

import pandas as pd
import requests
from PIL import Image
from openai import OpenAI


INPUT_CSV_CANDIDATES = [
    "monsters_ai_v2.csv",
    "monsters_ai.csv",
]

OUTPUT_CSV = "monsters_ai_v3.csv"

# 먼저 30개만 테스트
# 전체 분석할 때는 None으로 변경
LIMIT = 30

SLEEP_SECONDS = 1.2
MAX_RETRIES = 3

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


TAG_VOCAB = [
    # face
    "round",
    "square",
    "sharp",
    "long",
    "small-face",
    "wide-face",
    "no-face",

    # eyes
    "big-eye",
    "small-eye",
    "sharp-eye",
    "sleepy-eye",
    "simple-eye",
    "angry-eye",
    "no-eye",

    # expression
    "happy",
    "gentle",
    "blank",
    "angry",
    "playful",
    "mysterious",
    "scary",
    "sleepy",

    # body/species
    "human",
    "animal",
    "blob",
    "ghost",
    "object",
    "plant",
    "fish",
    "insect",
    "mixed-body",

    # mood
    "cute",
    "cool",
    "dark",
    "bright",
    "soft",
    "hard",
    "wild",
    "calm",
    "elegant",
    "magic",
    "strong",
    "weak",

    # colors
    "green",
    "blue",
    "red",
    "yellow",
    "purple",
    "brown",
    "white",
    "black",
    "gray",
    "pink",
    "orange",
    "mixed-color",

    # size / silhouette
    "tiny",
    "small",
    "medium",
    "large",
    "huge",
    "chubby",
    "thin",
    "round-body",
    "tall",
    "short",

    # texture / visual
    "fluffy",
    "jelly",
    "wood",
    "stone",
    "metal",
    "fire",
    "ice",
    "water",
    "poison",
    "shadow",

    # character archetype
    "baby-like",
    "monster-like",
    "warrior",
    "mage",
    "forest",
    "robot",
    "undead",
    "pet-like",
    "boss-like",
    "npc-like",

    # render quality
    "render-risk",
]


COLUMNS = [
    "name",
    "mob_id",
    "source_url",
    "image_url",

    "face_shape",
    "eye_style",
    "expression",
    "body_type",
    "color_tone",

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

    "match_tags",
    "match_note",
]


def find_input_csv():
    for path in INPUT_CSV_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("monsters_ai_v2.csv 또는 monsters_ai.csv 파일이 없습니다.")


def safe_str(value, default=""):
    if pd.isna(value):
        return default
    return str(value).strip()


def get_value(row, names, default=""):
    for name in names:
        if name in row.index:
            value = safe_str(row[name])
            if value:
                return value
    return default


def download_image(image_url):
    res = requests.get(image_url, timeout=20)
    res.raise_for_status()

    image = Image.open(BytesIO(res.content)).convert("RGB")
    image.thumbnail((768, 768))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)

    return buffer.getvalue()


def image_bytes_to_base64(image_bytes):
    import base64
    return base64.b64encode(image_bytes).decode("utf-8")


def clean_json_text(text):
    text = text.strip()
    text = re.sub(r"```json|```", "", text).strip()

    match = re.search(r"\{.*\}", text, re.S)
    if match:
        text = match.group(0)

    return text


def parse_json(text):
    cleaned = clean_json_text(text)
    return json.loads(cleaned)


def normalize_score(value, default=5):
    try:
        num = int(float(value))
        return max(0, min(num, 10))
    except Exception:
        return default


def normalize_choice(value, allowed, default):
    value = str(value).strip().lower()
    if value in allowed:
        return value
    return default


def normalize_tags(tags):
    if not isinstance(tags, list):
        tags = []

    cleaned = []
    vocab_set = set(TAG_VOCAB)

    for tag in tags:
        tag = str(tag).strip().lower()
        if tag in vocab_set and tag not in cleaned:
            cleaned.append(tag)

    return cleaned[:8]


def fill_missing_tags(tags, analysis):
    tags = list(tags)

    def add(tag):
        if tag in TAG_VOCAB and tag not in tags and len(tags) < 8:
            tags.append(tag)

    face_shape = str(analysis.get("face_shape", "")).lower()
    eye_style = str(analysis.get("eye_style", "")).lower()
    expression = str(analysis.get("expression", "")).lower()
    body_type = str(analysis.get("body_type", "")).lower()
    color_tone = str(analysis.get("color_tone", "")).lower()

    face_map = {
        "round": "round",
        "square": "square",
        "sharp": "sharp",
        "long": "long",
        "small": "small-face",
        "wide": "wide-face",
        "none": "no-face",
    }

    eye_map = {
        "big": "big-eye",
        "small": "small-eye",
        "sharp": "sharp-eye",
        "sleepy": "sleepy-eye",
        "angry": "angry-eye",
        "simple": "simple-eye",
        "none": "no-eye",
    }

    body_map = {
        "humanoid": "human",
        "animal": "animal",
        "blob": "blob",
        "ghost": "ghost",
        "object": "object",
        "plant": "plant",
        "fish": "fish",
        "insect": "insect",
        "mixed": "mixed-body",
    }

    color_map = {
        "warm": "orange",
        "cool": "blue",
        "dark": "dark",
        "bright": "bright",
        "green": "green",
        "blue": "blue",
        "red": "red",
        "yellow": "yellow",
        "neutral": "gray",
        "mixed": "mixed-color",
    }

    add(face_map.get(face_shape, "round"))
    add(eye_map.get(eye_style, "simple-eye"))
    add(expression if expression in TAG_VOCAB else "blank")
    add(body_map.get(body_type, "mixed-body"))
    add(color_map.get(color_tone, "mixed-color"))

    cute = normalize_score(analysis.get("cute_level"), 5)
    dark = normalize_score(analysis.get("dark_level"), 3)
    power = normalize_score(analysis.get("power_level"), 4)
    energy = normalize_score(analysis.get("energy"), 5)

    if cute >= 7:
        add("cute")
    if dark >= 6:
        add("dark")
    if power >= 7:
        add("strong")
    if energy >= 7:
        add("wild")
    if energy <= 3:
        add("calm")

    if normalize_score(analysis.get("blob_like"), 0) >= 7:
        add("soft")
    if normalize_score(analysis.get("animal_like"), 0) >= 7:
        add("pet-like")
    if normalize_score(analysis.get("plant_like"), 0) >= 7:
        add("forest")
    if normalize_score(analysis.get("ghost_like"), 0) >= 7:
        add("mysterious")

    while len(tags) < 8:
        add("monster-like")
        add("medium")
        add("mixed-color")
        add("blank")
        if len(tags) >= 8:
            break

    return tags[:8]


def analyze_monster_with_gpt(name, image_url, image_base64):
    tag_vocab_text = ", ".join(TAG_VOCAB)

    prompt = f"""
너는 메이플스토리 몬스터 이미지를 보고, 실제 사람 사진과 닮은꼴 매칭을 하기 위한
'몬스터 캐릭터 DNA'를 만드는 분석기야.

분석 대상 몬스터 이름: {name}

중요 규칙:
- 아래 JSON만 출력해.
- 설명 문장 없이 JSON만 출력해.
- 모든 수치 점수는 0~10 정수.
- match_tags는 반드시 제공된 태그 사전에서만 골라.
- match_tags는 정확히 8개.
- match_tags 중복 금지.
- 태그 사전에 없는 단어 절대 사용 금지.
- 실제 게임 강함이 아니라 사진 닮은꼴 매칭용 시각적/분위기 특성 기준으로 판단해.
- 콜라보/이벤트/검정 배경/렌더링 문제가 있어 보이면 match_tags에 render-risk 포함.
- 사람형이면 human, npc-like, mage, warrior, elegant 같은 태그를 적극 사용.
- 말랑하거나 동그란 몬스터면 blob, jelly, soft, round-body 같은 태그를 적극 사용.
- 동물형이면 animal, wild, pet-like, sharp-eye 같은 태그를 적극 사용.
- 식물/나무형이면 plant, forest, wood 태그를 적극 사용.
- 유령/어둠 계열이면 ghost, dark, mysterious, shadow 태그를 적극 사용.

사용 가능한 match_tags 태그 사전:
{tag_vocab_text}

선택지:
face_shape: round, square, sharp, long, small, wide, none
eye_style: big, small, sharp, sleepy, angry, simple, none
expression: gentle, happy, angry, blank, sleepy, playful, scary, mysterious, none
body_type: humanoid, animal, blob, ghost, object, plant, fish, insect, mixed
color_tone: warm, cool, dark, bright, green, blue, red, yellow, neutral, mixed

JSON 형식:
{{
  "face_shape": "round",
  "eye_style": "simple",
  "expression": "happy",
  "body_type": "blob",
  "color_tone": "green",

  "human_like": 0,
  "animal_like": 0,
  "blob_like": 10,
  "ghost_like": 0,
  "object_like": 0,
  "plant_like": 0,

  "cute_level": 8,
  "dark_level": 1,
  "power_level": 3,
  "energy": 5,

  "match_tags": ["round", "blob", "green", "soft", "cute", "happy", "simple-eye", "jelly"],
  "match_note": "둥글고 말랑한 외형이라 귀엽고 단순한 인상의 사람과 매칭하기 좋음"
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
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
    )

    return parse_json(response.choices[0].message.content)


def make_output_row(original_row, analysis):
    allowed_face = ["round", "square", "sharp", "long", "small", "wide", "none"]
    allowed_eye = ["big", "small", "sharp", "sleepy", "angry", "simple", "none"]
    allowed_expression = ["gentle", "happy", "angry", "blank", "sleepy", "playful", "scary", "mysterious", "none"]
    allowed_body = ["humanoid", "animal", "blob", "ghost", "object", "plant", "fish", "insect", "mixed"]
    allowed_color = ["warm", "cool", "dark", "bright", "green", "blue", "red", "yellow", "neutral", "mixed"]

    tags = normalize_tags(analysis.get("match_tags", []))
    tags = fill_missing_tags(tags, analysis)
    match_tags = ",".join(tags)

    return {
        "name": get_value(original_row, ["name", "monster_name", "몬스터명"]),
        "mob_id": get_value(original_row, ["mob_id", "id", "monster_id"]),
        "source_url": get_value(original_row, ["source_url", "source"]),
        "image_url": get_value(original_row, ["image_url", "img_url", "url", "image"]),

        "face_shape": normalize_choice(analysis.get("face_shape"), allowed_face, "none"),
        "eye_style": normalize_choice(analysis.get("eye_style"), allowed_eye, "simple"),
        "expression": normalize_choice(analysis.get("expression"), allowed_expression, "blank"),
        "body_type": normalize_choice(analysis.get("body_type"), allowed_body, "mixed"),
        "color_tone": normalize_choice(analysis.get("color_tone"), allowed_color, "mixed"),

        "human_like": normalize_score(analysis.get("human_like"), 0),
        "animal_like": normalize_score(analysis.get("animal_like"), 0),
        "blob_like": normalize_score(analysis.get("blob_like"), 0),
        "ghost_like": normalize_score(analysis.get("ghost_like"), 0),
        "object_like": normalize_score(analysis.get("object_like"), 0),
        "plant_like": normalize_score(analysis.get("plant_like"), 0),

        "cute_level": normalize_score(analysis.get("cute_level"), 5),
        "dark_level": normalize_score(analysis.get("dark_level"), 3),
        "power_level": normalize_score(analysis.get("power_level"), 4),
        "energy": normalize_score(analysis.get("energy"), 5),

        "match_tags": match_tags,
        "match_note": str(analysis.get("match_note", "")).strip(),
    }


def load_done_names():
    if not os.path.exists(OUTPUT_CSV):
        return set()

    try:
        done_df = pd.read_csv(OUTPUT_CSV).fillna("")
        if "name" not in done_df.columns:
            return set()
        return set(done_df["name"].astype(str).str.strip())
    except Exception:
        return set()


def append_row(row):
    exists = os.path.exists(OUTPUT_CSV)

    out_df = pd.DataFrame([row], columns=COLUMNS)
    out_df.to_csv(
        OUTPUT_CSV,
        mode="a",
        header=not exists,
        index=False,
        encoding="utf-8-sig",
    )


def main():
    input_csv = find_input_csv()
    print(f"입력 CSV: {input_csv}")
    print(f"출력 CSV: {OUTPUT_CSV}")
    print(f"LIMIT: {LIMIT}")

    df = pd.read_csv(input_csv).fillna("")
    done_names = load_done_names()

    count = 0

    for idx, row in df.iterrows():
        name = get_value(row, ["name", "monster_name", "몬스터명"])
        image_url = get_value(row, ["image_url", "img_url", "url", "image"])

        if not name or not image_url:
            print(f"[SKIP] {idx + 1}: name/image_url 없음")
            continue

        if name in done_names:
            print(f"[DONE] {idx + 1}: {name}")
            continue

        if LIMIT is not None and count >= LIMIT:
            print(f"LIMIT {LIMIT}개 도달. 종료.")
            break

        print(f"\n[{idx + 1}/{len(df)}] 분석 중: {name}")

        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                image_bytes = download_image(image_url)
                image_base64 = image_bytes_to_base64(image_bytes)

                analysis = analyze_monster_with_gpt(name, image_url, image_base64)
                out_row = make_output_row(row, analysis)

                append_row(out_row)
                done_names.add(name)

                print(f"완료: {name}")
                print(f"tags: {out_row['match_tags']}")
                print(f"note: {out_row['match_note']}")

                success = True
                count += 1
                time.sleep(SLEEP_SECONDS)
                break

            except Exception as e:
                print(f"실패 {attempt}/{MAX_RETRIES}: {name} / {e}")
                time.sleep(2)

        if not success:
            print(f"[FAILED] {name}")

    print("\n끝!")
    print(f"생성/누적 파일: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
