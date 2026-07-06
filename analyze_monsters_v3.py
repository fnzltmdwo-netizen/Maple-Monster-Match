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

# 처음엔 30개만 테스트 추천
# 전체 돌릴 때는 None으로 바꾸면 됨
LIMIT = 30

SLEEP_SECONDS = 1.2
MAX_RETRIES = 3

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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

    # 너무 큰 이미지는 줄이기
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


def analyze_monster_with_gpt(name, image_url, image_base64):
    prompt = f"""
너는 메이플스토리 몬스터 이미지를 보고, 실제 사람 얼굴 사진과 닮은꼴 매칭을 하기 위한
'몬스터 캐릭터 DNA'를 만드는 분석기야.

분석 대상 몬스터 이름: {name}

중요 규칙:
- 아래 JSON만 출력해.
- 설명 문장 없이 JSON만 출력해.
- 모든 수치 점수는 0~10 정수.
- 실제 게임 강함이 아니라, 사진 닮은꼴 매칭에 필요한 시각적/분위기 특성 기준으로 판단해.
- 사람과 닮기 쉬운지, 동물 같은지, 말랑한지, 유령 같은지, 물건 같은지 구분해.
- 외형이 잘 안 보이면 face_visibility가 낮다고 판단하되, 이번 JSON에는 face_visibility 대신 match_note에 적어.
- 콜라보/이벤트/검정 배경으로 보일 가능성이 있으면 match_tags에 "render-risk" 추가.

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

  "match_tags": ["round", "soft", "cute", "blob", "simple-face"],
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

    match_tags = analysis.get("match_tags", [])
    if isinstance(match_tags, list):
        match_tags = ",".join([str(x).strip() for x in match_tags if str(x).strip()])
    else:
        match_tags = str(match_tags)

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
