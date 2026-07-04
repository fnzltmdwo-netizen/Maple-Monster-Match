import os
import json
import time
import pandas as pd
from openai import OpenAI

INPUT_CSV = "monsters_full.csv"
OUTPUT_CSV = "monsters_ai.csv"

START = int(os.environ.get("START", "0"))
LIMIT = int(os.environ.get("LIMIT", "50"))

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def extract_json(text):
    text = text.strip().replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= 0:
        raise Exception("JSON 파싱 실패")
    return json.loads(text[start:end])


def analyze_monster(name, image_url):
    prompt = f"""
너는 메이플스토리 몬스터 DB 분류기다.
절대 귀여움 편향을 가지면 안 된다.

몬스터 이름:
{name}

face_shape: round, oval, long, square, triangle 중 하나.
vibe: cute, dark, strong, calm, mysterious 중 하나.

중요 규칙:
- 나무, 로봇, 골렘, 바위, 갑옷 계열은 calm 또는 strong.
- 보스형, 용, 악마형은 strong 또는 dark.
- 유령, 마법형, 실험체, 시계/기계 계열은 mysterious 또는 dark.
- 동글동글해도 무섭거나 기괴하면 dark/mysterious.
- power_level은 외형상 강해 보이는 정도.

cute_level, dark_level, power_level은 0~10 정수.

반드시 JSON만 출력:
{{
  "face_shape": "square",
  "vibe": "calm",
  "cute_level": 2,
  "dark_level": 4,
  "power_level": 6,
  "description": "각지고 단단하며 무표정한 인상"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        max_tokens=250,
    )

    return extract_json(response.choices[0].message.content)


def analyze_with_retry(name, image_url, max_retries=3):
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return analyze_monster(name, image_url)
        except Exception as e:
            last_error = e
            print(f"  retry {attempt}/{max_retries} failed: {e}")
            time.sleep(10 * attempt)

    raise last_error


def load_output_dataframe(base_df):
    out_df = base_df.copy()

    if not os.path.exists(OUTPUT_CSV):
        print("기존 AI CSV 없음. 새로 생성")
        return out_df

    old_df = pd.read_csv(OUTPUT_CSV)
    print(f"기존 {OUTPUT_CSV} 불러옴: {len(old_df)} rows")

    copy_cols = [
        "face_shape",
        "vibe",
        "cute_level",
        "dark_level",
        "power_level",
        "description",
    ]

    max_len = min(len(old_df), len(out_df))

    for col in copy_cols:
        if col in old_df.columns:
            out_df.loc[:max_len - 1, col] = old_df.loc[:max_len - 1, col].values

    return out_df


def main():
    base_df = pd.read_csv(INPUT_CSV)
    out_df = load_output_dataframe(base_df)

    end = min(START + LIMIT, len(base_df))
    print(f"전체 몬스터 수: {len(base_df)}")
    print(f"분석 범위: {START} ~ {end - 1}")

    for idx in range(START, end):
        row = base_df.iloc[idx]
        name = str(row["name"])
        image_url = str(row["image_url"]).strip()

        print(f"[{idx + 1}/{len(base_df)}] analyzing {name}")

        if not image_url or image_url == "nan":
            print("  skip: no image_url")
            continue

        try:
            ai = analyze_with_retry(name, image_url)

            out_df.loc[idx, "face_shape"] = ai["face_shape"]
            out_df.loc[idx, "vibe"] = ai["vibe"]
            out_df.loc[idx, "cute_level"] = int(ai["cute_level"])
            out_df.loc[idx, "dark_level"] = int(ai["dark_level"])
            out_df.loc[idx, "power_level"] = int(ai["power_level"])
            out_df.loc[idx, "description"] = ai["description"]

            print("  ok")

        except Exception as e:
            print("  FINAL ERROR:", e)
            out_df.loc[idx, "description"] = f"분석 실패: {e}"

        time.sleep(3)

    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"완료! 저장됨: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
