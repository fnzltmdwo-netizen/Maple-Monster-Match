import os
import json
import time
import pandas as pd
from openai import OpenAI

INPUT_CSV = "monsters_full.csv"
OUTPUT_CSV = "monsters_ai.csv"

START = int(os.environ.get("START", "0"))
LIMIT = int(os.environ.get("LIMIT", "100"))

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def extract_json(text):
    text = text.strip()
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end <= 0:
        raise Exception("JSON 파싱 실패")

    return json.loads(text[start:end])


def analyze_monster(name, image_url):
    prompt = f"""
너는 메이플스토리 몬스터 DB 분류기다.

절대 귀여움 편향을 가지면 안 된다.
몬스터를 사람 닮은꼴 매칭에 쓰기 위해 외형과 분위기를 냉정하게 분류한다.

몬스터 이름:
{name}

face_shape는 반드시 하나:
- round: 둥글거나 말랑한 외형
- oval: 타원형, 부드러운 세로형
- long: 길쭉하거나 몸통이 긴 외형
- square: 각지거나 단단한 외형
- triangle: 뾰족하거나 날카로운 외형

vibe는 반드시 하나:
- cute: 귀엽고 순한 인상
- dark: 음산함, 악마, 공포, 좀비 느낌
- strong: 전투적, 거칠고 강한 인상
- calm: 무표정, 차분함, 단단함
- mysterious: 기묘함, 마법적, 알 수 없는 분위기

중요 규칙:
- 나무, 로봇, 골렘, 바위, 갑옷 계열은 cute를 주지 말고 calm 또는 strong으로 분류해.
- 보스형, 용, 악마형은 strong 또는 dark로 분류해.
- 유령, 마법형, 실험체, 시계/기계 계열은 mysterious 또는 dark로 분류해.
- 동글동글해도 표정이 무섭거나 기괴하면 cute가 아니라 dark/mysterious야.
- power_level은 실제 전투력이 아니라 외형상 강해 보이는 정도야.

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
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        max_tokens=250,
    )

    return extract_json(response.choices[0].message.content)


def load_output_dataframe(base_df):
    out_df = base_df.copy()

    if not os.path.exists(OUTPUT_CSV):
        print("기존 AI CSV 없음. monsters_full.csv 기준으로 새로 생성")
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
            ai = analyze_monster(name, image_url)

            out_df.loc[idx, "face_shape"] = ai["face_shape"]
            out_df.loc[idx, "vibe"] = ai["vibe"]
            out_df.loc[idx, "cute_level"] = int(ai["cute_level"])
            out_df.loc[idx, "dark_level"] = int(ai["dark_level"])
            out_df.loc[idx, "power_level"] = int(ai["power_level"])
            out_df.loc[idx, "description"] = ai["description"]

            print("  ok")

        except Exception as e:
            print("  ERROR:", e)

        time.sleep(1)

    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"완료! 저장됨: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
