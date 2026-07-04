import os
import json
import time
import pandas as pd
from openai import OpenAI

INPUT_CSV = "monsters_full.csv"
OUTPUT_CSV = "monsters_ai.csv"

LIMIT = 10   # 테스트용. 성공하면 647로 변경

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def extract_json(text):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end <= 0:
        raise Exception("JSON 파싱 실패")

    return json.loads(text[start:end])


def analyze_monster(name, image_url):
    prompt = f"""
너는 메이플 몬스터 DB 분류기다.

절대 귀여움 편향을 가지면 안 된다.

몬스터 이름:
{name}

규칙:

1. face_shape는 반드시 하나 선택
- round (둥글다)
- oval (타원형)
- long (길쭉하다)
- square (각지다)
- triangle (뾰족하다)

2. vibe는 반드시 하나 선택
- cute
- dark
- strong
- calm
- mysterious

판정 기준:
- cute = 귀엽고 순함
- dark = 음산, 악마, 공포
- strong = 전투적, 강함
- calm = 무표정, 차분
- mysterious = 기묘, 마법적

중요:
나무/로봇/골렘 계열은 cute를 주지 마라.
보스형은 strong 또는 dark.
유령/마법형은 mysterious 또는 dark.

JSON만 출력:
{{
 "face_shape":"square",
 "vibe":"calm",
 "cute_level":2,
 "dark_level":4,
 "power_level":6,
 "description":"각지고 단단하며 무표정한 인상"
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
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        max_tokens=250
    )

    text = response.choices[0].message.content
    return extract_json(text)


def main():
    df = pd.read_csv(INPUT_CSV)
    results = []

    for idx, row in df.head(LIMIT).iterrows():
        name = str(row["name"])
        image_url = str(row["image_url"])

        print(f"[{idx+1}] analyzing {name}")

        try:
            ai = analyze_monster(name, image_url)

            row["face_shape"] = ai["face_shape"]
            row["vibe"] = ai["vibe"]
            row["cute_level"] = int(ai["cute_level"])
            row["dark_level"] = int(ai["dark_level"])
            row["power_level"] = int(ai["power_level"])
            row["description"] = ai["description"]

        except Exception as e:
            print("ERROR:", e)

        results.append(row)
        time.sleep(1)

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("완료!")


if __name__ == "__main__":
    main()
