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
너는 메이플스토리 몬스터 외형 분석기다.

몬스터 이름:
{name}

이미지를 보고 반드시 JSON만 출력.

가능한 face_shape:
round, oval, long, square, triangle

가능한 vibe:
cute, dark, strong, calm, mysterious

cute_level, dark_level, power_level:
0~10 정수

출력 예시:
{{
 "face_shape":"round",
 "vibe":"cute",
 "cute_level":8,
 "dark_level":2,
 "power_level":4,
 "description":"둥글고 귀엽고 순한 인상"
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
