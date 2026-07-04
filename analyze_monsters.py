import os
import json
import time
import pandas as pd
from openai import OpenAI

INPUT_CSV = "monsters_full.csv"
OUTPUT_CSV = "monsters_ai.csv"

LIMIT = 50   # 테스트용. 성공하면 647로 변경

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
