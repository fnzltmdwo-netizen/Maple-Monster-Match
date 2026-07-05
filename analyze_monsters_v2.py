import os
import json
import time
import re
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_CSV = "monsters_ai.csv"
OUTPUT_CSV = "monsters_ai_v2.csv"


def safe_json(text):
    text = text.strip()
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        text = match.group(0)
    return json.loads(text)


def analyze_monster(row):
    name = str(row.get("name", ""))
    image_url = str(row.get("image_url", ""))
    description = str(row.get("description", ""))

    prompt = f"""
너는 메이플스토리 몬스터 이미지를 보고
실제 사람 얼굴 사진과 닮은꼴 매칭에 필요한 특징을 분석하는 AI야.

몬스터 이름: {name}
기존 설명: {description}

아래 JSON 형식으로만 답해.

{{
  "face_shape": "round/square/sharp/long/small/wide",
  "vibe": "cute/calm/dark/mysterious/playful/cool/strong/sleepy/bright/elegant",
  "cute_level": 1~10,
  "dark_level": 1~10,
  "power_level": 1~10,

  "human_match_score": 1~10,
  "face_visibility": 1~10,
  "object_like": 1~10,
  "animal_like": 1~10,
  "plant_like": 1~10,

  "eye_style": "big/small/sharp/sleepy/angry/none/simple",
  "mouth_style": "smile/neutral/angry/open/none",
  "body_type": "blob/animal/humanoid/object/plant/fish/insect/ghost",
  "color_tone": "bright/dark/warm/cool/green/red/blue/yellow/neutral",

  "match_tags": ["태그 3~6개"],
  "match_note": "사람 닮은꼴 매칭 관점에서 짧은 설명"
}}

기준:
- 사람 얼굴 닮은꼴 테스트에 잘 어울리면 human_match_score 높게
- 식물/사물/배경형이면 object_like 또는 plant_like 높게
- 눈/입이 잘 보이면 face_visibility 높게
- 얼굴 없는 몬스터는 face_visibility 낮게
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
                            "url": image_url
                        }
                    }
                ]
            }
        ]
    )

    return safe_json(response.choices[0].message.content)


def main():
    df = pd.read_csv(INPUT_CSV).fillna("")

    if os.path.exists(OUTPUT_CSV):
        out_df = pd.read_csv(OUTPUT_CSV).fillna("")
        done_names = set(out_df["name"].astype(str))
        results = out_df.to_dict("records")
        print(f"이어하기 모드: 이미 완료 {len(done_names)}개")
    else:
        done_names = set()
        results = []

    for index, row in df.iterrows():
        name = str(row.get("name", ""))

        if name in done_names:
            continue

        image_url = str(row.get("image_url", ""))

        if not image_url:
            print("skip no image:", name)
            continue

        try:
            print(f"[{index + 1}/{len(df)}] 분석 중: {name}")

            ai = analyze_monster(row)

            new_row = row.to_dict()

            new_row["face_shape"] = ai.get("face_shape", row.get("face_shape", "round"))
            new_row["vibe"] = ai.get("vibe", row.get("vibe", "calm"))
            new_row["cute_level"] = ai.get("cute_level", row.get("cute_level", 5))
            new_row["dark_level"] = ai.get("dark_level", row.get("dark_level", 3))
            new_row["power_level"] = ai.get("power_level", row.get("power_level", 4))

            new_row["human_match_score"] = ai.get("human_match_score", 5)
            new_row["face_visibility"] = ai.get("face_visibility", 5)
            new_row["object_like"] = ai.get("object_like", 1)
            new_row["animal_like"] = ai.get("animal_like", 1)
            new_row["plant_like"] = ai.get("plant_like", 1)

            new_row["eye_style"] = ai.get("eye_style", "simple")
            new_row["mouth_style"] = ai.get("mouth_style", "neutral")
            new_row["body_type"] = ai.get("body_type", "object")
            new_row["color_tone"] = ai.get("color_tone", "neutral")
            new_row["match_tags"] = ",".join(ai.get("match_tags", []))
            new_row["match_note"] = ai.get("match_note", "")

            results.append(new_row)

            pd.DataFrame(results).to_csv(
                OUTPUT_CSV,
                index=False,
                encoding="utf-8-sig"
            )

            print("완료:", name)

            time.sleep(0.8)

        except Exception as e:
            print("실패:", name, e)
            continue

    print("전체 완료!")
    print("저장 파일:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
