from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os, base64, json, re, math, random
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

CSV_PATH = "monsters_ai.csv"
df = pd.read_csv(CSV_PATH).fillna("")

class MatchRequest(BaseModel):
    image_base64: str


TAG_KEYWORDS = {
    "cute": ["귀여", "cute", "아기", "동글", "말랑", "슬라임", "핑크"],
    "dark": ["어둠", "dark", "악마", "유령", "좀비", "스켈", "저주", "그림자"],
    "sharp": ["날카", "sharp", "칼", "뿔", "가시", "늑대", "표범"],
    "round": ["동글", "round", "통통", "볼", "구름", "버섯"],
    "funny": ["웃긴", "funny", "장난", "코믹", "바보"],
    "mysterious": ["신비", "mysterious", "마법", "요정", "정령"],
    "strong": ["강한", "strong", "보스", "전사", "거대", "포스"],
    "soft": ["부드", "soft", "말랑", "순한", "따뜻"],
    "cold": ["차가", "cold", "얼음", "눈", "서늘"],
    "animal": ["동물", "돼지", "고양", "강아", "곰", "토끼", "새", "원숭"],
}


def clean_base64(image_base64: str):
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    return image_base64.strip()


def safe_json(text: str):
    text = text.strip()
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        text = match.group(0)
    return json.loads(text)


def monster_text(row):
    cols = ["name", "monster_name", "description", "reason", "tags", "type"]
    return " ".join(str(row.get(c, "")) for c in cols if c in row.index)


def infer_monster_tags(row):
    text = monster_text(row).lower()
    tags = set()

    for tag, words in TAG_KEYWORDS.items():
        if any(w.lower() in text for w in words):
            tags.add(tag)

    if not tags:
        tags.add("normal")

    return list(tags)


def get_value(row, possible_cols, default=""):
    for col in possible_cols:
        if col in row.index and str(row[col]).strip():
            return str(row[col])
    return default


def score_match(user_tags, user_scores, monster_tags, row):
    tag_overlap = len(set(user_tags) & set(monster_tags))
    tag_score = tag_overlap * 18

    score = tag_score

    # 기존 CSV에 cute/dark/power/mood 같은 점수가 있으면 같이 사용
    for key in ["cute", "dark", "power", "soft", "sharp", "round", "funny", "mysterious"]:
        user_v = float(user_scores.get(key, 5))
        monster_v = None

        for col in [key, f"{key}_score"]:
            if col in row.index:
                try:
                    monster_v = float(row[col])
                    break
                except:
                    pass

        if monster_v is not None:
            score += max(0, 10 - abs(user_v - monster_v)) * 2

    # 너무 같은 애만 나오는 것 방지용 아주 작은 다양성
    score += random.uniform(0, 4)

    return score


@app.get("/")
def home():
    return {
        "message": "Maple Monster Match API is running!",
        "monster_count": len(df),
        "mode": "A3 tag matching"
    }


@app.post("/match")
def match_monster(req: MatchRequest):
    try:
        image_base64 = clean_base64(req.image_base64)

        prompt = """
너는 실제 사람 얼굴 사진을 보고 메이플스토리 몬스터 닮은꼴을 찾기 위한 분석기야.

중요:
- 얼굴의 신원/이름/성별 추정 금지
- 외모를 비하하지 말 것
- 닮은 몬스터 매칭용 특징만 뽑기

아래 JSON 형식으로만 답해.

{
  "tags": ["cute", "round", "soft"],
  "scores": {
    "cute": 1~10,
    "dark": 1~10,
    "power": 1~10,
    "soft": 1~10,
    "sharp": 1~10,
    "round": 1~10,
    "funny": 1~10,
    "mysterious": 1~10
  },
  "vibe": "짧은 분위기 설명"
}

태그 후보:
cute, dark, sharp, round, funny, mysterious, strong, soft, cold, animal, calm, playful, sleepy, bright
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )

        analysis = safe_json(response.choices[0].message.content)

        user_tags = analysis.get("tags", [])
        user_scores = analysis.get("scores", {})
        user_vibe = analysis.get("vibe", "")

        results = []

        for _, row in df.iterrows():
            m_tags = infer_monster_tags(row)
            score = score_match(user_tags, user_scores, m_tags, row)

            name = get_value(row, ["name", "monster_name", "몬스터명"], "이름 없음")
            image_url = get_value(row, ["image_url", "img_url", "url", "image"], "")
            desc = get_value(row, ["description", "desc", "reason"], "")

            results.append({
                "name": name,
                "image_url": image_url,
                "score": round(score, 2),
                "tags": m_tags,
                "reason": desc
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        # 이름 중복 제거
        unique = []
        seen = set()

        for r in results:
            if r["name"] in seen:
                continue
            seen.add(r["name"])
            unique.append(r)
            if len(unique) >= 3:
                break

        # reason이 비어있으면 새로 생성
        for r in unique:
            if not r["reason"]:
                common = list(set(user_tags) & set(r["tags"]))
                if common:
                    r["reason"] = f"{', '.join(common)} 분위기가 비슷해서 닮은 몬스터로 매칭됐어요."
                else:
                    r["reason"] = f"{user_vibe} 느낌과 몬스터의 전체 분위기가 비슷해서 매칭됐어요."

        return {
            "analysis": {
                "tags": user_tags,
                "scores": user_scores,
                "vibe": user_vibe
            },
            "results": unique
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
