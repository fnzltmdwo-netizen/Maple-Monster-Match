from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import re
import hashlib
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


COMMON_MONSTER_PENALTY = [
    "단지",
    "달팽이",
    "스포아",
    "슬라임",
    "주황버섯",
    "파란버섯",
    "리본돼지",
    "울트라 코-크 달팽이",
    "코-크 달팽이",
    "여신 탑의 로얄 네펜데스",
]


TAG_KEYWORDS = {
    "cute": ["귀여", "cute", "아기", "동글", "말랑", "핑크"],
    "dark": ["어둠", "dark", "악마", "유령", "좀비", "스켈", "저주", "그림자"],
    "sharp": ["날카", "sharp", "칼", "뿔", "가시", "늑대", "표범"],
    "round": ["동글", "round", "통통", "볼", "구름", "버섯", "달팽이"],
    "funny": ["웃긴", "funny", "장난", "코믹", "바보"],
    "mysterious": ["신비", "mysterious", "마법", "요정", "정령"],
    "strong": ["강한", "strong", "보스", "전사", "거대", "포스"],
    "soft": ["부드", "soft", "말랑", "순한", "따뜻"],
    "cold": ["차가", "cold", "얼음", "눈", "서늘"],
    "animal": ["동물", "돼지", "고양", "강아", "곰", "토끼", "새", "원숭"],
    "bright": ["밝", "bright", "해", "빛", "노랑"],
    "calm": ["차분", "calm", "조용", "평온"],
    "playful": ["장난", "playful", "개구", "활발"],
    "sleepy": ["졸린", "sleepy", "나른", "멍"],
    "cool": ["쿨", "cool", "시크", "무심"],
    "elegant": ["우아", "elegant", "고급", "귀족"],
}


def clean_base64(image_base64: str):
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    return image_base64.strip().replace("\n", "").replace("\r", "").replace(" ", "")


def make_image_hash(image_base64: str):
    cleaned = clean_base64(image_base64)
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()


def stable_tiebreaker(image_hash: str, monster_name: str):
    key = f"{image_hash}-{monster_name}"
    value = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(value[:6], 16) / 0xFFFFFF


def safe_json(text: str):
    text = text.strip()
    text = re.sub(r"```json|```", "", text).strip()

    match = re.search(r"\{.*\}", text, re.S)
    if match:
        text = match.group(0)

    return json.loads(text)


def get_value(row, possible_cols, default=""):
    for col in possible_cols:
        if col in row.index and str(row[col]).strip():
            return str(row[col]).strip()
    return default


def get_float(row, possible_cols, default=None):
    for col in possible_cols:
        if col in row.index:
            try:
                return float(row[col])
            except:
                pass
    return default


def monster_text(row):
    cols = [
        "name",
        "monster_name",
        "몬스터명",
        "description",
        "desc",
        "reason",
        "tags",
        "type",
    ]
    return " ".join(str(row.get(c, "")) for c in cols if c in row.index)


def infer_monster_tags(row):
    text = monster_text(row).lower()
    tags = set()

    if "tags" in row.index and str(row["tags"]).strip():
        raw_tags = str(row["tags"]).replace("|", ",").replace("/", ",")
        for tag in raw_tags.split(","):
            tag = tag.strip().lower()
            if tag:
                tags.add(tag)

    for tag, words in TAG_KEYWORDS.items():
        if any(w.lower() in text for w in words):
            tags.add(tag)

    if not tags:
        tags.add("normal")

    return sorted(list(tags))


def normalize_user_tags(tags):
    allowed = set(TAG_KEYWORDS.keys())
    cleaned = []

    for tag in tags:
        tag = str(tag).strip().lower()
        if tag in allowed and tag not in cleaned:
            cleaned.append(tag)

    return cleaned


def score_match(user_tags, user_scores, monster_tags, row, image_hash):
    name = get_value(row, ["name", "monster_name", "몬스터명"], "")

    user_tag_set = set(user_tags)
    monster_tag_set = set(monster_tags)

    overlap = len(user_tag_set & monster_tag_set)
    union = len(user_tag_set | monster_tag_set) or 1

    tag_score = (overlap / union) * 55

    score_score = 0
    score_count = 0

    keys = [
        "cute",
        "dark",
        "power",
        "soft",
        "sharp",
        "round",
        "funny",
        "mysterious",
        "strong",
        "cold",
        "bright",
        "calm",
        "playful",
        "sleepy",
        "cool",
        "elegant",
    ]

    for key in keys:
        try:
            user_v = float(user_scores.get(key, 5))
        except:
            user_v = 5

        monster_v = get_float(row, [key, f"{key}_score"], None)

        if monster_v is not None:
            score_score += max(0, 10 - abs(user_v - monster_v)) * 4
            score_count += 1

    if score_count > 0:
        score_score = score_score / score_count
    else:
        score_score = 18

    final_score = tag_score + score_score

    if any(common in name for common in COMMON_MONSTER_PENALTY):
        final_score -= 25

    if overlap == 0:
        final_score -= 12
    elif overlap == 1:
        final_score -= 5

    # soft/calm/round만으로 먹고 들어가는 몬스터 방지
    if set(user_tags).issubset({"soft", "calm", "round"}):
        if {"soft", "calm", "round"} & monster_tag_set:
            final_score -= 8

    final_score += stable_tiebreaker(image_hash, name) * 2

    return round(final_score, 4)


def add_percent(unique_results):
    if not unique_results:
        return unique_results

    max_score = unique_results[0]["score"] or 1

    for index, r in enumerate(unique_results):
        base_percent = int((r["score"] / max_score) * 96)

        if index == 0:
            percent = max(90, min(base_percent, 98))
        elif index == 1:
            percent = max(80, min(base_percent, 89))
        else:
            percent = max(70, min(base_percent, 79))

        r["percent"] = percent

    return unique_results


def generate_reason(monster_name, user_tags, vibe):
    try:
        prompt = f"""
사람 얼굴 분위기 분석 결과:
태그: {", ".join(user_tags)}
분위기: {vibe}

매칭 몬스터: {monster_name}

조건:
- 2문장
- 귀엽고 재밌게
- 메이플 닮은꼴 테스트 결과처럼 작성
- 외모 비하 금지
- 신원/성별/나이 추정 금지
- 80자 이내
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return f"{monster_name}와 전체 분위기가 비슷해요!"


@app.get("/")
def home():
    return {
        "message": "Maple Monster Match API is running!",
        "monster_count": len(df),
        "mode": "A3 stable tag matching enhanced v2",
    }


@app.post("/match")
def match_monster(req: MatchRequest):
    try:
        image_base64 = clean_base64(req.image_base64)
        image_hash = make_image_hash(image_base64)

        prompt = """
너는 실제 사람 얼굴 사진을 보고 메이플스토리 몬스터 닮은꼴을 찾기 위한 분석기야.

중요:
- 얼굴의 신원/이름/성별/나이 추정 금지
- 외모 비하 금지
- 닮은 몬스터 매칭용 특징만 뽑기
- 같은 사진이면 최대한 같은 분석이 나오게 일관적으로 판단하기
- 단순히 귀엽다/차분하다만 보지 말고 눈매, 인상, 실루엣, 분위기를 나눠서 판단하기
- 반드시 tags는 4개 이상 6개 이하로 선택하기
- soft, calm, round만 반복하지 말 것
- sharp, cool, bright, sleepy, playful, mysterious, elegant 중 해당되는 특징도 적극 포함하기

아래 JSON 형식으로만 답해.

{
  "tags": ["cute", "round", "soft", "calm"],
  "scores": {
    "cute": 1~10,
    "dark": 1~10,
    "power": 1~10,
    "soft": 1~10,
    "sharp": 1~10,
    "round": 1~10,
    "funny": 1~10,
    "mysterious": 1~10,
    "strong": 1~10,
    "cold": 1~10,
    "bright": 1~10,
    "calm": 1~10,
    "playful": 1~10,
    "sleepy": 1~10,
    "cool": 1~10,
    "elegant": 1~10
  },
  "vibe": "짧은 분위기 설명"
}

태그 후보:
cute, dark, sharp, round, funny, mysterious, strong, soft, cold, animal, calm, playful, sleepy, bright, cool, elegant
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

        analysis = safe_json(response.choices[0].message.content)

        user_tags = normalize_user_tags(analysis.get("tags", []))
        user_scores = analysis.get("scores", {})
        user_vibe = analysis.get("vibe", "")

        results = []

        for _, row in df.iterrows():
            monster_tags = infer_monster_tags(row)
            score = score_match(user_tags, user_scores, monster_tags, row, image_hash)

            name = get_value(row, ["name", "monster_name", "몬스터명"], "이름 없음")
            image_url = get_value(row, ["image_url", "img_url", "url", "image"], "")

            results.append({
                "name": name,
                "image_url": image_url,
                "score": score,
                "tags": monster_tags,
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

            if len(unique) >= 3:
                break

        unique = add_percent(unique)

        for r in unique:
            r["reason"] = generate_reason(
                r["name"],
                user_tags,
                user_vibe
            )

        return {
            "image_hash": image_hash,
            "analysis": {
                "tags": user_tags,
                "scores": user_scores,
                "vibe": user_vibe,
            },
            "results": unique,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
