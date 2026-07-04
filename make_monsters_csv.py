import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

BASE_URL = "https://mapledb.kr/mob.php"

headers = {
    "User-Agent": "Mozilla/5.0"
}

res = requests.get(BASE_URL, headers=headers, timeout=20)
res.raise_for_status()

soup = BeautifulSoup(res.text, "html.parser")

monsters = []

# 페이지 안의 모든 링크 검사
for a in soup.find_all("a"):
    name = a.get_text(strip=True)
    href = a.get("href", "")

    if not name:
        continue

    # mob 상세 링크만 추정
    if "mob.php" in href or "mob_id" in href or "id=" in href:
        full_url = urljoin(BASE_URL, href)

        monsters.append({
            "name": name,
            "source_url": full_url,
            "face_shape": "round",
            "vibe": "cute",
            "cute_level": 5,
            "dark_level": 3,
            "power_level": 5,
            "description": f"{name}의 메이플 몬스터 이미지와 분위기를 기반으로 한 후보",
            "image_url": ""
        })

# 중복 제거
df = pd.DataFrame(monsters)
df = df.drop_duplicates(subset=["name"])

df.to_csv("monsters_full.csv", index=False, encoding="utf-8-sig")

print(f"완료! 몬스터 {len(df)}마리 저장됨")
print("파일명: monsters_full.csv")
