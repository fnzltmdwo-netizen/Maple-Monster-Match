import pandas as pd
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
import re

BASE_URL = "https://mapledb.kr/mob.php"


def clean_monster_name(raw_name):
    name = raw_name.replace("\n", " ").strip()
    name = re.sub(r"\s+", " ", name)

    # LEVEL / HP / EXP 뒤 정보 제거
    name = re.sub(r"\s+LEVEL\s+\d+.*$", "", name)
    name = re.sub(r"\s+Lv\.\s*\d+.*$", "", name)
    name = re.sub(r"\s+HP\s+[\d,]+.*$", "", name)
    name = re.sub(r"\s+EXP\s+[\d,]+.*$", "", name)

    return name.strip()


def main():
    monsters = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(3000)

        items = page.evaluate("""
        () => {
          const results = [];
          const links = Array.from(document.querySelectorAll('a'));

          for (const a of links) {
            const rawName = a.innerText.trim();
            const href = a.getAttribute('href') || '';
            if (!rawName) continue;

            const isMonsterLink =
              href.includes('search.php') &&
              href.includes('t=mob');

            if (!isMonsterLink) continue;

            const parent = a.closest('tr, li, div, .card') || a.parentElement;
            const img = parent ? parent.querySelector('img') : null;
            const image = img ? (img.getAttribute('src') || '') : '';

            results.push({
              raw_name: rawName,
              href,
              image
            });
          }

          return results;
        }
        """)

        browser.close()

    seen = set()

    for item in items:
        name = clean_monster_name(item["raw_name"])
        source_url = urljoin(BASE_URL, item["href"])
        image_url = urljoin(BASE_URL, item["image"]) if item["image"] else ""

        if not name or name in seen:
            continue

        seen.add(name)

        monsters.append({
            "name": name,
            "source_url": source_url,
            "face_shape": "round",
            "vibe": "cute",
            "cute_level": 5,
            "dark_level": 3,
            "power_level": 5,
            "description": f"{name}의 외형과 분위기를 기반으로 한 메이플 몬스터 후보",
            "image_url": image_url
        })

    df = pd.DataFrame(monsters)
    df.to_csv("monsters_full.csv", index=False, encoding="utf-8-sig")

    print(f"완료! 몬스터 {len(df)}마리 저장됨")
    print(df.head(20))


if __name__ == "__main__":
    main()
