import pickle

PKL_PATH = "monster_clip_vectors.pkl"

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

print("================================")
print("monster_clip_vectors.pkl 확인")
print("================================")
print("벡터 개수:", len(data))
print()

keywords = ["포장마차", "로미오", "핀호브", "프릴드", "헬레나", "켄타우로스"]

for keyword in keywords:
    found = [x["name"] for x in data if keyword in x["name"]]
    print(f"[{keyword}] :", found if found else "없음")

print()
print("처음 30개 몬스터:")
for i, x in enumerate(data[:30], start=1):
    print(f"{i}. {x['name']}")

input("\n엔터를 누르면 종료됩니다...")
