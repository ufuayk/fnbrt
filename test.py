import json
from transformers import pipeline

clf = pipeline("text-classification", model="ProsusAI/finbert")

with open("examples.json", encoding="utf-8") as f: # ai write
    tests = json.load(f)

passed = 0
for case in tests:
    scores = clf(case["text"], top_k=None)
    predicted = scores[0]["label"]
    ok = predicted in case["expected"]
    passed += ok

    status = "OK  " if ok else "FAIL"
    top = ", ".join(f'{s["label"]}={s["score"]:.2f}' for s in scores)
    print(f'[{status}] #{case["id"]:>2} tahmin={predicted:<8} beklenen={case["expected"]}')
    print(f'        {top}')
    print(f'        {case["text"][:90]}')
    print()

print("")
print(f"{passed}/{len(tests)}")