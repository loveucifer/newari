import json

with open("translations.json", "r") as f:
    data = json.load(f)
print(f"Total dataset size: {len(data)}")