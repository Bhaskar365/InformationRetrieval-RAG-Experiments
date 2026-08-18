
import json
from pathlib import Path

def load_eval_dataset(path:str):
    path = Path(path)

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


dataset = load_eval_dataset(Path("D:\\mlTesting\\FAISS\\productionRAG\\data\\eval_dataset.json"))

if dataset is None:
    raise ValueError

for example in dataset:
    print(example["question"])


