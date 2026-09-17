import json
from datasets import Dataset
from trl import PPOTrainer, PPOConfig

def normalize_tests(ex):
    return [{"assertion": "assert 1 == 1"}]

data = [{"question": "test problem", "input_output": {"inputs": [["1"]], "outputs": [["1"]]}}]
ds = Dataset.from_list(data)

def encode(ex):
    return {"input_ids": [1, 2, 3], "benchmark_tests": json.dumps(normalize_tests(ex))}

ds = ds.map(encode)

collate = lambda rows: {k: [r[k] for r in rows] for k in rows[0]}

print("Dataset features:", ds.features)
print("Sample row:", ds[0])

row = ds[0]
print("benchmark_tests in row:", "benchmark_tests" in row)
parsed = json.loads(row["benchmark_tests"])
print("Parsed tests:", parsed)
