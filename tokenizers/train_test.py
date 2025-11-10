#script to do train/validation split
import json
from sklearn.model_selection import train_test_split

input_path = "babycosmofine_10M.jsonl"
train_out = "train.jsonl"
valid_out = "valid.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

train_lines, valid_lines = train_test_split(
    lines, 
    test_size=0.1,
    shuffle=True
)

with open(train_out, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open(valid_out, "w", encoding="utf-8") as f:
    f.writelines(valid_lines)

print(f"Train: {len(train_lines)} lines")
print(f"Valid: {len(valid_lines)} lines")
