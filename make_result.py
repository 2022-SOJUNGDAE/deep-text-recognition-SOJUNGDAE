import json
import pandas as pd

def read_json(path):
    return json.load(open(path, 'r'))

dict = read_json("result.json")


df = pd.read_csv("open/sample_submission.csv")
list = []
count = 0
for index in range(len(df)):
  data = df.iloc[index]
  file_name = data["img_path"]
  if f"open/{file_name[2:]}" not in dict:
    count += 0
  else:
    list.append(dict[f"open/{file_name[2:]}"])
    count += 1
df["text"] = list
df.to_csv("grad_clip10_2_valscore079_sub.csv", index=False)
