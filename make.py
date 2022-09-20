import os
import pandas as pd
import random

path = "/mnt/c/datasets/open_structure"

data = pd.read_csv(f"{path}/train_without_outlier.csv")
os.mkdir(f"{path}/validation")


train_index = list(range(len(data)))

random.shuffle(train_index)

val_index = train_index[:int(0.2*len(train_index))]
train_index = train_index[int(0.2*len(train_index)):]

data_train = data.iloc[train_index]
data_val = data.iloc[val_index]

data_train.reset_index(drop=True, inplace=True)
data_val.reset_index(drop=True, inplace=True)

image_list = []
text_list = []

for index in range(len(data_val)):
    single = data_val.iloc[index]

    image_path = single["img_path"].split("/")[-1]
    text = single["text"]
    os.rename(f"{path}/train/{image_path}", f"{path}/validation/{image_path}")
    
    image_list.append(f"validation/{image_path}")
    text_list.append(text)

data_val = pd.DataFrame(list(zip(image_list, text_list)),
               columns =['img_path', 'text'])


image_list = []
text_list = []

for index in range(len(data_train)):
    single = data_train.iloc[index]

    image_path = single["img_path"].split("/")[-1]
    text = single["text"]
    
    image_list.append(f"train/{image_path}")
    text_list.append(text)

data_train = pd.DataFrame(list(zip(image_list, text_list)),
                        columns =['img_path', 'text'])

data_train.to_csv("train.tsv", sep="\t", index=False)
data_val.to_csv("validation.tsv", sep="\t", index=False)