from openai import OpenAI
import os
from utils.net_utils import stringtolist
import json
import time


client = OpenAI(
    base_url='https://xiaoai.plus/v1',
    api_key='sk-40LQIyWIzbonUqeKcHokPvsW4mhrGI7nhqtoBTc15dXHRILh'
)

# VisDA
# folder_path = os.path.join("envisioned_classes", "VisDA")
# test_labels = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle',
#                   'person', 'plant']

# Office
folder_path = os.path.join("envisioned_classes", "Office")
test_labels = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle',
                  'person', 'plant']

file_path5 = os.path.join(folder_path, f"{40}_gpt-4-turbo_{5}.json")
# file_path6 = os.path.join(folder_path, f"{40}_gpt-4-turbo_{6}.json")

gen_class_num = 40

dissimilar5 = f'''Q: Given the image category [water jug], please suggest visually dissimilar categories that are not directly related or belong to the same primary group as [water jug]. Provide suggestions that do not share visual characteristics but are from broader and different domains than [water jug]. A: There are three classes dissimilar to [water jug], and they are from broader and different domains than [water jug]: - trumpets - helmets - rucksacks
  Q: Given the image category [{test_labels}], please suggest visually dissimilar categories that are not directly related or belong to the same primary group as [{test_labels}]. Provide suggestions that do not share visual characteristics but are from broader and different domains than [{test_labels}]. A: There are {gen_class_num} classes dissimilar to [{test_labels}], and they are from broader and different domains than [{test_labels}]:'''

# ir6 = f'''Q: Given the image category [water jug], please suggest categories that are not directly related or belong to the same primary group as [water jug]. A: There are three classes from broader and different domains than [water jug]: - trumpets - helmets - rucksacks
#   Q: Given the image category [{test_labels}], please suggest categories that are not directly related or belong to the same primary group as [{test_labels}]. A: There are {gen_class_num} classes from broader and different domains than [{test_labels}]:'''

prefix = "Before my question, I will give you an example; please strictly follow my answer format(just use '-' before every answer) and just give me the answer (the answer starts with -), no other words!!!  Do not include my questions and examples in your answer.\n"

# 5
start = time.time()
prompts = prefix + dissimilar5
completion = client.chat.completions.create(
  model="gpt-4-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompts}
  ]
)
response_texts = completion.choices[0].message.content
descriptors_list = stringtolist(response_texts)
descriptors = {"unknown": descriptors_list}
with open(file_path5, 'w') as fp:
    json.dump(descriptors, fp)
print(f"5:{descriptors} ")
endtime = time.time()
with open("TMM_MR/time.txt", "a+") as f:
  f.write(f"LLM use Time: {endtime - start:.0f}")
# 6
# prompts = prefix + ir6
# completion = client.chat.completions.create(
#   model="gpt-4-turbo",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompts}
#   ]
# )
# response_texts = completion.choices[0].message.content
# descriptors_list = stringtolist(response_texts)
# descriptors = {"unknown": descriptors_list}
# with open(file_path6, 'w') as fp:
#     json.dump(descriptors, fp)
# print(f"6:{descriptors} ")
