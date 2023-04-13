import os
from datasets import load_dataset
import pymongo
import time
import random
# from tqdm import tqdm

username = os.environ["MONGO_DB_USERNAME"] 
password = os.environ["MONGO_DB_PASSWORD"] 

available_symbols = set("АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!$%()*+,-./:;<=>?@[]^_{|}№~ ")
def clean_text(text):
    text = text.replace("\n", " ").replace("#", "")
    text = ''.join([c for c in text if c in available_symbols])
    return text

myclient = pymongo.MongoClient(f"mongodb://{username}:{password}@135.181.98.162:33333/?authMechanism=DEFAULT")
mydb = myclient["habr_dataset"]

def send_data_split(split_name, data):
    collections = mydb.list_collection_names()
    if split_name not in collections:
        col = mydb[split_name]
        col.create_index("id", unique=True)
    else:
        col = mydb[split_name]
    try:
        col.insert_many(data,ordered=False)
    except:
        pass 

    
dataset = load_dataset('IlyaGusev/habr', split="train", streaming=True)

thrash_timestamp =  1577836800    #  01/01/2020    
data = []
for idx, item in enumerate(dataset):
    if idx % 10000==0:
        print(idx)
    if item["language"] == "ru" and item["time_published"] >= thrash_timestamp and item["type"] == "article":
        new_item = {"id":item["id"], "title":item["title"],
                    "text_markdown":item["text_markdown"], "statistics":item["statistics"],
                    "hubs":item["hubs"], "flows":item["flows"], "tags":item["tags"], "time_published": item["time_published"],"time_added_to_db":int(time.time())}
        new_item["text_markdown"] = clean_text(new_item["text_markdown"])
        data.append(new_item)
random.shuffle(data)
split_idx = int(len(data)*0.2)
send_data_split("test",data[:split_idx])
send_data_split("train",data[split_idx:])

