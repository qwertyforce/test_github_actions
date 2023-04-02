import os
from datasets import load_dataset
import pymongo
from tqdm import tqdm

username = os.environ["MONGO_DB_USERNAME"] 
password = os.environ["MONGO_DB_PASSWORD"] 

available_symbols = set("АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!$%()*+,-./:;<=>?@[]^_{|}№~ ")
def clean_text(text):
    text = text.replace("\n", " ").replace("#", "")
    text = ''.join([c for c in text if c in available_symbols])
    return text

myclient = pymongo.MongoClient(f"mongodb://{username}:{password}@135.181.98.162:33333/?authMechanism=DEFAULT")
mydb = myclient["habr_dataset"]

def get_clean_send_data(split):
    collections = mydb.list_collection_names()
    if split not in collections:
        col = mydb[split]
        col.create_index("id", unique=True)
    else:
        col = mydb[split]
    
    dataset = load_dataset('IlyaGusev/habr', split=split, streaming=True)

    thrash_timestamp = 1577836800 # 01/01/2020
    data = []
    for item in tqdm(dataset,miniters=10000):
        if item["language"] == "ru" and item["time_published"] >= thrash_timestamp and item["type"] == "article":
            new_item = {"id":item["id"], "title":item["title"],
                        "text_markdown":item["text_markdown"], "statistics":item["statistics"],
                        "hubs":item["hubs"], "flows":item["flows"], "tags":item["tags"], "time_published": item["time_published"]}
            new_item["text_markdown"] = clean_text(new_item["text_markdown"])
            data.append(new_item)
    try:
        col.insert_many(data,ordered=False)
    except:
        pass

get_clean_send_data("train")
get_clean_send_data("test")