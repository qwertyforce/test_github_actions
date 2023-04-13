import os
import pymongo
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2").cuda()
MAX_SEQUENCE_LENGTH=2048
model.max_seq_length = MAX_SEQUENCE_LENGTH

username = os.environ["MONGO_DB_USERNAME"] 
password = os.environ["MONGO_DB_PASSWORD"] 

myclient = pymongo.MongoClient(f"mongodb://{username}:{password}@135.181.98.162:33333/?authMechanism=DEFAULT")
mydb = myclient["habr_dataset"]

batch_size = 8

def get_embs(batch):
    tokens = tokenizer(batch, max_length=MAX_SEQUENCE_LENGTH, padding="max_length", truncation=True)["input_ids"]
    tokens = torch.from_numpy(np.array(tokens)).cuda()
    with torch.no_grad():
        embeddings = model(tokens)
        embeddings = embeddings.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings.cpu().numpy()

for split in ["train","test"]:
    col = mydb[split]
    data = list(col.find({}))
    for start_pos in range(0,len(data),batch_size):
        batch = data[start_pos:start_pos + batch_size]
        text_data =  [x["text_markdown"] for x in batch]
        embs = [x.tolist() for x in get_embs(text_data)]
        # print([x["text_markdown"] for x in batch])
        # data = [x.tolist() for x in get_embs()]
        # # print(data)
        for i in range(len(batch)):
            col.find_one_and_update({'id':batch[i]["id"]}, {"$set": {'emb':embs[i] }}, upsert=False)