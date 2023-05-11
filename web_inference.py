
# import uvicorn
# if __name__ == '__main__':
#     uvicorn.run('web_inference:app', host='0.0.0.0', port=32846, log_level="info")
#     exit()

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
emb_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2").cuda()
MAX_SEQUENCE_LENGTH=2048
emb_model.max_seq_length = MAX_SEQUENCE_LENGTH

class LinearRegressor(torch.nn.Module):
  def __init__(self, input_dim=312, output_dim=1):
    super(LinearRegressor, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, 256)
    self.linear2 = torch.nn.Linear(256, 128)
    self.linear3 = torch.nn.Linear(128, output_dim)
  def forward(self, x):
    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    x = F.relu(x)
    x = self.linear3(x)
    return x

import sys
thismodule = sys.modules["__main__"]
setattr(thismodule, "LinearRegressor", LinearRegressor)

import pymongo
import pickle
import os 

username = os.environ["MONGO_DB_USERNAME"] 
password = os.environ["MONGO_DB_PASSWORD"] 

myclient = pymongo.MongoClient(f"mongodb://{username}:{password}@127.0.0.1:33333/?authMechanism=DEFAULT")
mydb = myclient["habr_dataset"]

col = mydb["models"]
models = list(col.find({}))
best_mse = 99999
best_model=None

for model in models:
    if model["mse"]<best_mse:
        best_mse=model["mse"]
        best_model=model

model_lin_reg = pickle.loads(best_model["model"])
if best_model["type"] == "pytorch":
   model_lin_reg.eval()

available_symbols = set("АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!$%()*+,-./:;<=>?@[]^_{|}№~ ")
def clean_text(text):
    text = text.replace("\n", " ").replace("#", "")
    text = ''.join([c for c in text if c in available_symbols])
    return text

def predict(text):
    text = clean_text(text)
    with torch.no_grad():
        tokens = tokenizer([text], max_length=MAX_SEQUENCE_LENGTH, padding="max_length", truncation=True)["input_ids"]
        tokens = torch.from_numpy(np.array(tokens)).cuda()
        embeddings = emb_model(tokens)
        embeddings = embeddings.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)

    if best_model["type"] == "pytorch":
        with torch.no_grad():
            res = model_lin_reg(embeddings)
            res=res.cpu().numpy()[0][0]
    else:
        res = model_lin_reg.predict(embeddings.cpu().numpy())
        res=res[0]
    return res

from fastapi import FastAPI, Body
from fastapi.responses import FileResponse

app = FastAPI()

@app.get("/")
def root():
    return FileResponse("templates/index.html")

@app.post("/hello")
def hello(data = Body()):
    text = data["text"]
    return {"message": f"{predict(text)}"}
