import numpy as np
import torch
import torch.nn.functional as F
import torch_pruning as tp
import time 
import io

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
models = list(col.find({"type":"pytorch","optimized":False}))
best_mse = 99999
best_model=None

for model in models:
    if model["mse"]<best_mse:
        best_mse=model["mse"]
        best_model=model

model_lin_reg = pickle.loads(best_model["model"])
model_lin_reg.eval()

example_inputs = torch.randn(1, 312).cuda()

# 0. importance criterion for parameter selections
imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')


iterative_steps = 10 # You can prune your model to the target sparsity iteratively.
pruner = tp.pruner.MagnitudePruner(
    model_lin_reg, 
    example_inputs, 
    global_pruning=True, # If False, a uniform sparsity will be assigned to different layers.
    importance=imp, # importance criterion for parameter selection
    iterative_steps=iterative_steps, # the number of iterations to achieve target sparsity
    ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=[],
)

for i in range(iterative_steps):
    pruner.step()


test_embs=[]
test_scores=[]
for x in mydb["test"].find({},{"emb":1,"statistics":1}):
    test_embs.append(np.array(x["emb"]))
    test_scores.append(np.array(x["statistics"]["score"]))
test_embs=np.array(test_embs,dtype=np.float32)
test_scores=np.array(test_scores,dtype=np.float32)

from sklearn.metrics import mean_squared_error, mean_absolute_error
with torch.no_grad():
    res = model_lin_reg(torch.from_numpy(test_embs).cuda()).cpu().numpy()
mse,mae = mean_squared_error(test_scores,res),mean_absolute_error(test_scores,res)
print(mse)
print(mae)


col = mydb["models"]
buff = io.BytesIO()
torch.save(model_lin_reg, buff)
buff.seek(0)
col.insert_one({"training_date":int(time.time()),"type":"pytorch","mse":int(mse),"mae":int(mae),"model":pickle.dumps(buff),"optimized":True})