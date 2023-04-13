import os
import pymongo
import torch 
from tqdm import tqdm
from statistics import mean
import torch.nn.functional as F
import numpy as np
# import xgboost as xgb 
import pickle
import time

username = os.environ["MONGO_DB_USERNAME"] 
password = os.environ["MONGO_DB_PASSWORD"] 

myclient = pymongo.MongoClient(f"mongodb://{username}:{password}@127.0.0.1:33333/?authMechanism=DEFAULT")
mydb = myclient["habr_dataset"]

col = mydb["models"]
models = list(col.find({"type":"pytorch"}))
best_mae = 99999
best_model=None

for model in models:
    if model["mae"]<best_mae:
        best_mae=model["mae"]
        best_model=model

train_embs=[]
train_scores=[]
for x in mydb["train"].find({"time_added_to_db":{"$gte":best_model["training_date"]}},{"emb":1,"statistics":1}):
    train_embs.append(np.array(x["emb"]))
    train_scores.append(np.array(x["statistics"]["score"]))
train_embs=np.array(train_embs,dtype=np.float32)
train_scores=np.array(train_scores,dtype=np.float32)

if len(train_embs)==0:
   print("no new training data found")
   exit()

test_embs=[]
test_scores=[]
for x in mydb["test"].find({},{"emb":1,"statistics":1}):
    test_embs.append(np.array(x["emb"]))
    test_scores.append(np.array(x["statistics"]["score"]))
test_embs=np.array(test_embs,dtype=np.float32)
test_scores=np.array(test_scores,dtype=np.float32)


class LinearRegressor(torch.nn.Module):
  def __init__(self, input_dim=312, output_dim=1):
    super(LinearRegressor, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, 256)
    self.linear2 = torch.nn.Linear(256, 128)
    self.linear3 = torch.nn.Linear(128, output_dim)
    # self.linear4 = torch.nn.Linear(256, 1)
  def forward(self, x):
    x = self.linear1(x)
    x = F.relu(x)
    # x = torch.nn.functional.dropout(x, p=0.3)
    x = self.linear2(x)
    x = F.relu(x)
    # x = torch.nn.functional.dropout(x, p=0.3)
    x = self.linear3(x)
    # x = F.relu(x)
    # x = self.linear4(x)
    return x

model_lin_reg = pickle.loads(best_model["model"])

criterion = torch.nn.MSELoss()
# criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_lin_reg.parameters(), lr=0.001)
model_lin_reg = model_lin_reg.cuda()


X_train_cuda = torch.from_numpy(train_embs).cuda()
y_train_cuda = torch.from_numpy(train_scores).cuda()
X_test_cuda = torch.from_numpy(test_embs).cuda()
y_test_cuda = torch.from_numpy(test_scores).cuda()

losses = []
losses_test = []
batch_size=32
acc_test = []
for epoch in tqdm(range(10)):
    cur_loss_idx = len(losses)
    cur_loss_text_idx = len(losses_test)
    permutation = torch.randperm(X_train_cuda.size()[0])
    # losses_ep = []
    # losses_ep_test = []
    for i in range(0,X_train_cuda.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_cuda[indices], y_train_cuda[indices]

        outputs = model_lin_reg.forward(batch_x)
        # print(outputs.shape)
        # print(batch_y.shape)
        # loss = criterion(outputs,batch_y)
        loss = criterion(outputs,batch_y.unsqueeze(1))

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        with torch.no_grad():
            output = model_lin_reg(X_test_cuda)
            # loss = criterion(output, y_test_cuda)
            loss = criterion(output, y_test_cuda.unsqueeze(1))
            losses_test.append(loss.item())
    print("=========================")
    print(f"train loss = {mean(losses[cur_loss_idx:])}")
    print(f"test loss = {mean(losses_test[cur_loss_text_idx:])}")
model_lin_reg.eval()

from sklearn.metrics import mean_squared_error, mean_absolute_error
with torch.no_grad():
    res = model_lin_reg(X_test_cuda).cpu().numpy()
mse,mae = mean_squared_error(test_scores,res),mean_absolute_error(test_scores,res)
print(mse)
print(mae)


col = mydb["models"]
col.insert_one({"training_date":int(time.time()),"type":"pytorch","mse":int(mse),"mae":int(mae),"model":pickle.dumps(model_lin_reg)})

# print('xgb')
# regressor=xgb.XGBRegressor(eval_metric='rmse')
# regressor.fit(train_embs, train_scores)
# predictions = regressor.predict(test_embs)
# mse,mae =  mean_squared_error(test_scores,predictions), mean_absolute_error(test_scores,predictions)
# print(mse)
# print(mae)
# col.insert_one({"training_date":int(time.time()),"type":"xgb","mse":int(mse),"mae":int(mae),"model":pickle.dumps(regressor)})

# print('svr')
# from sklearn.svm import SVR
# svr = SVR()
# svr.fit(train_embs,train_scores)
# preds = svr.predict(test_embs) 
# mse,mae = mean_squared_error(test_scores,preds), mean_absolute_error(test_scores,preds)
# print(mse)
# print(mae)
# col.insert_one({"training_date":int(time.time()),"type":"svr","mse":int(mse),"mae":int(mae),"model":pickle.dumps(svr)})

