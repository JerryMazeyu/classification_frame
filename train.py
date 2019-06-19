from model.model_0 import *
from data_process import *
import torch as t
from config import opt
from torch import nn


device = t.device('cuda:0' if opt.use_gpu else 'cpu')
model = densenet121


criterion = nn.CrossEntropyLoss()
lr = opt.lr
optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)


densenet121_finetune_wt = model.state_dict()
for epoch in range(opt.max_epoch):
    print("epoch is: ", epoch)
    for ii, (input, target) in enumerate(train_loader):
        print("ii is: ", ii)
        input.to(device)
        target.to(device)
        optimizer.zero_grad()
        score = model(input)
        loss = criterion(score, target)
        loss.backward()
        optimizer.step()
        print("loss is: ", loss)

t.save(densenet121_finetune_wt, opt.net_path)
