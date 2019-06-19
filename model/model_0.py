import torch as t
from config import opt
from torchvision import models
from torch import nn
from cnn_finetune import make_model


resnet34 = models.resnet34(pretrained=True, num_classes=1000)
resnet34.fc = nn.Linear(512, opt.num_class)
device = t.device('cuda:0' if opt.use_gpu else 'cpu')
resnet34.to(device)


densenet121 = make_model('densenet121', num_classes=opt.num_class, pretrained=True)
densenet121.to(device)



