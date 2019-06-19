import torch as t
import visdom
from model.model_0 import *
from data_process import *
import numpy as np


# vis = visdom.Visdom(env=opt.env)
model = densenet121
model.load_state_dict(t.load(opt.net_path))
n = len(test_loader)
print(n)
acc_freq = 0
for ii, (input, target) in enumerate(test_loader):
    print("+"*78)
    print("this is the %s time." % (ii+1))
    output = model(input)
    output_numpy = output.detach().numpy()
    output_batch = np.argmax(output_numpy, axis=1)
    target = target.numpy()
    print(output_batch, target)
    delta = target - output_batch
    acc_freq += (opt.batch_size - np.count_nonzero(delta))
    print("the accuracy is : %.2f" % (acc_freq/(n*opt.batch_size)))
