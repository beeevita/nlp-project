#coding=utf8
import torch.nn as nn
import torch
import numpy as np

# model = nn.LSTM(input_size=6, hidden_size=9, num_layers=2, batch_first=True)
# model = model.double()
#
# x = np.random.randn(2, 10, 6)  # 100个10*6的tensor
# print(x)
#
# x = torch.from_numpy(x)
# print(x.shape)
#
# y, (hn, cn) = model(x)  # 不提供h0和c0，默认全0
# print('y:', y.shape)
# print('hn:', hn.shape)
# print('cn:', cn.shape)
x = np.random.randn(2, 10, 1)
print(x.squeeze(-1).shape)
