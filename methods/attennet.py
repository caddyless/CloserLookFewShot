import backbone
import torch
import torch.nn as nn
from io_utils import device
from torch.autograd import Variable
import numpy as np
from methods.meta_template import MetaTemplate


class AttenNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(AttenNet, self).__init__(model_func, n_way, n_support)
        self.feature = model_func(n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        x = Variable(x.to(device))
        x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
        scores = self.feature(x)
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.to(device))

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query)


