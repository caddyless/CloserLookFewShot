import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate


class DenseNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(DenseNet, self).__init__(model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous()
        # the shape of z is [n_data, n_dim]
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        dists = cosine(z_query, z_proto)
        scores = dists
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query)


def cosine(x, y, scale: int = 10):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    if len(x.size()) == 2:
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return scale * F.cosine_similarity(x, y, dim=2)
    elif len(x.size()) == 3:
        f = x.size(2)
        x = x.unsqueeze(1).expand(n, m, d, f)
        y.unsqueeze_(0)
        y = y.unsqueeze(3).expand(n, m, d, f)
        return scale * F.cosine_similarity(x, y, dim=2).sum(2)
