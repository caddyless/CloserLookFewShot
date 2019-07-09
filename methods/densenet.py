import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from io_utils import device
import torch.nn.functional as F
from methods.meta_template import MetaTemplate


class DenseNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(DenseNet, self).__init__(model_func, n_way, n_support, flatten=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False, is_flat=True):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous()
        # the shape of z is [n_data, n_dim]
        z_proto = torch.mean(z_support, (1, 3, 4))
        z_query = z_query.contiguous().view(self.n_way * self.n_query, z_query.size(2), -1)

        dists = cosine(z_query, z_proto)
        if is_flat:
            scores = dists.mean(2)
        else:
            scores = dists
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.to(device))

        scores = self.set_forward(x, is_flat=False)
        f_num = scores.size(2)
        scores = scores.transpose(1, 2)
        scores = scores.contiguous()
        scores = scores.view(scores.size(0) * scores.size(1), -1)

        y_query = y_query.repeat(f_num, 1)
        y_query = y_query.transpose(0, 1)
        y_query = y_query.contiguous()
        y_query = y_query.view(-1)

        return self.loss_fn(scores, y_query)


def cosine(x, y, scale: int = 10):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1) and len(x.size()) == 3, print('The dim of x must be 3')
    f = x.size(2)
    x = x.unsqueeze(1).expand(n, m, d, f)
    y.unsqueeze_(0)
    y = y.unsqueeze(3).expand(n, m, d, f)
    return scale * F.cosine_similarity(x, y, dim=2)