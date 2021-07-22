import torch
import torch.nn.functional as F

# Loss functions
def loss_rank(x1, x2, target):
    batch_size = x1.shape[0]
    loss1 = F.cross_entropy(x1, target, reduction='none')
    ind_1_sorted = torch.argsort(loss1)

    loss2 = F.cross_entropy(x2, target, reduction='none')
    ind_2_sorted = torch.argsort(loss2)

    remember_rate = 0.2
    num_remember = int(remember_rate * batch_size)
    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    sample1 = set(ind_1_update.tolist())
    sample2 = set(ind_2_update.tolist())
    common = sample1 & sample2
    diff1 = sample1 - common
    diff2 = sample2 - common

    return torch.LongTensor(list(common)), torch.LongTensor(list(diff1)), torch.LongTensor(list(diff2)), num_remember




