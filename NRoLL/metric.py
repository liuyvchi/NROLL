
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F

def mix_fc(t, x, w, label, fc_type, scale=32, margin=0.55, **kwargs):
    """
    The mix_fc is the function version of MixFC
    :param x: input tensor of shape batch_size x feat_dim
    :param w: weight tensor of shape feat_dim x num_class
    :param label: label
    :param fc_type: str, can only be either svfc or arcfc
    :param scale: default is 32
    :param margin: default is 0.55
    :param kwargs: some other parameters needed to make this function work
    :return:
    """
    assert fc_type in ['svfc', 'arcfc', 'softmax']
    kernel_norm = F.normalize(w, dim=0)
    # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
    # note x is assumed to be l2-normalized
    cos_theta = torch.mm(x, kernel_norm)

    cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
    cos_m = math.cos(margin)
    sin_m = math.sin(margin)
    mm = math.sin(math.pi - margin) * margin  # issue 1
    threshold = math.cos(math.pi - margin)  # threshould = 0.3
    batch_size = label.size(0)
    if fc_type == 'softmax':
        return cos_theta*scale
    elif fc_type == 'svfc':
        if 'is_am' not in kwargs:
            kwargs['is_am'] = False
        if 'mask' not in kwargs:
            kwargs['mask'] = 1.2
        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)  # get ground truth score of cos distance
        
        if kwargs['is_am']:  # AM
            mask = cos_theta > gt - margin
            final_gt = torch.where(gt > margin, gt - margin, gt)
        else:
            sin_theta = torch.sqrt(1.0- torch.pow(gt, 2))
            cos_theta_m = gt * cos_m - sin_theta * sin_m  # cos(gt+theta)
            mask = cos_theta > cos_theta_m
            final_gt = torch.where(gt>0, cos_theta_m, gt)       
            # final_gt = gt
            
        # process hard example.
        hard_example = cos_theta[mask]
        cos_theta[mask] = t * hard_example + t - 1.0

        cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        cos_theta = cos_theta * scale
        return cos_theta
    elif fc_type == 'arcfc':
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = (cos_theta * cos_m - sin_theta * sin_m)
        if 'easy_margin' not in kwargs:
            kwargs['easy_margin'] = True
        if kwargs['easy_margin']:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
            #cos_theta_m = torch.where(cos_theta > -1, cos_theta_m, cos_theta)
        else:
            cos_theta_m = torch.where(cos_theta > threshold, cos_theta_m, cos_theta - mm)
        output = cos_theta * 1.0
        output[torch.arange(0, batch_size), label] = cos_theta_m[torch.arange(0, batch_size), label]
        output = output * scale
        return output
    else:
        raise Exception('Unknown input flag.')


def cos_value(x, w):

    kernel_norm = F.normalize(w, dim=0)
    # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
    # note x is assumed to be l2-normalized
    cos_theta = torch.mm(x, kernel_norm)

    cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
    return cos_theta