import torch
import torch.nn.functional as F

# Loss functions
def group_decsion(logit_list, target, r0=0.5, r1=0.5, r2=0.8):
    batch_size = target.size(0)
    M = len(logit_list)
    labels_hc_ms = []
    ind_hc_ms = []
    for i in range(M):
        loss_m = F.cross_entropy(logit_list[i], target, reduction='none')
        ind_sorted_m = torch.argsort(loss_m)
        ind_hc_m = ind_sorted_m[:int(r0*batch_size)]
        ind_hc_ms.append(ind_hc_m)
        labels_hc_m = torch.zeros(batch_size)
        labels_hc_m[ind_hc_m]=1
        labels_hc_ms.append(labels_hc_m)
    
    labels_hc_ms = torch.stack(labels_hc_ms, dim=0)
    hc_score = torch.sum(labels_hc_ms, dim =0)

    ind_hc = torch.nonzero((r1*M<hc_score)&(hc_score<=r2*M)).view(-1)
    ind_clean = torch.nonzero(hc_score>r2*M).view(-1)
    ind_hc_set = set(ind_hc.tolist())
    ind_clean_set = set(ind_clean.tolist())
    
    feeds_hc_ms = []
    feeds_clean_ms = []
    for i in range(M):
        hc_from_m = set(ind_hc_ms[i].tolist()) & ind_hc_set
        feeds_hc_ms.append(torch.LongTensor((list(hc_from_m))))
        clean_from_m = set(ind_hc_ms[i].tolist()) & ind_clean_set
        feeds_clean_ms.append(torch.LongTensor(list(clean_from_m)))
        
    return feeds_clean_ms, feeds_hc_ms

def group_decsion_avgloss(logit_list, target, r0=0.5):
    batch_size = target.size(0)
    M = len(logit_list)
    loss_ms = []
    ind_hc_ms = []
    for i in range(M):
        loss_m = F.cross_entropy(logit_list[i], target, reduction='none')
        loss_ms.append(loss_m)
        ind_hc_m = torch.argsort(loss_m)[:int(r0*batch_size)]
        ind_hc_ms.append(ind_hc_m)
    
    mean_sm = torch.mean(torch.stack(loss_ms, 0), 0)
    ind_clean = torch.argsort(mean_sm)[:int(r0*batch_size)]
    ind_clean_set = set(ind_clean.tolist())

    feeds_hc_ms = []
    feeds_clean_ms = []
    
    for i in range(M):
        clean_from_m = set(ind_hc_ms[i].tolist()) & ind_clean_set   # constrain the feed source of clean
        # clean_from_m = ind_clean_set                                # do not constrain the feed source of the clean
        feeds_clean_ms.append(torch.LongTensor((list(clean_from_m))))
        hc_from_m = set(ind_hc_ms[i].tolist()) - ind_clean_set
        feeds_hc_ms.append(torch.LongTensor(list(hc_from_m)))

    return feeds_clean_ms, feeds_hc_ms

def group_decsion_intersect(logit_list, target, r0=0.5):
    batch_size = target.size(0)
    M = len(logit_list)
    loss_ms = []
    ind_hc_ms = []
    for i in range(M):
        loss_m = F.cross_entropy(logit_list[i], target, reduction='none')
        loss_ms.append(loss_m)
        ind_hc_m = torch.argsort(loss_m)[:int(r0*batch_size)]
        ind_hc_ms.append(set(ind_hc_m.tolist()))
    
    ind_clean_set = set.intersection(*ind_hc_ms)

    feeds_hc_ms = []
    feeds_clean_ms = []
    
    for i in range(M):
        clean_from_m = ind_hc_ms[i] & ind_clean_set   # constrain the feed source of clean
        # clean_from_m = ind_clean_set                                # do not constrain the feed source of the clean
        feeds_clean_ms.append(torch.LongTensor((list(clean_from_m))))
        hc_from_m = ind_hc_ms[i]- ind_clean_set
        feeds_hc_ms.append(torch.LongTensor(list(hc_from_m)))

    return feeds_clean_ms, feeds_hc_ms

def get_feed_source(m_indx, M, alpha = 1, shuffle = False):
    assert (alpha < M)
    model_seq = torch.arange(M)
    if m_indx+alpha+1 <= M:
        source_m_ind = model_seq[m_indx + 1 : m_indx + alpha + 1]
        return source_m_ind
    else:
        front_num = (m_indx+alpha+1)%M
        fron_ind = model_seq[:front_num]
        end_ind = model_seq[m_indx + 1:]
        source_m_ind = torch.cat((fron_ind, end_ind), dim=-1)

    return source_m_ind
    



    
    
    
        

        