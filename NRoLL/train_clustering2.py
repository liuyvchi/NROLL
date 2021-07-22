import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import Parameter
from loss import loss_rank
import numpy as np
import os
import lmdb_utils
# import lmdb_utils_synthetic
import argparse
import sys
import logging as logger
import torch.nn.functional as F
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
from mobilefacenet import MobileFaceNet, EncoderPlusArc
from metric import mix_fc, cos_value
from group_mining import group_decsion, get_feed_source, group_decsion_avgloss, group_decsion_intersect
import random
from tensorboardX import SummaryWriter
import shutil
from test_lfw import *
import pickle 
import collections
import torchvision.transforms as transforms
import utils.threshould as threshould


# ------------------- test LFW ---------------------------
DATA_ROOT = '/home/liuyuchi3/data/'

lfw_filename = DATA_ROOT + '781/lfw_image_list.txt'
lfw_root_dir = DATA_ROOT + '781/lfw_cropped_part34/lfw_cropped_part34/'
lfw_pairs_file = DATA_ROOT + '781/lfw_test_pairs.txt'
lfw_test_pairs = []

lfw_test_pairs, test_data_loader = prepare_test(lfw_filename, lfw_root_dir, lfw_pairs_file, lfw_test_pairs)

# ------------------- test LFW ---------------------------

# transformer_train = transforms.Compose([
#     # transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#     transforms.ToTensor()
# ])

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# def introspection(feat_l_q, label, prototypes):
#     print(prototypes.size(), label.size())
#     intra_sim = torch.mm(feat_l_q, prototypes[label].transpose(0,1)).diagonal(0)
#     mean_intra_sim = torch.mean(intra_sim)
#     inter_sim = torch.mm(prototypes[label], prototypes[label].transpose(0,1))
#     max_inter_sim = torch.max(inter_sim[inter_sim.triu(diagonal=1) > 0])

#     if mean_intra_sim < max_inter_sim:
#         return True
#     else:
#         return False 

def update_model_k(model_q, model_k, m):
    for k_param, q_param in zip(model_k.parameters(), model_q.parameters()):
        k_param.data.mul_(m).add_((1 - m), q_param.detach().data)


def clustering(conf, labeled_loader, unlabel_loader, loss_fn, model_q, model_k, pre_centers, optimizer, saved_dir, print_freq=200, introspection_freq=1000):
    # switch to train mode
    
    m_m = 0.999
    m_t = 0.999
    m_p = 0.9
    
    pre_classes = conf.num_class 

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabel_loader)
    
    cluster_dic = {}
    introspection_times = 0

    T_up, T_down = None, None
    
    if pre_centers is None:
        prototypes = torch.Tensor(conf.num_class, conf.feat_dim).cuda()
        prototypes.requires_grad = False
        prototypes.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
    else:
        prototypes = pre_centers.cuda()

    for batch_idx in range(conf.total_iter): #conf.full_loader_size
        try:
            img_lq, img_lk, label, lmdbkeys_l, lmdbkeys_2 = labeled_iter.next()
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            img_lq, img_lk, label, lmdbkeys_l, lmdbkeys_2 = labeled_iter.next()

        try:
            img_uq, img_uk, _, lmdbkeys_u = unlabeled_iter.next()
        except StopIteration:
            unlabeled_iter = iter(unlabel_loader)
            img_uq, img_uk, _, lmdbkeys_u = unlabeled_iter.next()
            print("introspection")
            with open(os.path.join(saved_dir,"results.txt"), 'w') as wf:
                for key, value in cluster_dic.items():
                    wf.write(key+' '+str(value)+"\n")
#             cluster_dic = {}
#             prototypes = prototypes[:pre_classes]
#             introspection_times+=1
#             conf.writer.add_scalar('introspection_times', introspection_times, batch_idx)


        model_q = model_q.train()
        img_lq = img_lq.cuda()
        label = label.squeeze().cuda()
        feat_l_q = model_q(img_lq) 
        
        model_k.eval()
        def set_bn_train(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.train()
        model_k.apply(set_bn_train)
        with torch.no_grad():
            img_lk = img_lk.cuda()                
            feat_l_k = model_k(img_lk) 
        
        # update prototypes
        prototypes[label] = m_p*prototypes[label] + (1-m_p)*feat_la_k.detach() # update old prototypes in the momentum fashion by using labeled data
        prototypes[label] = F.normalize(prototypes[label], dim=1)
        
#         logits_l = mix_fc(1.2, feat_l_q, prototypes.transpose(0,1), label, fc_type='softmax', margin=0.5, easy_margin=True)

        # compute loss for labeled data
    
#         logits_l = torch.mm(feat_l_q, prototypes.transpose(0,1).detach()) 
#         T = 0.033
#         logits_l = torch.div(logits_l, T)
#         loss_l = loss_fn(logits_l, label)
        
        prototypes_copy = prototypes.detach()
        prototypes_copy[label]= 0
        prototypes_copy = prototypes_copy[prototypes_copy[:, 0] != 0]
        
        N = len(label)
        C = conf.feat_dim
        all_idx = range(len(prototypes_copy))
        K = len(prototypes_copy)
        T = 0.033
#         random_n_idx = random.sample(all_idx, K)
        logits_p = torch.bmm(feat_l_q.view(N,1,C), feat_l_k.view(N, C, 1)).view(N, 1)   # positive logits
        logits_n = torch.mm(feat_l_q.view(N,C), prototypes_copy.transpose(0,1).view(C, K)) # negative logits
        logits_l = torch.cat((logits_p, logits_n), dim =1) 
        logits_l = logits_l.clamp(-1, 1)  # for numerical stability
        logits_l = torch.div(logits_l, T)      
        label_l = torch.zeros(N).long().cuda() # label for new classes are all 0-th
        loss_l = loss_fn(logits_l, label_l) 

        # total loss
        loss = loss_l
        
        conf.writer.add_scalar('loss', loss, batch_idx)
        conf.writer.add_scalar('loss_l', loss_l, batch_idx)

        if batch_idx % print_freq == 0:
            lr = get_lr(optimizer)
            print('introspection %d, iter %d, lr %f, loss %f， loss_l %f' % (introspection_times, batch_idx, lr, loss, loss_l))

        # update model_q
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update model_K    
        update_model_k(model_q, model_k, m_m)

        # update cluster_dic
        prototypes[label] = feat_l_k.detach()

        conf.writer.add_scalar('prototypes_length', len(prototypes), batch_idx)

        if batch_idx % conf.test_freq == 0 and batch_idx != 0:
            print("test model on LFW")
            model_q.eval()
            res = test(model_q, lfw_test_pairs, test_data_loader, blufr=False)
            lfw = res[0]
            print("lfw:", lfw, "cur_introspection_batch:", introspection_times, batch_idx)
            conf.writer.add_scalar('lfw', lfw, batch_idx)
        
        if batch_idx != 0 and batch_idx % conf.check_point_size == 0:
            saved_name = 'iter_%d.pt' % (batch_idx)
            torch.save({'state_dict': model_q.module.state_dict()}, os.path.join(saved_dir, saved_name))
            logger.info('save checkpoint %s to disk...' % (saved_name))
            logger.info('Start evaluating the model on LFW...')

        conf.lr_schedule.step()

    saved_name = 'model_q.pt'
    torch.save({'state_dict': model_q.module.state_dict()}, os.path.join(saved_dir, saved_name))
    print('save checkpoint %s to disk...' % (saved_name))
    

def train(conf):

    model_q = MobileFaceNet(conf.feat_dim)
    model_k = MobileFaceNet(conf.feat_dim)
    pre_centers = None
    conf.USE_PRETRAIN = False
    if conf.USE_PRETRAIN:
        model_path = './pre-trained/model1_epoch_44.pt'
        prototypes_path = './pre-trained/model1_epoch_44_fc.pickle'
        model_q.load_state_dict(torch.load(model_path)['state_dict'])
        model_k.load_state_dict(torch.load(model_path)['state_dict'])
        with open (prototypes_path, 'rb') as f:
            pre_centers = torch.tensor(pickle.load(f)).transpose(0,1)
            pre_centers = F.normalize(pre_centers, dim=1)
            print(pre_centers.size())
        
    model_q = torch.nn.DataParallel(model_q).cuda()
    model_k = torch.nn.DataParallel(model_k).cuda()
    
#     pre_centers = Parameter(torch.Tensor(conf.feat_dim, conf.num_class).cuda())
#     pre_centers.requires_grad = True
#     pre_centers.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    parameters = [p for p in model_q.parameters() if p.requires_grad]
#     parameters.append(pre_centers)
    optimizer = optim.SGD(parameters, lr=conf.lr, momentum=conf.momentum, weight_decay=5e-4, nesterov=True)
    loss_fn = torch.nn.CrossEntropyLoss() 

    conf.milestones = [p*int(conf.total_iter/5) for p in conf.milestones]
    conf.lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=conf.milestones, gamma=0.1)
    
    labeled_file_list = ['deepglint_opensplit_2_2_labeled.txt']
    unlabeled_file_list = ['deepglint_opensplit_2_2_unlabeled.txt']
    db_labeled = lmdb_utils.SingleLMDBDataset_1ID2img(conf.source_lmdb_list[0], labeled_file_list[0], conf.num_class, conf.key)
    db_unlabeled = lmdb_utils.MultiLMDBDataset(conf.source_lmdb_list, unlabeled_file_list, conf.key)
    print("db_labeled length:", len(db_labeled))
    loader_labeled = DataLoader(db_labeled, conf.batch_size, True, num_workers=4)   
    loader_unlabeled = DataLoader(db_unlabeled, conf.batch_size, True, num_workers=4)   
    
    clustering(conf, loader_labeled, loader_unlabeled, loss_fn, model_q, model_k, pre_centers, optimizer, conf.saved_dir, conf.print_freq, conf.introspection_freq)

    db_labeled.close()
    db_unlabeled.close()
    
    
def train_on_neuhub(argv):
    conf = argparse.ArgumentParser(description='train arcface on msceleb provided by deepglint.')
    conf.add_argument('--key', type=int, default=None, help="you must send a key into the script before training.")
    conf.add_argument("--input_db_dir", type=str, default='/home/liuyuchi3/data/', help="input database name")
#     conf.add_argument("--train_db_name", type=str, default='CASIA-Clean-R_part34_lmdb_enc', help="comma separated list of training database.")
#     conf.add_argument("--train_db_name", type=str, default='vggface2_part34_lmdb', help="comma separated list of training database.")
    conf.add_argument("--train_db_name", type=str, default='deepglint_msra_unoverlap_part34', help="comma separated list of training database.")
    conf.add_argument("--train_id", type=str, default='0', help="id for current train")
   
    conf.add_argument("--is_am", type=int, default=0, help="use AM if 1, otherwise arcface.")
    conf.add_argument('--margin', type=float, default=0.5, help='only take effect when trained on svfc.')
    conf.add_argument('--mask', type=float, default=1.1, help='only take effect when trained on svfc')
    conf.add_argument('--num_class', type=int, default=36389, help='number of classes') # CASIA clean 9869 vggface2 9131 msceleb 85173
    conf.add_argument('--batch_size', type=int, default=512, help='batch size over all gpus.')
    

    conf.add_argument('--easy_margin', type=int, default=1, help="1 if use easy margin.")
    conf.add_argument('--step', type=str, default='2, 3, 4',
                      help='similar to step specified in caffe solver, but in epoch mechanism')
    conf.add_argument('--loss_type', type=str, default='svfc', help="loss type, can only be arcfc or svfc")
    conf.add_argument('--gpu', type=str, default='', help='useless value, only for neuhub.')
    conf.add_argument('--feat_dim', type=int, default=512, help='feature dimension.')
    conf.add_argument('--epochs', type=int, default=5, help='how many epochs you want to train.')
    conf.add_argument('--total_iter', type=int, default=60000, help='how many epochs you want to train.')
    conf.add_argument('--print_freq', type=int, default=200, help='frequency of displaying current training state.')
    conf.add_argument('--in_channels', type=int, default=3, help='color image, it should be 3, gray scale, 1')
    conf.add_argument('--momentum', type=float, default=0.9, help='momentum')
    conf.add_argument('--p', type=int, default=1, help='control the depth of attention net.')
    conf.add_argument('--t', type=int, default=2, help='control the depth of attention net.')
    conf.add_argument('--r', type=int, default=1, help='control the depth of attention net.')
    conf.add_argument('--net_mode', type=str, default='irse', help='ir or irse')
    
    conf.add_argument('--lr', type=float, default=0.1, help='initial learning rate.')
    conf.add_argument('--exponent', type=float, default=1,
                       help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T).')

    conf.add_argument('--M', type=int, default=4, help='number of models in group mining')
    conf.add_argument('--alpha', type=int, default=1, help='alpha of feeds in group mining')
    conf.add_argument('--r0', type=float, default=0.7, help='threshould0 for intra model decision')
    conf.add_argument('--r1', type=float, default=0.5, help='threshould1 for inter model decision')
    conf.add_argument('--r2', type=float, default=0.8, help='threshould2 for inter model decision')
    conf.add_argument('--shuffle', type=str, default='close', help='shuffle the order of models')
    conf.add_argument('--noise_rate', type=float, default=0.3, help='symmetric noise rate')
    conf.add_argument('--mining_type', type=str, default='intersect', help='shuffle the order of models')
    conf.add_argument('--notificaiton', type=str, default='hc is not indluded in clean', help='notification')
    conf.add_argument('--split_num', default=5, type=int)
    conf.add_argument('--introspection_freq', default=1000, type=int)
    conf.add_argument('--test_freq', default=2000, type=int)
    conf.add_argument('--check_point_size', default=5000, type=int)
    
    

    # conf.add_argument('--tensorboardx_logdir', type=str, default='', help="dir name for tesnsorborad log")
    args = conf.parse_args()
    
    args.output_model_dir = "snapshot/"+args.train_id
    args.r0 = (1-args.noise_rate)
    print(args)
    print('PID:', os.getpid())

    cur_log_dir = os.path.join('notespace', args.train_id)
    if os.path.exists(cur_log_dir):
        shutil.rmtree(cur_log_dir)
    writer = SummaryWriter(log_dir=cur_log_dir)
    args.writer = writer

    args.t = np.ones(args.split_num) * (args.mask)
    # args.t = np.ones(args.epochs) * (args.mask - 1.0) 
    # args.t[:args.epochs] = 1.0 + np.linspace(0, (args.mask - 1.0) ** args.exponent, args.epochs) #adaptive t based on epoch 

    args.milestones = [int(p) for p in args.step.split(',')]
    
    tmp_train_dbs = [p.lstrip().rstrip() for p in args.train_db_name.split(',')]
    train_dbs = set()
    for e in tmp_train_dbs:
        train_dbs.add(e)
    print('%d database are(is) given to train this task.' % len(train_dbs))
    args.source_file_list = []
    args.source_lmdb_list = []
    args.saved_dir = args.output_model_dir  
    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)
    assert os.path.isdir(args.input_db_dir)
    for d in train_dbs:
        abs_db_path = os.path.join(args.input_db_dir, d)
        if os.path.isdir(abs_db_path):
            args.source_lmdb_list.append(abs_db_path)
            tmp_txt = os.listdir(abs_db_path)
            for e in tmp_txt:
                if e.endswith('.txt') and 'train' in e:
                    train_file_list = os.path.join(abs_db_path, e)
                    args.source_file_list.append(train_file_list)
                    break
    
    print('Start optimization.')
    train(args)
    print('Optimization done!')
    writer.close()

if __name__ == '__main__':
    print()
    train_on_neuhub(sys.argv)

