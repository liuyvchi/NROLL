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
from mobilefacenet import MobileFaceNet
from attention_network_def import AttentionNet
from metric import mix_fc
from group_mining import group_decsion, get_feed_source, group_decsion_avgloss, group_decsion_intersect
import random
from tensorboardX import SummaryWriter
import shutil
from test_lfw import * 
import collections

# ------------------- test LFW ---------------------------
DATA_ROOT = '/export/home/liuyuchi3/data/'

lfw_filename = DATA_ROOT + '781/lfw_image_list.txt'
lfw_root_dir = DATA_ROOT + '781/lfw_cropped_part34/lfw_cropped_part34/'
lfw_pairs_file = DATA_ROOT + '781/lfw_test_pairs.txt'
lfw_test_pairs = []

lfw_test_pairs, test_data_loader = prepare_test(lfw_filename, lfw_root_dir, lfw_pairs_file, lfw_test_pairs)

# ------------------- test LFW ---------------------------

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(conf, data_loader, org_t, loss_fn, models, fc_ms, optimizers, rate_schedule, cur_epoch, saved_dir, print_freq=200):
    # switch to train mode
    device = torch.device('cuda:0')
    M = conf.M
    alpha = conf.alpha
    
    loader_size = len(data_loader)
    check_point_size = (loader_size // 3, loader_size // 2, loader_size // 3 * 2)

    for batch_idx, (image, label) in enumerate(data_loader):
        ## warm up ##
        cur_iteration = cur_epoch*loader_size+batch_idx
        r_e = min(((cur_iteration+1)/(2*loader_size))*conf.noise_rate, conf.noise_rate)
        conf.r0 = (1-r_e)
        ##
        ## warm up on t##
        t = min(((cur_iteration+1)/(conf.epochs*loader_size))*org_t, org_t)
        ##
        assert conf.shuffle in ['open', 'close']
        if conf.shuffle == 'open':
            agents = list(zip(models, fc_ms, optimizers))
            random.shuffle(agents)
            models, fc_ms, optimizers = zip(*agents)
        
        label = label.squeeze()
        label = label.to(device)

        # #############################
        # start mining. explore hc and clean index
        # ##############################    
        pred_ms = []
        with torch.no_grad():
            for i in range(M):
                models[i].eval()
                image_detach = image.detach()
                image_detach = image_detach.to(device)
                feat_norm = models[i](image_detach)
                pred = mix_fc(1.0, feat_norm, fc_ms[i], label.detach(), fc_type='arcfc', margin=0.5, easy_margin=True)
                pred_detach = pred.detach()
                pred_ms.append(pred_detach)
                del pred 

            assert (conf.mining_type in ['vote', 'avgloss', 'intersect'])
            if conf.mining_type=='vote':
                feeds_clean_ms, feeds_hc_ms = group_decsion(pred_ms, label.detach(), r0=conf.r0, r1=conf.r1, r2=conf.r2)
            elif conf.mining_type=='avgloss':
                feeds_clean_ms, feeds_hc_ms = group_decsion_avgloss(pred_ms, label.detach(), r0=conf.r0)    
            elif conf.mining_type == 'intersect':
                feeds_clean_ms, feeds_hc_ms = group_decsion_intersect(pred_ms, label.detach(), r0=conf.r0)
            del pred_ms

        # ########################
        # start update
        # ###########################
        for i in range(M):
            models[i].train()
            fc_ms[i].requires_grad=True
            clean_ind = []
            hc_ind = []
            hc_ind_repeats = []
            loss_clean = 0
            loss_hc = 0
            clean_weight = 0
            hc_weight = 0

            image = image.to(device)
            feat_norm = models[i](image)

            models_ind = get_feed_source(i, M = M, alpha = alpha)
            for j in (models_ind.tolist()):
                clean_ind.append(set(feeds_clean_ms[j].tolist()))
                hc_ind.append(set(feeds_hc_ms[j].tolist()))
                hc_ind_repeats += feeds_hc_ms[j].tolist()
                
            clean_ind = torch.LongTensor(list(set.union(*clean_ind)))
            hc_ind = torch.LongTensor(list(set.union(*hc_ind)))   

            # constrain the the number of hc
            if clean_ind.numel()+hc_ind.numel()>int(conf.batch_size*conf.r0):
                over_size = clean_ind.numel() + hc_ind.numel() - int(conf.batch_size*conf.r0)
                hc_remain_num = hc_ind.numel()- over_size
                conf.writer.add_scalar('over_size', over_size, cur_iteration)
                count_result = torch.LongTensor(collections.Counter(hc_ind_repeats).most_common())
                hc_ind = count_result[:,0][:hc_remain_num].view(-1)
            
            if clean_ind.numel()>0:
                label_clean = label[clean_ind]
                feat_clean_norm = feat_norm[clean_ind]
#                 pred_clean = mix_fc(t, feat_clean_norm, fc_ms[i], label_clean, fc_type=conf.loss_type, margin=0.5, is_am=False)
                pred_clean = mix_fc(t, feat_clean_norm, fc_ms[i], label_clean, fc_type='arcfc', margin=0.5, is_am=False)
                loss_clean = F.cross_entropy(pred_clean, label_clean)
                clean_weight = len(clean_ind)
                conf.writer.add_scalars('loss/loss_clean_m', {'loss_clean_m%d'%(i):loss_clean} , cur_iteration)
                conf.writer.add_scalars('sample_num/num_clean_m', {'num_clean_m%d'%(i):len(clean_ind)} , cur_iteration)    
            
            if hc_ind.numel()>0:
                label_hc = label[hc_ind]
                feat_hc_norm = feat_norm[hc_ind]
#                 pred_hc = mix_fc(1.0, feat_hc_norm, fc_ms[i], label_hc, fc_type='arcfc', margin=0.5, easy_margin=True)
                pred_hc = mix_fc(1.0, feat_hc_norm, fc_ms[i], label_hc,fc_type=conf.loss_type, margin=0.5, easy_margin=True)
                loss_hc = F.cross_entropy(pred_hc, label_hc)
                hc_weight = len(hc_ind)
                conf.writer.add_scalars('loss/loss_hc_m', {'loss_hc_m%d'%(i):loss_hc} , cur_iteration)
                conf.writer.add_scalars('sample_num/num_hc_m', {'num_hc_m%d'%(i):len(hc_ind)} , cur_iteration)
            
            num_update = len(clean_ind)+len(hc_ind) 

            if i == 0:
                conf.real_sample += num_update

            loss_m = (hc_weight * loss_hc + clean_weight * loss_clean)/num_update
            conf.writer.add_scalars('loss/loss_m', {'loss_m%d'%(i):loss_m}, cur_iteration)
            conf.writer.add_scalars('sample_num/num_m', {'num_m%d'%(i):num_update}, cur_iteration)


            optimizers[i].zero_grad()
            loss_m.backward()
            optimizers[i].step()

            if batch_idx % print_freq == 0:
                loss_m_val = loss_m.item()
                lr = get_lr(optimizers[i])
                print('epoch %d, iter %d, lr %f, model_m%f, loss_clean %f' % (cur_epoch, batch_idx, lr, i, loss_clean)) 
                print('epoch %d, iter %d, lr %f, model_m%f, loss_hc %f' % (cur_epoch, batch_idx, lr, i, loss_hc)) 
                print('epoch %d, iter %d, lr %f, model_m%f, loss_m %f' % (cur_epoch, batch_idx, lr, i, loss_m_val))   
                print(len(hc_ind), len(clean_ind))
                real_epoch = conf.real_sample/len(data_loader.dataset)
                print("real_epoch:", real_epoch)

            if batch_idx != 0 and batch_idx in check_point_size:
                if cur_epoch > conf.epochs - 3:
                    saved_name = 'model_%d_epoch_%d_batch_%d.pt' % (i, cur_epoch, batch_idx)
                    torch.save({'state_dict': models[i].module.state_dict()}, os.path.join(saved_dir, saved_name))
                    logger.info('save checkpoint %s to disk...' % (saved_name))
                if i == 0:
                    logger.info('Start evaluating the model on LFW...')
                    models[i].eval()
                    res = test(models[i], lfw_test_pairs, test_data_loader, blufr=False)
                    lfw = res[0]
                    print("lfw:", lfw, "cur_epoch_batch:", cur_epoch, batch_idx)
                    conf.writer.add_scalar('lfw_m:', lfw, cur_iteration)


    if cur_epoch > conf.epochs - 5:
        for i in range(M):
            saved_name = 'model_%d_epoch_%d.pt' % (i, cur_epoch)
            torch.save({'state_dict': models[i].module.state_dict()}, os.path.join(saved_dir, saved_name))
            print('save checkpoint %s to disk...' % (saved_name))


def train(conf):
#     db = lmdb_utils.MultiLMDBDataset_noise(conf.source_lmdb_list, conf.source_file_list, conf.num_class, conf.noise_rate, conf.key) 
    db = lmdb_utils.MultiLMDBDataset(conf.source_lmdb_list, conf.source_file_list, conf.key)
    
    data_loader = DataLoader(db, conf.batch_size, True, num_workers=4)

    models=[]
    fc_ms = []
    optimizers = []
    for i in range(conf.M):
        model = MobileFaceNet(conf.feat_dim)
#         model = AttentionNet()
        model = torch.nn.DataParallel(model).cuda()
        fc_m = Parameter(torch.Tensor(conf.feat_dim, conf.num_class).cuda())
        fc_m.requires_grad = True
        fc_m.data.uniform_(-1-(i/100), 1+(i/100)).renorm_(2, 1, 1e-5).mul_(1e5)
        parameter = [p for p in model.parameters() if p.requires_grad]
        parameter.append(fc_m)
        optimizer = optim.SGD(parameter, lr=conf.lr, momentum=conf.momentum, weight_decay=5e-4, nesterov=True)
        models.append(model)
        fc_ms.append(fc_m)
        optimizers.append(optimizer)

    loss_fn = torch.nn.MSELoss()

    lr_schedules = []
    for i in range(conf.M):
        lr_schedule = optim.lr_scheduler.MultiStepLR(optimizers[i], milestones=conf.milestones, gamma=0.1)
        lr_schedules.append(lr_schedule)
        
    conf.real_sample = 0
    for epoch in range(conf.epochs):
#         r_e = min(((epoch+1)/10)*conf.noise_rate, conf.noise_rate)
#         conf.r0 = (1-r_e)
        print("r0:", conf.r0)
        train_one_epoch(conf, data_loader, conf.t[epoch], loss_fn, models, fc_ms, optimizers, conf.rate_schedule[epoch], epoch, conf.saved_dir, conf.print_freq)
        for i in range(conf.M):
            lr_schedules[i].step()
    db.close()
    
    
def train_on_neuhub(argv):
    conf = argparse.ArgumentParser(description='train arcface on msceleb provided by deepglint.')
    conf.add_argument('--key', type=int, default=134, help="you must send a key into the script before training.")
    conf.add_argument("--input_db_dir", type=str, default='/export/home/liuyuchi3/data/', help="input database name")
#     conf.add_argument("--train_db_name", type=str, default='CASIA-Clean-R_part34_lmdb_enc', help="comma separated list of training database.")
#     conf.add_argument("--train_db_name", type=str, default='vggface2_part34_lmdb', help="comma separated list of training database.")
    conf.add_argument("--train_db_name", type=str, default='MC-R_part34_lmdb_enc', help="comma separated list of training database.")
    conf.add_argument("--train_id", type=str, default='0', help="id for current train")
    # conf.add_argument("--output_model_dir", type=str, default='snapshot/CASIA_M4_alpha1_shuffle', help="Comma separated list of lmdb paths")    
   
    conf.add_argument("--is_am", type=int, default=0, help="use AM if 1, otherwise arcface.")
    conf.add_argument('--margin', type=float, default=0.5, help='only take effect when trained on svfc.')
    conf.add_argument('--mask', type=float, default=1.1, help='only take effect when trained on svfc')
    conf.add_argument('--num_class', type=int, default=85173, help='number of classes') # CASIA clean 9869 vggface2 9131 msceleb 85173
    conf.add_argument('--batch_size', type=int, default=512, help='batch size over all gpus.')
    

    conf.add_argument('--easy_margin', type=int, default=1, help="1 if use easy margin.")
    conf.add_argument('--step', type=str, default='2,3,4',
                      help='similar to step specified in caffe solver, but in epoch mechanism')
    conf.add_argument('--loss_type', type=str, default='svfc', help="loss type, can only be arcfc or svfc")
    conf.add_argument('--gpu', type=str, default='', help='useless value, only for neuhub.')
    conf.add_argument('--feat_dim', type=int, default=512, help='feature dimension.')
    conf.add_argument('--epochs', type=int, default=5, help='how many epochs you want to train.')
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
    
    forget_rate = 0.2
    args.rate_schedule = np.ones(args.epochs) * forget_rate
    args.t = np.ones(args.epochs) * (args.mask)
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

