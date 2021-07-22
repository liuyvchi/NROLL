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
# from mobilefacenet import MobileFaceNet
from model import MobileFaceNet
from metric import mix_fc
from tensorboardX import SummaryWriter
import shutil
from test_lfw import *

# ------------------- test LFW ---------------------------
DATA_ROOT = '/home/liuyuchi3/data/'

lfw_filename = DATA_ROOT + '781/lfw_image_list.txt'
lfw_root_dir = DATA_ROOT + '781/lfw_cropped_part34/lfw_cropped_part34/'
lfw_pairs_file = DATA_ROOT + '781/lfw_test_pairs.txt'
lfw_test_pairs = []

lfw_test_pairs, test_data_loader = prepare_test(lfw_filename, lfw_root_dir, lfw_pairs_file, lfw_test_pairs)

# ------------------- test LFW ---------------------------

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(conf, data_loader, t, model1, fc_net1_w, optimizer1, rate_schedule, cur_epoch, saved_dir, print_freq=100):
    # switch to train mode
    device = torch.device('cuda:0')
    model1.train()
    # model2.train()
    db_size = len(data_loader)
    check_point_size = (db_size // 3, db_size // 2, db_size // 3 * 2)

    for batch_idx, (image, label) in enumerate(data_loader):
        cur_iteration = cur_epoch*(int(len(data_loader.dataset)/conf.batch_size))+batch_idx
        image = image.to(device)
        label = label.squeeze()
        label = label.to(device)
        feat1_norm = model1(image)
        
        pred1 = mix_fc(t, feat1_norm, fc_net1_w, label, fc_type=conf.loss_type, margin=0.5, easy_margin=True)
        
        loss1 = F.cross_entropy(pred1, label)

        conf.writer.add_scalar('loss/loss_m', loss1, cur_iteration)

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        if batch_idx % print_freq == 0:
            loss_val1 = loss1.item()
            lr = get_lr(optimizer1)
            logger.info('epoch %d, iter %d, lr %f, loss1 %f' % (cur_epoch, batch_idx, lr, loss_val1))

        if batch_idx != 0 and batch_idx in check_point_size:
            saved_name = 'model1_epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
            torch.save({'state_dict': model1.module.state_dict()}, os.path.join(saved_dir, saved_name))
            logger.info('save checkpoint %s to disk...' % (saved_name))
            logger.info('Start evaluating the model on LFW...')
            model1.eval()
            res = test(model1, lfw_test_pairs, test_data_loader, blufr=True)
            lfw, lfw3, lfw4, lfw5 = res[0], res[2], res[3], res[4]
            print("lfw:",lfw, lfw3, lfw4, lfw5)
            conf.writer.add_scalars('lfw', {'lfw%':lfw}, cur_iteration)
            conf.writer.add_scalars('lfw', {'lfw1e-3':lfw3}, cur_iteration)
            conf.writer.add_scalars('lfw', {'lfw1e-4':lfw4}, cur_iteration)
            conf.writer.add_scalars('lfw', {'lfw1e-5':lfw5}, cur_iteration)
                
    saved_name1 = 'model1_epoch_%d.pt' % cur_epoch
    torch.save({'state_dict': model1.module.state_dict()}, os.path.join(saved_dir, saved_name1))
    logger.info('save checkpoint %s to disk...' % (saved_name1))


def train(conf):
    if conf.noise_rate > 0:
        db = lmdb_utils.MultiLMDBDataset_noise(conf.source_lmdb_list, conf.source_file_list, conf.num_class, conf.noise_rate, conf.key) 
    else:
        db = lmdb_utils.MultiLMDBDataset(conf.source_lmdb_list, conf.source_file_list, conf.key)
      
    data_loader = DataLoader(db, conf.batch_size, True, num_workers=4)

    model1 = MobileFaceNet(conf.feat_dim)
    model1 = torch.nn.DataParallel(model1).cuda()
    fc_net1_w = Parameter(torch.Tensor(conf.feat_dim, conf.num_class).cuda())
    fc_net1_w.requires_grad = True
    fc_net1_w.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
    parameters1 = [p for p in model1.parameters() if p.requires_grad]
    parameters1.append(fc_net1_w)
    optimizer1 = optim.SGD(parameters1, lr=conf.lr, momentum=conf.momentum, weight_decay=5e-4, nesterov=True)

    lr_schedule1 = optim.lr_scheduler.MultiStepLR(optimizer1, milestones=conf.milestones, gamma=0.1)
    for epoch in range(conf.epochs):
        train_one_epoch(conf, data_loader, conf.t[epoch], model1, fc_net1_w, optimizer1, conf.rate_schedule[epoch], epoch, conf.saved_dir, conf.print_freq)
        lr_schedule1.step()
    db.close()
    
    
def train_on_neuhub(argv):
    conf = argparse.ArgumentParser(description='train arcface on msceleb provided by deepglint.')
    conf.add_argument('--key', type=int, default=134, help="you must send a key into the script before training.")
    conf.add_argument("--input_db_dir", type=str, default='/home/liuyuchi3/data/CASIA/crop/', help="input database name")
    conf.add_argument("--train_db_name", type=str, default='CASIA-Clean-R_part34_lmdb_enc', help="comma separated list of training database.")
    conf.add_argument("--train_id", type=str, default='0', help="Comma separated list of lmdb paths")    
   
    conf.add_argument("--is_am", type=int, default=0, help="use AM if 1, otherwise arcface.")
    conf.add_argument('--margin', type=float, default=0.5, help='only take effect when trained on svfc.')
    conf.add_argument('--mask', type=float, default=1.1, help='only take effect when trained on svfc')
    conf.add_argument('--num_class', type=int, default=9869, help='number of classes') # webface_clean_unoverlap
    conf.add_argument('--batch_size', type=int, default=128, help='batch size over all gpus.')
    

    conf.add_argument('--easy_margin', type=int, default=1, help="1 if use easy margin.")
    conf.add_argument('--step', type=str, default='6,12,17',
                      help='similar to step specified in caffe solver, but in epoch mechanism')
    conf.add_argument('--loss_type', type=str, default='svfc', help="loss type, can only be arcface or svfc")
    conf.add_argument('--gpu', type=str, default='', help='useless value, only for neuhub.')
    conf.add_argument('--feat_dim', type=int, default=512, help='feature dimension.')
    conf.add_argument('--epochs', type=int, default=20, help='how many epochs you want to train.')
    conf.add_argument('--print_freq', type=int, default=200, help='frequency of displaying current training state.')
    conf.add_argument('--in_channels', type=int, default=3, help='color image, it should be 3, gray scale, 1')
    conf.add_argument('--momentum', type=float, default=0.9, help='momentum')
    conf.add_argument('--p', type=int, default=1, help='control the depth of attention net.')
    conf.add_argument('--t', type=int, default=2, help='control the depth of attention net.')
    conf.add_argument('--r', type=int, default=1, help='control the depth of attention net.')
    conf.add_argument('--net_mode', type=str, default='irse', help='ir or irse')
    conf.add_argument('--noise_rate', type=float, default=0, help='symmetric noise rate')
    
    conf.add_argument('--lr', type=float, default=0.1, help='initial learning rate.')
    conf.add_argument('--exponent', type=float, default=1,
                       help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T).')
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
    args.t = np.ones(args.epochs) * (args.mask - 1.0)
    args.t[:args.epochs] = 1.0 + np.linspace(0, (args.mask - 1.0) ** args.exponent, args.epochs)

    args.milestones = [int(p) for p in args.step.split(',')]
    
    tmp_train_dbs = [p.lstrip().rstrip() for p in args.train_db_name.split(',')]
    train_dbs = set()
    for e in tmp_train_dbs:
        train_dbs.add(e)
    logger.info('%d database are(is) given to train this task.' % len(train_dbs))
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
    logger.info('Start optimization.')
    train(args)
    logger.info('Optimization done!')

if __name__ == '__main__':
    train_on_neuhub(sys.argv)

