import numpy as np
import torch
from torch.utils.data import DataLoader
from attention_network_def import AttentionNet
from torch.utils.data import Dataset
from lmdb_utils import DenoiseLMDBDataset
import lmdb
import random
from caffe_pb2 import Datum
import cv2
import argparse
import logging as logger
import matplotlib
import pickle
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')


def transform_image(image):
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
    image = torch.from_numpy(image.astype(np.float32))

    return image


def denoise(pretrained_model_path, denoise_loader, multiFace_loader, id_feature, image_feature, all_person, multiFaceKey_list, output_dir, p=1, t=2, r=1, net_mode='irse', key=None):
    ####
    # feature extraction, comment out this part if features are already extracted
    ####
    model = AttentionNet(3, p, t, r, net_mode, (2, 4, 2))
    model.load_state_dict(torch.load(pretrained_model_path)['state_dict'], strict=True)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    device = torch.device('cuda:0')  # % device_ids[0])

    # with torch.no_grad():
    #     for batch_idx, (image, lmdb_key) in enumerate(denoise_loader):            
    #         image = image.to(device)
    #         feature = model(image).cpu().numpy()
    #         feat_dim = feature.shape[1]
    #         batch_size = feature.shape[0]

    #         # sotre feature for persons
    #         for i in range(batch_size):
    #             person_id = lmdb_key[i].split('_')[-2]
    #             image_id = lmdb_key[i].split('_')[-1]
    #             if id_feature[person_id]['feat'] is None:
    #                 num_images = id_feature[person_id]['num_images']
    #                 id_feature[person_id]['idx2key']=[]
    #                 id_feature[person_id]['feat'] = np.zeros((num_images, feat_dim), dtype=np.float32)
    #             idx = id_feature[person_id]['fid']
    #             assert(idx==len(id_feature[person_id]['idx2key']))
    #             id_feature[person_id]['idx2key'].append(image_id)
    #             id_feature[person_id]['feat'][idx] = feature[i]
    #             id_feature[person_id]['fid'] += 1
    #         if batch_idx > 0 and batch_idx % 100 == 0:
    #             logger.info('extracting %d batch images...' % batch_idx)

    #     logger.info('Dump features to disk...')
    #     output_path = os.path.join(output_dir, 'person_features.pkl')
    #     output = open(output_path, 'wb')
    #     pickle.dump(id_feature, output)
    #     output.close()
        
    #     for batch_idx, (image, lmdb_key) in enumerate(multiFace_loader):
    #         image = image.to(device)
    #         feature = model(image).cpu().numpy()
    #         feat_dim = feature.shape[1] 
    #         batch_size = feature.shape[0]
            
    #         # store feature for multiface images
    #         for i in range(batch_size):
    #             multiFaceKey = lmdb_key[i]
    #             if multiFaceKey not in multiFaceKey_list:
    #                 print(multiFaceKey)
    #                 assert(0)
    #             image_feature[multiFaceKey] = feature[i]

    #         if batch_idx > 0 and batch_idx % 100 == 0:
    #             logger.info('Extract features for %d query faces...' % batch_idx)

    #     output_path = os.path.join(output_dir, 'image_features.pkl')
    #     output = open(output_path, 'wb')
    #     pickle.dump(image_feature, output)
    #     output.close()

    logger.info('Finish extracting features.')
    logger.info('Computing scores for each multiface...')

    logger.info('Dump finished!')

    ### 
    # detect noise
    ###
    with open (os.path.join(output_dir,'person_features.pkl'), 'rb') as f:
        id_feature= pickle.load(f)
    
    with open (os.path.join(output_dir,'image_features.pkl'), 'rb') as f:
        image_feature= pickle.load(f)

    logger.info('Start to compute similarities...')
    noisy_candidates = []
    for idx, multiFaceKey in enumerate(multiFaceKey_list):      
        person_id = multiFaceKey.split('_')[-2]
        person_feat = id_feature[person_id]['feat']
        multiFace_feat = image_feature[multiFaceKey]
        score = np.dot(multiFace_feat, person_feat.T)
        if len(score)<5:
            continue
        score = score.tolist()
        noisy_times = 0
        for i in range(len(score)):
            if score[i] < 0.2 and score[i]!=0:
                if i%100==0:
                    logger.info('face %s has similarity scores %f.' % (multiFaceKey, score[i]))
                # noisy_candidates.append([multiFaceKey, id_feature[person_id]['idx2key'][i], score[i]])
                noisy_times += 1
        if noisy_times>2:
            noisy_candidates.append(multiFaceKey)

    noise_list = os.path.join(output_dir, 'deepglint_part34_noise_list.txt') 
    with open(noise_list, 'w') as f:
        for candidate in noisy_candidates:
            # f.write(candidate[0]+" " + candidate[1] + " " + str(candidate[2])+"\n")
            f.write(candidate +"\n")
    
    logger.info('Noise detection complete!')   

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Caffe2: contrastive training")
    parser.add_argument("--model_dir", type=str, default='/export/home/liuyc/372/', help='where is the model in')
    parser.add_argument("--model_file", type=str, default='arcface_epoch_8_batch_15319.pt', help='the model in model_dir and ends with .pt')
    parser.add_argument("--db_dir", type=str, default='/export/home/liuyc/data/', help='where is the dataset in')
    parser.add_argument("--db_name", type=str, default='deepglintmsra_part_34', help='the name of dataset')
    parser.add_argument("--filelist_folder", type=str, default='/export/home/liuyc/data/deepglintmsra_part_34', help='where train_file in')
    parser.add_argument("--filelist_file", type=str, default='only_key_filelist.txt', help='column 1 is key word, colum 2 is label(person id)')
    parser.add_argument("--multiFaceKey_file", type=str, default='only_key_filelist.txt', help='the keys for iamges with multiple faces')
    parser.add_argument("--output_dir", type=str, default='./deeglint_part32_out', help='the dir to save feat')
    parser.add_argument("--gpu", type=str, default='', help="useless argument, only for neuhub")

    args = parser.parse_args()
    
    model_file = os.path.join(args.model_dir,args.model_file)
    if not os.path.exists(model_file):
        raise Exception("Cannot find %s." % model_file)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    filelist_file = os.path.join(args.filelist_folder, args.filelist_file)
    multiFaceKey_file = os.path.join(args.filelist_folder, args.multiFaceKey_file)
    db_file = os.path.join(args.db_dir, args.db_name)

    
    multiFaceKey_list=[]
    with open(multiFaceKey_file, 'r') as f:
        for line in f:
            line = line.rstrip().lstrip()
            if not len(line):
                continue
            multiFaceKey = line
            multiFaceKey_list.append(multiFaceKey)

    all_person={}
    with open(filelist_file, 'r') as f:
        for line in f:
            line = line.rstrip().lstrip()
            if not len(line):
                continue
            lmdb_key = line
            person_id= lmdb_key.split('_')[-2]
            if person_id not in all_person:
                all_person[person_id]=[lmdb_key]
            else:
                all_person[person_id].append(lmdb_key)
    
    image_feature={}
    with open(multiFaceKey_file, 'r') as f:
        for line in f:
            line = line.rstrip().lstrip()
            if not len(line):
                continue
            multiFaceKey = line
            image_feature[multiFaceKey]=None
            


    id_feature={}
    with open (filelist_file, 'r') as f:
        for line in f:
            line = line.rstrip().lstrip()
            lmdb_key = line
            if not len(line):
                continue
            person_id = lmdb_key.split('_')[-2]
            if person_id in all_person:
                if person_id not in id_feature:
                    id_feature[person_id] = {'fid':0, 'num_images': len(all_person[person_id]), 'feat': None}
            else:
                raise Exception("Cannot find %s in file list." % person_id)

    denoise_db = DenoiseLMDBDataset(db_file, filelist_file)
    denoise_loader = DataLoader(denoise_db, batch_size=1024, num_workers=4)

    multiFace_db = DenoiseLMDBDataset(db_file, multiFaceKey_file)
    multiFace_loader = DataLoader(multiFace_db, batch_size=1024, num_workers=4)

    db_path = os.path.join(args.db_dir, args.db_name)
    logger.info('Start denoising...')
    denoise(model_file, denoise_loader, multiFace_loader, id_feature, image_feature, all_person, multiFaceKey_list, args.output_dir)
    logger.info('Finish denoising...')