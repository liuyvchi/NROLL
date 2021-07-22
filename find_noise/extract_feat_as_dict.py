import numpy as np
import torch
from torch.utils.data import DataLoader
from attention_network_def import AttentionNet
import lmdb_utils
import os
import argparse
import logging as logger
import matplotlib
import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')


def extract_features(pretrained_model_path, test_data_loader, label2files, output_dir, p=1, t=2, r=1, net_mode='irse'):
    model = AttentionNet(3, p, t, r, net_mode, (2, 4, 2))
    model.load_state_dict(torch.load(pretrained_model_path)['state_dict'], strict=True)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    device = torch.device('cuda:0')  # % device_ids[0])

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_data_loader):
            image = image.to(device)
            feature = model(image).cpu().numpy()
            feat_dim = feature.shape[1]
            label = label.squeeze()
            batch_size = feature.shape[0]
            for i in range(batch_size):
                this_label = label[i].item()
                if label2files[this_label]['feat'] is None:
                    num_images = label2files[this_label]['num_images']
                    label2files[this_label]['feat'] = np.zeros((num_images, feat_dim), dtype=np.float32)
                idx = label2files[this_label]['fid']
                label2files[this_label]['feat'][idx] = feature[i]
                label2files[this_label]['fid'] += 1
            if batch_idx > 0 and batch_idx % 50 == 0:
                logger.info('extracting %d batch images...' % batch_idx)
              
    logger.info('Finish extracting features.')
    logger.info('Computing scores within each figure...')
    
    logger.info('Dump features to disk...')
    output_path = os.path.join(output_dir, 'deepdlint_part34.pkl')
    output = open(output_path, 'wb')
    pickle.dump(label2files, output)
    output.close()
    logger.info('Dump finished!')
    
    logger.info('Start to compute similarities...')
    similarities = []
    for label in label2files:
        feat = label2files[label]['feat']
        #np.save('person_'+str(label)+'.feat', feat)
        sim = np.dot(feat, feat.T)
        score = np.triu(sim, 1)
        score = score[score != 0]
        similarities.extend(score.tolist())
    logger.info('There are %d similarity scores.' % len(similarities))
    plt.hist(similarities, bins=10000)
    plt.xlabel('cos')
    plt.ylabel('#of examples')
    plt.title('positive score distribution')
    plt.savefig('distribution_deepglint.jpg')
    logger.info('Finish compute similarities!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Caffe2: contrastive training")
    parser.add_argument("--model_dir", type=str, help='where is the model in')
    parser.add_argument("--model_file", type=str, help='the model in model_dir and ends with .pt')
    parser.add_argument("--db_dir", type=str, help='where is the dataset in')
    parser.add_argument("--db_name", type=str, help='the name of dataset')
    parser.add_argument("--train_file_folder", type=str, help='where train_file in')
    parser.add_argument("--train_file", type=str, help='column 1 is key word, colum 2 is label(person id)')
    parser.add_argument("--output_dir", type=str, help='the dir to save feat')
    parser.add_argument("--gpu", type=str, default='', help="useless argument, only for neuhub")
    args = parser.parse_args()
    
    model_file = os.path.join(args.model_dir,args.model_file)
    if not os.path.exists(model_file):
        raise Exception("Cannot find %s." % model_file)
    
    train_file = os.path.join(args.train_file_folder, args.train_file)
    label2files = {}
    with open(train_file, 'r') as fin:
        for line in fin:
            l = line.rstrip().lstrip()
            if len(l) > 0:
                lmdb_key, label = l.split(' ')
                label = int(label)
                if label not in label2files:
                    label2files[label] = {'fid': 0, 'num_images': 0, 'feat': None}
                label2files[label]['num_images'] += 1
     
    train_db = os.path.join(args.db_dir, args.db_name)
    db = lmdb_utils.SingleLMDBDataset(train_db, train_file, key=None)
    data_loader = DataLoader(db, batch_size=1024, num_workers=4)
    logger.info('Start extracting features...')
    extract_features(model_file, data_loader, label2files, args.output_dir)
    logger.info('Finish extracting features.')
