import sys
# sys.path.append('/home/wangjun492/wj_armory')
from mobilefacenet import MobileFaceNet
from attention_network_def import AttentionNet
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import os
from prettytable import PrettyTable
import copy
import re
import logging as logger
import torch.nn.functional as F
# from model import MobileFaceNet

logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

import scipy.io as sio

blufr_file = '/home/liuyuchi3/data/lfw_evaluate/blufr_lfw_config_gjz.mat'
blufr_info = sio.loadmat(blufr_file)


def find_score(far, vr, target=1e-4):
    # far is an array in descending order, find the index of far whose element is closest to target,
    # and return vr[index]
    l = 0
    u = far.size - 1
    e = -1
    while u - l > 1:
        mid = (l + u) // 2
        # print far[mid]
        if far[mid] == target:
            if target != 0:
                return vr[mid]
            else:
                e = mid
                break
        elif far[mid] < target:
            u = mid
        else:
            l = mid
    if target == 0:
        i = e
        while i >= 0:
            if far[i] != 0:
                break
            i -= 1
        if i >= 0:
            return vr[i + 1]
        else:
            return vr[u]
    # Actually, either array[l] or both[u] is not equal to target, so choose a closer one.

    if target != 0 and far[l] / target >= 8:  # cannot find points that's close enough to target.
        return 0.0
    nearest_point = (target - far[l]) / (far[u] - far[l]) * (vr[u] - vr[l]) + vr[l]
    return nearest_point
    # if far[l] - target > target - far[u]:
    #     return vr[u]
    # else:
    #     return vr[l]


def compute_roc(score, label, num_thresholds=1000, show_sample_hist=False):
    pos_dist = score[label == 1]
    neg_dist = score[label == 0]

    num_pos_samples = pos_dist.size
    num_neg_samples = neg_dist.size
    data_max = np.max(score)
    data_min = np.min(score)
    unit = (data_max - data_min) * 1.0 / num_thresholds
    threshold = data_min + (data_max - data_min) * np.array(range(1, num_thresholds + 1)) / num_thresholds
    new_interval = threshold - unit / 2.0 + 2e-6
    new_interval = np.append(new_interval, np.array(new_interval[-1] + unit))
    P = np.triu(np.ones(num_thresholds))

    pos_hist, dummy = np.histogram(pos_dist, new_interval)
    neg_hist, dummy2 = np.histogram(neg_dist, new_interval)
    pos_mat = pos_hist[:, np.newaxis]
    neg_mat = neg_hist[:, np.newaxis]

    assert pos_hist.size == neg_hist.size == num_thresholds
    far = np.dot(P, neg_mat) / num_neg_samples
    far = np.squeeze(far)
    vr = np.dot(P, pos_mat) / num_pos_samples
    vr = np.squeeze(vr)
    if show_sample_hist is False:
        return far, vr, threshold
    else:
        return far, vr, threshold, pos_hist, neg_hist


def test_lfw(mask, score):
    acc_list = np.zeros(10, np.float32)
    for i in range(10):
        test_label = mask[i * 600: (i + 1) * 600]
        test_score = score[i * 600: (i + 1) * 600]
        if i == 0:
            train_label = mask[600:]
            train_score = score[600:]
        elif i == 9:
            train_label = mask[:5400]
            train_score = score[:5400]
        else:
            train_label_1 = mask[:i * 600]
            train_label_2 = mask[(i + 1) * 600:]
            train_label = np.hstack([train_label_1, train_label_2])
            train_score_1 = score[:i * 600]
            train_score_2 = score[(i + 1) * 600:]
            train_score = np.hstack([train_score_1, train_score_2])

        far, vr, threshold = compute_roc(train_score, train_label)
        train_accuracy = (vr + 1 - far) / 2.0
        tr = threshold[np.argmax(train_accuracy)]
        num_right_samples = 0
        for j in range(600):
            if test_score[j] >= tr and test_label[j] == 1:
                num_right_samples += 1
            elif test_score[j] < tr and test_label[j] == 0:
                num_right_samples += 1
        acc_list[i] = num_right_samples * 1.0 / 600
    mean = np.mean(acc_list)
    std = np.std(acc_list) / np.sqrt(10)
    return mean, std

def test_CFP(mask, score):
    acc_list = np.zeros(10, np.float32)
    for i in range(10):
        test_label = mask[i * 700: (i + 1) * 700]
        test_score = score[i * 700: (i + 1) * 700]
        if i == 0:
            train_label = mask[700:]
            train_score = score[700:]
        elif i == 9:
            train_label = mask[:5300]
            train_score = score[:5300]
        else:
            train_label_1 = mask[:i * 700]
            train_label_2 = mask[(i + 1) * 700:]
            train_label = np.hstack([train_label_1, train_label_2])
            train_score_1 = score[:i * 700]
            train_score_2 = score[(i + 1) * 700:]
            train_score = np.hstack([train_score_1, train_score_2])

        far, vr, threshold = compute_roc(train_score, train_label)
        train_accuracy = (vr + 1 - far) / 2.0
        tr = threshold[np.argmax(train_accuracy)]
        num_right_samples = 0
        for j in range(700):
            if test_score[j] >= tr and test_label[j] == 1:
                num_right_samples += 1
            elif test_score[j] < tr and test_label[j] == 0:
                num_right_samples += 1
        acc_list[i] = num_right_samples * 1.0 / 700
    mean = np.mean(acc_list)
    std = np.std(acc_list) / np.sqrt(10)
    return mean, std

def test_blufr(descriptors):
    test_index_tuple = blufr_info['testIndex']
    labels = blufr_info['labels']
    num_trials = 10
    # descriptors = np.zeros((13233, feat_dim), dtype=np.float32)
    vr = np.zeros([num_trials, 3], dtype=np.float32)
    for i in range(num_trials):
        test_index = test_index_tuple[i, 0].squeeze() - 1
        test_label = labels[test_index, 0].squeeze()
        x = descriptors[test_index, :]
        score = np.dot(x, x.T)
        # logger.info("running blufr %s fold..." % (i + 1))
        num = score.shape[0]
        label_this_fold = np.zeros((num, num), dtype=np.bool)
        for j in range(num):
            label_this_fold[j] = test_label[j] == test_label
        far_array, vr_array, threshold = compute_roc(score.flat, label_this_fold.flat, 5000)
        vr[i, 0] = find_score(far_array, vr_array, 1e-3)
        vr[i, 1] = find_score(far_array, vr_array, 1e-4)
        vr[i, 2] = find_score(far_array, vr_array, 1e-5)
        # vr[i, 3] = find_score(far_array, vr_array, 1e-6)
    mean_vr = vr.mean(axis=0)
    return mean_vr

def test_blufr_multiple(descriptors):
    test_index_tuple = blufr_info['testIndex']
    labels = blufr_info['labels']
    num_trials = 10
    # descriptors = np.zeros((13233, feat_dim), dtype=np.float32)
    vr = np.zeros([num_trials, 4], dtype=np.float32)
    for i in range(num_trials):
        test_index = test_index_tuple[i, 0].squeeze() - 1
        test_label = labels[test_index, 0].squeeze()
        score = 0
        for m in range(len(descriptors)):
            x = descriptors[m][test_index, :]
            score += np.dot(x, x.T)
        score = score/len(descriptors)
        # logger.info("running blufr %s fold..." % (i + 1))
        num = score.shape[0]
        label_this_fold = np.zeros((num, num), dtype=np.bool)
        for j in range(num):
            label_this_fold[j] = test_label[j] == test_label
        far_array, vr_array, threshold = compute_roc(score.flat, label_this_fold.flat, 5000)
        vr[i, 0] = find_score(far_array, vr_array, 1e-3)
        vr[i, 1] = find_score(far_array, vr_array, 1e-4)
        vr[i, 2] = find_score(far_array, vr_array, 1e-5)
        vr[i, 3] = find_score(far_array, vr_array, 1e-6)
        
    mean_vr = vr.mean(axis=0)
    return mean_vr



def test_all_pairs(filename2feat, filename2label):
    num_images = 0
    feat_dim = 0
    for k, v in filename2feat.items():
        num_images += 1
        if feat_dim == 0:
            feat_dim = v.size
        else:
            assert feat_dim == v.size
    feat = np.zeros((num_images, feat_dim), dtype=np.float32)
    i = 0
    labels = []
    for k, v in filename2feat.items():
        feat[i] = v
        i += 1
        labels.append(filename2label[k])
    scores = np.dot(feat, feat.T)
    sim = np.triu(scores, 1)
    sim = sim[sim != 0]
    num_pairs = sim.size
    assert num_pairs == num_images * (num_images - 1) // 2
    label_arr = np.zeros(num_pairs, dtype=np.uint8)
    k = 0
    for i in range(num_images):
        for j in range(i + 1, num_images):
            if labels[i] == labels[j]:
                label_arr[k] = 1
            else:
                label_arr[k] = 0
            k += 1
    assert k == num_pairs
    far_array, vr_array, threshold = compute_roc(sim, label_arr, 5000)
    vr1 = find_score(far_array, vr_array, 1e-6)
    vr2 = find_score(far_array, vr_array, 1e-7)
    vr3 = find_score(far_array, vr_array, 1e-8)
    vr4 = find_score(far_array, vr_array, 1e-9)
    return vr1, vr2, vr3, vr4

def load_image(filename, color=True, mean=127.5, std=128.0):
    """
    Load an image && convert it to gray-scale or BGR image as needed.

    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as ile False
        loads as intensity (if image is already gray-scale
    mean: pre-process, default is minus 127.5, divided by 128.0
    std: pre-process.

    Returns
    -------
    image : an image with type np.uint8 in range [0,255]
        of size (3 x H x W ) in BGR or
        of size (1 x H x W ) in gray-scale, if order == CHW
        else return H X W X 3 in BGR or H X W X 1 in gray-scale
    """
    order = 'CHW'
    assert order.upper() in ['CHW', 'HWC']
    if not os.path.exists(filename):
        raise Exception('%s does not exist.' % filename)

    flags = cv2.IMREAD_COLOR
    if color is False:
        flags = cv2.IMREAD_GRAYSCALE
    python_version = sys.version_info.major
    if python_version == 2:
        img = cv2.imread(filename, flags)
    elif python_version == 3:
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # img = cv2.resize(img, (120, 120))
    else:
        raise Exception('Unknown python version.')

    if img.ndim == 2:
        assert color is False
        img = img[:, :, np.newaxis]
    if order.upper() == 'CHW':
        img = (img.transpose((2, 0, 1)) - mean) / std
    else:
        img = (img - mean) / std
    return img.astype(np.float32)


class LFWDataset(Dataset):
    def __init__(self, lfw_filename, lfw_root_dir):
        self.lfw_file_list = []
        with open(lfw_filename, 'r') as fin:
            for line in fin:
                l = line.rstrip().lstrip()
                if len(l) > 0:
                    self.lfw_file_list.append(l)
        self.lfw_root_dir = lfw_root_dir

    def __len__(self):
        return len(self.lfw_file_list)

    def __getitem__(self, index):
        filename = self.lfw_file_list[index]
        image = load_image(os.path.join(self.lfw_root_dir, filename))
        image = torch.from_numpy(image)
        return image, filename


def test(model, lfw_test_pairs, test_data_loader, p=1, t=2, r=1, net_mode='irse', blufr=False, TEST_CFP = False):

    filename2feat = {}
    device = torch.device('cuda:0')  # % device_ids[0])
    with torch.no_grad():
        for batch_idx, (image, filenames) in enumerate(test_data_loader):
            image = image.to(device)
            feature = model(image).cpu().numpy()
            # feature = F.normalize(model(image)).cpu().numpy() # norm feature
            for fid, e in enumerate(filenames):
                # assert e not in filename2feat
                filename2feat[e] = copy.deepcopy(feature[fid])

    logger.info('running lfw evaluation procedure...')
    score_list = []
    label_list = []
    for pairs in lfw_test_pairs:
        feat1 = filename2feat[pairs[0]]
        feat2 = filename2feat[pairs[1]]
        dist = np.dot(feat1, feat2) / np.sqrt(np.dot(feat1, feat1) * np.dot(feat2, feat2))
        score_list.append(dist)
        label_list.append(pairs[2])
    score = np.array(score_list)
    label = np.array(label_list)
    if TEST_CFP:
        lfw_acc, lfw_std = test_CFP(label, score)
    else:
        lfw_acc, lfw_std = test_lfw(label, score)
    # vr1, vr2, vr3, vr4 = test_all_pairs(filename2feat, filename2label)
    # mean_vr = [vr1, vr2, vr3, vr4]
    # mean_vr = [0, 0, 0, 0]

    # --------------test blufr protocal-----------

    if blufr:
        image_dim = 512
        
        blufr_image_list = blufr_info['imageList']
        assert blufr_image_list.shape[0] == 13233
        num_images_lfw = blufr_image_list.shape[0]
        descriptor = np.zeros((num_images_lfw, image_dim), dtype=np.float32)
        for i in range(num_images_lfw):
            k = blufr_image_list[i, 0][0]
            descriptor[i] = filename2feat[k]
        mean_vr = test_blufr(descriptor)

        return lfw_acc, lfw_std, mean_vr[0], mean_vr[1], mean_vr[2]
    else:
        return lfw_acc, lfw_std

def test_multiple(models, lfw_test_pairs, test_data_loader, p=1, t=2, r=1, net_mode='irse', blufr=False, TEST_CFP = False):

    filename2feats = []
    for m in range(len(models)):
        filename2feats.append({})
    device = torch.device('cuda:0')  # % device_ids[0])
    with torch.no_grad():
        for batch_idx, (image, filenames) in enumerate(test_data_loader):
            image = image.to(device)
            for m in range(len(models)):
                feature = models[m](image).cpu().numpy()
                # feature = F.normalize(model(image)).cpu().numpy() # norm feature
                for fid, e in enumerate(filenames):
                    # assert e not in filename2feat
                    filename2feats[m][e] = copy.deepcopy(feature[fid])

    logger.info('running lfw evaluation procedure...')
    score_list = []
    label_list = []
    for pairs in lfw_test_pairs:
        dist = 0
        for m in range(len(models)):
            feat1 = filename2feats[m][pairs[0]]
            feat2 = filename2feats[m][pairs[1]]
            dist += np.dot(feat1, feat2) / np.sqrt(np.dot(feat1, feat1) * np.dot(feat2, feat2))
        dist = dist/len(models)
        score_list.append(dist)
        label_list.append(pairs[2])
    score = np.array(score_list)
    label = np.array(label_list)
    if TEST_CFP:
        lfw_acc, lfw_std = test_CFP(label, score)
    else:
        lfw_acc, lfw_std = test_lfw(label, score)
    # vr1, vr2, vr3, vr4 = test_all_pairs(filename2feat, filename2label)
    # mean_vr = [vr1, vr2, vr3, vr4]
    # mean_vr = [0, 0, 0, 0]

    # --------------test blufr protocal-----------

    if blufr:
        image_dim = 512
        
        blufr_image_list = blufr_info['imageList']
        assert blufr_image_list.shape[0] == 13233
        num_images_lfw = blufr_image_list.shape[0]
        descriptors = []
        descriptor = np.zeros((num_images_lfw, image_dim), dtype=np.float32)
        for m in range(len(models)):
            for i in range(num_images_lfw):
                k = blufr_image_list[i, 0][0]
                descriptor[i] = filename2feats[m][k]
            descriptors.append(descriptor)
        mean_vr = test_blufr_multiple(descriptors)

        return lfw_acc, lfw_std, mean_vr[0], mean_vr[1], mean_vr[2], mean_vr[3]
    else:
        return lfw_acc, lfw_std



def prepare_test(lfw_filename, lfw_root_dir, lfw_pairs_file, lfw_test_pairs):
    pat = re.compile(r'(\S+)\s+(\S+)\s+(\S+)')
    with open(lfw_pairs_file, 'r') as infile:
        for line in infile:
            l = line.rstrip()
            l = l.lstrip()
            if len(l) > 0:
                obj = pat.search(l)
                if obj:
                    file1 = obj.group(1)
                    file2 = obj.group(2)
                    label = int(obj.group(3))
                    lfw_test_pairs.append([file1, file2, label])
                else:
                    raise Exception('Cannot parse line %s, expected format: file1 file2 image_label' % l)
    test_data_loader = DataLoader(LFWDataset(lfw_filename, lfw_root_dir), batch_size=512, num_workers=4)
    
    return lfw_test_pairs, test_data_loader

if __name__ == '__main__':


    """
    filename2label = {}
    with open(lfw_all_pairs_file, 'r') as fin:
        for line in fin:
            l = line.rstrip().lstrip()
            if len(l) > 0:
                filename, label = l.split(' ')
                assert filename not in filename2label
                filename2label[filename] = int(label)
    """
    

    model_suffix = '.pt'
    trained_model_dir = './snapshot/1024_1/best'
    # trained_model_dir = '../372/'
    files = os.listdir(trained_model_dir)
    trained_models_name = []
    for filename in files:
        if filename.endswith(model_suffix):
            trained_models_name.append(filename)

    trained_models = []
    for model_name in trained_models_name:
        model_path = os.path.join(trained_model_dir, model_name)
        model = MobileFaceNet(512)
        # model = AttentionNet(3, 1, 2, 1, 'irse', (2, 4, 2))
        # model.load_state_dict(torch.load(model_path), strict=True)
        model.load_state_dict(torch.load(model_path)['state_dict'], strict=True)
        # model.cuda()
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        trained_models.append(model)

    DATA_ROOT = '/home/liuyuchi3/data/'

    # ------------------- test LFW ---------------------------

    lfw_filename = DATA_ROOT + '781/lfw_image_list.txt'
    lfw_root_dir = DATA_ROOT + '781/lfw_cropped_part34/lfw_cropped_part34/'
    lfw_pairs_file = DATA_ROOT + '781/lfw_test_pairs.txt'
    lfw_test_pairs = []

    lfw_test_pairs, test_data_loader = prepare_test(lfw_filename, lfw_root_dir, lfw_pairs_file, lfw_test_pairs)

    field_name_list = ['model name', 'mean accuracy', 'std', '1e-3', '1e-4', '1e-5', '1e-6']
    table_disk = PrettyTable(field_name_list)
    logger.info('Start evaluating the model on LFW...')
    final_result = []

    # for i in range(len(trained_models_name)):
    #     test_all_pairs
    #     res = test(trained_models[i], lfw_test_pairs, test_data_loader, blufr=True)
    #     final_result.append([trained_models_name[i]] + list(res)) 

    res = test_multiple(trained_models, lfw_test_pairs, test_data_loader, blufr=True)
    final_result.append([trained_models_name[0]] + list(res)) 

    final_result.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    for r in final_result:
        table_disk.add_row(r)
    logger.info('Evaluation done!')
    print(table_disk)


    # # ------------------- test CALFW ---------------------------

    CALFW_filename= DATA_ROOT + 'calfw/calfw_image_list.txt'
    CALFW_root_dir = DATA_ROOT+'calfw/aligned_images_part34/'
    CALFW_pairs_file = DATA_ROOT+'calfw/calfw_test_pairs.txt'
    CALFW_test_pairs = []

    lfw_test_pairs, test_data_loader = prepare_test(CALFW_filename, CALFW_root_dir, CALFW_pairs_file, CALFW_test_pairs)

    field_name_list = ['model name', 'mean accuracy', 'std']
    table_disk = PrettyTable(field_name_list)
    logger.info('Start evaluating the model on CALFW...')
    final_result = []
    # for i in range(len(trained_models_name)):
    #     test_all_pairs
    #     res = test(trained_models[i], lfw_test_pairs, test_data_loader)
    #     final_result.append([trained_models_name[i]] + list(res)) 

    res = test_multiple(trained_models, lfw_test_pairs, test_data_loader)
    final_result.append([trained_models_name[0]] + list(res)) 

    final_result.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    for r in final_result:
        table_disk.add_row(r)
    logger.info('Evaluation done!')
    print(table_disk)


    # ------------------- test CPLFW ---------------------------

    CPLFW_filename= DATA_ROOT + 'cplfw/cplfw_image_list.txt'
    CPLFW_root_dir = DATA_ROOT+'cplfw/aligned_images_part34/'
    CPLFW_pairs_file = DATA_ROOT+'cplfw/cplfw_test_pairs.txt'
    CPLFW_test_pairs = []

    lfw_test_pairs, test_data_loader = prepare_test(CPLFW_filename, CPLFW_root_dir, CPLFW_pairs_file, CPLFW_test_pairs)

    field_name_list = ['model name', 'mean accuracy', 'std']
    table_disk = PrettyTable(field_name_list)
    logger.info('Start evaluating the model on CPLFW...')
    final_result = []
    # for i in range(len(trained_models_name)):
    #     test_all_pairs
    #     res = test(trained_models[i], lfw_test_pairs, test_data_loader)
    #     final_result.append([trained_models_name[i]] + list(res)) 

    res = test_multiple(trained_models, lfw_test_pairs, test_data_loader)
    final_result.append([trained_models_name[0]] + list(res)) 

    final_result.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    for r in final_result:
        table_disk.add_row(r)
    logger.info('Evaluation done!')
    print(table_disk)


    # ------------------- test AgeDB ---------------------------

    AgeDB_filename= DATA_ROOT + 'agedb/agedb30_imgs_list.txt'
    AgeDB_root_dir = DATA_ROOT+'agedb/AgeDB_part34/'
    AgeDB_pairs_file = DATA_ROOT+'agedb/agedb30_pairs_list.txt'
    AgeDB_test_pairs = []

    lfw_test_pairs, test_data_loader = prepare_test(AgeDB_filename, AgeDB_root_dir, AgeDB_pairs_file, AgeDB_test_pairs)

    field_name_list = ['model name', 'mean accuracy', 'std']
    table_disk = PrettyTable(field_name_list)
    logger.info('Start evaluating the model on AgeDB...')
    final_result = []
    # for i in range(len(trained_models_name)):
    #     test_all_pairs
    #     res = test(trained_models[i], lfw_test_pairs, test_data_loader)
    #     final_result.append([trained_models_name[i]] + list(res)) 

    res = test_multiple(trained_models, lfw_test_pairs, test_data_loader)
    final_result.append([trained_models_name[0]] + list(res)) 

    final_result.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    for r in final_result:
        table_disk.add_row(r)
    logger.info('Evaluation done!')
    print(table_disk)
    

    # ------------------- test CFP ---------------------------

    CFP_filename= DATA_ROOT + 'cfp_fp/cfp_img_list.txt'
    CFP_root_dir = DATA_ROOT+'cfp_fp/cfp_part_34/'
    CFP_pairs_file = DATA_ROOT+'cfp_fp/cfp_pairs_list.txt'
    CFP_test_pairs = []

    lfw_test_pairs, test_data_loader = prepare_test(CFP_filename, CFP_root_dir, CFP_pairs_file, CFP_test_pairs)

    field_name_list = ['model name', 'mean accuracy', 'std']
    table_disk = PrettyTable(field_name_list)
    logger.info('Start evaluating the model on CFP...')
    final_result = []
    # for i in range(len(trained_models_name)):
    #     test_all_pairs
    #     res = test(trained_models[i], lfw_test_pairs, test_data_loader)
    #     final_result.append([trained_models_name[i]] + list(res)) 

    res = test_multiple(trained_models, lfw_test_pairs, test_data_loader, TEST_CFP=True)
    final_result.append([trained_models_name[0]] + list(res))   

    final_result.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    for r in final_result:
        table_disk.add_row(r)
    logger.info('Evaluation done!')
    print(table_disk)
  

    # ------------------- test rfw caucasian ---------------------------

    CFP_filename= DATA_ROOT + 'rfw/txts/Caucasian/Caucasian_new_images.txt'
    CFP_root_dir = DATA_ROOT+'rfw/images/Caucasian/'
    CFP_pairs_file = DATA_ROOT+'rfw/txts/Caucasian/Caucasian_new_pairs.txt'
    CFP_test_pairs = []

    lfw_test_pairs, test_data_loader = prepare_test(CFP_filename, CFP_root_dir, CFP_pairs_file, CFP_test_pairs)

    field_name_list = ['model name', 'mean accuracy', 'std']
    table_disk = PrettyTable(field_name_list)
    logger.info('Start evaluating the model on rfw Caucasian...')
    final_result = []
    # for i in range(len(trained_models_name)):
    #     test_all_pairs
    #     res = test(trained_models[i], lfw_test_pairs, test_data_loader)
    #     final_result.append([trained_models_name[i]] + list(res)) 

    res = test_multiple(trained_models, lfw_test_pairs, test_data_loader)
    final_result.append([trained_models_name[0]] + list(res))   

    final_result.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    for r in final_result:
        table_disk.add_row(r)
    logger.info('Evaluation done!')
    print(table_disk)


    # ------------------- test rfw Indian ---------------------------

    CFP_filename= DATA_ROOT + 'rfw/txts/Indian/Indian_new_images.txt'
    CFP_root_dir = DATA_ROOT+'rfw/images/Indian/'
    CFP_pairs_file = DATA_ROOT+'rfw/txts/Indian/Indian_new_pairs.txt'
    CFP_test_pairs = []

    lfw_test_pairs, test_data_loader = prepare_test(CFP_filename, CFP_root_dir, CFP_pairs_file, CFP_test_pairs)

    field_name_list = ['model name', 'mean accuracy', 'std']
    table_disk = PrettyTable(field_name_list)
    logger.info('Start evaluating the model on rfw Indian...')
    final_result = []
    # for i in range(len(trained_models_name)):
    #     test_all_pairs
    #     res = test(trained_models[i], lfw_test_pairs, test_data_loader)
    #     final_result.append([trained_models_name[i]] + list(res)) 

    res = test_multiple(trained_models, lfw_test_pairs, test_data_loader)
    final_result.append([trained_models_name[0]] + list(res))   

    final_result.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    for r in final_result:
        table_disk.add_row(r)
    logger.info('Evaluation done!')
    print(table_disk)

    # ------------------- test rfw Asian ---------------------------

    CFP_filename= DATA_ROOT + 'rfw/txts/Asian/Asian_new_images.txt'
    CFP_root_dir = DATA_ROOT+'rfw/images/Asian/'
    CFP_pairs_file = DATA_ROOT+'rfw/txts/Asian/Asian_new_pairs.txt'
    CFP_test_pairs = []

    lfw_test_pairs, test_data_loader = prepare_test(CFP_filename, CFP_root_dir, CFP_pairs_file, CFP_test_pairs)

    field_name_list = ['model name', 'mean accuracy', 'std']
    table_disk = PrettyTable(field_name_list)
    logger.info('Start evaluating the model on rfw Asian...')
    final_result = []
    # for i in range(len(trained_models_name)):
    #     test_all_pairs
    #     res = test(trained_models[i], lfw_test_pairs, test_data_loader)
    #     final_result.append([trained_models_name[i]] + list(res)) 

    res = test_multiple(trained_models, lfw_test_pairs, test_data_loader)
    final_result.append([trained_models_name[0]] + list(res))   

    final_result.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    for r in final_result:
        table_disk.add_row(r)
    logger.info('Evaluation done!')
    print(table_disk)

    # ------------------- test rfw African ---------------------------

    CFP_filename= DATA_ROOT + 'rfw/txts/African/African_new_images.txt'
    CFP_root_dir = DATA_ROOT+'rfw/images/African/'
    CFP_pairs_file = DATA_ROOT+'rfw/txts/African/African_new_pairs.txt'
    CFP_test_pairs = []

    lfw_test_pairs, test_data_loader = prepare_test(CFP_filename, CFP_root_dir, CFP_pairs_file, CFP_test_pairs)

    field_name_list = ['model name', 'mean accuracy', 'std']
    table_disk = PrettyTable(field_name_list)
    logger.info('Start evaluating the model on rfw African...')
    final_result = []
    # for i in range(len(trained_models_name)):
    #     test_all_pairs
    #     res = test(trained_models[i], lfw_test_pairs, test_data_loader)
    #     final_result.append([trained_models_name[i]] + list(res)) 

    res = test_multiple(trained_models, lfw_test_pairs, test_data_loader)
    final_result.append([trained_models_name[0]] + list(res))   

    final_result.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    for r in final_result:
        table_disk.add_row(r)
    logger.info('Evaluation done!')
    print(table_disk)
