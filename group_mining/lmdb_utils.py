# parse lmdb script, it's used for either classification or pairwise siamese task. It can be modified easily for
# inference.
# Author: Shuo Wang
# Version: 1.0.0
# encoding: utf-8
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import lmdb
from caffe_pb2 import Datum
import random

class MultiLMDBDataset(Dataset):
    def __init__(self, source_lmdbs, source_files, key=None, need_label_offset=True, transform=None):
        assert isinstance(source_files, list) or isinstance(source_files, tuple)
        assert isinstance(source_lmdbs, list) or isinstance(source_lmdbs, tuple)
        assert len(source_lmdbs) == len(source_files)
        assert len(source_files) > 0

        self.envs = []
        self.txns = []
        for lmdb_path in source_lmdbs:
            self.envs.append(lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False,
                                       readahead=False, meminit=False))
            self.txns.append(self.envs[-1].begin(write=False))

        self.train_list = []
        self.labels = []

        if need_label_offset:
            last_label = 0
            max_label = -1
            for db_id, lmdb_train_file in enumerate(source_files):
                with open(lmdb_train_file, 'r') as infile:
                    for line in infile:
                        l = line.rstrip().lstrip()
                        if len(l) > 0:
                            # lmdb_key, label = l.split(' ')
                            groups = l.split(' ')
                            lmdb_key = groups[0]
                            label = groups[1]
                            self.train_list.append([lmdb_key, int(label) + last_label, db_id])
                            self.labels.append(int(label) + last_label)
                            max_label = max(max_label, int(label) + last_label)
                max_label += 1
                last_label = max_label
                # print('last label = %d' % last_label)
        else:
            for db_id, lmdb_train_file in enumerate(source_files):
                with open(lmdb_train_file, 'r') as infile:
                    for line in infile:
                        l = line.rstrip().lstrip()
                        if len(l) > 0:
                            lmdb_key, label = l.split(' ')
                            self.train_list.append([lmdb_key, int(label), db_id])
                            self.labels.append(int(label))

        # if transform is None:
        self.transform = transform
        self.key = key

    def close(self):
        for i in range(len(self.txns)):
            self.txns[i].abort()
        for j in range(len(self.envs)):
            self.envs[j].close()

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        lmdb_key, label, db_id = self.train_list[index]
        datum = Datum()
        # label = torch.zeros(50 * 5)
        raw_byte_buffer = self.txns[db_id].get(lmdb_key.encode('utf-8'))
        if self.key is None:
            real_byte_buffer = raw_byte_buffer
        else:
            real_byte_buffer = bytes([e ^ self.key for e in raw_byte_buffer])
        datum.ParseFromString(real_byte_buffer)
        image = cv2.imdecode(np.fromstring(datum.data, dtype=np.uint8), -1)
        # image = cv2.resize(image, (144, 144))
        # if image.ndim == 2:
        #     image = image[:, :, np.newaxis]
        if self.transform is None:
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
            image = torch.from_numpy(image.astype(np.float32))
            pass
        else:
            image = self.transform(image)

        label_tensor = torch.zeros(1, dtype=torch.long)
        label_tensor[0] = label
        return image, label_tensor

class MultiLMDBDataset_noise(MultiLMDBDataset):
    def __init__(self, source_lmdbs, source_files, num_classes, noise_rate, key=None, need_label_offset=True, transform=None):
        super(MultiLMDBDataset_noise, self).__init__(source_lmdbs, source_files, key, need_label_offset, transform)

        self.num_classes = num_classes

        self.symmetric_noise(noise_rate)

    def symmetric_noise(self, noise_rate):
        """Insert symmetric noise.
        For all classes, ground truth labels are replaced with uniform random
        classes. use the way from co-teaching. 
        """
        np.random.seed(0)
        targets=np.zeros(len(self.train_list))
        for i in range(len(self.train_list)):
            targets[i]=self.train_list[i][1]
        mask = np.random.rand(len(targets)) <= noise_rate
        rnd_targets = np.random.choice(self.num_classes, mask.sum())
        targets[mask] = rnd_targets
        targets = [int(target) for target in targets]
        for i in range(len(self.train_list)):
            self.train_list[i][1]=targets[i]

class MultiLMDBData_semi(MultiLMDBDataset):
    def __init__(self, source_lmdbs, source_files, full_db, islabeled=True, indexs=None, key=None, need_label_offset=True, transform=None):
        super(MultiLMDBData_semi, self).__init__(source_lmdbs, source_files, key, need_label_offset, transform)

        self.selected_ind = indexs
        if islabeled:
            for idx in self.selected_ind:
                self.train_list[idx][1] = full_db.train_list[idx][1]


    def __len__(self):
        return len(self.selected_ind)

    def __getitem__(self, index):
        lmdb_key, label, db_id = self.train_list[self.selected_ind[index]]
        datum = Datum()
        # label = torch.zeros(50 * 5)
        raw_byte_buffer = self.txns[db_id].get(lmdb_key.encode('utf-8'))
        if self.key is None:
            real_byte_buffer = raw_byte_buffer
        else:
            real_byte_buffer = bytes([e ^ self.key for e in raw_byte_buffer])
        datum.ParseFromString(real_byte_buffer)
        image = cv2.imdecode(np.fromstring(datum.data, dtype=np.uint8), -1)
        # image = cv2.resize(image, (144, 144))
        # if image.ndim == 2:
        #     image = image[:, :, np.newaxis]
        if self.transform is None:
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
            image = torch.from_numpy(image.astype(np.float32))
            pass
        else:
            image = self.transform(image)

        label_tensor = torch.zeros(1, dtype=torch.long)
        label_tensor[0] = label
        return image, label_tensor, lmdb_key, self.selected_ind[index]


def semi_split(labels, num_classes, plit_offset):
    labels = np.array(labels)
    labeled_ind = []
    unlabeled_ind = []
    
    for i in range(num_classes):
        ind = np.where(labels == i)[0]
        random.Random(1).shuffle(ind)  #fixed split with the same random
        labeled_ind.extend(ind[:int(len(ind)*plit_offset)])
        unlabeled_ind.extend(ind[int(len(ind)*plit_offset):])

    return labeled_ind, unlabeled_ind


def semi_split_n(labels, num_classes, n):
    labels = np.array(labels)
    split_result = []
    for j in range(n):
        split_result.append([])
    
    for i in range(num_classes):
        if i%100==0:
            print("begin split class:", i)
        ind = np.where(labels == i)[0]
        num_unit = int(len(ind)/n)
        random.Random(1).shuffle(ind)  #fixed split with the same random
        for j in range(n):
            if j == n-1:
                split_result[j].extend(ind[j*num_unit:]) 
            else:
                split_result[j].extend(ind[j*num_unit:(j+1)*num_unit]) 

    ## assert #
    total_samples = 0
    for j in range(n):
        print(len(split_result[j]))
        total_samples += len(split_result[j])
    assert(total_samples == len(labels))

    return split_result