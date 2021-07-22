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
from PIL import Image

def transform(image):
    if random.random() > 0.5:
        x1_offset = np.random.randint(0, 8, size=1)[0]
        y1_offset = np.random.randint(0, 8, size=1)[0]
        x2_offset = np.random.randint(112, 120, size=1)[0]
        y2_offset = np.random.randint(112, 120, size=1)[0]
        image = image[x1_offset:x2_offset,y1_offset:y2_offset]
        image = cv2.resize(image,(120,120))  
    if random.random() > 0.9:
        image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if random.random() > 0.5:
        image = cv2.flip(image, 1) 
    if random.random() > 0.5:
        theta = (random.randint(-10,10)) * np.pi / 180
        M_rotate = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0]], dtype=np.float32)
        image = cv2.warpAffine(image, M_rotate, (120, 120))
    if image.ndim == 2:
        image = (image - 127.5) * 0.0078125
        new_image = np.zeros([3,120, 120], np.float32)
        new_image[0,:,:] = image
        # new_image[1,:,:] = 0
        # new_image[2,:,:] = 0
        image = torch.from_numpy(new_image.astype(np.float32))
    else:
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))
    return image

class SingleLMDBDataset_1ID2img(Dataset):
    def __init__(self, source_lmdb, source_filelists, label_num, transform=None, key=None):
        self.env = lmdb.open(source_lmdb, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        self.train_list = []
        with open(source_filelists, 'r') as infile:
            for line in infile:
                l = line.rstrip().lstrip()
                if len(l) > 0:
                    lmdb_key, label = l.split(' ')
                    self.train_list.append([lmdb_key, int(label)])
        self.transform = transform
        self.key = key
        
        self.label_num = label_num
        self.id2lmdbkeys = {}  
        self.gen_id2lmdbkeys()
    
    def gen_id2lmdbkeys(self):
        for index in range(len(self.train_list)):
            lmdb_key, label = self.train_list[index]
            if label not in self.id2lmdbkeys.keys():
                self.id2lmdbkeys[label] = []
            self.id2lmdbkeys[label].append(lmdb_key)
    
    def close(self):
        self.txn.abort()
        self.env.close()

    def __len__(self):
        return self.label_num

    def __getitem__(self, index):
        if index > self.label_num:
            raise IndexError("index exceeds the max-length of the dataset.")
        if len(self.id2lmdbkeys[index])==1:
            lmdb_key1 = self.id2lmdbkeys[index][0]
            lmdb_key2 = lmdb_key1
        else:    
            lmdb_keys= random.sample(self.id2lmdbkeys[index], 2)
            lmdb_key1, lmdb_key2 = lmdb_keys[0], lmdb_keys[1]
        label = index
        
        # process the first img
        datum1 = Datum()
        raw_byte_buffer1 = self.txn.get(lmdb_key1.encode('utf-8'))
        if self.key is None:
            real_byte_buffer1 = raw_byte_buffer1
        else:
            real_byte_buffer1 = bytes([e ^ self.key for e in raw_byte_buffer1])
        datum1.ParseFromString(real_byte_buffer1)
        image1 = cv2.imdecode(np.fromstring(datum1.data, dtype=np.uint8), -1)
        image1 = transform(image1)

        
        # process the sencond img
        datum2 = Datum()
        raw_byte_buffer2 = self.txn.get(lmdb_key2.encode('utf-8'))
        if self.key is None:
            real_byte_buffer2 = raw_byte_buffer2
        else:
            real_byte_buffer2 = bytes([e ^ self.key for e in raw_byte_buffer2])
        datum2.ParseFromString(real_byte_buffer2)
        image2 = cv2.imdecode(np.fromstring(datum2.data, dtype=np.uint8), -1)
        image2 = transform(image2)

        label_tensor = torch.zeros(1, dtype=torch.long)
        label_tensor[0] = label
        
        return image1, image2, label_tensor, lmdb_key1, lmdb_key2
    
class SingleLMDBDataset_ID2aug(Dataset):
    def __init__(self, source_lmdb, source_filelists, label_num, transform=None, key=None):
        self.env = lmdb.open(source_lmdb, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        self.train_list = []
        with open(source_filelists, 'r') as infile:
            for line in infile:
                l = line.rstrip().lstrip()
                if len(l) > 0:
                    lmdb_key, label = l.split(' ')
                    self.train_list.append([lmdb_key, int(label)])
        self.transform = transform
        self.key = key
        
        self.label_num = label_num
        self.id2lmdbkeys = {}  
        self.gen_id2lmdbkeys()
    
    def gen_id2lmdbkeys(self):
        for index in range(len(self.train_list)):
            lmdb_key, label = self.train_list[index]
            if label not in self.id2lmdbkeys.keys():
                self.id2lmdbkeys[label] = []
            self.id2lmdbkeys[label].append(lmdb_key)
    
    def close(self):
        self.txn.abort()
        self.env.close()

    def __len__(self):
        return self.label_num

    def __getitem__(self, index):
        if index > self.label_num:
            raise IndexError("index exceeds the max-length of the dataset.")
        lmdb_key = random.choice(self.id2lmdbkeys[index])
        label = index
        datum = Datum()
        raw_byte_buffer = self.txn.get(lmdb_key.encode('utf-8'))
        if self.key is None:
            real_byte_buffer = raw_byte_buffer
        else:
            real_byte_buffer = bytes([e ^ self.key for e in raw_byte_buffer])
        datum.ParseFromString(real_byte_buffer)
        image = cv2.imdecode(np.fromstring(datum.data, dtype=np.uint8), -1)
        # if image.ndim == 2:
        #     image = image[:, :, np.newaxis]
        image1 = transform(image)
        image2 = transform(image)
        
        label_tensor = torch.zeros(1, dtype=torch.long)
        label_tensor[0] = label
        
        return image1, image2, label_tensor, lmdb_key 

class SingleLMDBDataset_2aug(Dataset):
    def __init__(self, source_lmdb, source_filelists, transform=None, key=None):
        self.env = lmdb.open(source_lmdb, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        self.train_list = []
        with open(source_filelists, 'r') as infile:
            for line in infile:
                l = line.rstrip().lstrip()
                if len(l) > 0:
                    lmdb_key, label = l.split(' ')
                    self.train_list.append([lmdb_key, int(label)])
        self.transform = transform
        self.key = key
    
    def close(self):
        self.txn.abort()
        self.env.close()

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        lmdb_key, label = self.train_list[index]
        datum = Datum()
        raw_byte_buffer = self.txn.get(lmdb_key.encode('utf-8'))
        if self.key is None:
            real_byte_buffer = raw_byte_buffer
        else:
            real_byte_buffer = bytes([e ^ self.key for e in raw_byte_buffer])
        datum.ParseFromString(real_byte_buffer)
        image = cv2.imdecode(np.fromstring(datum.data, dtype=np.uint8), -1)
        # if image.ndim == 2:
        #     image = image[:, :, np.newaxis]
        image1 = transform(image)
        image2 = transform(image)

        label_tensor = torch.zeros(1, dtype=torch.long)
        label_tensor[0] = label
        
        return image1, image2, label_tensor, lmdb_key 

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
        # if image.ndim == 2:
        #     image = image[:, :, np.newaxis]
        if self.transform is None:
            image1 = image
            if random.random() > 0.5:
                image1 = cv2.flip(image1, 1)
            if image.ndim == 2:
                image1 = image1[:, :, np.newaxis]
            image1 = (image1.transpose((2, 0, 1)) - 127.5) * 0.0078125
            image1 = torch.from_numpy(image1.astype(np.float32))
            pass
        else:
            image1 = self.transform(image)

        if self.transform is None:
            image2 = image
            if random.random() > 0.5:
                image2 = cv2.flip(image2, 1)
            if image.ndim == 2:
                image2 = image2[:, :, np.newaxis]
            image2 = (image2.transpose((2, 0, 1)) - 127.5) * 0.0078125
            image2 = torch.from_numpy(image2.astype(np.float32))
            pass
        else:
            image2 = self.transform(image)

        label_tensor = torch.zeros(1, dtype=torch.long)
        label_tensor[0] = label
        return image1, image2, label_tensor, lmdb_key 
    

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