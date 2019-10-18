from __future__ import print_function
from PIL import Image
import os
from os import path
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import random

class depth_segmentation_dataset(data.Dataset):
    base_folder = 'depth-classification-dataset'

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 load_pickle=True, train_dir=None, label_dir=None, test_dir=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        if train:
            pickle_dir = path.join(train_dir, "pickle")
        else:
            pickle_dir = path.join(test_dir, "pickle")
        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            if load_pickle:
                for pckl in os.listdir(pickle_dir):
                    if pckl.endswith(".pickle"):
                        pckl_path = path.join(pickle_dir, pckl)
                        pckl_file = open(pckl_path,'rb')
                        entry = pickle.load(pckl_file,encoding='latin1')
                        self.train_data.append(entry['data'])
                        self.train_labels.append(entry['labels'])
                        pckl_file.close()
                self.train_data = np.concatenate(self.train_data)
                self.train_labels = np.concatenate(self.train_labels)
            else:
                for patch in os.listdir(train_dir):
                    if patch.endswith('.jpg'):
                        patch_path = path.join(train_dir, patch)
                        base = path.splitext(patch)[0]
                        label = base + '.png'
                        label_path = path.join(label_dir,label)
                        patch_file = Image.open(patch_path)
                        patch_arr = np.array(patch_file)
                        self.train_data.append(patch_arr)
                        label_file = Image.open(label_path)
                        label_arr = np.array(label_file)
                        #Make the classes 0-14 instead of 1-15
                        label_arr -= 1
                        self.train_labels.append(label_arr)
                self.train_data = np.expand_dims(self.train_data,axis=0)
                self.train_labels = np.expand_dims(self.train_labels,axis=0)
                self.train_data = np.squeeze(self.train_data, axis=0)
                self.train_labels = np.squeeze(self.train_labels, axis=0)
                entry = dict()
                entry['data'] = self.train_data
                entry['labels'] = self.train_labels
                pckl_file = path.join(pickle_dir, 'train_seg.pickle')
                with open(pckl_file, 'wb') as pckl:
                    pickle.dump(entry, pckl)
        else:
            self.test_data = []
            self.test_labels = []
            if load_pickle:
                for pckl in os.listdir(pickle_dir):
                    if pckl.endswith(".pickle"):
                        pckl_path = path.join(pickle_dir, pckl)
                        pckl_file = open(pckl_path, 'rb')
                        entry = pickle.load(pckl_file, encoding='latin1')
                        self.test_data.append(entry['data'])
                        self.test_labels.append(entry['labels'])
                        pckl_file.close()
                self.test_data = np.concatenate(self.test_data)
                self.test_labels = np.concatenate(self.test_labels)
            else:
                for patch in os.listdir(test_dir):
                    if patch.endswith('.jpg'):
                        patch_path = path.join(test_dir, patch)
                        base = path.splitext(patch)[0]
                        label = base + '.png'
                        label_path = path.join(label_dir, label)
                        patch_file = Image.open(patch_path)
                        patch_arr = np.array(patch_file)
                        self.test_data.append(patch_arr)
                        label_file = Image.open(label_path)
                        label_arr = np.array(label_file)
                        # Make the classes 0-14 instead of 1-15
                        label_arr -= 1
                        self.test_labels.append(label_arr)
                self.test_data = np.expand_dims(self.test_data, axis=0)
                self.test_labels = np.expand_dims(self.test_labels, axis=0)
                self.test_data = np.squeeze(self.test_data, axis=0)
                self.test_labels = np.squeeze(self.test_labels, axis=0)
                entry = dict()
                entry['data'] = self.test_data
                entry['labels'] = self.test_labels
                pckl_file = path.join(pickle_dir, 'test_seg.pickle')
                if not path.isdir(pickle_dir):
                    os.mkdir(pickle_dir)
                with open(pckl_file, 'wb') as pckl:
                    pickle.dump(entry, pckl)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def mv_30_percent_for_testing(train_dir=None, test_dir=None):
    file_list = os.listdir(train_dir)
    nof_files = len(file_list)
    files_for_test = round(0.3 * nof_files)
    files_for_test = np.random.choice(file_list, files_for_test,replace=False)
    if not path.isdir(test_dir):
        os.makedirs(test_dir)
    for file in files_for_test:
        src = path.join(train_dir, file)
        dst = path.join(test_dir, file)
        os.rename(src, dst)


if __name__ == '__main__':
    # mv_30_percent_for_testing(train_dir='/home/yotamg/data/jpg_images/patches', test_dir='/home/yotamg/data/jpg_images/test/patches')
    # cls1 = depth_segmentation_dataset('a', train=True,load_pickle=False,train_dir='/home/yotamg/data/jpg_images/patches', label_dir='/home/yotamg/data/depth_pngs/patches')
    # cls2 = depth_segmentation_dataset('a', train=False, load_pickle=False,test_dir='/home/yotamg/data/jpg_images/test/patches',label_dir='/home/yotamg/data/depth_pngs/patches')
    mv_30_percent_for_testing(train_dir='/home/yotamg/data/jpg_images/', test_dir='/home/yotamg/data/jpg_images/test/')

