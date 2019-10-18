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


class depth_classification_dataset(data.Dataset):
    base_folder = 'depth-classification-dataset'

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 load_pickle=True, train_dir=None, test_dir=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            if load_pickle:
                for pckl in os.listdir(train_dir):
                    if pckl.endswith(".pickle"):
                        pckl_path = path.join(train_dir, pckl)
                        pckl_file = open(pckl_path,'rb')
                        entry = pickle.load(pckl_file,encoding='latin1')
                        self.train_data.append(entry['data'])
                        self.train_labels += entry['labels']
                        pckl_file.close()
                self.train_data = np.concatenate(self.train_data)
            else:
                for dir in os.listdir(train_dir):
                    label_dir = path.join(train_dir, dir)
                    if os.path.isdir(label_dir):
                        label = int(dir)-1
                        self.train_data = []
                        self.train_labels = []
                        for img_file in os.listdir(label_dir):
                            if img_file.endswith('.png'):
                                img = Image.open(path.join(label_dir,img_file))
                                img_arr = np.array(img)
                                self.train_data.append(img_arr)
                                self.train_labels.append(label)
                        self.train_data = np.expand_dims(self.train_data, axis=0)
                        self.train_data = np.squeeze(self.train_data, axis=0)
                        entry = dict(())
                        entry['data'] = self.train_data
                        entry['labels'] = self.train_labels
                        pckl_file = path.join(train_dir,'train_'+str(label)+'.pickle')
                        with open(pckl_file, 'wb') as pckl:
                            pickle.dump(entry, pckl)
        else:
            self.test_data = []
            self.test_labels = []
            if load_pickle:
                for pckl in os.listdir(test_dir):
                    if pckl.endswith(".pickle"):
                        pckl_path = path.join(test_dir, pckl)
                        pckl_file = open(pckl_path, 'rb')
                        entry = pickle.load(pckl_file, encoding='latin1')
                        self.test_data.append(entry['data'])
                        self.test_labels += entry['labels']
                        pckl_file.close()
                self.test_data = np.concatenate(self.test_data)
            else:
                for dir in os.listdir(test_dir):
                    label_dir = path.join(test_dir, dir)
                    if os.path.isdir(label_dir):
                        label = int(dir)-1
                        self.test_data = []
                        self.test_labels = []
                        for img_file in os.listdir(label_dir):
                            if img_file.endswith('.png'):
                                img = Image.open(path.join(label_dir, img_file))
                                img_arr = np.array(img)
                                self.test_data.append(img_arr)
                                self.test_labels.append(label)
                        self.test_data = np.expand_dims(self.test_data, axis=0)
                        self.test_data = np.squeeze(self.test_data, axis=0)
                        entry = dict(())
                        entry['data'] = self.test_data
                        entry['labels'] = self.test_labels
                        pckl_file = path.join(test_dir, 'test_' + str(label) + '.pickle')
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


