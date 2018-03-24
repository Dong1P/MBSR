import os
import random
import math

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class DIV2K(data.Dataset):
    def __init__(self, args, train=True):
        self._init_basic(args, train)

        split = 'train'
        dir_HR = 'DIV2K_{}_HR'.format(split)
        dir_LR = 'DIV2K_{}_LR_bicubic'.format(split)
        x_scale = ['X{}'.format(s) for s in args.scale]
        Three_scale = [4,2]
        Tree_scale = ['X{}'.format(s) for s in Three_scale]

        if self.args.ext != 'pack':
            self.dir_in = [
                os.path.join(self.apath, dir_LR, xs) for xs in x_scale]
            self.dir_tar = os.path.join(self.apath, dir_HR)
        else:
            print('Preparing binary packages...')
            packname = 'pack.pt' if self.train else 'packv.pt'
            name_tar = os.path.join(self.apath, dir_HR, packname)
            print('\tLoading ' + name_tar)
            self.pack_in = []
            self.pack_tar = torch.load(name_tar)
            if self.train:
                self._save_partition(
                    self.pack_tar,
                    os.path.join(self.apath, dir_HR, 'packv.pt'))

            for i, xs in enumerate(x_scale):
                name_in = os.path.join(self.apath, dir_LR, xs, packname)
                print('\tLoading ' + name_in)
                self.pack_in.append(torch.load(name_in))
                if self.train:
                    self._save_partition(
                        self.pack_in[i],
                        os.path.join(self.apath, dir_LR, xs, 'packv.pt'))

            for i, xs in enumerate(Tree_scale):
                name_in = os.path.join(self.apath, dir_LR, xs, packname)
                print('\tLoading ' + name_in)
                self.pack_in.append(torch.load(name_in))

    def __getitem__(self, idx):
        scale = self.scale[self.idx_scale]
        idx = self._get_index(idx)
        img_in, img_tar,img_in4,img_in2 = self._load_file(idx)
        
        img_in, img_tar,img_in4,img_in2, pi, ai = self._get_patch(img_in, img_tar,img_in4,img_in2)
        if self.train:
            img_in, img_tar,img_in4,img_in2 = common.set_channel_Three(
            img_in, img_tar,img_in4,img_in2, self.args.n_colors)
            
            return common.np2Tensor_Three(img_in, img_tar,img_in4,img_in2, self.args.rgb_range)
            #return common.np2Tensor(img_in, img_tar, self.args.rgb_range)

        else:
            img_in, img_tar = common.set_channel(
            img_in, img_tar, self.args.n_colors)
            
            return common.np2Tensor(img_in, img_tar, self.args.rgb_range)
    def __len__(self):
        if self.train:
            return self.args.n_train * self.repeat
        else:
            return self.args.n_val

    def _init_basic(self, args, train):
        self.args = args
        self.train = train
        self.scale = args.scale
        self.idx_scale = 0

        self.repeat = args.test_every // (args.n_train // args.batch_size)

        if args.ext == 'png':
            self.apath = args.dir_data + '/DIV2K'
            self.ext = '.png'
        else:
            self.apath = args.dir_data + '/DIV2K_decoded'
            self.ext = '.pt'

    def _get_index(self, idx):
        if self.train:
            idx = (idx % self.args.n_train) + 1
        else:
            idx = (idx + self.args.offset_val) + 1

        return idx

    def _load_file(self, idx):
        def _get_filename():
            filename = '{:0>4}'.format(idx)
            name_in = '{}/{}x{}{}'.format(
                self.dir_in[self.idx_scale],
                filename,
                self.scale[self.idx_scale],
                self.ext)
            name_tar = os.path.join(self.dir_tar, filename + self.ext)

            return name_in, name_tar

        if self.args.ext == 'png':
            name_in, name_tar = _get_filename()
            img_in = misc.imread(name_in)
            img_tar = misc.imread(name_tar)
        elif self.args.ext == 'pt':
            name_in, name_tar = _get_filename()
            img_in = torch.load(name_in).numpy()
            img_tar = torch.load(name_tar).numpy()
        elif self.args.ext == 'pack':
            img_in = self.pack_in[self.idx_scale][idx].numpy()
            img_in_4 = self.pack_in[1][idx].numpy()
            img_in_2 = self.pack_in[2][idx].numpy()
            img_tar = self.pack_tar[idx].numpy()
        return img_in, img_tar,img_in_4,img_in_2

    def _get_patch(self, img_in, img_tar,img_in4,img_in2):
        scale = self.scale[self.idx_scale]
        if self.train:
            img_in, img_tar,img_in4,img_in2, pi = common.get_patch(
                img_in, img_tar,img_in4,img_in2, self.args, scale)
            img_in, img_tar,img_in4,img_in2, ai = common.augment(img_in, img_tar,img_in4,img_in2)

            return img_in, img_tar,img_in4,img_in2, pi, ai
        else:
            ih, iw, c = img_in.shape
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]

            return img_in, img_tar, None, None,None,None

    def _save_partition(self, dict_full, name):
        dict_val = {}
        for i in range(self.args.n_train, self.args.n_train + self.args.n_val):
            dict_val[i + 1] = dict_full[i + 1]
        torch.save(dict_val, name)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

