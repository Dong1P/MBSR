import os
import random
import math

from data import common_DW
#import common_DW as common
import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
##Three

class DIV2K_DW(data.Dataset):
    def __init__(self, args, train=True):
        self._init_basic(args, train)
        self.fullTrain = args.fullTrain
        self.fullTargetScale = args.fullTargetScale
        self.fullInputScale = args.fullInputScale
        split = 'train'
        dir_HR = 'DIV2K_{}_HR'.format(split)
       # dir_LR = 'DIV2K_{}_LR_bicubic'.format(split)
        dir_LR = 'DIV2K_{}_difficult'.format(split)
        x_NofullTrain_Input = ['X{}'.format(s) for s in args.fullInputScale]
        x_NofullTrain_Out = ['X{}'.format(s) for s in args.fullTargetScale]
        x_scale = ['X{}'.format(s) for s in args.scale]
        Three_scale = [4,2]
        Tree_scale = ['X{}'.format(s) for s in Three_scale]
        self.DIV2K = '/home/kky/Dongwon/SuperResolution/Data/NTIRE/DIV2K/DIV2K_train_LR_difficult_register'

        if self.args.ext != 'pack':
            self.dir_in = [
                os.path.join(self.apath, dir_LR, xs) for xs in x_scale]
            self.dir_tar = os.path.join(self.apath, dir_HR)
        else:
            print('Preparing binary packages...')
            if self.fullTrain:
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
                    if self.train:
                        self._save_partition(
                                self.pack_in[i+1],
                                os.path.join(self.apath, dir_LR, xs, 'packv.pt'))

            else:
                if args.fullTargetScale == '1':    
                    dir_HR = 'DIV2K_{}_HR'.format(split)
                    packname = 'pack.pt' if self.train else 'packv.pt'
                    name_tar = os.path.join(self.apath, dir_HR, packname)
                    print('\tLoading ' + name_tar)
                    self.pack_in = []
                    self.pack_tar = torch.load(name_tar)
                    if self.train:
                        self._save_partition(
                                self.pack_tar,
                                os.path.join(self.apath, dir_HR, 'packv.pt'))
                else:
                    #xs = x_NofullTrain_Out[0]
                    xs = 'X2_decoded'
                    packname = 'pack.pt' if self.train else 'packv.pt'
                    name_tar = os.path.join(self.DIV2K,str(xs), packname)
                    print('\tLoading ' + name_tar)
                    self.pack_in = []
                    self.pack_tar = torch.load(name_tar)
                    if self.train:
                        self._save_partition(
                                self.pack_tar,
                                os.path.join(self.DIV2K, xs, 'packv.pt'))                    
                                        
                #for i, xs in enumerate(x_NofullTrain_Input):
                    
                xs = 'X4_4'
                name_in = os.path.join(self.apath,dir_LR, str(xs), packname)
                print('\tLoading ' + name_in)
                self.pack_in.append(torch.load(name_in))
                if self.train:
                    self._save_partition(
                            self.pack_in[0],
                            os.path.join(self.apath,dir_LR, xs, 'packv.pt'))

                
                
                

    def __getitem__(self, idx):
        scale = self.scale[self.idx_scale]
        idx = self._get_index(idx)
        img_in, img_tar,img_in4,img_in2 = self._load_file(idx)
        
        img_in, img_tar,img_in4,img_in2, pi, ai = self._get_patch(img_in, img_tar,img_in4,img_in2)
        if self.train:
            if self.fullTrain:
                img_in, img_tar,img_in4,img_in2 = common_DW.set_channel_Three(
                        img_in, img_tar,img_in4,img_in2, self.args.n_colors)
                return common_DW.np2Tensor_DW(img_in, img_tar,img_in4,img_in2, self.args.rgb_range)
            else:
                img_in, img_tar = common_DW.set_channel(
                        img_in, img_tar, self.args.n_colors)
		#print(img_in.shape)
		#print(img_tar.shape)
                return common_DW.np2Tensor(img_in, img_tar, self.args.rgb_range)
            #return common.np2Tensor(img_in, img_tar, self.args.rgb_range)
        else:
           # print(img_in.shape)
           # print(img_tar.shape)
            img_in, img_tar = common_DW.set_channel(
            img_in, img_tar, self.args.n_colors)
            #print(img_in.shape)
	    #print(img_tar.shape)
            return common_DW.np2Tensor(img_in, img_tar, self.args.rgb_range)
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
            if self.fullTrain:
                img_in = self.pack_in[self.idx_scale][idx].numpy()
                img_in_4 = self.pack_in[1][idx].numpy()
                img_in_2 = self.pack_in[2][idx].numpy()
                img_tar = self.pack_tar[idx].numpy()
                return img_in, img_tar,img_in_4,img_in_2
            else:
                img_in = self.pack_in[self.idx_scale][idx].numpy()
                img_tar = self.pack_tar[idx].numpy()
                
                return img_in, img_tar,None,None

    def _get_patch(self, img_in, img_tar,img_in4,img_in2):
        if self.train:
            if self.fullTrain:
                scale = self.scale[self.idx_scale]
                img_in, img_tar,img_in4,img_in2, pi = common_DW.get_patch(
                        img_in, img_tar,img_in4,img_in2, self.args, scale)
                img_in, img_tar,img_in4,img_in2, ai = common_DW.augment(img_in, img_tar,img_in4,img_in2)

                return img_in, img_tar,img_in4,img_in2, pi, ai
            else:
                scale = int(self.fullInputScale[0])/int(self.fullTargetScale[0])
                img_in, img_tar,img_in4,img_in2, pi = common_DW.get_patch_DW(
                        img_in, img_tar, self.args, int(scale))
                img_in, img_tar,img_in4,img_in2, ai = common_DW.augment_DW(img_in, img_tar)

                return img_in, img_tar,None,None, pi, ai                
        else:
            scale = self.scale[self.idx_scale] ##0209 revised.
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

