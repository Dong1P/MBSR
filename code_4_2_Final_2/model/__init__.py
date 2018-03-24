from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

class model:
    def __init__(self, args):
        self.module = import_module('model.' + args.model)
        self.args = args
        self.n_GPU_number = args.n_GPU_number
    def get_model(self):
        print('Making model...')
        module = import_module('model.' + self.args.model)
        my_model = module.make_model(self.args)
        #if self.args.pre_train_1 != '.':
        print('Loading model from {}...'.format(self.args.pre_train_1))
        print('Loading model from {}...'.format(self.args.pre_train_2))
        print('Loading model from {}...'.format(self.args.pre_train_3))
        my_model.load_state_dict_4_2(torch.load(self.args.pre_train_1))
            #my_model.load_state_dict_4_2(torch.load(self.args.pre_train_2))
 #       my_model.load_state_dict_8_4(torch.load(self.args.pre_train_3))

        if not self.args.no_cuda:
            print('\tCUDA is ready!')
            torch.cuda.manual_seed(self.args.seed)
            if self.n_GPU_number != 100:
               my_model.cuda(self.n_GPU_number)
            else:
                my_model.cuda(1)
            if self.args.n_GPUs > 1:
                my_model = nn.DataParallel(my_model, range(0,2))

        if self.args.print_model:
            print(my_model)

        return my_model

