import torch

import utils
from option import args
from data import data
if args.fullTrain:
    from trainer import Trainer
else:
    from preTrainer import Trainer
    
torch.manual_seed(args.seed)
checkpoint = utils.checkpoint(args)

if checkpoint.ok:
    my_loader = data(args).get_loader()
    t = Trainer(my_loader, checkpoint, args)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()
    
    

 
    
""" 
my_loader = data(args).get_loader()

loader_train, loader_test = my_loader
check = 0

#for batch, (input, target,input_4,input_2, idx_scale) in enumerate(loader_train):
#    check = check+1
for batch, (input, target, idx_scale) in enumerate(loader_train):
    check = check+1
    
print(check)
"""
