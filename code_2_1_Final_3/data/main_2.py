import torch


from option import args
from data import data

    
    
my_loader = data(args).get_loader()

loader_train, loader_test = my_loader
check = 0

for batch, (input, target,input_4,input_2, idx_scale) in enumerate(loader_train):
    check = check+1
    
print(check)
