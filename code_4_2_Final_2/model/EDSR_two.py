from model import common

import torch.nn as nn

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()
        self.fullTrain = args.fullTrain
        self.fullInputScale = args.fullInputScale
        self.fullTargetScale = args.fullTargetScale
        self.fulltrainSecondScale = int(self.fullInputScale[0])
        self.fulltrainFirstScale = int(8/int(self.fullTargetScale[0]))
        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        #scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, -1)
        
        # define head module
        
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale) \
            for _ in range(n_resblock)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
         # define tail module

        modules_tail_first = [
            common.Upsampler(conv, self.fulltrainFirstScale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]
    
        modules_tail_second = [
            common.Upsampler(conv, self.fulltrainSecondScale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, 1)

        self.head_first = nn.Sequential(*modules_head)
        self.body_first = nn.Sequential(*modules_body)
        self.tail_first = nn.Sequential(*modules_tail_first)
        
        #self.head_2 = nn.Sequential(*modules_head)
        #self.body_2 = nn.Sequential(*modules_body)
        #self.tail_2 = nn.Sequential(*modules_tail)

        self.head_second = nn.Sequential(*modules_head)
        self.body_second = nn.Sequential(*modules_body)
        self.tail_second = nn.Sequential(*modules_tail_second)
    def forward(self, x):
        if self.fullTrain:
            x = self.sub_mean(x)
            x = self.head_first(x)
            res = self.body_first(x)
            res += x
            x = self.tail_first(res)
            x2_1 = self.add_mean(x)

            x2 = self.sub_mean(x2_1)
            x2 = self.head_second(x2)
            res2 = self.body_second(x2)
            res2 += x2
            x2 = self.tail_second(res2)
            x_output = self.add_mean(x2)
            return x_output,x2_1
        else:
            if self.fullInputScale == '2':
                x2_1 = x
                x2 = self.sub_mean(x2_1)
                x2 = self.head_second(x2)
                res2 = self.body_second(x2)
                res2 += x2
                x2 = self.tail_second(res2)
                x_output = self.add_mean(x2)
                return x_output
            
            elif self.fullInputScale == '4':
                x2_1 = x
                x2 = self.sub_mean(x2_1)
                x2 = self.head_second(x2)
                res2 = self.body_second(x2)
                res2 += x2
                x2 = self.tail_second(res2)
                x_output = self.add_mean(x2)
                return x_output
############################################################3         
       
            elif self.fullInputScale == '8':
                x = self.sub_mean(x)
                x = self.head_first(x)
                res = self.body_first(x)
                res += x
                x = self.tail_first(res)
                x2_1 = self.add_mean(x)
                return x2_1
            
            
            
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

