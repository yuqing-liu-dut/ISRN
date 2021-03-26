from model import common

import torch
import torch.nn as nn

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

def make_model(args, parent=False):
    return Net(args)

class Conv_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv_ReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
        )
    def forward(self, x):
        return x + self.conv(x)

class AdaptiveFM(nn.Module):
    def __init__(self, n_channels, kernel_size=3):
        super(AdaptiveFM, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=n_channels)
    def forward(self, x):
        return self.conv(x)+x

class AdaptiveResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(AdaptiveResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            AdaptiveFM(out_channels, kernel_size),
        )
    def forward(self, x):
        return x + self.conv(x)

class AdaptiveResGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks):
        super(AdaptiveResGroup, self).__init__()
        module_group = [AdaptiveResBlock(in_channels, out_channels, kernel_size=kernel_size) for _ in range(n_blocks)]
        module_group.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
        module_group.append(nn.ReLU(True))
        module_group.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
        module_group.append(AdaptiveFM(out_channels, kernel_size))
        self.conv = nn.Sequential(*module_group)
    def forward(self, x):
        return x + self.conv(x)


class ResGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks):
        super(ResGroup, self).__init__()
        module_group = [ResBlock(in_channels, out_channels, kernel_size=kernel_size) for _ in range(n_blocks)]
        module_group.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
        module_group.append(nn.ReLU(True))
        module_group.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
        self.conv = nn.Sequential(*module_group)
    def forward(self, x):
        return x + self.conv(x)

# LR -> HR
class SolverHR(nn.Module):
    def __init__(self, scale, n_channels=64, n_blocks=5, n_groups=3):
        super(SolverHR, self).__init__()
        module_head = [nn.Conv2d(3, n_channels, kernel_size=3, padding=3//2)]
        module_body = [AdaptiveResGroup(n_channels, n_channels, kernel_size=3, n_blocks=n_blocks) for _ in range(n_groups)]
        module_tail = [nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=3//2),
                        nn.Conv2d(n_channels, 3*scale*scale, kernel_size=3, padding=3//2),
                        nn.PixelShuffle(scale),
                        nn.Conv2d(3, 3, kernel_size=5, padding=5//2)]
        module_skip = [nn.Conv2d(3, 3*scale*scale, kernel_size=5, padding=5//2),
                        nn.PixelShuffle(scale),
                        nn.Conv2d(3, 3, kernel_size=5, padding=5//2)]
        self.head = nn.Sequential(*module_head)
        self.body = nn.Sequential(*module_body)
        self.tail = nn.Sequential(*module_tail)
        self.skip = nn.Sequential(*module_skip)
    def forward(self, x):
        head = self.head(x)
        body = self.body(head)
        tail = self.tail(body)
        skip = self.skip(x)
        return tail+skip

# HR -> LR
class Downsampler(nn.Module):
    def __init__(self, scale):
        super(Downsampler, self).__init__()
        if scale == 2:
            module_downsampler = [nn.Conv2d(3, 32, kernel_size=3, padding=3//2),
                                  nn.Conv2d(32, 32, kernel_size=3, padding=3//2,),
                                  nn.ReLU(True),
                                  nn.Conv2d(32, 32, kernel_size=3, padding=3//2, stride=2),
                                  nn.ReLU(True),
                                  nn.Conv2d(32, 3, kernel_size=3, padding=3//2)]
        elif scale == 3:
            module_downsampler = [nn.Conv2d(3, 32, kernel_size=3, padding=3//2),
                                  nn.Conv2d(32, 32, kernel_size=3, padding=3//2,),
                                  nn.ReLU(True),
                                  nn.Conv2d(32, 32, kernel_size=3, padding=3//2, stride=3),
                                  nn.ReLU(True),
                                  nn.Conv2d(32, 3, kernel_size=3, padding=3//2)]
        elif scale == 4:
            module_downsampler = [nn.Conv2d(3, 32, kernel_size=3, padding=3//2),
                                  nn.Conv2d(32, 32, kernel_size=3, padding=3//2,),
                                  nn.ReLU(True),
                                  nn.Conv2d(32, 32, kernel_size=3, padding=3//2, stride=2),
                                  nn.ReLU(True),
                                  nn.Conv2d(32, 3, kernel_size=3, padding=3//2, stride=2)]
        elif scale == 8:
            module_downsampler = [nn.Conv2d(3, 32, kernel_size=3, padding=3//2),
                                  nn.Conv2d(32, 32, kernel_size=3, padding=3//2,),
                                  nn.ReLU(True),
                                  nn.Conv2d(32, 32, kernel_size=3, padding=3//2, stride=2),
                                  nn.ReLU(True),
                                  nn.Conv2d(32, 32, kernel_size=3, padding=3//2, stride=2),
                                  nn.ReLU(True),
                                  nn.Conv2d(32, 3, kernel_size=3, padding=3//2, stride=2)]
        self.downsampler = nn.Sequential(*module_downsampler)
    def forward(self, x):
        return self.downsampler(x)

# LR, u -> LR
class SolverLR(nn.Module):
    def __init__(self):
        super(SolverLR, self).__init__()
        module_conv = [nn.Conv2d(3*2, 64, kernel_size=3, padding=3//2),
                       nn.Conv2d(64, 64, kernel_size=3, padding=3//2),
                       nn.ReLU(True),
                       nn.Conv2d(64, 3, kernel_size=3, padding=3//2),]
        self.conv = nn.Sequential(*module_conv)
    def forward(self, u, lr):
        return self.conv(torch.cat([u, lr], 1))

# HR_1, ..., HR_k -> HR
class SolverEM(nn.Module):
    def __init__(self, n_inputs):
        super(SolverEM, self).__init__()
        module_em = [nn.Conv2d(3*n_inputs, 64, kernel_size=3, padding=3//2),
                     nn.ReLU(True),
                     nn.Conv2d(64, 3, kernel_size=3, padding=3//2),]
        self.em = nn.Sequential(*module_em)
    def forward(self, x):
        return self.em(x)

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        scale = args.scale[0]
        stage = 5
        self.stage = stage
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        
        module_SolverSR = [SolverHR(scale, n_blocks=6, n_groups=6)]
        module_Downsampler = [Downsampler(scale) for _ in range(stage-1)]
        module_SolverLR = [SolverLR() for _ in range(stage-1)]
        module_SolverEM = [SolverEM(stage)]

        self.SolverSR = nn.Sequential(*module_SolverSR)
        self.DownSampler = nn.Sequential(*module_Downsampler)
        self.SolverLR = nn.Sequential(*module_SolverLR)
        self.SolverEM = nn.Sequential(*module_SolverEM)
        
    def forward(self, x):
        lr = self.sub_mean(x)
        sr_out = []
        u = lr
        for i in range(self.stage):
            sr = self.SolverSR(u)
            sr_out.append(sr)
            if i != self.stage-1:
                u = self.DownSampler[i](sr)
                u = self.SolverLR[i](lr, u)
        sr = self.SolverEM(torch.cat(sr_out, 1))
        x = self.add_mean(sr)
        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


if __name__ == '__main__':
    with torch.no_grad():
        class ARGs:
            def __init__(self):
                self.scale=[4]
                self.rgb_range=255
        from thop import profile
        net = Net(ARGs()).cuda()
        inputs = torch.rand((1, 3, 720//4, 1280//4)).cuda()
        macs, param = profile(net, (inputs,))
        print('MACs (G):', macs/1000/1000/1000)
        print('Param (M):', param/1000/1000)
        # MACs(G): 988.8344309759999
        # Param(M): 3.459563
