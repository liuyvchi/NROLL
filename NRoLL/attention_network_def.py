from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter, ModuleList
import torch
import torch.nn.functional as F
import math


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def l2_norm(x, axis=1):
    norm = torch.norm(x, 2, axis, True)
    output = x / norm
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.global_avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.global_avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BottleneckIR(Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckIR, self).__init__()
        self.identity = 0
        if in_channels == out_channels:
            if stride == 1:
                self.identity = 1
            else:
                self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channels, out_channels, (1, 1), stride, bias=False),
                                             BatchNorm2d(out_channels))
        self.res_layer = Sequential(BatchNorm2d(in_channels),
                                    Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
                                    BatchNorm2d(out_channels),
                                    PReLU(out_channels),
                                    Conv2d(out_channels, out_channels, (3, 3), stride, 1, bias=False),
                                    BatchNorm2d(out_channels))

    def forward(self, x):
        shortcut = x if self.identity == 1 else self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class BottleneckIRSE(Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckIRSE, self).__init__()
        self.identity = 0
        if in_channels == out_channels and stride == 1:
            # if stride == 1:
            self.identity = 1
            # else:
            #    self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channels, out_channels, (1, 1), stride, bias=False),
                                             BatchNorm2d(out_channels))
        self.res_layer = Sequential(BatchNorm2d(in_channels),
                                    Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
                                    BatchNorm2d(out_channels),
                                    PReLU(out_channels),
                                    Conv2d(out_channels, out_channels, (3, 3), stride, 1, bias=False),
                                    BatchNorm2d(out_channels),
                                    SEModule(out_channels, 16))

    def forward(self, x):
        shortcut = self.shortcut_layer(x) if self.identity != 1 else x
        res = self.res_layer(x)
        return res + shortcut


class BasicResBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResBlock, self).__init__()
        self.identity = 0
        if in_channels == out_channels:
            if stride == 1:
                self.identity = 1
            else:
                self.shortcut_layer = MaxPool2d(2, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                             BatchNorm2d(out_channels))
        self.res_layer = Sequential(Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
                                    BatchNorm2d(out_channels),
                                    ReLU(inplace=True),
                                    Conv2d(out_channels, out_channels, (3, 3), stride, 1, bias=False),
                                    BatchNorm2d(out_channels))

    def forward(self, x):
        shortcut = x if self.identity == 1 else self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class MaskModule(Module):
    def __init__(self, down_sample_times, out_channels, r, net_mode='ir'):
        super(MaskModule, self).__init__()
        assert net_mode in ('ir', 'basic', 'irse')
        func = {'ir': BottleneckIR, 'irse': BottleneckIRSE, 'basic': BasicResBlock}

        self.max_pool_layers = ModuleList()
        for i in range(down_sample_times):
            self.max_pool_layers.append(MaxPool2d(2, 2))

        self.prev_res_layers = ModuleList()
        for i in range(down_sample_times):
            tmp_prev_res_block_layers = []
            for j in range(r):
                tmp_prev_res_block_layers.append(func[net_mode](out_channels, out_channels, 1))
            self.prev_res_layers.append(Sequential(*tmp_prev_res_block_layers))

        self.mid_res_layers = None
        self.post_res_layers = None
        if down_sample_times > 1:
            self.mid_res_layers = ModuleList()
            for i in range(down_sample_times - 1):
                self.mid_res_layers.append(func[net_mode](out_channels, out_channels, 1))

            self.post_res_layers = ModuleList()
            for i in range(down_sample_times - 1):
                tmp_post_res_block_layers = []
                for j in range(r):
                    tmp_post_res_block_layers.append(func[net_mode](out_channels, out_channels, 1))
                self.post_res_layers.append(Sequential(*tmp_post_res_block_layers))

        self.r = r
        self.out_channels = out_channels
        self.down_sample_times = down_sample_times

    def mask_branch(self, x, cur_layers, down_sample_times):
        h = x.shape[2]
        w = x.shape[3]

        cur_layers.append(self.max_pool_layers[self.down_sample_times - down_sample_times](x))

        cur_layers.append(self.prev_res_layers[self.down_sample_times - down_sample_times](cur_layers[-1]))
        # down_sample_times -= 1
        if down_sample_times - 1 <= 0:

            cur_layers.append(F.interpolate(cur_layers[-1], (h, w), mode='bilinear'))
            return cur_layers[-1]
        else:
            cur_layers.append(self.mid_res_layers[self.down_sample_times - down_sample_times](cur_layers[-1]))

            shortcut_layer = cur_layers[-1]
            v = self.mask_branch(cur_layers[-1], cur_layers, down_sample_times - 1)
            cur_layers.append(shortcut_layer + v)

            cur_layers.append(self.post_res_layers[self.down_sample_times - down_sample_times](cur_layers[-1]))
            cur_layers.append(F.interpolate(cur_layers[-1], (h, w), mode='bilinear'))
            return cur_layers[-1]

    def forward(self, x):
        cur_layers = []
        return self.mask_branch(x, cur_layers, self.down_sample_times)


class AttentionModule(Module):
    def __init__(self, in_channels, out_channels, input_spatial_dim, p=1, t=2, r=1, net_mode='ir'):
        super(AttentionModule, self).__init__()
        self.func = {'ir': BottleneckIR, 'irse': BottleneckIRSE, 'basic': BasicResBlock}

        # start branch
        self.start_branch = ModuleList()
        self.start_branch.append(self.func[net_mode](in_channels, out_channels, 1))
        for i in range(p - 1):
            self.start_branch.append(self.func[net_mode](out_channels, out_channels, 1))

        # trunk branch
        self.trunk_branch = ModuleList()
        for i in range(t):
            self.trunk_branch.append(self.func[net_mode](out_channels, out_channels, 1))

        # mask branch
        # 1st, determine how many down-sample operations should be executed.
        num_down_sample_times = 0
        resolution = input_spatial_dim
        while resolution > 4 and resolution not in (8, 7, 6, 5):
            num_down_sample_times += 1
            resolution = (resolution - 2) / 2 + 1
        self.num_down_sample_times = min(num_down_sample_times, 100)
        self.mask_branch = MaskModule(num_down_sample_times, out_channels, r, net_mode)

        self.mask_helper = Sequential(Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
                                      BatchNorm2d(out_channels),
                                      ReLU(inplace=True),
                                      Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
                                      BatchNorm2d(out_channels),
                                      Sigmoid())
        # output branch
        self.out_branch = ModuleList()
        for i in range(p):
            self.out_branch.append(self.func[net_mode](out_channels, out_channels, 1))
        self.p = p
        self.t = t
        self.r = r

    def forward(self, x):
        for i in range(self.p):
            x = self.start_branch[i](x)
        y = x
        for i in range(self.t):
            x = self.trunk_branch[i](x)

        trunk = x
        mask = self.mask_branch(y)
        mask = self.mask_helper(mask)
        out = trunk * (mask + 1)
        for i in range(self.p):
            out = self.out_branch[i](out)
        return out


class AttentionNet(Module):
    def __init__(self, in_channels=3, p=1, t=2, r=1, net_mode='ir', attention_stages=(1, 1, 1)):
        super(AttentionNet, self).__init__()
        final_res_block = 3
        func = {'ir': BottleneckIR, 'irse': BottleneckIRSE, 'basic': BasicResBlock}
        self.input_layer = Sequential(Conv2d(in_channels, 64, 3, 1, 1),
                                      BatchNorm2d(64),
                                      ReLU(inplace=True),
                                      func[net_mode](64, 64, 2))
        input_spatial_dim = (120 - 1) // 2 + 1
        modules = []

        # stage 1
        for i in range(attention_stages[0]):
            modules.append(AttentionModule(64, 64, input_spatial_dim, p, t, r, net_mode))

        modules.append(func[net_mode](64, 128, 2))
        input_spatial_dim = (input_spatial_dim - 1) // 2 + 1

        # stage 2
        for i in range(attention_stages[1]):
            modules.append(AttentionModule(128, 128, input_spatial_dim, p, t, r, net_mode))

        modules.append(func[net_mode](128, 256, 2))
        input_spatial_dim = (input_spatial_dim - 1) // 2 + 1

        # stage 3
        for i in range(attention_stages[2]):
            modules.append(AttentionModule(256, 256, input_spatial_dim, p, t, r, net_mode))

        modules.append(func[net_mode](256, 512, 2))
        input_spatial_dim = (input_spatial_dim - 1) // 2 + 1

        for i in range(final_res_block):
            modules.append(func[net_mode](512, 512, 1))

        self.body = Sequential(*modules)
        self.output_layer = Sequential(Flatten(),
                                       Linear(512 * input_spatial_dim * input_spatial_dim, 512, False),
                                       BatchNorm1d(512))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)




