import torch.nn as nn
import torch
import torch.functional as F

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):#num_input_features特征层数
        super(_DenseLayer, self).__init__()#growth_rate=32增长率 bn_size=4
        #（56 * 56 * 64）
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        #（56 * 56 * 32）
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)#（56 * 56 * 64*3）
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        # bn1 + relu1 + conv1
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        # type(List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)#按通道合并
        # bn1 + relu1 + conv1
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function



class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()#num_layers层重复次数
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,   #for一次层数增加32
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)    # 追加denselayer层到字典里面

    def forward(self, init_features):
        features = [init_features]  #原来的特征，64
        for name, layer in self.named_children():   # 依次遍历添加的6个layer层，
            new_features = layer(*features) #计算特征
            features.append(new_features)   # 追加特征
        return torch.cat(features, 1)   # 按通道数合并特征

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 1))

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.conv1_prelu1 = nn.PReLU()
        self.LN1 = nn.LayerNorm()

        # self.dense1 = DenseLayer
    pass

class T_Decoder(nn.Module):

    pass

class F_Decoder(nn.Module):

    pass

class Fusion_net(nn.Module):
    """
    一个 encoder，两个decoder
    """

