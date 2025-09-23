import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models import create_model
from .backbones.model_convnext import convnext_tiny
from .backbones.resnet import Resnet
# from .backbones.resnet import Resnet50
import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter


# class Gem_heat(nn.Module):
#     def __init__(self, dim = 768, p=3, eps=1e-6):
#         super(Gem_heat, self).__init__()
#         self.p = nn.Parameter(torch.ones(dim) * p)
#         self.eps = eps
#
#     def forward(self, x):
#         return self.gem(x, p=self.p, eps=self.eps)
#
#
#     def gem(self, x, p=3):
#         p = F.softmax(p).unsqueeze(-1)
#         x = torch.matmul(x,p)
#         x = x.view(x.size(0), x.size(1))
#         return x
#
#
# def position(H, W, is_cuda=True):
#     if is_cuda:
#         loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
#         loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
#     else:
#         loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
#         loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
#     loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
#     return loc
#
#
# def stride(x, stride):
#     b, c, h, w = x.shape
#     return x[:, :, ::stride, ::stride]
#
#
# def init_rate_half(tensor):
#     if tensor is not None:
#         tensor.data.fill_(0.5)
#
#
# def init_rate_0(tensor):
#     if tensor is not None:
#         tensor.data.fill_(0.)


# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
# class ZPool(nn.Module):
#     def forward(self, x):
#         return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
#
# class AttentionGate(nn.Module):
#     def __init__(self):
#         super(AttentionGate, self).__init__()
#         kernel_size = 7
#         self.compress = ZPool()
#         self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.conv(x_compress)
#         scale = torch.sigmoid_(x_out)
#         return x * scale
#
# class TripletAttention(nn.Module):
#     def __init__(self):
#         super(TripletAttention, self).__init__()
#         self.cw = AttentionGate()
#         self.hc = AttentionGate()
#     def forward(self, x):
#         print('x',x.shape)
#         x_perm1 = x.permute(0,2,1,3).contiguous()
#         x_out1 = self.cw(x_perm1)
#         x_out11 = x_out1.permute(0,2,1,3).contiguous()
#         x_perm2 = x.permute(0,3,2,1).contiguous()
#         x_out2 = self.hc(x_perm2)
#         x_out21 = x_out2.permute(0,3,2,1).contiguous()
#         print(x_out11.shape),print(x_out21.shape)  # torch.Size([8, 768, 12, 12])
#         return x_out11, x_out21


# class SpatialAttention(nn.Module):
#     def __init__(self, in_channels, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         # print('avg_out',avg_out.shape)
#         # print('x.shape',x.shape)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         # print('max_out',max_out.shape)
#         cat_out = torch.cat([avg_out, max_out], dim=1)
#         # print('cat_out',cat_out.shape)
#         attention = self.sigmoid(self.conv(cat_out))
#         # print('attention',attention.shape)
#         return attention * x
# #
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels * 2, in_channels // reduction, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # print('x.shape', x.shape)
#         avg_out = self.avg_pool(x)
#         max_out = self.max_pool(x)
#         avg_max_out = torch.cat([avg_out, max_out], dim=1)
#         # print('avg_max_out', avg_max_out.shape)
#         # 通过卷积层获取通道注意力权重
#         attention = self.conv(avg_max_out)
#         # print('attention', attention.shape)
#         # 返回注意力加权后的张量
#         return attention * x
# class FeatureFusionModule(nn.Module):
#     def __init__(self, in_channels):
#         super(FeatureFusionModule, self).__init__()
#
#         self.conv1x1 = nn.Conv2d(in_channels, in_channels//4, kernel_size=1)
#         self.conv1x1.apply(weights_init_kaiming)
#
#         self.attention = nn.Sequential(
#             nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         out = self.conv1x1(x)
#         out = self.attention(out)
#
#         return out
class DEE_module(nn.Module):
    def __init__(self, channel, reduction=16,spatial_kernel_size=7,flag=False):
        super(DEE_module, self).__init__()
        # self.FC = nn.Conv2d(channel, channel, kernel_size=1)
        # self.FC.apply(weights_init_kaiming)
        self.FC11 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC11.apply(weights_init_kaiming)
        self.FC12 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC12.apply(weights_init_kaiming)
        self.FC13 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC13.apply(weights_init_kaiming)
        self.FC1 = nn.Conv2d(channel//4, channel, kernel_size=1)
        self.FC1.apply(weights_init_kaiming)
        # self.FeatureFusionModule = FeatureFusionModule(in_channels=channel)
        # self.channel_attention = ChannelAttention(channel, reduction)
        # self.spatial_attention = SpatialAttention(channel, kernel_size=spatial_kernel_size)
        self.flag = flag
        self.dropout = nn.Dropout(p=0.01)
        self.CPM = ImprovedCBAM(768)
        # self.CPM1 = ImprovedCBAM1(768)
        # self.CPM1 = ImprovedCBAM1(768)
    def forward(self, x):
        # print('x1',x1.shape)

        x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x))/3
        x1 = self.FC1(F.relu(x1))
        out1 = (x + x1)/2
        # out2 = self.channel_attention(x1)
        # out2 = self.spatial_attention(x1)
        out2 = (x1 + x)/2
        # attention = self.FeatureFusionModule(x1)
        # out2 = self.FeatureFusionModule(out2)
        # out1 = (attention + out2)/2
        # attention_out = self.attention_module(cat_features)
        # # print('attention_out.shape',attention_out.shape)
        # # print(x.shape)
        # out = torch.cat((x, attention_out), 0)
        # x2 = (self.FC21(x) + self.FC22(x) + self.FC23(x))/3
        # x2 = self.FC2(F.relu(x2))
        # print('x2',x2.shape)

        # out = torch.cat((x, x1, x2), 0)
        if self.flag:
            out1 = self.CPM(out2)
            # out2 = self.CPM1(out1)
            out1 = self.dropout(out1)
            out2 = self.dropout(out2)
            return out1, out2
        else:
            out1 = self.dropout(out1)
            return out1
        # print(out2.shape)

# class NonLocalModule(nn.Module):
#     def __init__(self, in_channels, reduction=2):
#         super(NonLocalModule, self).__init__()
#         self.inter_channels = max(1, in_channels // reduction)
#
#         self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
#         self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
#         self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
#
#         self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
#
#     def forward(self, x):
#         B, C, H, W = x.size()
#
#         g_x = self.g(x).view(B, self.inter_channels, -1)
#         g_x = g_x.permute(0, 2, 1)
#         # print('g_x',g_x.shape)
#         theta_x = self.theta(x).view(B, self.inter_channels, -1)
#         theta_x = theta_x.permute(0, 2, 1)
#
#         phi_x = self.phi(x).view(B, self.inter_channels, -1)
#
#         energy = torch.matmul(theta_x, phi_x)
#         attention = F.softmax(energy, dim=-1)
#
#         y = torch.matmul(attention, g_x)
#         y = y.permute(0, 2, 1).contiguous()
#         y = y.view(B, self.inter_channels, H, W)
#         W_y = self.W(y)
#         z = W_y + x
#         return z
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
            # print(input_dim)
            # print(num_bottleneck)
            # add_block += [nn.Conv1d(input_dim, num_bottleneck, kernel_size=1)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        # print('x_initial',x.shape)
        x = self.add_block(x)
        # print('self.add_block',self.add_block)
        # print('self.classifier',self.classifier)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                # print('f',f.shape)
                # print('x',x.shape)
                return x,f
            else:
                x = self.classifier(x)
                return x
        else:
            return x

# SE注意力机制模块

# class MultiScaleClassBlock(nn.Module):
#     def __init__(self, input_dim, class_num, droprate, bnorm=True, num_bottleneck=512, scales=[1, 2, 3],
#                  return_f=False):
#         super(MultiScaleClassBlock, self).__init__()
#         self.scales = scales
#         self.return_f = return_f
#
#         self.multi_scale_blocks = nn.ModuleList()
#         for scale in scales:
#             scale_block = []
#             scale_block.append(
#                 nn.Conv1d(in_channels=input_dim, out_channels=num_bottleneck, kernel_size=scale, stride=1,
#                           padding=scale // 2))
#             if bnorm:
#                 scale_block.append(nn.BatchNorm1d(num_bottleneck))
#             scale_block.append(nn.ReLU(inplace=True))
#             if droprate > 0:
#                 scale_block.append(nn.Dropout(p=droprate))
#             self.multi_scale_blocks.append(nn.Sequential(*scale_block))
#
#         self.classifier = nn.Linear(num_bottleneck * len(scales), class_num)
#         self.classifier.apply(self.weights_init_classifier)
#
#     def forward(self, x):
#         multi_scale_features = []
#         print(x.shape)
#         for scale_block in self.multi_scale_blocks:
#             x_scaled = scale_block(x)  # 修改后的代码，去掉了 unsqueeze(1)
#             multi_scale_features.append(x_scaled)
#
#         x = torch.cat(multi_scale_features, dim=1)
#
#         if self.return_f:
#             f = x
#             x = self.classifier(x)
#             return x, f
#         else:
#             x = self.classifier(x)
#             return x



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)

# class build_convnext(nn.Module):
#     def __init__(self, num_classes, block = 4, return_f=False, resnet=False):
#         super(build_convnext, self).__init__()
#         self.return_f = return_f
#         if resnet:
#             convnext_name = "resnet101"
#             print('using model_type: {} as a backbone'.format(convnext_name))
#             self.in_planes = 2048
#             self.convnext = Resnet(pretrained=True)
#         else:
#             convnext_name = "convnext_tiny"
#             print('using model_type: {} as a backbone'.format(convnext_name))
#             if 'base' in convnext_name:
#                 self.in_planes = 1024
#             elif 'large' in convnext_name:
#                 self.in_planes = 1536
#             elif 'xlarge' in convnext_name:
#                 self.in_planes = 2048
#             else:
#                 self.in_planes = 768
#             self.convnext = create_model(convnext_name, pretrained=True)
#
#         self.num_classes = num_classes
#         self.classifier1 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
#         self.block = block
#         # self.tri_layer = TripletAttention()
#         self.DEE = DEE_module(768)
#         self.NLM = NonLocalModule(768)
#         for i in range(self.block):
#             name = 'classifier_mcb' + str(i + 1)
#             setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))
#
#
#     def forward(self, x):
#         # print('x_initial',x.shape)
#         gap_feature, part_features = self.convnext(x)
#         # print('gap_feature.shape',gap_feature.shape)
#
#         part_features = self.NLM(part_features)
#         # print('part_features.shape',part_features.shape)
#         # tri_features = self.tri_layer(part_features)
#         tri_features = self.DEE(part_features)
#         # print('tri_features',tri_features[0].shape)
#         convnext_feature = self.classifier1(gap_feature)
#
#         # print('convnext_feature[0]',convnext_feature[0].shape)
#         # print('convnext_feature[1]',convnext_feature[1].shape)
#         tri_list = []
#         for i in range(self.block):
#             # print(self.block)
#             tri_list.append(tri_features[i].mean([-2, -1]))
#         triatten_features = torch.stack(tri_list, dim=2)
#         # print('triatten_features',triatten_features.shape)
#
#         if self.block == 0:
#             y = []
#         else:
#             y = self.part_classifier(self.block, triatten_features, cls_name='classifier_mcb')
#
#         if self.training:
#             y = y + [convnext_feature]
#             if self.return_f:
#                 cls, features = [], []
#                 for i in y:
#                     cls.append(i[0])
#                     features.append(i[1])
#                 return cls, features
#         else:
#             ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
#             y = torch.cat([y, ffeature], dim=2)
#
#         return y
# class SelfAttention(nn.Module):
#     def __init__(self, input_dim):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(input_dim, input_dim)
#         self.key = nn.Linear(input_dim, input_dim)
#         self.value = nn.Linear(input_dim, input_dim)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
#         attention = self.softmax(torch.bmm(Q, K.transpose(1, 2)) / (x.size(-1) ** 0.5))
#         out = torch.bmm(attention, V)
#         return out + x  # 残差连接

class MultiScaleFusion(nn.Module):
    def __init__(self, input_channels_list, output_channels,flag=False):
        super(MultiScaleFusion, self).__init__()
        self.conv = nn.Conv2d(sum(input_channels_list), output_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.flag = flag
        # self.norm = nn.LayerNorm(768, eps=1e-6)
    def forward(self, *inputs):
        # 在通道维度（即dim=1）上拼接特征
        x = torch.cat(inputs, dim=1)
        # 使用1x1卷积进行通道压缩
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print('x', inputs[1].shape)
        if self.flag:
            x= x.mean([-2, -1])
            # x= self.norm(x.mean([-2, -1]))
        else:
            x
        return x



# 使用逐层融合



# 使用该优化模块

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print('avg_out',avg_out.shape)
        # print('x.shape',x.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print('max_out',max_out.shape)
        cat_out = torch.cat([avg_out, max_out], dim=1)
        # print('cat_out',cat_out.shape)
        attention = self.sigmoid(self.conv(cat_out))
        # print('attention',attention.shape)
        return attention * x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print('x.shape', x.shape)
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_max_out = torch.cat([avg_out, max_out], dim=1)
        # print('avg_max_out', avg_max_out.shape)
        # 通过卷积层获取通道注意力权重
        attention = self.conv(avg_max_out)
        # print('attention', attention.shape)
        # 返回注意力加权后的张量
        return attention * x
# #
class ImprovedCBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel_size=7):
        super(ImprovedCBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(in_channels, kernel_size=spatial_kernel_size)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_attention = self.channel_attention(x)
        # print('channel_attention', channel_attention.shape)
        spatial_attention = self.spatial_attention(x)
        # print('spatial_attention', channel_attention.shape)
        channel_gate = self.channel_gate(torch.cat([x, channel_attention], dim=1))
        spatial_gate = self.spatial_gate(torch.cat([x, spatial_attention], dim=1))
        # print('channel_gate', channel_gate.shape)
        # print('spatial_gate', spatial_gate.shape)
        improved_features = channel_gate * x + spatial_gate * x
        # print('improved_features',improved_features.shape)
        return improved_features
#channel_attention torch.Size([8, 768, 12, 12])
# spatial_attention torch.Size([8, 768, 12, 12])
# channel_gate torch.Size([8, 768, 12, 12])
# spatial_gate torch.Size([8, 1, 12, 12])

class OptimizedChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(OptimizedChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # mid_channels = in_channels // reduction
        mid_channels = in_channels
        # 组卷积减少计算量
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=4)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=1, groups=4)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        # 合并后做一次卷积操作
        out = avg_out + max_out
        # out = self.relu(self.conv1(out))
        out = self.sigmoid(self.conv2(out))
        return out * x


class OptimizedSpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(OptimizedSpatialAttention, self).__init__()
        padding = kernel_size // 2
        # 使用深度可分离卷积来提高效率
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, groups=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.conv(out)


class ImprovedCBAM1(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel_size=7):
        super(ImprovedCBAM1, self).__init__()
        self.channel_attention = OptimizedChannelAttention(in_channels, reduction)
        self.spatial_attention = OptimizedSpatialAttention(kernel_size=spatial_kernel_size)

        # 同时为通道和空间门加上残差连接
        self.channel_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        # 通道注意力
        channel_attention = self.channel_attention(x)
        channel_gate = self.channel_gate(channel_attention)
        channel_gate = channel_gate +residual
        # print((channel_gate * x).shape)
        # 空间注意力
        spatial_attention = self.spatial_attention(x)
        # print(spatial_attention .shape)
        spatial_gate = self.spatial_gate(spatial_attention)
        # print((spatial_gate * x).shape)
        # 使用残差连接改进特征融合
        improved_features = channel_gate * x + spatial_gate * x + residual
        # print('improved_features',improved_features.shape)
        return improved_features
#channel_attention torch.Size([8, 768, 12, 12])
# spatial_attention torch.Size([8, 768, 12, 12])
# channel_gate torch.Size([8, 768, 12, 12])
# spatial_gate torch.Size([8, 1, 12, 12])

class build_convnext(nn.Module):
    def __init__(self, num_classes, block=4, return_f=False, resnet=False):
        super(build_convnext, self).__init__()
        self.return_f = return_f
        if resnet:
            convnext_name = "resnet101"
            print('using model_type: {} as a backbone'.format(convnext_name))
            self.in_planes = 2048
            self.convnext = Resnet(pretrained=True)
        else:
            convnext_name = "convnext_tiny"
            print('using model_type: {} as a backbone'.format(convnext_name))
            if 'base' in convnext_name:
                self.in_planes = 1024
            elif 'large' in convnext_name:
                self.in_planes = 1536
            elif 'xlarge' in convnext_name:
                self.in_planes = 2048
            else:
                self.in_planes = 768
            self.convnext = create_model(convnext_name, pretrained=True)

        self.num_classes = num_classes
        self.classifier1 =ClassBlock(self.in_planes, num_classes, droprate=0.5, return_f=return_f)
        self.block = block
        self.DEE = DEE_module(768,flag=True)
        self.DEE1 = DEE_module(96,flag=False)
        self.block_fusion = MultiScaleFusion([768, 96], output_channels=768,flag=True)
        self.block_fusion2 = MultiScaleFusion([768, 96], output_channels=768, flag=False)
        for i in range(self.block):
            name = 'classifier_mcb' + str(i + 1)
            setattr(self, name,
                    ClassBlock(self.in_planes, num_classes, droprate=0.5, return_f=self.return_f))

    def forward(self, x):
        x1,part_features = self.convnext(x)
        # gap_feature = self.ATE(gap_feature)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # x_ = self.MFA1(x2,x1)
        # print(x_.shape)
        # x_1 = self.MFA2(x3,x_)
        # part_features = self.MFA3(part_features,x_1)
        # x_ = self.FAP(gap_feature_expanded)
        # print('x_.shape', x_.shape)
        # print('gap_feature.shape',gap_feature.shape)
        # print('part_features.shape',part_features.shape)
        # part_features = self.NLM(part_features)
        # gap_feature = self.MFA(gap_feature_expanded,x_)
        # gap_feature = self.avgpool(gap_feature)  # 现在 x 的形状为 [8, 768, 1, 1]
        # gap_feature = gap_feature.view(x.size(0), -1)
        # print(gap_feature.shape)

        # print(x1.shape)
        x1 = F.interpolate(x1, (part_features.shape[-2], part_features.shape[-1]), mode='bilinear')

        x1 = self.DEE1(x1)

        # # print(x1.shape)
        # x2 = F.interpolate(x2, (part_features.shape[-2], part_features.shape[-1]), mode='bilinear')
        # x3 = F.interpolate(x3, (part_features.shape[-2], part_features.shape[-1]), mode='bilinear')
        # fused_features = torch.cat((x1, x2), dim=1)
        # x1 = self.conv_fusion(fused_features)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # x1 = self.CPM1(x1)

        # x1 = self.CPM1(x1)
        # print('x1',x1.shape)
        # print(part_features.shape)
        # print('x2',x2.shape)
        gap_feature=self.block_fusion(x1,part_features)
        # gap_feature = self.norm(part_features.mean([-2, -1]))
        # x_ = self.MFA1(x2,x1)
        # print(gap_feature.shape)
        # x_1 = self.MFA2(x3,x_)
        # part_features = self.MFA3(part_features,x3)
        # print('gap_feature',gap_feature.shape)
        # print(x3.shape)

        part_features = self.block_fusion2(x1,part_features)

        # part_features = self.CPM1(part_features)
        # print(xp.shape)


        tri_features = self.DEE(part_features)
        # print('part_features.shape',part_features.shape)
        # gap_feature = self.ATE(gap_feature)
        convnext_feature = self.classifier1(gap_feature)
        # print(convnext_feature[0].shape)
        # print(convnext_feature[1].shape)
        tri_list = []
        for i in range(self.block):
            tri_list.append(tri_features[i].mean([-2, -1]))
        triatten_features = torch.stack(tri_list, dim=2)

        if self.block == 0:
            y = []
        else:
            y = self.part_classifier(self.block, triatten_features, cls_name='classifier_mcb')

        if self.training:
            y = y + [convnext_feature]
            if self.return_f:
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                # print("Returning:", len([cls, features, loss_ort]))
                return cls, features
            return y
        else:
            ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
            y = torch.cat([y, ffeature], dim=2)

            return y



    def part_classifier(self, block, x, cls_name='classifier_mcb'):
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            name = cls_name + str(i+1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y



def make_convnext_model(num_class,block = 4,return_f=False,resnet=False):
    print('===========building convnext===========')
    model = build_convnext(num_class,block=block,return_f=return_f,resnet=resnet)
    return model


