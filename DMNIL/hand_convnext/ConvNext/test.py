# University: Undergraduate CAUC|Graduate XJTU
# Developers: ISChenawei
# Edit time: 2024/10/2 11:27
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x):
        # 输入尺寸需要转为 (sequence_length, batch_size, embed_dim)
        x = x.permute(2, 0, 1)
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output.permute(1, 2, 0)  # 转回 (batch_size, embed_dim, sequence_length)

class DomainSpaceAlignmentWithAttention(nn.Module):
    def __init__(self, in_channels=768, out_channels=256, num_heads=8, scale=1.0):
        super(DomainSpaceAlignmentWithAttention, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.l2_norm = True
        self.attention = MultiHeadAttention(in_channels, num_heads)

    def forward(self, part_features):
        b, c, h, w = part_features.shape
        pfeat = part_features.flatten(2)  # (bs, c, h*w)
        W = self.attention(pfeat)  # 使用多头注意力替代MLP
        W = F.normalize(W, dim=1) if self.l2_norm else W
        W *= (1 / self.scale)
        W = torch.sigmoid(W)  # 使用sigmoid来增强弱特征
        pfeat_align = torch.cat([pfeat, W], dim=1)
        return pfeat_align




class GatedDomainAlignment(nn.Module):
    def __init__(self, in_channels, out_channels, scale=1.0):
        super(GatedDomainAlignment, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.l2_norm = True
        # 门控机制，用于动态调节对齐的强度
        self.gate = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, part_features):
        b, c, h, w = part_features.shape
        pfeat = part_features.flatten(2)  # (bs, c, h*w)
        W = self.gate(pfeat)  # 通过门控机制动态调整对齐的权重
        W = F.normalize(W, dim=1) if self.l2_norm else W
        W *= (1 / self.scale)
        W = torch.sigmoid(W)
        pfeat_align = torch.cat([pfeat, W], dim=1)
        return pfeat_align




class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        x_out = self.channel_attention(x) * x
        # 空间注意力
        max_out, _ = torch.max(x_out, dim=1, keepdim=True)
        avg_out = torch.mean(x_out, dim=1, keepdim=True)
        x_out = torch.cat([max_out, avg_out], dim=1)
        x_out = self.spatial_attention(x_out) * x_out
        return x_out


class DomainSpaceAlignmentWithCBAM(nn.Module):
    def __init__(self, in_channels=768, out_channels=256, reduction=16, kernel_size=7, scale=1.0):
        super(DomainSpaceAlignmentWithCBAM, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.l2_norm = True
        self.cbam = CBAM(in_channels, reduction, kernel_size)

    def forward(self, part_features):
        b, c, h, w = part_features.shape
        pfeat = part_features.flatten(2)  # (bs, c, h*w)
        W = self.cbam(pfeat.view(b, c, h, w))  # 使用CBAM增强特征对齐
        W = F.normalize(W, dim=1) if self.l2_norm else W
        W *= (1 / self.scale)
        W = torch.sigmoid(W)
        pfeat_align = torch.cat([pfeat, W], dim=1)
        return pfeat_align





class LightweightAlignment(nn.Module):
    def __init__(self, in_channels, out_channels, scale=1.0):
        super(LightweightAlignment, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.l2_norm = True
        # 使用深度可分离卷积代替标准卷积
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, part_features):
        b, c, h, w = part_features.shape
        # 深度可分离卷积处理特征
        depthwise_out = self.depthwise_conv(part_features)
        W = self.pointwise_conv(depthwise_out)
        W = F.normalize(W, dim=1) if self.l2_norm else W
        W *= (1 / self.scale)
        W = torch.sigmoid(W)
        return W




class ResidualDomainAlignment(nn.Module):
    def __init__(self, in_channels, out_channels, scale=1.0):
        super(ResidualDomainAlignment, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.l2_norm = True
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, part_features):
        b, c, h, w = part_features.shape
        pfeat = part_features.flatten(2)  # (bs, c, h*w)
        W = self.proj(pfeat)
        W = F.normalize(W, dim=1) if self.l2_norm else W
        W *= (1 / self.scale)
        W = torch.sigmoid(W)
        # 残差连接，保留原始特征
        pfeat_align = torch.cat([pfeat, W], dim=1)
        return pfeat_align + pfeat


class ImprovedMLP1D(nn.Module):
    """
    Enhanced MLP1D with skip connections and increased depth.
    """

    def __init__(self, in_channels, hid_channels, out_channels, norm_layer=None, bias=False, num_mlp=3):
        super(ImprovedMLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.layers = nn.ModuleList()

        # 输入到隐藏层的降维
        self.layers.append(nn.Sequential(
            nn.Conv1d(in_channels, hid_channels, 1, bias=bias),
            norm_layer(hid_channels),
            nn.ReLU(inplace=True)
        ))

        # 增加深度的隐藏层
        for _ in range(num_mlp - 2):
            self.layers.append(nn.Sequential(
                nn.Conv1d(hid_channels, hid_channels, 1, bias=bias),
                norm_layer(hid_channels),
                nn.ReLU(inplace=True)
            ))

        # 隐藏层到输出层
        self.layers.append(nn.Sequential(
            nn.Conv1d(hid_channels, out_channels, 1, bias=bias)
        ))

    def forward(self, x):
        residual = x
        for layer in self.layers[:-1]:
            x = layer(x) + residual  # 引入跳跃连接
        x = self.layers[-1](x)
        return x



class MultiHeadFeatureAlignment(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(MultiHeadFeatureAlignment, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(out_channels, num_heads)

        # 线性变换将输入维度从768降到256
        self.query_proj = nn.Conv1d(in_channels, out_channels, 1)
        self.key_proj = nn.Conv1d(in_channels, out_channels, 1)
        self.value_proj = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        b, c, seq_len = x.shape
        query = self.query_proj(x).permute(2, 0, 1)  # 转换为 (seq_len, batch_size, embed_dim)
        key = self.key_proj(x).permute(2, 0, 1)
        value = self.value_proj(x).permute(2, 0, 1)

        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output.permute(1, 2, 0)  # 恢复为 (batch_size, embed_dim, seq_len)


class SparseMLP1D(nn.Module):
    """
    MLP with sparse regularization (Dropout or L1 regularization).
    """
    def __init__(self, in_channels, hid_channels, out_channels, norm_layer=None, bias=False, num_mlp=2, sparsity=0.05):
        super(SparseMLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.sparsity = sparsity  # 控制稀疏度
        mlps = []
        for _ in range(num_mlp - 1):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def forward(self, x):
        x = self.mlp(x)
        # 应用稀疏正则化（例如Dropout）
        x = F.dropout(x, p=self.sparsity, training=self.training)
        return x
class FeatureAlignmentWithTemperatureScaling(nn.Module):
    def __init__(self, in_channels, out_channels, temperature=0.1):
        super(FeatureAlignmentWithTemperatureScaling, self).__init__()
        self.temperature = temperature
        self.proj = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        W = self.proj(x)
        # 使用温度缩放的Softmax
        W = F.softmax(W / self.temperature, dim=2)
        return W


class ImprovedDomainSpaceAlignment(nn.Module):
    def __init__(self, in_channels=768, mid_channels=256, out_channels=256, num_heads=8, temperature=0.1):
        super(ImprovedDomainSpaceAlignment, self).__init__()

        # 使用多头注意力模块替代MLP
        self.multihead_alignment = MultiHeadFeatureAlignment(in_channels=in_channels, out_channels=mid_channels,
                                                             num_heads=num_heads)

        # 使用温度缩放Softmax进行特征对齐
        self.feature_scaling = FeatureAlignmentWithTemperatureScaling(in_channels=mid_channels,
                                                                      out_channels=out_channels,
                                                                      temperature=temperature)

        self.scale = 1.0
        self.l2_norm = True

    def forward(self, part_features):
        b, c, h, w = part_features.shape
        pfeat = part_features.flatten(2)  # 将特征展平为 (batch_size, channels, sequence_length)

        # 使用多头注意力模块进行特征对齐
        W = self.multihead_alignment(pfeat)

        # L2归一化
        W = F.normalize(W, dim=1) if self.l2_norm else W

        # 使用温度缩放的Softmax进行特征权重归一化
        W *= (1 / self.scale)
        W = self.feature_scaling(W)

        # 将原始特征与对齐后的特征进行拼接
        pfeat_align = torch.cat([pfeat, W], dim=1)
        return pfeat_align


class ImprovedSparseMLP1D(nn.Module):
    """
    Improved MLP with skip connections and sparse regularization (Dropout).
    Combines depth enhancement with feature sparsity control.
    """

    def __init__(self, in_channels, hid_channels, out_channels, norm_layer=None, bias=False, num_mlp=3, sparsity=0.05):
        super(ImprovedSparseMLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.sparsity = sparsity  # 控制稀疏度
        self.layers = nn.ModuleList()

        # 输入到隐藏层的降维
        self.layers.append(nn.Sequential(
            nn.Conv1d(in_channels, hid_channels, 1, bias=bias),
            norm_layer(hid_channels),
            nn.ReLU(inplace=True)
        ))

        # 增加深度的隐藏层
        for _ in range(num_mlp - 2):
            self.layers.append(nn.Sequential(
                nn.Conv1d(hid_channels, hid_channels, 1, bias=bias),
                norm_layer(hid_channels),
                nn.ReLU(inplace=True)
            ))

        # 隐藏层到输出层
        self.layers.append(nn.Sequential(
            nn.Conv1d(hid_channels, out_channels, 1, bias=bias)
        ))

    def forward(self, x):
        residual = x
        for layer in self.layers[:-1]:
            x = layer(x) + residual  # 引入跳跃连接
        x = self.layers[-1](x)

        # 稀疏正则化，通过Dropout进行稀疏性控制
        x = F.dropout(x, p=self.sparsity, training=self.training)
        return x


class ImprovedMultiScaleFusion1D(nn.Module):
    def __init__(self, input_channels_list, output_channels, flag=False):
        super(ImprovedMultiScaleFusion1D, self).__init__()

        # 对通道数进行对齐的 1D 卷积操作
        self.conv1 = nn.Conv1d(input_channels_list[0], output_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(input_channels_list[1], output_channels, kernel_size=1, stride=1, padding=0)

        # 融合后的特征压缩
        self.conv_fusion = nn.Conv1d(2 * output_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU(inplace=True)

        self.flag = flag

    def forward(self, x, W):
        # 先将通道对齐为相同的尺寸
        x = self.conv1(x)  # [16, output_channels, 144]
        W = self.conv2(W)  # [16, output_channels, 144]

        # 在通道维度上拼接特征
        fused = torch.cat([x, W], dim=1)  # [16, 2 * output_channels, 144]

        # 使用1x1卷积进行通道压缩
        fused = self.conv_fusion(fused)
        fused = self.bn(fused)
        fused = self.relu(fused)

        if self.flag:
            # 如果flag为True，进行全局平均池化
            fused = fused.mean(-1)  # 对序列长度维度求均值，得到 [16, output_channels]

        return fused




## 融合策略

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    def __init__(self, input_channels_list, output_channels):
        super(MultiScaleFusion, self).__init__()

        # 通道对齐的 1x1 卷积
        self.conv1 = nn.Conv2d(input_channels_list[0], output_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(input_channels_list[1], output_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(input_channels_list[2], output_channels, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(input_channels_list[3], output_channels, kernel_size=1, stride=1, padding=0)

        # 上采样，将所有特征图的尺寸调整为 96x96
        self.upsample2 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)

        # 通道融合后的卷积
        self.conv_fusion = nn.Conv2d(output_channels * 4, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2, x3, x4):
        # 通道对齐
        x1 = self.conv1(x1)  # [16, output_channels, 96, 96]
        x2 = self.upsample2(self.conv2(x2))  # 将x2上采样到 [16, output_channels, 96, 96]
        x3 = self.upsample3(self.conv3(x3))  # 将x3上采样到 [16, output_channels, 96, 96]
        x4 = self.upsample4(self.conv4(x4))  # 将x4上采样到 [16, output_channels, 96, 96]

        # 在通道维度上拼接所有尺度的特征
        fused = torch.cat([x1, x2, x3, x4], dim=1)  # [16, output_channels*4, 96, 96]

        # 通过 3x3 卷积进一步融合特征
        fused = self.conv_fusion(fused)
        fused = self.bn(fused)
        fused = self.relu(fused)

        # 全局平均池化
        fused = F.adaptive_avg_pool2d(fused, (1, 1)).flatten(1)  # 得到 [16, output_channels]

        return fused
# 定义模型
input_channels_list = [96, 192, 384, 768]  # 对应 x1, x2, x3, x4 的通道数
output_channels = 512  # 假设我们要压缩到512通道
fusion_layer = MultiScaleFusion(input_channels_list, output_channels)

# 假设 x1, x2, x3, x4 的输入大小分别为 [16, 96, 96, 96], [16, 192, 48, 48], [16, 384, 24, 24], [16, 768, 12, 12]
x1 = torch.randn(16, 96, 96, 96)
x2 = torch.randn(16, 192, 48, 48)
x3 = torch.randn(16, 384, 24, 24)
x4 = torch.randn(16, 768, 12, 12)

# 前向传播，得到融合后的特征
fused_output = fusion_layer(x1, x2, x3, x4)
print(fused_output.shape)  # 输出应为 [16, 512]，用于分类任务

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        return x * avg_out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_out))


class DynamicFusion(nn.Module):
    def __init__(self, num_features, reduction=16):
        super(DynamicFusion, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction, num_features),
            nn.Sigmoid()
        )

    def forward(self, features):
        # 计算每个特征图的加权权重
        avg_pooled = [torch.mean(f, dim=[2, 3], keepdim=False) for f in features]  # 每个特征图的全局平均池化
        concatenated = torch.cat(avg_pooled, dim=1)  # 拼接所有特征图的全局平均池化结果
        weights = self.fc(concatenated).unsqueeze(-1).unsqueeze(-1)  # 计算动态权重

        # 对每个特征图进行加权
        weighted_features = [f * w for f, w in
                             zip(features, torch.split(weights, [f.size(1) for f in features], dim=1))]
        return weighted_features


class InnovativeMultiScaleFusion(nn.Module):
    def __init__(self, input_channels_list, output_channels):
        super(InnovativeMultiScaleFusion, self).__init__()

        # 通道对齐的 1x1 卷积
        self.conv1 = nn.Conv2d(input_channels_list[0], output_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(input_channels_list[1], output_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(input_channels_list[2], output_channels, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(input_channels_list[3], output_channels, kernel_size=1, stride=1, padding=0)

        # 上采样，将所有特征图的尺寸调整为 96x96
        self.upsample2 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)

        # 融合后的通道压缩卷积
        self.conv_fusion = nn.Conv2d(output_channels * 4, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

        # 通道注意力
        self.channel_attention = ChannelAttention(output_channels)
        # 空间注意力
        self.spatial_attention = SpatialAttention()
        # 动态融合
        self.dynamic_fusion = DynamicFusion(num_features=output_channels * 4)

    def forward(self, x1, x2, x3, x4):
        # 1. 通道对齐
        x1 = self.conv1(x1)  # [16, output_channels, 96, 96]
        x2 = self.upsample2(self.conv2(x2))  # [16, output_channels, 96, 96]
        x3 = self.upsample3(self.conv3(x3))  # [16, output_channels, 96, 96]
        x4 = self.upsample4(self.conv4(x4))  # [16, output_channels, 96, 96]

        # 2. 使用动态融合模块根据输入特征的内容自动调整每个特征图的权重
        features = self.dynamic_fusion([x1, x2, x3, x4])

        # 3. 在通道维度上拼接
        fused = torch.cat(features, dim=1)  # [16, output_channels*4, 96, 96]

        # 4. 融合后的特征通过3x3卷积进行通道压缩
        fused = self.conv_fusion(fused)
        fused = self.bn(fused)
        fused = self.relu(fused)

        # 5. 通过通道注意力机制调整特征的通道重要性
        fused = self.channel_attention(fused)

        # 6. 通过空间注意力机制调整特征的空间重要性
        fused = self.spatial_attention(fused)

        # 7. 全局平均池化用于分类
        fused = F.adaptive_avg_pool2d(fused, (1, 1)).flatten(1)  # 得到 [16, output_channels]

        return fused
# 定义模型
input_channels_list = [96, 192, 384, 768]  # 对应 x1, x2, x3, x4 的通道数
output_channels = 512  # 假设我们要压缩到512通道
fusion_layer = InnovativeMultiScaleFusion(input_channels_list, output_channels)

# 假设 x1, x2, x3, x4 的输入大小分别为 [16, 96, 96, 96], [16, 192,
class LightweightMultiScaleFusion(nn.Module):
    def __init__(self, input_channels_list, output_channels):
        super(LightweightMultiScaleFusion, self).__init__()

        # 使用 1x1 卷积对通道数进行对齐
        self.conv1 = nn.Conv2d(input_channels_list[0], output_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(input_channels_list[1], output_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(input_channels_list[2], output_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(input_channels_list[3], output_channels, kernel_size=1)

        # 上采样，将较小特征图调整为96x96
        self.upsample2 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)

    def forward(self, x1, x2, x3, x4):
        # 使用 1x1 卷积对齐通道数
        x1 = self.conv1(x1)
        x2 = self.upsample2(self.conv2(x2))
        x3 = self.upsample3(self.conv3(x3))
        x4 = self.upsample4(self.conv4(x4))

        # 逐像素平均池化融合，减少参数量
        fused = (x1 + x2 + x3 + x4) / 4.0

        # 全局平均池化
        fused = F.adaptive_avg_pool2d(fused, (1, 1)).flatten(1)

        return fused
input_channels_list = [96, 192, 384, 768]
output_channels = 512
fusion_layer = LightweightMultiScaleFusion(input_channels_list, output_channels)

x1 = torch.randn(16, 96, 96, 96)
x2 = torch.randn(16, 192, 48, 48)
x3 = torch.randn(16, 384, 24, 24)
x4 = torch.randn(16, 768, 12, 12)

fused_output = fusion_layer(x1, x2, x3, x4)
print(fused_output.shape)  # 输出应为 [16, 512]


class PyramidFusion(nn.Module):
    def __init__(self, input_channels_list, output_channels):
        super(PyramidFusion, self).__init__()

        # 对每个输入通道的特征图进行 1x1 卷积
        self.conv1 = nn.Conv2d(input_channels_list[0], output_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(input_channels_list[1], output_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(input_channels_list[2], output_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(input_channels_list[3], output_channels, kernel_size=1)

        # 对较大特征图进行下采样
        self.downsample2 = nn.MaxPool2d(kernel_size=2)
        self.downsample3 = nn.MaxPool2d(kernel_size=2)
        self.downsample4 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x1, x2, x3, x4):
        # 对齐通道
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        # 对较大的特征图逐步下采样
        x2_down = self.downsample2(x2)
        x3_down = self.downsample3(x3)
        x4_down = self.downsample4(x4)

        # 逐层融合
        fusion1 = x1 + x2_down  # 第一层融合
        fusion2 = fusion1 + x3_down  # 第二层融合
        fusion3 = fusion2 + x4_down  # 第三层融合

        # 全局平均池化
        fused = F.adaptive_avg_pool2d(fusion3, (1, 1)).flatten(1)

        return fused
input_channels_list = [96, 192, 384, 768]
output_channels = 512
fusion_layer = PyramidFusion(input_channels_list, output_channels)

x1 = torch.randn(16, 96, 96, 96)
x2 = torch.randn(16, 192, 48, 48)
x3 = torch.randn(16, 384, 24, 24)
x4 = torch.randn(16, 768, 12, 12)

fused_output = fusion_layer(x1, x2, x3, x4)
print(fused_output.shape)  # 输出应为 [16, 512]


class WeightedMultiScaleFusion(nn.Module):
    def __init__(self, input_channels_list, output_channels):
        super(WeightedMultiScaleFusion, self).__init__()

        # 通道对齐的 1x1 卷积
        self.conv1 = nn.Conv2d(input_channels_list[0], output_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(input_channels_list[1], output_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(input_channels_list[2], output_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(input_channels_list[3], output_channels, kernel_size=1)

        # 上采样
        self.upsample2 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)

        # 加权融合
        self.weight_fc = nn.Linear(4, 4, bias=False)

    def forward(self, x1, x2, x3, x4):
        # 通道对齐
        x1 = self.conv1(x1)
        x2 = self.upsample2(self.conv2(x2))
        x3 = self.upsample3(self.conv3(x3))
        x4 = self.upsample4(self.conv4(x4))

        # 计算每个特征图的全局平均池化，用于生成加权系数
        avg_pools = torch.stack([F.adaptive_avg_pool2d(x, 1).flatten(1) for x in [x1, x2, x3, x4]], dim=1)

        # 通过全连接层生成权重
        weights = F.softmax(self.weight_fc(avg_pools.mean(dim=-1)), dim=1).unsqueeze(-1).unsqueeze(-1)

        # 对每个特征图进行加权融合
        fused = x1 * weights[:, 0:1] + x2 * weights[:, 1:2] + x3 * weights[:, 2:3] + x4 * weights[:, 3:4]

        # 全局平均池化
        fused = F.adaptive_avg_pool2d(fused, (1, 1)).flatten(1)

        return fused
input_channels_list = [96, 192, 384, 768]
output_channels = 512
fusion_layer = WeightedMultiScaleFusion(input_channels_list, output_channels)

x1 = torch.randn(16, 96, 96, 96)
x2 = torch.randn(16, 192, 48, 48)
x3 = torch.randn(16, 384, 24, 24)
x4 = torch.randn(16, 768, 12, 12)

fused_output = fusion_layer(x1, x2, x3, x4)
print(fused_output.shape)  # 输出应为 [16, 512]

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model


class ConvNeXtTinyWithPyramidFusion(nn.Module):
    def __init__(self, num_classes, output_channels=512):
        super(ConvNeXtTinyWithPyramidFusion, self).__init__()

        # 使用预训练的ConvNeXt Tiny作为主干
        self.backbone = create_model('convnext_tiny', pretrained=True, features_only=True)

        # 1x1 卷积对齐通道数
        self.conv1 = nn.Conv2d(96, output_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(192, output_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(384, output_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(768, output_channels, kernel_size=1)

        # 下采样卷积，用于金字塔融合
        self.downsample1 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1)
        self.downsample3 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1)

        # 加权机制，自动为不同尺度的特征分配权重
        self.weight_fc = nn.Linear(output_channels * 4, 4, bias=False)

        # 最终融合后的分类器
        self.classifier = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        # 提取多尺度特征
        features = self.backbone(x)
        x1, x2, x3, x4 = features  # 分别是 [16, 96, 96, 96], [16, 192, 48, 48], [16, 384, 24, 24], [16, 768, 12, 12]

        # 通道对齐
        x1 = self.conv1(x1)  # [16, output_channels, 96, 96]in
        x2 = self.conv2(x2)  # [16, output_channels, 48, 48]
        x3 = self.conv3(x3)  # [16, output_channels, 24, 24]
        x4 = self.conv4(x4)  # [16, output_channels, 12, 12]

        # 使用金字塔逐层融合
        # 第一层融合：x4直接进入，x3下采样后融合
        fusion1 = x4 + self.downsample3(x3)
        # 第二层融合：fusion1和x2下采样后融合
        fusion2 = fusion1 + self.downsample2(x2)
        # 第三层融合：fusion2和x1下采样后融合
        fusion3 = fusion2 + self.downsample1(x1)

        # 使用全局平均池化计算每个特征的权重
        avg_pools = torch.cat([F.adaptive_avg_pool2d(fusion3, 1).flatten(1)], dim=1)

        # 通过全连接层生成权重
        weights = F.softmax(self.weight_fc(avg_pools), dim=1).unsqueeze(-1).unsqueeze(-1)

        # 加权融合
        final_fusion = fusion3 * weights[:, 0:1]

        # 全局平均池化
        out = F.adaptive_avg_pool2d(final_fusion, (1, 1)).flatten(1)

        # 分类输出
        return self.classifier(out)


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = input_dim ** -0.5  # 缩放因子

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        out = torch.matmul(attn_weights, v)
        return out + x  # 残差连接





class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False, num_heads=3, threshold=0.5):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        self.num_heads = num_heads
        self.threshold = threshold

        # 动态多层感知机部分
        self.add_block1 = nn.Sequential(
            nn.Linear(input_dim, num_bottleneck),
            nn.BatchNorm1d(num_bottleneck) if bnorm else nn.Identity(),
            nn.LeakyReLU(0.1) if relu else nn.Identity(),
            nn.Dropout(p=droprate) if droprate > 0 else nn.Identity()
        )

        self.add_block2 = nn.Sequential(
            nn.Linear(num_bottleneck, num_bottleneck),
            nn.BatchNorm1d(num_bottleneck),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=droprate)
        )

        # 用于动态选择层的门控机制
        self.gate = nn.Linear(num_bottleneck, 1)  # 用于判断是否使用add_block2

        # 多头分类器
        self.classifiers = nn.ModuleList([nn.Linear(num_bottleneck, class_num) for _ in range(num_heads)])
        for classifier in self.classifiers:
            classifier.apply(weights_init_classifier)

    def forward(self, x):
        # 首先经过add_block1
        x = self.add_block1(x)

        # 通过门控机制决定是否激活add_block2
        gate_value = torch.sigmoid(self.gate(x))
        if gate_value.mean() > self.threshold:
            x = self.add_block2(x)  # 只有在条件满足时，才进入更深的层次

        # 经过多头分类器
        outputs = [classifier(x) for classifier in self.classifiers]
        x = torch.mean(torch.stack(outputs), dim=0)  # 平均融合多头分类结果

        if self.training:
            if self.return_f:
                return x, gate_value
            else:
                return x
        else:
            return x



class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // reduction, bias=False)
        self.fc2 = nn.Linear(input_dim // reduction, input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=-1, keepdim=True)  # 全局平均池化
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y  # 加权特征

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False, sparsity_ratio=0.5):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        self.sparsity_ratio = sparsity_ratio

        # 稀疏连接层
        self.add_block = nn.Sequential(
            nn.Linear(input_dim, num_bottleneck),
            nn.BatchNorm1d(num_bottleneck) if bnorm else nn.Identity(),
            nn.LeakyReLU(0.1) if relu else nn.Identity(),
            nn.Dropout(p=droprate) if droprate > 0 else nn.Identity()
        )

        # SE注意力模块
        self.se = SEBlock(num_bottleneck)

        # 分类层
        self.classifier = nn.Linear(num_bottleneck, class_num)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        # 稀疏化处理
        mask = (torch.rand_like(x) > self.sparsity_ratio).float()
        x = x * mask.to(x.device)

        # 经过稀疏连接和SE注意力
        x = self.add_block(x)
        x = self.se(x)  # 注意力加权

        # 分类
        x = self.classifier(x)

        if self.training:
            if self.return_f:
                return x, mask  # 返回稀疏化的mask以便分析
            else:
                return x
        else:
            return x



import math

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(GhostModule, self).__init__()
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Linear(in_channels, init_channels, bias=False)
        self.cheap_operation = nn.Linear(init_channels, new_channels, bias=False)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)  # 生成 ghost 特征

class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        gamma = (self.block_size ** 2) / (x.size(-1) ** 2)
        mask = (torch.rand_like(x) < gamma).float().to(x.device)
        return x * mask

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f

        # Ghost 模块
        self.ghost_module = GhostModule(input_dim, num_bottleneck)

        # DropBlock 正则化
        self.dropblock = DropBlock(block_size=3)

        # 分类层
        self.classifier = nn.Linear(num_bottleneck, class_num)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        # 通过 Ghost 模块生成特征
        x = self.ghost_module(x)

        # DropBlock 正则化
        x = self.dropblock(x)

        # 分类
        x = self.classifier(x)

        if self.training:
            if self.return_f:
                return x, x  # 返回经过 Ghost 模块处理后的特征和最终结果
            else:
                return x
        else:
            return x


import torch.nn.functional as F

class DeformableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DeformableConv1d, self).__init__()
        self.offset_conv = nn.Conv1d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=padding)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = F.conv1d(x, self.conv.weight, self.conv.bias, padding=self.conv.padding, dilation=self.conv.dilation, stride=self.conv.stride)
        return x

class ClassBlockWithDeformableConv(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlockWithDeformableConv, self).__init__()
        self.return_f = return_f

        # Deformable Convolution 层
        self.deform_conv = DeformableConv1d(input_dim, num_bottleneck)

        # DropBlock 正则化
        self.dropblock = DropBlock(block_size=3)

        # 分类层
        self.classifier = nn.Linear(num_bottleneck, class_num)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        # 通过 Deformable Conv 层处理
        x = self.deform_conv(x)

        # DropBlock 正则化
        x = self.dropblock(x)

        # 分类
        x = self.classifier(x)

        if self.training:
            if self.return_f:
                return x, x  # 返回经过 Ghost 模块处理后的特征和最终结果
            else:
                return x
        else:
            return x



class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(SelfSupervisedClassBlock, self).__init__()
        self.return_f = return_f

        # 使用 BYOL 风格的自监督 MLP
        self.projector = nn.Sequential(
            nn.Linear(input_dim, num_bottleneck),
            nn.BatchNorm1d(num_bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(num_bottleneck, num_bottleneck)
        )

        # 分类层
        self.classifier = nn.Linear(num_bottleneck, class_num)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        # 使用自监督投影
        x = self.projector(x)

        # 分类
        x = self.classifier(x)

        if self.training:
            if self.return_f:
                return x, x  # 返回特征
            else:
                return x
        else:
            return x



import torch.fft

class FrequencyClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(FrequencyClassBlock, self).__init__()
        self.return_f = return_f

        # 频域特征提取
        self.freq_transform = lambda x: torch.fft.fft(x, dim=-1)

        # 分类层
        self.classifier = nn.Linear(input_dim, class_num)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        # 频域变换
        x_freq = self.freq_transform(x)

        # 频域信息分类
        x = self.classifier(x_freq.real)

        if self.training:
            if self.return_f:
                return x, x_freq.real  # 返回频域特征
            else:
                return x
        else:
            return x



class FPNClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(FPNClassBlock, self).__init__()
        self.return_f = return_f

        # 特征金字塔层
        self.fpn1 = nn.Conv1d(input_dim, num_bottleneck // 2, kernel_size=1)
        self.fpn2 = nn.Conv1d(num_bottleneck // 2, num_bottleneck, kernel_size=1)

        # 分类层
        self.classifier = nn.Linear(num_bottleneck, class_num)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        # 特征金字塔提取
        x = self.fpn1(x.unsqueeze(-1))  # 先降维处理
        x = self.fpn2(x.squeeze(-1))

        # 分类
        x = self.classifier(x)

        if self.training:
            if self.return_f:
                return x, x  # 返回特征
            else:
                return x
        else:
            return x





class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f

        # 多尺度融合模块
        self.multi_scale_block = MultiScaleFusion(input_dim, num_bottleneck)

        add_block = []
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck * 3)]  # 融合后的维度增加，乘以3是因为我们用了3个卷积
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)

        classifier = []
        classifier += [nn.Linear(num_bottleneck * 3, class_num)]  # 同样，线性层输入维度也要乘以3
        classifier = nn.Sequential(*classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        # 多尺度特征提取
        x = self.multi_scale_block(x)
        x = self.add_block(x)

        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


class GlobalLocalFusion(nn.Module):
    def __init__(self, input_dim, num_bottleneck):
        super(GlobalLocalFusion, self).__init__()
        # 局部特征通过卷积提取
        self.local_conv = nn.Conv1d(input_dim, num_bottleneck, kernel_size=3, padding=1)
        # 全局特征通过全局平均池化提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(input_dim, num_bottleneck)

    def forward(self, x):
        # 提取局部特征
        local_features = self.local_conv(x)

        # 提取全局特征
        global_features = self.global_pool(x)
        global_features = global_features.view(global_features.size(0), -1)  # Flatten
        global_features = self.fc(global_features)
        global_features = global_features.unsqueeze(2)  # 重新调整形状

        # 局部和全局特征融合
        out = local_features + global_features  # 或使用 torch.cat([local_features, global_features], dim=1)
        return out


class MultiScaleFusionWithAtrous(nn.Module):
    def __init__(self, input_dim, num_bottleneck):
        super(MultiScaleFusionWithAtrous, self).__init__()
        # 使用不同的空洞率
        self.conv_dilated1 = nn.Conv1d(input_dim, num_bottleneck, kernel_size=3, dilation=1, padding=1)
        self.conv_dilated2 = nn.Conv1d(input_dim, num_bottleneck, kernel_size=3, dilation=2, padding=2)
        self.conv_dilated3 = nn.Conv1d(input_dim, num_bottleneck, kernel_size=3, dilation=3, padding=3)

    def forward(self, x):
        x1 = self.conv_dilated1(x)
        x2 = self.conv_dilated2(x)
        x3 = self.conv_dilated3(x)

        # 将不同空洞率下的特征拼接
        out = torch.cat([x1, x2, x3], dim=1)
        return out


class MultiScaleFusion(nn.Module):
    def __init__(self, input_dim, num_bottleneck):
        super(MultiScaleFusion, self).__init__()
        # 不同卷积核的分支
        self.conv3x3 = nn.Conv1d(input_dim, num_bottleneck, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv1d(input_dim, num_bottleneck, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv1d(input_dim, num_bottleneck, kernel_size=7, padding=3)

    def forward(self, x):
        # 不同卷积核提取特征
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        x7 = self.conv7x7(x)

        # 将不同尺度的特征拼接
        out = torch.cat([x3, x5, x7], dim=1)
        return out



class MultiScaleFusionDilated(nn.Module):
    def __init__(self, input_dim, num_bottleneck, groups=4):
        super(MultiScaleFusionDilated, self).__init__()
        # 使用膨胀卷积来扩展感受野
        self.dilated1 = nn.Conv1d(1, num_bottleneck // groups, kernel_size=3, dilation=1, padding=1)
        self.dilated2 = nn.Conv1d(1, num_bottleneck // groups, kernel_size=3, dilation=2, padding=2)
        self.dilated3 = nn.Conv1d(1, num_bottleneck // groups, kernel_size=3, dilation=3, padding=3)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        # 使用不同膨胀率的卷积来提取不同尺度的特征
        x1 = self.dilated1(x)
        x2 = self.dilated2(x)
        x3 = self.dilated3(x)

        # 拼接不同尺度的特征，并通过全局池化汇聚
        out = torch.cat([x1, x2, x3], dim=1)
        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)  # (batch_size, num_bottleneck)
        return out


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False, groups=4):
        super(ClassBlock, self).__init__()
        self.return_f = return_f

        # 使用多尺度卷积模块进行特征提取
        self.multi_scale_block = MultiScaleFusionDilated(input_dim, num_bottleneck, groups=groups)

        add_block = []
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]  # Batch Normalization
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]  # Dropout 正则化
        self.add_block = nn.Sequential(*add_block)

        # 分类层
        self.classifier = nn.Linear(num_bottleneck, class_num)

    def forward(self, x):
        x = self.multi_scale_block(x)  # 提取多尺度特征

        x = self.add_block(x)

        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


class DSA_loss(nn.Module):
    """
    改进的 DSA_loss 损失函数，支持 MSE（带有 MLP 特征对齐）和 InfoNCE 损失。
    """

    def __init__(self, loss_function, hidden_dim=512, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function  # 默认 CrossEntropy
        self.device = device

        # 可学习的logit scale，初始值为1/0.07
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 控制是否使用InfoNCE
        self.if_infoNCE = False

        # MLP层用于特征空间对齐
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

    def cosine_similarity_loss(self, pred, target):
        """
        使用余弦相似度来替代简单的 MSE 计算。
        Args:
            pred (Tensor): 预测的特征 (batch_size, feature_dim)
            target (Tensor): 目标特征 (batch_size, feature_dim)
        """
        pred_norm = F.normalize(pred, dim=1)
        target_norm = F.normalize(target, dim=1)
        cosine_sim = torch.sum(pred_norm * target_norm, dim=1)
        return 1 - cosine_sim.mean()

    def forward(self, image_features1, image_features2):
        # 调整特征形状并应用 MLP 对齐
        b, c, n = image_features1.shape
        feat1 = image_features1.transpose(2, 1).reshape(b, c * n)  # (batch_size, c*n)
        feat2 = image_features2.transpose(2, 1).reshape(b, c * n)

        # 使用 MLP 特征对齐
        feat1 = self.mlp(feat1)
        feat2 = self.mlp(feat2)

        if not self.if_infoNCE:
            # 计算余弦相似度损失
            loss = self.cosine_similarity_loss(feat1, feat2)

        else:
            # 使用 InfoNCE 损失
            feat1 = F.normalize(feat1, dim=-1)
            feat2 = F.normalize(feat2, dim=-1)

            logits_per_image1 = self.logit_scale.exp() * feat1 @ feat2.T
            logits_per_image2 = logits_per_image1.T

            labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

            # 计算 InfoNCE 损失
            loss_infoNCE = (self.loss_function(logits_per_image1, labels) +
                            self.loss_function(logits_per_image2, labels)) / 2

            # 自适应 InfoNCE 损失权重
            loss = 0.5 * loss_infoNCE + 0.5 * self.cosine_similarity_loss(feat1, feat2)

        return loss