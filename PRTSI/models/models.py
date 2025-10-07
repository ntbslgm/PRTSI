
import torch
from torch import nn
from einops import rearrange
import lightning as L
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
import torch
from torch import nn
from einops import rearrange
import math
import torch.nn.functional as F
import lightning as L
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath


def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]

## Feature Extractor
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),  # kernel_size=5,有64个滤波器
            # configs.mid_channels是卷积核的数量，64个，即输出特征的通道数
            nn.BatchNorm1d(configs.mid_channels),  # 批归一化，其中参数为此层所归一化的特征通道数
            nn.ReLU(),  # 激活函数
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)  # 正则化技术，通过随机将一部分神经元的输出置为0来防止模型过拟合，其中configs.dropout是丢弃率即dropout的比例
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.aap = nn.AdaptiveAvgPool1d(configs.features_len)
        # 这是自适应平均池化层，可以根据输出目标大小自动调整输入的大小
        # configs.features_len是输出特征的长度
        # 在CNN类的构造函数中，它的作用是定义网络的结构，使得在网络的前向传播中输入经过卷积层处理后的特征图能够被自适应平均池化，从而得到统一大小的输出。

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # 将自适应平均池化层的输出展平
        x_flat = self.aap(x).view(x.shape[0], -1)
        # 通过tAPE生成位置编码并加到展平的特征上
        return x_flat, x

## Feature Extractor
class CNN_tAPE_back(nn.Module):
    def __init__(self, configs):
        super(CNN_tAPE_back, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),  # kernel_size=5,有64个滤波器
            # configs.mid_channels是卷积核的数量，64个，即输出特征的通道数
            nn.BatchNorm1d(configs.mid_channels),  # 批归一化，其中参数为此层所归一化的特征通道数
            nn.ReLU(),  # 激活函数
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)  # 正则化技术，通过随机将一部分神经元的输出置为0来防止模型过拟合，其中configs.dropout是丢弃率即dropout的比例
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.aap = nn.AdaptiveAvgPool1d(configs.features_len)
        # 这是自适应平均池化层，可以根据输出目标大小自动调整输入的大小
        # configs.features_len是输出特征的长度
        # 在CNN类的构造函数中，它的作用是定义网络的结构，使得在网络的前向传播中输入经过卷积层处理后的特征图能够被自适应平均池化，从而得到统一大小的输出。
        self.tAPE = tAPE(d_model=configs.input_channels, max_len=configs.sequence_len, dropout=configs.dropout)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # 将自适应平均池化层的输出展平
        x_flat = self.aap(x).view(x.shape[0], -1)
        # 通过tAPE生成位置编码并加到展平的特征上
        x_flat = self.tAPE(x_flat.unsqueeze(1)).squeeze(1)
        return x_flat, x

## Feature Extractor
class CNN_tAPE_front(nn.Module):
    def __init__(self, configs):
        super(CNN_tAPE_front, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),  # kernel_size=5,有64个滤波器
            # configs.mid_channels是卷积核的数量，64个，即输出特征的通道数
            nn.BatchNorm1d(configs.mid_channels),  # 批归一化，其中参数为此层所归一化的特征通道数
            nn.ReLU(),  # 激活函数
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)  # 正则化技术，通过随机将一部分神经元的输出置为0来防止模型过拟合，其中configs.dropout是丢弃率即dropout的比例
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.aap = nn.AdaptiveAvgPool1d(configs.features_len)
        # 这是自适应平均池化层，可以根据输出目标大小自动调整输入的大小
        # configs.features_len是输出特征的长度
        # 在CNN类的构造函数中，它的作用是定义网络的结构，使得在网络的前向传播中输入经过卷积层处理后的特征图能够被自适应平均池化，从而得到统一大小的输出。
        self.tAPE = tAPE(d_model=configs.input_channels, max_len=configs.sequence_len, dropout=configs.dropout)

    def forward(self, x_in):
        x = self.tAPE(x_in)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.aap(x)
        # 将自适应平均池化层的输出展平
        x_flat = x.view(x.shape[0], -1)
        return x_flat, x

##  Classifier
class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions


# Temporal Imputer
class Temporal_Imputer1(nn.Module):
    def __init__(self, configs):
        super(Temporal_Imputer1, self).__init__()
        self.seq_length = configs.features_len
        self.num_channels = configs.final_out_channels
        self.hid_dim = configs.AR_hid_dim
        # input size: batch_size, 128 channel, 18 seq_length
        self.rnn = nn.LSTM(input_size=self.num_channels, hidden_size=self.hid_dim)

    def forward(self, x):  # x:(32,128,18)
        x = x.view(x.size(0), -1, self.num_channels)  # x:(32,18,128)
        out, (h, c) = self.rnn(x)  # out:(32,18,128) h:(1,18,128) c:(1,18,128)
        out = out.view(x.size(0), self.num_channels, -1)
        # take the last time step
        return out  # out:(32,18,128)


# temporal masking
def masking(x, num_splits=8, num_masked=4):
    # x:(32,9,128)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_splits)  # patches:(32,9,8,16)
    masked_patches = patches.clone()  # masked_patches:(32,9,8,16)
    # rand_indices:(9,8) [[1, 5, 2, 4, 7, 6, 3, 0]...]
    rand_indices = torch.rand(x.shape[1], num_splits).argsort(dim=-1)
    selected_indices = rand_indices[:, :num_masked]  # selected_indices:(9,1) num_masked:1
    masks = []
    for i in range(masked_patches.shape[1]):  # masked_patches.shape[1]:9
        # masked_patches[:, i, (selected_indices[i, :]), :] :(32,1,16)
        masks.append(masked_patches[:, i, (selected_indices[i, :]), :])
        masked_patches[:, i, (selected_indices[i, :]), :] = 0
        # orig_patches[:, i, (selected_indices[i, :]), :] =
    # masks:{list:9}  each list:(32,1,16)
    mask = rearrange(torch.stack(masks), 'b a p l -> a b (p l)')
    # mask:(32,9,16) masked_x:(32,9,128)
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=num_splits)

    return masked_x, mask


# ====================================== add
class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        # x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        # x = x.transpose(1, 2)
        return x

class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        threshold = torch.quantile(normalized_energy, self.threshold_param)
        dominant_frequencies = normalized_energy > threshold

        # Initialize adaptive mask
        adaptive_mask = torch.zeros_like(x_fft, device=x_fft.device)
        adaptive_mask[dominant_frequencies] = 1

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        # Adaptive High Frequency Mask (no need for dimensional adjustments)
        freq_mask = self.create_adaptive_high_freq_mask(x_fft)
        x_masked = x_fft * freq_mask.to(x.device)

        weight_high = torch.view_as_complex(self.complex_weight_high)
        x_weighted2 = x_masked * weight_high

        x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x

# 将输入序列分割成多个补丁（patch），然后将这些补丁通过卷积层映射到固定维度的嵌入表示中。
class PatchEmbed1(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        # seq_len:(128) stride:4    num_patches = int((128 - 8) / 4 + 1) = 31
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        # in_chans: 9 embed_dim: 128 patch_size: 8 stride = 4
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        # self.proj(x)  ==== x:(16,9,128)---> x:(16,128,31)
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out # x:(16,9,128) x_out:(16,31,128)


class PatchEmbed(L.LightningModule):
    # ====todo
    #  HAR patch_size=15 in_chans = 18    embed_dim=128
    #  FD patch_size=94 in_chans = 109   embed_dim=128
    def __init__(self, seq_len=3000, patch_size=94, in_chans=109, embed_dim=128):
        super().__init__()
        stride = patch_size // 2
        #  ======== todo
        # HAR seq_len:(128) stride:4    num_patches = int((128 - 8) / 4 + 1) = 31
        # EEG change  num_patches = int((seq_len - patch_size) / stride + 1) ===>  num_patches = int((3000 - patch_size) / stride + 1)
        # FD change  num_patches = int((seq_len - patch_size) / stride + 1) ===>  num_patches = int((5120 - patch_size) / stride + 1)
        num_patches = int((5120- patch_size) / stride + 1)
        self.num_patches = num_patches
        # HAR FD
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.pos_embed = nn.Parameter(torch.zeros
                                      (1, num_patches, 128), requires_grad=True)
        # # EEG
        # self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=7)
        # self.pos_embed = nn.Parameter(torch.zeros
        #                               (1, num_patches, 8), requires_grad=True)
        self.pos_drop = nn.Dropout(p=0.15)
    def forward(self, x):
        x=x.transpose(1,2)
        y=self.pos_embed
        x_out = self.proj(x)
        x_out = x_out.flatten(2)
        x_out = x_out.transpose(1, 2)
        x_out = x_out + self.pos_embed
        x_patched = self.pos_drop(x_out)

        x_patched=x_patched.transpose(1,2)
        last_column = x_patched[:, :, -1:]
        new_column=torch.zeros_like(last_column).expand(-1,-1,1)
        new_column.copy_(last_column)
        x_expanded = torch.cat((x_patched,new_column),dim=2)
        x_expanded = torch.cat((x_expanded, new_column), dim=2)
        # x_expanded = torch.cat((x_expanded, new_column), dim=2)#FD 2次 HAR 1次 EEG 3次
        return x_expanded

class Temporal_Imputer(L.LightningModule):  # HAR(32,128,18) dim = 18
    def __init__(self, configs, dim=18, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super(Temporal_Imputer, self).__init__()
        self.norm1 = norm_layer(configs.dim)
        self.asb = Adaptive_Spectral_Block(128) #HAR FD 128 EEG 8
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(configs.dim)
        mlp_hidden_dim = int(configs.dim * mlp_ratio)
        # self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.icb = ICB(in_features=configs.in_features, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # Check if both ASB and ICB are true
        # if args.ICB and args.ASB:
        # x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # x = self.norm1(x)
        x = self.asb(x)  # 32 749 128

        # x = self.norm2
        x = self.icb(x)
        # # If only ICB is true
        # elif args.ICB:
        #     x = x + self.drop_path(self.icb(self.norm2(x)))
        # # If only ASB is true
        # elif args.ASB:
        #     x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x

class tAPE(nn.Module):
    def __init__(self, d_model, dropout, max_len, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Create positional encoding matrix `pe` with dimensions [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Use sin and cos based on availability of indices, handling odd and even dimensions
        pe[:, ::2] = torch.sin(position * div_term[: (d_model + 1) // 2])
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        # Apply scale factor and register as buffer
        pe = scale_factor * pe.unsqueeze(0)
        pe=pe.permute(0,2,1)# Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):

        #--------FD EEG------------
        #获取输入序列的实际长度
        max_len_actual = x.size(1)
        # 获取位置编码的实际长度，并确保它不会超过输入序列的长度
        pe = self.pe[:, :, :max_len_actual]
        # 将位置编码加到输入上，并应用dropout
        # 由于pe的第二维度与x的第二维度匹配，我们可以直接在第二维度上进行相加操作
        x = x + pe.to(x.device)
        return self.dropout(x)
        #--------FD EEG------------


        #--------HAR HHAR WISDM---------------
        #Ensure positional encoding does not exceed input sequence length
        # max_len_actual = min(x.size(1), self.pe.size(1))
        # x = x + self.pe[:, :max_len_actual, :].to(x.device)
        # return self.dropout(x)
        #--------HAR---------------

# ====================================== end
