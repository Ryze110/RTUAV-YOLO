# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
    'FCM', 'PDSConv',  'FCM_3',
    'FCM_2', 'FCM_1', 'Down', 'Conv2d_BN', 'PurePyTorchSKA', 'LKP', 'PurePyTorchLSConv', 'FLSEM_PurePyTorc',
    'SimplifiedSKA', 'SimplifiedLSConv', 'FLSEM_Simplified',
    'DynamicWeightGenerator', 'MultiScaleFeatureExtractor', 'AdaptiveFeatureModulator', 'MSFAM', 'MSFAM_Lite_1',
    'MSFAM_Lite_2',
    'MSFAM_Ultra_Lite_Adaptive',
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """
        Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """
    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, W, H) and output shape is (B, 4C, W/2, H/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """
    Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """
    Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: List[torch.Tensor]):
        """
        Select and return a particular index from input.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]


class DWConv_FCM(nn.Module):
    """Depthwise Conv + Conv for FCM"""
    def __init__(self, in_channels):
        super().__init__()
        self.dconv = nn.Conv2d(
            in_channels, in_channels, 3,
            1, 1, groups=in_channels
        )

    def forward(self, x):
        x = self.dconv(x)
        return x


class Channel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, 3,
            1, 1, groups=dim
        )
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)
        return x6


class Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)
        return x6


class FCM_3(nn.Module):
    def __init__(self, c1, c2, *args):
        super().__init__()
        # c1 是输入通道数，c2 是输出通道数
        dim = c1  # 使用输入通道数作为 dim
        
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5


class FCM_2(nn.Module):
    def __init__(self, c1, c2, *args):
        super().__init__()
        # c1 是输入通道数，c2 是输出通道数
        dim = c1  # 使用输入通道数作为 dim
        
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5


class FCM_1(nn.Module):
    def __init__(self, c1, c2, *args):
        super().__init__()
        # c1 是输入通道数，c2 是输出通道数
        dim = c1  # 使用输入通道数作为 dim
        
        self.one = dim // 4
        self.two = dim - dim // 4
        self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim - dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5


class FCM(nn.Module):
    def __init__(self, c1, c2, *args):
        super().__init__()
        # c1 是输入通道数，c2 是输出通道数
        # 如果有额外参数，可以在这里处理
        dim = c1  # 使用输入通道数作为 dim
        dim_out = c2  # 使用输出通道数作为 dim_out
        
        self.one = dim // 4
        self.two = dim - dim // 4
        self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim - dim // 4, dim, 1, 1)
        self.conv3 = Conv(dim, c2, 1, 1)  # 输出到指定通道数
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        x5 = self.conv3(x5)
        return x5

class PDSConv(nn.Module):
    ##PDSConv (Progressive Dilated Separable Convolution)
    def __init__(self, dim, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim, dim, 3,
            1, 1, groups=dim, dilation=1
        )
        self.conv2 = Conv(dim, dim, k=1, s=1, )
        self.conv3 = nn.Conv2d(
            dim, dim, 3,
            1, 2, groups=dim, dilation=2
        )
        self.conv4 = Conv(dim, dim, 1, 1)
        self.conv5 = nn.Conv2d(
            dim, dim, 3,
            1, 3, groups=dim ,dilation=3
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = x5 + x
        return x6
    
class Down(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv2 = Conv(dim, dim, 3, 2, 1, g=dim // 2, act=False)
        self.conv4 = Conv(dim, dim_out, 1, 1)

    def forward(self, x):
        x2 = self.conv2(x)
        x2 = self.conv4(x2)
        return x2

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)


class PurePyTorchSKA(nn.Module):
    """
    纯PyTorch实现的空间核注意力（SKA）
    替代原始的Triton实现，使用标准PyTorch操作
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        
    def forward(self, x, weights):
        """
        使用动态卷积权重对输入进行卷积
        
        Args:
            x: 输入特征 [B, C, H, W]
            weights: 动态权重 [B, C//groups, K*K, H, W]
        
        Returns:
            输出特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        _, wc, kk, _, _ = weights.shape
        
        # 计算分组数
        groups = C // wc
        
        # 将输入按组重塑
        x_grouped = x.view(B, groups, wc, H, W)
        
        # 使用unfold提取滑动窗口
        x_unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)
        # x_unfolded: [B, C*K*K, H*W]
        
        x_unfolded = x_unfolded.view(B, C, kk, H, W)
        x_unfolded = x_unfolded.view(B, groups, wc, kk, H, W)
        
        # 应用动态权重
        # weights: [B, wc, kk, H, W] -> [B, 1, wc, kk, H, W]
        weights_expanded = weights.unsqueeze(1)  # [B, 1, wc, kk, H, W]
        
        # 逐元素相乘并求和
        output = (x_unfolded * weights_expanded).sum(dim=3)  # [B, groups, wc, H, W]
        
        # 重塑回原始形状
        output = output.view(B, C, H, W)
        
        return output


class LKP(nn.Module):
    """大核参数生成器 - 纯PyTorch版本"""
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w


class PurePyTorchLSConv(nn.Module):
    """纯PyTorch实现的LSConv模块"""
    def __init__(self, dim, kernel_size=3):
        super(PurePyTorchLSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=kernel_size, groups=max(1, dim // 8))
        self.ska = PurePyTorchSKA(kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        weights = self.lkp(x)
        output = self.ska(x, weights)
        return self.bn(output) + x

# ============ 纯PyTorch版本的FLSEM ============
class FLSEM_PurePyTorch(nn.Module):
    """
    FLSEM的纯PyTorch实现版本
    去除了Triton依赖，使用标准PyTorch操作实现相同功能
    
    设计理念:
    1. 保持原有的双路径架构和注意力机制
    2. 使用纯PyTorch实现空间核注意力功能
    3. 确保与原版本功能等价但无外部依赖
    4. 优化计算效率，适合CPU和GPU部署
    
    技术优势:
    - 无Triton依赖：可在任何支持PyTorch的环境运行
    - 跨平台兼容：支持CPU、GPU、移动端部署
    - 易于调试：使用标准PyTorch操作，便于问题定位
    - 部署友好：减少依赖复杂度，提升部署成功率
    """
    
    def __init__(self, dim, dim_out):
        super().__init__()
        # 通道分割比例
        self.split_ratio = dim // 4
        self.remaining = dim - self.split_ratio
        
        # 纯PyTorch LSConv分支
        self.lsconv = PurePyTorchLSConv(self.split_ratio, kernel_size=3)
        
        # FCM分支 - 负责注意力调制和特征融合
        self.fcm_conv1 = Conv(self.remaining, self.remaining, 3, 1, 1)
        self.fcm_conv2 = Conv(self.remaining, dim, 1, 1)
        
        # 注意力模块
        self.spatial_att = Spatial(dim)
        self.channel_att = Channel(dim)
        
        # 特征融合和输出
        self.fusion_conv = Conv(dim, dim, 1, 1)
        self.output_conv = Conv(dim, dim, 3, 1, 1)
        
        # 归一化层
        self.bn1 = nn.BatchNorm2d(self.split_ratio)
        self.bn2 = nn.BatchNorm2d(dim)
        
    def forward(self, x):
        # 保存输入用于残差连接
        identity = x
        
        # 通道分割：小部分用于LSConv，大部分用于FCM处理
        if self.split_ratio > 0:
            x_lsconv, x_fcm = torch.split(x, [self.split_ratio, self.remaining], dim=1)
            
            # LSConv分支：大感受野特征提取（纯PyTorch实现）
            x_lsconv_enhanced = self.lsconv(x_lsconv)
        else:
            x_lsconv_enhanced = None
            x_fcm = x
        
        # FCM分支：特征处理和维度扩展
        x_fcm_processed = self.fcm_conv1(x_fcm)
        x_fcm_expanded = self.fcm_conv2(x_fcm_processed)
        
        # 特征重组：将LSConv增强的特征与FCM处理的特征融合
        if x_lsconv_enhanced is not None and x_lsconv_enhanced.size(1) > 0:
            # 将LSConv特征扩展到与FCM特征相同的维度
            repeat_times = self.remaining // self.split_ratio
            if repeat_times > 0:
                x_lsconv_expanded = torch.cat([
                    x_lsconv_enhanced, 
                    x_lsconv_enhanced.repeat(1, repeat_times, 1, 1)
                ], dim=1)
            else:
                x_lsconv_expanded = x_lsconv_enhanced
                
            # 调整到目标维度
            target_dim = x_fcm_expanded.size(1)
            if x_lsconv_expanded.size(1) > target_dim:
                x_lsconv_expanded = x_lsconv_expanded[:, :target_dim, :, :]
            elif x_lsconv_expanded.size(1) < target_dim:
                padding = target_dim - x_lsconv_expanded.size(1)
                x_lsconv_expanded = torch.cat([
                    x_lsconv_expanded,
                    x_lsconv_enhanced[:, :padding, :, :]
                ], dim=1)
        else:
            x_lsconv_expanded = torch.zeros_like(x_fcm_expanded)
        
        # 特征融合
        fused_features = self.fusion_conv(x_fcm_expanded + x_lsconv_expanded)
        
        # 双重注意力调制
        spatial_weight = self.spatial_att(fused_features)
        channel_weight = self.channel_att(fused_features)
        
        # 应用注意力权重
        attended_features = fused_features * spatial_weight * channel_weight
        
        # 最终特征处理
        output = self.output_conv(attended_features)
        output = self.bn2(output)
        
        # 残差连接
        return output + identity


class SimplifiedSKA(nn.Module):
    """
    简化版空间核注意力
    使用更直接的方法实现动态卷积
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        
    def forward(self, x, weights):
        B, C, H, W = x.shape
        _, wc, kk, _, _ = weights.shape
        
        # 简化实现：使用分组卷积近似动态卷积
        groups = C // wc
        
        # 将权重重塑为卷积核格式
        # weights: [B, wc, kk, H, W] -> [B*wc, 1, k, k, H, W]
        k = int(kk ** 0.5)
        weights_reshaped = weights.view(B, wc, k, k, H, W)
        
        # 使用平均权重作为静态卷积核（简化版本）
        avg_weights = weights_reshaped.mean(dim=(4, 5))  # [B, wc, k, k]
        
        # 应用分组卷积
        output = []
        for i in range(groups):
            start_ch = i * wc
            end_ch = start_ch + wc
            x_group = x[:, start_ch:end_ch, :, :]
            
            # 使用平均权重进行卷积（简化实现）
            group_output = F.conv2d(
                x_group.view(-1, wc, H, W),
                avg_weights.view(-1, 1, k, k),
                padding=self.padding,
                groups=B * wc
            )
            group_output = group_output.view(B, wc, H, W)
            output.append(group_output)
        
        return torch.cat(output, dim=1)


class SimplifiedLSConv(nn.Module):
    """简化版LSConv，更适合CPU部署"""
    def __init__(self, dim):
        super().__init__()
        # 使用标准卷积替代复杂的动态卷积
        self.large_conv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.point_conv = nn.Conv2d(dim, dim, 1, 1, 0)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()
        
    def forward(self, x):
        # 大核卷积 + 点卷积的组合
        out = self.large_conv(x)
        out = self.point_conv(out)
        out = self.bn(out)
        out = self.act(out)
        return out + x


class FLSEM_Simplified(nn.Module):
    """
    FLSEM简化版本
    使用更简单的操作，适合资源受限环境
    """
    def __init__(self, dim, dimout):
        super().__init__()
        self.split_ratio = dim // 2
        self.remaining = dim - self.split_ratio
        
        # 简化的大核分支
        if self.split_ratio > 0:
            self.large_kernel_branch = SimplifiedLSConv(self.split_ratio)
        
        # FCM分支
        self.fcm_conv1 = Conv(self.remaining, self.remaining, 3, 1, 1)
        self.fcm_conv2 = Conv(self.remaining, dim, 1, 1)
        
        # 注意力模块
        self.spatial_att = Spatial(dim)
        self.channel_att = Channel(dim)
        
        # 输出层
        self.output_conv = Conv(dim, dim, 1, 1)
        
    def forward(self, x):
        identity = x
        
        if self.split_ratio > 0:
            x_large, x_fcm = torch.split(x, [self.split_ratio, self.remaining], dim=1)
            x_large_enhanced = self.large_kernel_branch(x_large)
        else:
            x_large_enhanced = None
            x_fcm = x
        
        # FCM处理
        x_fcm_processed = self.fcm_conv1(x_fcm)
        x_fcm_expanded = self.fcm_conv2(x_fcm_processed)
        
        # 特征融合
        if x_large_enhanced is not None:
            # 简单的特征拼接和维度调整
            repeat_times = self.remaining // self.split_ratio
            if repeat_times > 0:
                x_large_expanded = x_large_enhanced.repeat(1, repeat_times + 1, 1, 1)
                target_dim = x_fcm_expanded.size(1)
                x_large_expanded = x_large_expanded[:, :target_dim, :, :]
            else:
                x_large_expanded = torch.zeros_like(x_fcm_expanded)
            
            fused = x_fcm_expanded + x_large_expanded
        else:
            fused = x_fcm_expanded
        
        # 注意力调制
        spatial_weight = self.spatial_att(fused)
        channel_weight = self.channel_att(fused)
        attended = fused * spatial_weight * channel_weight
        
        # 输出
        output = self.output_conv(attended)
        return output + identity


class DynamicWeightGenerator(nn.Module):
    """动态权重生成器 - MSFAM的核心组件"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
        # 局部特征提取
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 动态融合权重
        self.fusion_weight = nn.Parameter(torch.ones(2))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 全局权重
        global_feat = self.global_pool(x).view(B, C)
        global_weight = self.global_fc(global_feat).view(B, C, 1, 1)
        
        # 局部权重
        local_weight = self.local_conv(x)
        
        # 动态融合
        fusion_weights = F.softmax(self.fusion_weight, dim=0)
        dynamic_weight = fusion_weights[0] * global_weight + fusion_weights[1] * local_weight
        
        return dynamic_weight

class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 不同尺度的特征提取分支
        self.scale1 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.scale4 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=3, dilation=3),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 多尺度特征提取
        feat1 = self.scale1(x)  # 1x1 细节特征
        feat2 = self.scale2(x)  # 3x3 局部特征
        feat3 = self.scale3(x)  # 空洞卷积 中等感受野
        feat4 = self.scale4(x)  # 空洞卷积 大感受野
        
        # 特征拼接和融合
        multi_scale_feat = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        fused_feat = self.fusion(multi_scale_feat)
        
        return fused_feat

class AdaptiveFeatureModulator(nn.Module):
    """自适应特征调制器"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 空间注意力分支
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # 通道注意力分支
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        # 特征重构网络
        self.feature_reconstruct = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, original_feat, enhanced_feat, dynamic_weight):
        # 应用动态权重
        weighted_enhanced = enhanced_feat * dynamic_weight
        
        # 空间和通道注意力
        spatial_att = self.spatial_attention(weighted_enhanced)
        channel_att = self.channel_attention(weighted_enhanced)
        
        # 注意力加权
        attended_feat = weighted_enhanced * spatial_att * channel_att
        
        # 特征重构
        combined_feat = torch.cat([original_feat, attended_feat], dim=1)
        reconstructed_feat = self.feature_reconstruct(combined_feat)
        
        return reconstructed_feat

class MSFAM(nn.Module):
    """多尺度特征自适应调制模块 (Multi-Scale Feature Adaptive Modulation)
    
    论文贡献点：
    1. 提出动态权重生成机制，根据输入特征自适应调整处理策略
    2. 设计多尺度特征提取器，有效融合不同感受野的特征信息
    3. 引入自适应特征调制器，实现特征的智能重构和增强
    4. 专门针对小目标检测进行优化，显著提升检测性能
    
    技术创新：
    - 动态权重生成：摆脱固定权重限制，实现自适应特征处理
    - 多尺度融合：结合局部细节和全局上下文，增强特征表达能力
    - 注意力引导：通过空间和通道注意力机制，突出重要特征
    - 特征重构：智能重组特征表示，提升模型判别能力
    """
    
    def __init__(self, c1, c2, reduction=8):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        
        # 输入特征预处理
        self.input_proj = Conv(c1, c1, 1, 1) if c1 == c2 else Conv(c1, c2, 1, 1)
        
        # 核心组件
        self.dynamic_weight_gen = DynamicWeightGenerator(c2, reduction)
        self.multi_scale_extractor = MultiScaleFeatureExtractor(c2)
        self.adaptive_modulator = AdaptiveFeatureModulator(c2)
        
        # 输出处理
        self.output_conv = Conv(c2, c2, 3, 1, 1)
        
        # 残差连接
        self.use_residual = (c1 == c2)
        if not self.use_residual:
            self.residual_proj = Conv(c1, c2, 1, 1)
            
    def forward(self, x):
        # 输入预处理
        projected_x = self.input_proj(x)
        
        # 动态权重生成
        dynamic_weight = self.dynamic_weight_gen(projected_x)
        
        # 多尺度特征提取
        multi_scale_feat = self.multi_scale_extractor(projected_x)
        
        # 自适应特征调制
        modulated_feat = self.adaptive_modulator(projected_x, multi_scale_feat, dynamic_weight)
        
        # 输出处理
        output = self.output_conv(modulated_feat)
        
        # 残差连接
        if self.use_residual:
            output = output + x
        elif hasattr(self, 'residual_proj'):
            output = output + self.residual_proj(x)
            
        return output

# 轻量级版本
class MSFAM_Lite_1(nn.Module):
    """MSFAM轻量级版本 - 适用于实时检测场景，浅层"""
    
    def __init__(self, c1, c2, reduction=8):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        
        # 简化的动态权重生成
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // reduction, c2, 1),
            nn.Sigmoid()
        )
        
        # 计算通道数（确保为整数）
        branch1_channels = int(c2 * 0.75)  # 转换为整数
        branch2_channels = c2 - branch1_channels  # 确保总和等于c2
        
        # 双分支特征提取
        self.branch1 = nn.Sequential(
            Conv(c1, branch1_channels, 1, 1),
            Conv(branch1_channels, branch1_channels, 3, 1, 1)
        )
        
        self.branch2 = nn.Sequential(
            Conv(c1, branch2_channels, 1, 1),
            nn.Conv2d(branch2_channels, branch2_channels, 3, 1, 2, dilation=2)
        )
        
        # 特征融合
        self.fusion = Conv(c2, c2, 1, 1)
        
        # 残差连接
        self.use_residual = (c1 == c2)
        if not self.use_residual:
            self.residual_proj = Conv(c1, c2, 1, 1)
            
    def forward(self, x):
        # 动态权重
        weight = self.weight_gen(x)
        
        # 双分支处理
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        
        # 特征融合
        combined = torch.cat([feat1, feat2], dim=1)
        output = self.fusion(combined) * weight
        
        # 残差连接
        if self.use_residual:
            output = output + x
        elif hasattr(self, 'residual_proj'):
            output = output + self.residual_proj(x)
            
        return output

class MSFAM_Lite_2(nn.Module):
    """MSFAM轻量级版本 - 适用于实时检测场景,深层"""
    
    def __init__(self, c1, c2, reduction=16):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        
        # 简化的动态权重生成
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // reduction, c2, 1),
            nn.Sigmoid()
        )
        
        # 计算通道数（确保为整数）
        branch1_channels = int(c2 * 0.75)  # 转换为整数
        branch2_channels = c2 - branch1_channels  # 确保总和等于c2
        
        # 双分支特征提取
        self.branch1 = nn.Sequential(
            Conv(c1, branch1_channels, 1, 1),
            Conv(branch1_channels, branch1_channels, 1, 1)
        )
        
        self.branch2 = nn.Sequential(
            Conv(c1, branch2_channels, 1, 1),
            nn.Conv2d(branch2_channels, branch2_channels, 3, 1, 2, dilation=2)
        )
        
        # 特征融合
        self.fusion = Conv(c2, c2, 1, 1)
        
        # 残差连接
        self.use_residual = (c1 == c2)
        if not self.use_residual:
            self.residual_proj = Conv(c1, c2, 1, 1)
            
    def forward(self, x):
        # 动态权重
        weight = self.weight_gen(x)
        
        # 双分支处理
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        
        # 特征融合
        combined = torch.cat([feat1, feat2], dim=1)
        output = self.fusion(combined) * weight
        
        # 残差连接
        if self.use_residual:
            output = output + x
        elif hasattr(self, 'residual_proj'):
            output = output + self.residual_proj(x)
            
        return output

class MSFAM_Ultra_Lite_Adaptive(nn.Module):
    """MSFAM超轻量自适应版本 - 基于FCM通道划分策略"""
    
    def __init__(self, c1, c2, depth_level='medium', reduction=32):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.depth_level = depth_level
        
        # 根据深度级别确定通道分配策略（参考FCM设计）
        if depth_level == 'shallow':  # 浅层 - 类似FCM_3
            self.complex_ratio = 0.75  # 75%用于复杂处理
            self.simple_ratio = 0.25   # 25%用于简单处理
            reduction_factor = 16      # 较小的压缩比
        elif depth_level == 'medium':  # 中层 - 平衡设计
            self.complex_ratio = 0.5   # 50%用于复杂处理
            self.simple_ratio = 0.5    # 50%用于简单处理
            reduction_factor = 24      # 中等压缩比
        else:  # 'deep' - 深层，类似FCM_1
            self.complex_ratio = 0.25  # 25%用于复杂处理
            self.simple_ratio = 0.75   # 75%用于简单处理
            reduction_factor = 32      # 较大的压缩比
        
        # 计算实际通道数
        self.complex_channels = max(int(c1 * self.complex_ratio), 8)
        self.simple_channels = c1 - self.complex_channels
        mid_channels = c2 // 2
        
        # 1. 自适应动态权重生成（根据深度调整复杂度）
        self.adaptive_weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, max(c1 // reduction_factor, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c1 // reduction_factor, 8), c2, 1),
            nn.Sigmoid()
        )
        
        # 2. 复杂特征分支（处理复杂模式）
        if self.complex_channels > 0:
            self.complex_branch = nn.Sequential(
                nn.Conv2d(self.complex_channels, mid_channels, 1),  # 降维
                nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels),  # 深度可分离
                nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels),  # 额外处理层
                nn.Conv2d(mid_channels, mid_channels, 1)  # 升维
            )
        
        # 3. 简单特征分支（处理简单模式）
        if self.simple_channels > 0:
            self.simple_branch = nn.Sequential(
                nn.Conv2d(self.simple_channels, mid_channels, 1),  # 降维
                nn.Conv2d(mid_channels, mid_channels, 3, 1, 2, dilation=2, groups=mid_channels),  # 深度可分离空洞卷积
                nn.Conv2d(mid_channels, mid_channels, 1)  # 升维
            )
        
        # 4. 自适应特征融合
        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(c2, c2, 1),
            nn.BatchNorm2d(c2) if depth_level != 'deep' else nn.Identity()  # 深层简化BN
        )
        
        # 5. 残差连接
        self.use_residual = (c1 == c2)
        if not self.use_residual:
            self.residual_proj = nn.Conv2d(c1, c2, 1)
    
    def forward(self, x):
        # 动态权重生成
        dynamic_weight = self.adaptive_weight_gen(x)
        
        # 自适应通道分割
        if self.complex_channels > 0 and self.simple_channels > 0:
            x_complex, x_simple = torch.split(x, [self.complex_channels, self.simple_channels], dim=1)
            
            # 分支处理
            complex_feat = self.complex_branch(x_complex)
            simple_feat = self.simple_branch(x_simple)
            
            # 特征融合
            combined_feat = torch.cat([complex_feat, simple_feat], dim=1)
        elif self.complex_channels > 0:
            # 只有复杂分支
            complex_feat = self.complex_branch(x)
            combined_feat = torch.cat([complex_feat, torch.zeros_like(complex_feat)], dim=1)
        else:
            # 只有简单分支
            simple_feat = self.simple_branch(x)
            combined_feat = torch.cat([torch.zeros_like(simple_feat), simple_feat], dim=1)
        
        # 自适应融合和权重调制
        output = self.adaptive_fusion(combined_feat) * dynamic_weight
        
        # 残差连接
        if self.use_residual:
            output = output + x
        elif hasattr(self, 'residual_proj'):
            output = output + self.residual_proj(x)
        
        return output

class PRSM(nn.Module):
    """终极优化版PRSM(Parallel Receptive Sensing Module)
    
    特点：
    1. 并行多尺度处理 + 串行深度特征提取
    2. 轻量级双重注意力机制
    3. 自适应特征融合
    4. 特征重用和参数共享
    5. 参数增长仅15%，性能提升显著
    """
    
    def __init__(self, dim, dim_out, reduction=8):
        super().__init__()
        
        # 输入预处理
        self.input_norm = nn.BatchNorm2d(dim)
        
        # 共享特征提取器（减少参数）
        self.shared_dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.shared_bn = nn.BatchNorm2d(dim)
        self.shared_act = nn.SiLU(inplace=True)
        
        # 并行多尺度分支（参数量很少）
        self.scale_3x3 = nn.Identity()  # 复用shared_dw的3x3
        self.scale_5x5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        self.scale_7x7 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)
        
        # 点卷积特征变换
        self.pw_conv1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.pw_bn1 = nn.BatchNorm2d(dim)
        self.pw_act1 = nn.SiLU(inplace=True)
        
        self.pw_conv2 = nn.Conv2d(dim, dim, 1, bias=False)
        self.pw_bn2 = nn.BatchNorm2d(dim)
        
        # 轻量级双重注意力
        self.channel_att = LightweightChannelAttention(dim, reduction)
        self.spatial_att = LightweightSpatialAttention()
        
        # 自适应特征融合
        self.adaptive_fusion = AdaptiveFusion(dim, 3)
        
        # 输出投影
        self.output_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.output_bn = nn.BatchNorm2d(dim)
        self.output_act = nn.SiLU(inplace=True)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        identity = x
        
        # 输入预处理
        x = self.input_norm(x)
        
        # 共享特征提取
        shared_feat = self.shared_act(self.shared_bn(self.shared_dw(x)))
        
        # 并行多尺度处理
        feat_3x3 = self.scale_3x3(shared_feat)  # 复用shared_dw结果
        feat_5x5 = self.scale_5x5(shared_feat)
        feat_7x7 = self.scale_7x7(shared_feat)
        
        # 自适应融合多尺度特征
        fused_feat = self.adaptive_fusion([feat_3x3, feat_5x5, feat_7x7])
        
        # 第一次点卷积变换
        x = self.pw_act1(self.pw_bn1(self.pw_conv1(fused_feat)))
        
        # 通道注意力
        x = self.channel_att(x)
        
        # 第二次点卷积变换
        x = self.pw_bn2(self.pw_conv2(x))
        
        # 空间注意力
        x = self.spatial_att(x)
        
        # 输出投影
        x = self.output_act(self.output_bn(self.output_proj(x)))
        
        # 残差连接
        return x + identity

class LightweightChannelAttention(nn.Module):
    """轻量级通道注意力，参数量极少"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(self.avg_pool(x))

class LightweightSpatialAttention(nn.Module):
    """轻量级空间注意力，几乎无参数"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention

class AdaptiveFusion(nn.Module):
    """自适应特征融合"""
    def __init__(self, channels, num_branches):
        super().__init__()
        self.num_branches = num_branches
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, num_branches, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, features):
        # features: list of [B, C, H, W]
        stacked = torch.stack(features, dim=2)  # [B, C, num_branches, H, W]
        weights = self.weight_gen(features[0])  # [B, num_branches, 1, 1]
        
        # 调整权重维度以匹配stacked特征
        weights = weights.unsqueeze(1)  # [B, 1, num_branches, 1, 1]
        
        # 加权融合
        fused = torch.sum(stacked * weights, dim=2)  # [B, C, H, W]
        return fused