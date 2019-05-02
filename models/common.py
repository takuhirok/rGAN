import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils as utils

from .condbatchnorm import CondBatchNorm2d


def weights_init(m):
    if isinstance(m, CustomConv2d):
        if m.conv.weight is not None:
            if m.residual_init:
                init.xavier_uniform_(m.conv.weight.data, gain=math.sqrt(2))
            else:
                init.xavier_uniform_(m.conv.weight.data)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias.data, 0.0)
    if isinstance(m, CustomLinear):
        if m.linear.weight is not None:
            init.xavier_uniform_(m.linear.weight.data)
        if m.linear.bias is not None:
            init.constant_(m.linear.bias.data, 0.0)
    if isinstance(m, CustomEmbedding):
        if m.embed.weight is not None:
            init.xavier_uniform_(m.embed.weight.data)
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, CondBatchNorm2d):
        if m.weight is not None:
            init.constant_(m.weight.data, 1.0)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


def global_pooling(input, pooling='mean'):
    if pooling == 'mean':
        return input.mean(3).mean(2)
    elif pooling == 'sum':
        return input.sum(3).sum(2)
    else:
        raise NotImplementedError()


class CustomConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=None,
                 bias=True,
                 spectral_norm=False,
                 residual_init=True):
        super(CustomConv2d, self).__init__()
        self.residual_init = residual_init
        if padding is None:
            padding = int((kernel_size - 1) / 2)

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        if spectral_norm:
            self.conv = utils.spectral_norm(self.conv)

    def forward(self, input):
        return self.conv(input)


class CustomEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, spectral_norm=False):
        super(CustomEmbedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        if spectral_norm:
            self.embed = utils.spectral_norm(self.embed)

    def forward(self, input):
        return self.embed(input)


class CustomLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 spectral_norm=False):
        super(CustomLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if spectral_norm:
            self.linear = utils.spectral_norm(self.linear)

    def forward(self, input):
        return self.linear(input)


class ConvMeanPool(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 spectral_norm=False,
                 residual_init=True):
        super(ConvMeanPool, self).__init__()
        self.conv = CustomConv2d(in_channels,
                                 out_channels,
                                 kernel_size,
                                 bias=bias,
                                 spectral_norm=spectral_norm,
                                 residual_init=residual_init)

    def forward(self, input):
        output = input
        output = self.conv(output)
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] +
                  output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        return output


class MeanPoolConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 spectral_norm=False,
                 residual_init=True):
        super(MeanPoolConv, self).__init__()
        self.conv = CustomConv2d(in_channels,
                                 out_channels,
                                 kernel_size,
                                 bias=bias,
                                 spectral_norm=spectral_norm,
                                 residual_init=residual_init)

    def forward(self, input):
        output = input
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] +
                  output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        output = self.conv(output)
        return output


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_square = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, in_height, in_width, in_depth) = output.size()
        out_depth = int(in_depth / self.block_size_square)
        out_width = int(in_width * self.block_size)
        out_height = int(in_height * self.block_size)
        output = output.contiguous().view(batch_size, in_height, in_width,
                                          self.block_size_square, out_depth)
        output_list = output.split(self.block_size, 3)
        output_list = [
            output_element.contiguous().view(batch_size, in_height, out_width,
                                             out_depth)
            for output_element in output_list
        ]
        output = torch.stack(output_list, 0).transpose(0, 1).permute(
            0, 2, 1, 3, 4).contiguous().view(batch_size, out_height, out_width,
                                             out_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 spectral_norm=False,
                 residual_init=True):
        super(UpSampleConv, self).__init__()
        self.conv = CustomConv2d(in_channels,
                                 out_channels,
                                 kernel_size,
                                 bias=bias,
                                 spectral_norm=spectral_norm,
                                 residual_init=residual_init)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 resample=None,
                 spectral_norm=False):
        super(ResidualBlock, self).__init__()
        if in_channels != out_channels or resample is not None:
            self.learnable_shortcut = True
        else:
            self.learnable_shortcut = False

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.conv_shortcut = ConvMeanPool(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              spectral_norm=spectral_norm,
                                              residual_init=False)
            self.conv1 = CustomConv2d(in_channels,
                                      in_channels,
                                      kernel_size=kernel_size,
                                      spectral_norm=spectral_norm)
            self.conv2 = ConvMeanPool(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      spectral_norm=spectral_norm)
        elif resample is None:
            if self.learnable_shortcut:
                self.conv_shortcut = CustomConv2d(in_channels,
                                                  out_channels,
                                                  kernel_size=1,
                                                  spectral_norm=spectral_norm,
                                                  residual_init=False)
            self.conv1 = CustomConv2d(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      spectral_norm=spectral_norm)
            self.conv2 = CustomConv2d(out_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      spectral_norm=spectral_norm)
        else:
            raise NotImplementedError()

    def forward(self, input):
        if self.learnable_shortcut:
            shortcut = self.conv_shortcut(input)
        else:
            shortcut = input

        output = input
        output = self.relu1(output)
        output = self.conv1(output)
        output = self.relu2(output)
        output = self.conv2(output)
        return shortcut + output


class OptimizedResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 spectral_norm=False):
        super(OptimizedResidualBlock, self).__init__()
        self.conv1 = CustomConv2d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  spectral_norm=spectral_norm)
        self.conv2 = ConvMeanPool(out_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  spectral_norm=spectral_norm)
        self.conv_shortcut = MeanPoolConv(in_channels,
                                          out_channels,
                                          kernel_size=1,
                                          spectral_norm=spectral_norm,
                                          residual_init=False)
        self.relu2 = nn.ReLU()

    def forward(self, input):
        shortcut = self.conv_shortcut(input)

        output = input
        output = self.conv1(output)
        output = self.relu2(output)
        output = self.conv2(output)
        return shortcut + output


class CondResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_classes,
                 resample=None,
                 spectral_norm=False):
        super(CondResidualBlock, self).__init__()
        if in_channels != out_channels or resample is not None:
            self.learnable_shortcut = True
        else:
            self.learnable_shortcut = False

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'up':
            self.norm1 = CondBatchNorm2d(in_channels, num_classes)
            self.norm2 = CondBatchNorm2d(out_channels, num_classes)
            self.conv_shortcut = UpSampleConv(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              spectral_norm=spectral_norm,
                                              residual_init=False)
            self.conv1 = UpSampleConv(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      spectral_norm=spectral_norm)
            self.conv2 = CustomConv2d(out_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      spectral_norm=spectral_norm)
        else:
            raise NotImplementedError()

    def forward(self, input, label):
        if self.learnable_shortcut:
            shortcut = self.conv_shortcut(input)
        else:
            shortcut = input

        output = input
        output = self.norm1(output, label)
        output = self.relu1(output)
        output = self.conv1(output)
        output = self.norm2(output, label)
        output = self.relu2(output)
        output = self.conv2(output)
        return shortcut + output
