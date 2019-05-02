import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class _CondBatchNorm(nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(_CondBatchNorm, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_classes, num_features))
            self.bias = Parameter(torch.Tensor(num_classes, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.uniform_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input, label=None, class_weight=None):
        self._check_input_dim(input)
        if label is None:
            if class_weight is None:
                raise ValueError(
                    'either label or class_weight must not be None')
        else:
            if class_weight is None:
                batch_size = label.size(0)
                class_weight = torch.zeros(batch_size,
                                           self.num_classes).to(label.device)
                class_weight.scatter_(1, label.unsqueeze(1), 1)
            else:
                raise ValueError('either label or class_weight must be None')

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = (
                        1.0 / self.num_batches_tracked.item())
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # apply batch norm without using weight and bias
        output = F.batch_norm(input, self.running_mean, self.running_var, None,
                              None, self.training
                              or not self.track_running_stats,
                              exponential_average_factor, self.eps)

        # scale and shift using weight and bias
        batch_size, channels = input.size(0), input.size(1)
        if self.weight is not None:
            weight = class_weight.mm(self.weight)
            output *= weight.contiguous().view(
                batch_size, channels, *([1] * (input.dim() - 2))).expand(
                    (batch_size, channels, *input.size()[2:]))
        if self.bias is not None:
            bias = class_weight.mm(self.bias)
            output += bias.contiguous().view(
                batch_size, channels, *([1] * (input.dim() - 2))).expand(
                    (batch_size, channels, *input.size()[2:]))
        return output

    def extra_repr(self):
        return (
            '{num_features}, {num_classes}, eps={eps}, momentum={momentum}, '
            'affine={affine}, track_running_stats={track_running_stats}'.
            format(**self.__dict__))


class CondBatchNorm1d(_CondBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(
                input.dim()))


class CondBatchNorm2d(_CondBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))


class CondBatchNorm3d(_CondBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(
                input.dim()))
