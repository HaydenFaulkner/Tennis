"""R(2+1)D nets implemented in Gluon.
A Closer Look at Spatiotemporal Convolutions for Action Recognition
https://arxiv.org/pdf/1711.11248.pdf

Official code: https://github.com/facebookresearch/VMZ

Must use standard relu for pretrained weights from VMZ
Slightly different results attained between implementations due to BatchNorm variations - see README.md

"""
from __future__ import division

__all__ = ['R21DV1',
           'BasicBlockV1',
           'BottleneckV1',
           'get_r21d']

import math
import os

import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
import gluoncv

# Helpers
def _conv3d(out_channels, kernel, strides=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
            groups=1, use_bias=False, prefix=''):
    """A common 3dconv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='3D')
    cell.add(nn.Conv3D(out_channels, kernel_size=kernel, strides=strides, padding=padding, dilation=dilation,
                       groups=groups, use_bias=use_bias, prefix=prefix))
    return cell

def _conv21d(out_channels, kernel, strides=(1, 1, 1), padding=(0, 0, 0),
             in_channels=0, mid_channels=None, norm_layer=nn.BatchNorm, norm_kwargs=None, prefix=''):
    """R(2+1)D from 'A Closer Look at Spatiotemporal Convolutions for Action Recognition'"""
    cell = nn.HybridSequential(prefix='R(2+1)D')
    if mid_channels is None:
        mid_channels = int(math.floor((kernel[0] * kernel[1] * kernel[2] * in_channels * out_channels) /
                           (kernel[1] * kernel[2] * in_channels + kernel[0] * out_channels)))


    cell.add(_conv3d(mid_channels, (1, kernel[1], kernel[2]),
                     strides=(1, strides[1], strides[2]),
                     padding=(0, padding[1], padding[2]),
                     prefix=prefix+'middle_'))

    cell.add(norm_layer(epsilon=1e-3, momentum=0.9, prefix=prefix+'middle_',
                        **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.0))

    cell.add(_conv3d(out_channels, (kernel[0], 1, 1),
                     strides=(strides[0], 1, 1),
                     padding=(padding[0], 0, 0),
                     prefix=prefix))

    return cell


class BasicBlockV1(HybridBlock):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    Modified with R(2+1)D convs.
    This is used for R21DV1 for 18, 34 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, prefix='', **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix=prefix)
        self.body.add(_conv21d(channels, kernel=[3,3,3], strides=[stride,stride,stride], padding=[1, 1, 1],
                               in_channels=in_channels, prefix=prefix+'conv_1_'))
        self.body.add(nn.BatchNorm(epsilon=1e-3, momentum=0.9, prefix=prefix+'conv_1_'))
        self.body.add(nn.LeakyReLU(0.0))
        self.body.add(_conv21d(channels, kernel=[3,3,3], strides=[1,1,1], padding=[1, 1, 1],
                               in_channels=channels, prefix=prefix+'conv_2_'))
        self.body.add(nn.BatchNorm(epsilon=1e-3, momentum=0.9, prefix=prefix+'conv_2_'))

        if downsample:
            self.downsample = nn.HybridSequential(prefix=prefix)
            self.downsample.add(_conv3d(channels, kernel=[1,1,1], strides=[stride,stride,stride],
                                        padding=[0, 0, 0], prefix=prefix+'down_'))
            self.downsample.add(nn.BatchNorm(epsilon=1e-3, momentum=0.9, prefix=prefix+'down_'))
        else:
            self.downsample = None

        self.final_relu = nn.LeakyReLU(0.0)

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = self.final_relu(residual+x)

        return x

class BottleneckV1(HybridBlock):
    r"""Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    Modified with R(2+1)D convs.
    This is used for R21DV1 for 50, 101, 152 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, prefix='',**kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix=prefix)
        self.body.add(_conv3d(channels//4, [1, 1, 1], strides=[stride,stride,stride], prefix=prefix+'conv_1_'))
        self.body.add(nn.BatchNorm(epsilon=1e-3, momentum=0.9, prefix=prefix+'conv_1_'))
        self.body.add(nn.LeakyReLU(0.0))
        self.body.add(_conv21d(channels//4, [3, 3, 3], strides=[1,1,1], padding=[1,1,1],
                               in_channels=channels//4, prefix=prefix+'conv_2_'))
        self.body.add(nn.BatchNorm(epsilon=1e-3, momentum=0.9, prefix=prefix+'conv_2_'))
        self.body.add(nn.LeakyReLU(0.0))
        self.body.add(_conv3d(channels, [1, 1, 1], strides=[1,1,1], prefix=prefix+'conv_3_'))
        self.body.add(nn.BatchNorm(epsilon=1e-3, momentum=0.9, prefix=prefix+'conv_3_'))
        if downsample:
            self.downsample = nn.HybridSequential(prefix=prefix)
            self.downsample.add(_conv3d(channels, [1, 1, 1], strides=[stride, stride, stride], prefix=prefix+'down_'))
            self.downsample.add(nn.BatchNorm(epsilon=1e-3, momentum=0.9, prefix=prefix+'down_'))
        else:
            self.downsample = None

        self.final_relu = nn.LeakyReLU(0.0)

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = self.final_relu(x + residual)
        return x

# Nets
class R21DV1(HybridBlock):
    r"""R(2+1)D model from
    `"A Closer Look at Spatiotemporal Convolutions for Action Recognition"
    <http://arxiv.org/pdf/1711.11248>`_ paper.
    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    t : int, default 1
        number of timesteps.
    """
    def __init__(self, block, layers, channels, classes=400, **kwargs):
        super(R21DV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(_conv21d(channels[0], [3, 7, 7], strides=[1, 2, 2], padding=[1, 3, 3],
                                       mid_channels=45, prefix='init_'))
            self.features.add(nn.BatchNorm(epsilon=1e-3, momentum=0.9, use_global_stats=True, prefix='init_'))
            self.features.add(nn.LeakyReLU(0.0))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i]))
            self.avg = nn.GlobalAvgPool3D()

            self.dense = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels, prefix='block1_'))
            for i in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix='block%d_'%(i+2)))
        return layer

    def hybrid_forward(self, F, x):
        x = F.swapaxes(x, 1, 2)  # moved into features as block
        x = self.features(x)
        avg = self.avg(x)
        sm = F.softmax(self.dense(avg))

        return x, avg, sm


# Constructor
def get_r21d(num_layers, n_classes, t=8, pretrained=False, ctx=mx.cpu(),
             root=os.path.join('models', 'vision', 'rdnet', 'weights'), **kwargs):

    net_layers = { 34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
                  152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

    assert num_layers in net_layers, \
        "Invalid number of layers: %d. Options are %s" % (num_layers, str(net_layers.keys()))

    assert n_classes in [387, 400, 487]
    assert t in [8, 32]
    block_type, layers, channels = net_layers[num_layers]

    if block_type == 'basic_block':
        block_class = BasicBlockV1
    else:
        block_class = BottleneckV1

    net = R21DV1(block_class, layers, channels, classes=n_classes, **kwargs)

    if pretrained:
        if num_layers == 34:
            if n_classes == 400:
                if t == 8:
                    pretrained_path = '34_8_kinetics_from_ig65m_f128022400.params'
                else:
                    pretrained_path = '34_32_kinetics_from_ig65m_f1061696810.params'
            else:
                if t == 8:
                    pretrained_path = '34_8_ig65m_from_scratch_f79708462.params'
                else:
                    pretrained_path = '34_32_ig65m_from_scratch_f102649996.params'
        else:
            assert n_classes == 487
            assert t == 32
            pretrained_path = '152_sports1m_f127111290.params'

        net.load_parameters(os.path.join(root, pretrained_path), ctx=ctx)
    else:
        net.initialize()

    return net

if __name__ == '__main__':
    from utils import convert_weights, get_test_frames
    # just for debugging

    # pkl_path = "models/definitions/rdnet/weights/r2plus1d_152_sports1m_from_scratch_f127111290.pkl"
    # save_path = "models/definitions/rdnet/weights/152_sports1m_f127111290.params"
    # n_layers = 152
    # t = 32
    # dataset = 'sports1m'

    # pkl_path = "models/definitions/rdnet/weights/r2plus1d_34_clip32_ig65m_from_scratch_f102649996.pkl"
    # save_path = "models/definitions/rdnet/weights/34_32_ig65m_from_scratch_f102649996.params"
    # n_layers = 34
    # t = 32
    # n_classes = 359 # 487 sports, # 400 kinetics

    pkl_path = "models/definitions/rdnet/weights/r2plus1d_34_clip8_ft_kinetics_from_ig65m_ f128022400.pkl"
    save_path = "models/definitions/rdnet/weights/34_8_kinetics_from_ig65m_f128022400.params"
    n_layers = 34
    t = 8
    n_classes = 400 # 487 sports, # 400 kinetics

    model = get_r21d(n_layers, n_classes, t=t, pretrained=True)

    out = model.summary(mx.nd.ones((2, 3, t, 112, 112)))

    # gluoncv.utils.viz.plot_network(model, shape=(2, 3, length_rgb, 112, 112))

    # convert_weights(model, load_path=pkl_path, n_classes=n_classes, n_layers=n_layers, save_path=save_path)

    # model.load_parameters(save_path)
    # frames = get_test_frames("/path/to/test.mp4", length_rgb=length_rgb)
    # out, avg, sm = model(frames)

    print('DONE')
