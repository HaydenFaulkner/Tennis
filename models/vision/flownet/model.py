"""
FlowNet: Learning Optical Flow with Convolutional Networks
Philipp Fischer et al.

from https://arxiv.org/pdf/1504.06852.pdf
"""
import os

import mxnet as mx
from mxnet import autograd
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class FlowNetS(HybridBlock):
    """
    FlowNet S without batch norm
    """
    def __init__(self, prefix='flownetS', **kwargs):
        super(FlowNetS, self).__init__(**kwargs)

        with self.name_scope():
            self.conv1 = nn.HybridSequential(prefix=prefix+'_conv_1.')
            self.conv1.add(nn.Conv2D(channels=64, kernel_size=7, strides=2, padding=3, prefix='conv1.0.'))
            self.conv1.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU1.'))

            self.conv2 = nn.HybridSequential(prefix=prefix + '_conv_2.')
            self.conv2.add(nn.Conv2D(channels=128, kernel_size=5, strides=2, padding=2, prefix='conv2.0.'))
            self.conv2.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU2.'))

            self.conv3 = nn.HybridSequential(prefix=prefix+'_conv_3.')
            self.conv3.add(nn.Conv2D(channels=256, kernel_size=5, strides=2, padding=2, prefix='conv3.0.'))
            self.conv3.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU3.'))
            self.conv3.add(nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1, prefix='conv3_1.0.'))
            self.conv3.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU4.'))

            self.conv4 = nn.HybridSequential(prefix=prefix+'_conv_4.')
            self.conv4.add(nn.Conv2D(channels=512, kernel_size=3, strides=2, padding=1, prefix='conv4.0.'))
            self.conv4.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU5.'))
            self.conv4.add(nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1, prefix='conv4_1.0.'))
            self.conv4.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU6.'))

            self.conv5 = nn.HybridSequential(prefix=prefix+'_conv_5.')
            self.conv5.add(nn.Conv2D(channels=512, kernel_size=3, strides=2, padding=1, prefix='conv5.0.'))
            self.conv5.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU7.'))
            self.conv5.add(nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1, prefix='conv5_1.0.'))
            self.conv5.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU8.'))

            self.conv6 = nn.HybridSequential(prefix=prefix+'_conv_6.')
            self.conv6.add(nn.Conv2D(channels=1024, kernel_size=3, strides=2, padding=1, prefix='conv6.0.'))
            self.conv6.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU9.'))
            self.conv6.add(nn.Conv2D(channels=1024, kernel_size=3, strides=1, padding=1, prefix='conv6_1.0.'))
            self.conv6.add(nn.LeakyReLU(alpha=0.1, prefix='ReLU10.'))

            self.predict_flow6 = nn.Conv2D(channels=2, kernel_size=3, strides=1, padding=1, prefix='predict_flow6.')
            self.deconv5 = nn.Conv2DTranspose(channels=512, kernel_size=4, strides=2, padding=1, prefix='deconv5.0.')
            self.relu11 = nn.LeakyReLU(alpha=0.1, prefix='ReLU11.')
            self.upsampled_flow6_to_5 = nn.Conv2DTranspose(channels=2, kernel_size=4, strides=2, padding=1,
                                                           prefix='upsampled_flow6_to_5.', use_bias=False)

            self.predict_flow5 = nn.Conv2D(channels=2, kernel_size=3, strides=1, padding=1, prefix='predict_flow5.')
            self.deconv4 = nn.Conv2DTranspose(channels=256, kernel_size=4, strides=2, padding=1, prefix='deconv4.0.')
            self.relu12 = nn.LeakyReLU(alpha=0.1, prefix='ReLU12.')
            self.upsampled_flow5_to_4 = nn.Conv2DTranspose(channels=2, kernel_size=4, strides=2, padding=1,
                                                           prefix='upsampled_flow5_to_4.', use_bias=False)

            self.predict_flow4 = nn.Conv2D(channels=2, kernel_size=3, strides=1, padding=1, prefix='predict_flow4.')
            self.deconv3 = nn.Conv2DTranspose(channels=128, kernel_size=4, strides=2, padding=1, prefix='deconv3.0.')
            self.relu13 = nn.LeakyReLU(alpha=0.1, prefix='ReLU13.')
            self.upsampled_flow4_to_3 = nn.Conv2DTranspose(channels=2, kernel_size=4, strides=2, padding=1,
                                                           prefix='upsampled_flow4_to_3.', use_bias=False)

            self.predict_flow3 = nn.Conv2D(channels=2, kernel_size=3, strides=1, padding=1, prefix='predict_flow3.')
            self.deconv2 = nn.Conv2DTranspose(channels=64, kernel_size=4, strides=2, padding=1, prefix='deconv2.0.')
            self.relu14 = nn.LeakyReLU(alpha=0.1, prefix='ReLU14.')
            self.upsampled_flow3_to_2 = nn.Conv2DTranspose(channels=2, kernel_size=4, strides=2, padding=1,
                                                           prefix='upsampled_flow3_to_2.', use_bias=False)

            self.predict_flow2 = nn.Conv2D(channels=2, kernel_size=3, strides=1, padding=1, prefix='predict_flow2.')

    def hybrid_forward(self, F, x):

        x = F.reshape(x, shape=(0, -3, -2))  # concat the two imgs on the channels dim b,i,c,h,w -> b,i*c,h,w

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.relu11(self.deconv5(out_conv6))

        concat5 = F.concat(out_conv5, out_deconv5, flow6_up)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.relu12(self.deconv4(concat5))

        concat4 = F.concat(out_conv4, out_deconv4, flow5_up)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.relu13(self.deconv3(concat4))

        concat3 = F.concat(out_conv3, out_deconv3, flow4_up)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.relu14(self.deconv2(concat3))

        concat2 = F.concat(out_conv2, out_deconv2, flow3_up)
        flow2 = self.predict_flow2(concat2)

        if autograd.is_training():
            return flow2, flow3, flow4, flow5, flow6

        return flow2  # do upsampling in numpy outside as F.UpSampling() bilinear is very broken


def get_flownet(ctx=mx.cpu(), root=os.path.join('flownet'), **kwargs):

    net = FlowNetS(**kwargs)
    net.load_parameters(os.path.join(root, 'FlowNet2-S_checkpoint.params'), ctx=ctx)

    return net


if __name__ == '__main__':
    model = get_flownet()
    out = model.summary(mx.nd.random_normal(shape=(1, 2, 3, 384, 512)))
