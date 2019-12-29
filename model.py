"""
Defines the model specifications
"""
import mxnet as mx
from mxnet.gluon import HybridBlock, nn

from utils.layers import TimeDistributed


class FrameModel(HybridBlock):
    def __init__(self, backbone, num_classes=-1, **kwargs):
        """
        A framewise model (just the backbone CNN with a single dense layer to the classes)

        Args:
            backbone: the backbone CNN model
            num_classes (int): the number of classes
        """
        super(FrameModel, self).__init__(**kwargs)
        with self.name_scope():
            self.backbone = backbone
            self.classes = None
            if num_classes > 0:
                self.classes = nn.Dense(num_classes, flatten=True, activation='sigmoid')

    def hybrid_forward(self, F, x):
        x = self.backbone(x)
        if self.classes:
            x = self.classes(x)
        return x


class TemporalPooling(HybridBlock):
    def __init__(self, backbone, num_classes=-1, pool='max', **kwargs):
        """
        A temporal pooling model

        Args:
            backbone: the backbone CNN model
            num_classes (int): the number of classes,
                               -1 meaning the backbone output is a softmax (default)
                               0 means we apply the pool between the backbone softmax
        """
        super(TemporalPooling, self).__init__(**kwargs)
        self.pool = pool
        with self.name_scope():
            self.classes = None
            if num_classes == 0:
                self.td = TimeDistributed(backbone.backbone)
                self.classes = backbone.classes
            else:
                self.td = TimeDistributed(backbone)
                if num_classes > 0:
                    self.classes = nn.Dense(num_classes, flatten=True, activation='sigmoid')

    def hybrid_forward(self, F, x):
        x = self.td(x)
        if self.pool == 'mean':
            x = F.mean(x, axis=1)
        else:
            x = F.max(x, axis=1)
        if self.classes:
            x = self.classes(x)
        return x


class TimeModel(HybridBlock):
    def __init__(self, backbone, num_classes=-1, hidden_size=128, **kwargs):
        """
        A temporal CNN+RNN(GRU) model

        Args:
            backbone: the backbone CNN model
            num_classes (int): the number of classes
            hidden_size (int): the hidden size of the GRU (default is 128)
        """
        super(TimeModel, self).__init__(**kwargs)
        with self.name_scope():
            self.td = TimeDistributed(backbone)
            self.gru = mx.gluon.rnn.GRU(hidden_size, layout="NTC", bidirectional=True)
            self.classes = None
            if num_classes > 0:
                self.classes = nn.Dense(num_classes, flatten=True, activation='sigmoid')

    def hybrid_forward(self, F, x):
        x = self.td(x)
        x = self.gru(x.squeeze(axis=-1).squeeze(axis=-1))
        x = F.max(x, axis=1)
        if self.classes:
            x = self.classes(x)
        return x


class Debug(HybridBlock):
    def __init__(self, **kwargs):
        """
        Useful model for debugging
        """
        super(Debug, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(channels=4, kernel_size=2)

    def hybrid_forward(self, F, x):
        x = F.relu(self.conv1(x))
        return x


class TwoStreamModel(HybridBlock):
    def __init__(self, backbone_rgb, backbone_flow, num_classes, **kwargs):
        """
        A two stream model (just the backbone CNNs concatenated before a single dense layer to the classes)

        Args:
            backbone_rgb: the rgb backbone CNN model
            backbone_flow: the flow backbone CNN model
            num_classes (int): the number of classes
        """
        super(TwoStreamModel, self).__init__(**kwargs)
        with self.name_scope():
            self.backbone_rgb = backbone_rgb
            self.backbone_flow = backbone_flow
            self.classes = nn.Dense(num_classes, flatten=True, activation='sigmoid')

    def hybrid_forward(self, F, x):
        rgb = F.slice_axis(x, axis=-3, begin=0, end=3)
        flow = F.slice_axis(x, axis=-3, begin=3, end=6)
        rgb = self.backbone_rgb(rgb)
        flow = self.backbone_flow(flow)
        x = F.concat(rgb, flow, dim=-3)
        x = self.classes(x)
        return x


if __name__ == '__main__':
    # just for debugging
    from mxnet import gluon, autograd

    mod = Debug()
    td = TimeDistributed(mod)
    td.initialize()
    td.hybridize()
    mse_loss = gluon.loss.L2Loss()
    with autograd.record():
        out = td(mx.nd.ones((3, 2, 3, 2, 2)))
        loss = mse_loss(out, mx.nd.ones((3, 2, 4, 1, 1)))
    loss.backward()
