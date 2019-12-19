"""
Defines the model specifications
"""
import mxnet as mx
from mxnet.gluon import HybridBlock, nn

from utils.layers import TimeDistributed


class FrameModel(HybridBlock):
    def __init__(self, backbone, num_classes, **kwargs):
        """
        A framewise model (just the backbone CNN with a single dense layer to the classes)

        Args:
            backbone: the backbone CNN model
            num_classes (int): the number of classes
        """
        super(FrameModel, self).__init__(**kwargs)
        with self.name_scope():
            self.backbone = backbone.features
            self.classes = nn.Dense(num_classes, flatten=True, activation='sigmoid')

    def hybrid_forward(self, F, x):
        x = self.backbone(x)
        x = self.classes(x)
        return x


class TimeModel(HybridBlock):
    def __init__(self, backbone, num_classes, hidden_size=128, **kwargs):
        """
        A temporal CNN+RNN(GRU) model

        Args:
            backbone: the backbone CNN model
            num_classes (int): the number of classes
            hidden_size (int): the hidden size of the GRU (default is 128)
        """
        super(TimeModel, self).__init__(**kwargs)
        with self.name_scope():
            self.td = TimeDistributed(backbone.features)
            self.gru = mx.gluon.rnn.GRU(hidden_size, layout="NTC", bidirectional=True)
            self.classes = nn.Dense(num_classes, flatten=True, activation='sigmoid')

    def hybrid_forward(self, F, x):
        x = self.td(x)
        x = self.gru(x.squeeze(axis=-1).squeeze(axis=-1))
        x = F.max(x, axis=1)
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
