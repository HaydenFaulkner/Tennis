"""
Defines the custom model and layer specification
"""
import mxnet as mx
from mxnet.gluon import HybridBlock, nn


class TimeDistributed(HybridBlock):
    def __init__(self, model, style='reshape', **kwargs):
        """
        A time distributed layer like that seen in Keras

        Args:
            model: the backbone model that will be repeated over time
            style (str): either 'reshape' or 'for' for the implementation to use (default is reshape)
        """
        super(TimeDistributed, self).__init__(**kwargs)
        assert style in ['reshape', 'for']
        self._style = style
        with self.name_scope():
            self.model = model

    def apply_model(self, x, _):
        return self.model(x), []

    def hybrid_forward(self, F, x):
        if self._style == 'for':
            # For loop style
            x = F.swapaxes(x, 0, 1)  # swap batch and seqlen channels
            x, _ = F.contrib.foreach(self.apply_model, x, [])  # runs on first channel, which is now seqlen
            x = F.swapaxes(x, 0, 1)  # swap seqlen and batch channels
        else:
            # Reshape style, doesn't work with symbols cause no shape
            batch_size = x.shape[0]
            input_length = x.shape[1]
            x = F.reshape(x, (batch_size * input_length,) + x.shape[2:])  # (num_samples * timesteps, ...)
            x = self.model(x)
            x = F.reshape(x, (batch_size, input_length,) + x.shape[1:])  # (num_samples, timesteps, ...)

        return x


class FrameModel(HybridBlock):
    def __init__(self, backbone, classes, **kwargs):
        """
        A framewise model (just the backbone CNN with a single dense layer to the classes)

        Args:
            backbone: the backbone CNN model
            classes (int): the number of classes
        """
        super(FrameModel, self).__init__(**kwargs)
        with self.name_scope():
            self.backbone = backbone.features
            self.dense = nn.Dense(classes, flatten=True)

    def hybrid_forward(self, F, x):
        x = self.backbone(x)
        x = self.dense(x)
        return x


class TimeModel(HybridBlock):
    def __init__(self, backbone, classes, hidden_size=128, **kwargs):
        """
        A temporal CNN+RNN(GRU) model

        Args:
            backbone: the backbone CNN model
            classes (int): the number of classes
            hidden_size (int): the hidden size of the GRU (default is 128)
        """
        super(TimeModel, self).__init__(**kwargs)
        with self.name_scope():
            self.td = TimeDistributed(backbone)
            self.gru = mx.gluon.rnn.GRU(hidden_size, layout="NTC", bidirectional=True)
            self.dense = nn.Dense(classes, flatten=True)

    def hybrid_forward(self, F, x):
        x = self.td(x)
        x = self.gru(x)
        x = F.max(x, axis=1)
        x = self.dense(x)
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
    mod.initialize()
    td = TimeDistributed(mod)
    td.initialize()
    td.hybridize()
    mse_loss = gluon.loss.L2Loss()
    with autograd.record():
        out = td(mx.nd.ones((3, 2, 3, 2, 2)))
        loss = mse_loss(out, mx.nd.ones((3, 2, 4, 1, 1)))
    loss.backward()
