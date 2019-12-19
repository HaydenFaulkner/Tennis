"""
Defines the custom layer specifications
"""

from mxnet.gluon import HybridBlock


class TimeDistributed(HybridBlock):
    def __init__(self, model, style='reshape', **kwargs):
        """
        A time distributed layer like that seen in Keras
        Args:
            model: the backbone model that will be repeated over time
            style (str): either 'reshape' or 'for' for the implementation to use (default is reshape1)
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
            if isinstance(x, tuple):  # for handling multiple outputs
                x = (F.swapaxes(xi, 0, 1) for xi in x)
            elif isinstance(x, list):
                x = [F.swapaxes(xi, 0, 1) for xi in x]
            else:
                x = F.swapaxes(x, 0, 1)  # swap seqlen and batch channels
        else:
            shp = x  # can use this to keep shapes for reshape back to (batch, timesteps, ...)
            x = F.reshape(x, (-3, -2))  # combines batch and timesteps dims
            x = self.model(x)
            if isinstance(x, tuple):  # for handling multiple outputs
                x = (F.reshape_like(xi, shp, lhs_end=1, rhs_end=2) for xi in x)
            elif isinstance(x, list):
                x = [F.reshape_like(xi, shp, lhs_end=1, rhs_end=2) for xi in x]
            else:
                x = F.reshape_like(x, shp, lhs_end=1, rhs_end=2)  # (num_samples, timesteps, ...)

        return x
