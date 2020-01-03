# R(2+1)D Models
We have attempted to convert the [official R(2+1)D](https://github.com/facebookresearch/VMZ)
 models and weights **from Caffe2 over to Gluon**.

There is also a newer very nicely written [_unnoficial_ PyTorch implementation](https://github.com/moabitcoin/ig65m-pytorch), 
however we don't directly use it.

### Model Definition
We modify [Gluon's ResNet implementation](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/resnet.py) 
to handle R(2+1)D models. The definition can be found in `r21d.py`.

### Pre-trained weights
The original `.pkl` pre-trained models can be downloaded from their 
[model zoo](https://github.com/facebookresearch/VMZ/blob/master/tutorials/model_zoo.md).

We have converted two models into `.params`, which you can download from 
[Google Drive](https://drive.google.com/open?id=1j3x7OTvz_MFviJCFxEnLjyuF0lyGemBO).

`r2plus1d_34_clip8_ft_kinetics_from_ig65m_ f128022400.pkl` --> `34_8_kinetics_from_ig65m_f128022400.params`

`r2plus1d_152_sports1m_from_scratch_f127111290.pkl` --> `152_sports1m_f127111290.params`

We store the models in the `/VidDet/models/definitions/rdnet/weights/` directory.

### The Minor Differences
We tested our implementation in comparison to the official code, and despite similar setups 
we found some minor differences:
1. In data loading the OpenCV resize operation gives a slightly different result, even with 
the same interpolation method selected. These differences result in many pixels being 1-off 
in value;

2. The [BatchNorm](https://beta.mxnet.io/api/gluon/_autogen/mxnet.gluon.nn.BatchNorm.html) 
calculation in Gluon also gives a slightly different result in comparison to the 
[SpatialBN](https://caffe2.ai/docs/operators-catalogue.html#spatialbn) of Caffe2;

3. We needed to keep the ReLUs as non-leaky for the pre-trained weights.
