# FlowNet Model
I use the FlowNet-S model described in the paper: 
["FlowNet: Learning Optical Flow with Convolutional Networks"](https://arxiv.org/pdf/1504.06852.pdf) 
to generate our Optical Flow images.

I have converted the weights from the [NVIDIA FlowNet2](https://github.com/NVIDIA/flownet2-pytorch) models from
PyTorch to Gluon.

Please download the params from my [Google Drive](https://drive.google.com/open?id=1wA3lPxSPc4rKQoz6-8Pr077MulnHx0Sf), 
and place it in this directory (you will only need `FlowNet2-S_checkpoint.params`). 

To generate the flow images just execute the `run.py` script. I will also allow for the download of the flow images 
directly soon.