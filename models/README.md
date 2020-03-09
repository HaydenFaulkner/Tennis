<h1 align="center">Models</h1>
<p align="center">A selection of the best models are available for <a href="https://drive.google.com/open?id=1vODBn10jtQ_MEpxdKXLlG6h0j0JchTBD">download from my Google Drive</a>. After downloading simply store the pre-trained model directories either the <a href="vision/experiments"><code>vision/experiments</code></a> or <a href="captioning/experiments"><code>captioning/experiments</code></a> directory.</p>
<p align="center">A summary of the models and their results is below</p>
<p align="center">TODO Add summary table</p>


<h2></h2>
<h2 align="center">Vision</h2>
<h3 align="center">Framewise CNN</h3>
<p align="center">The first model (with ID <code>0006</code>) and basis for many other experiments was a framewise <a href="https://arxiv.org/pdf/1608.06993.pdf">DenseNet-121</a> architecture, this can be evaluated with</p>
<p align="center"><code>$ python evaluate.py --model_id 0006 --backbone DenseNet121</code></p>
<p align="center"><img src="../img/densenet.svg"></p>

<p align="center">.......</p>
<h3 align="center">Temporal Pooling</h3>
<p align="center", color="red">TODO</p>


<p align="center">.......</p>
<h3 align="center">CNN - RNN</h3>
<p align="center">The CNN-RNN model (with ID <code>0042</code>) utilises the pretrained framewise <a href="https://arxiv.org/pdf/1608.06993.pdf">DenseNet-121</a> architecture (<code>0006</code>), this can be evaluated with</p>
<p align="center"><code>$ python evaluate.py --model_id 0042 --backbone DenseNet121 --temp_pool gru --window 30 --backbone_from_id 0006 --feats_model 0006 --freeze_backbone</code></p>
<p align="center"><img src="../img/cnnrnn.svg"></p>


<p align="center">.......</p>
<h3 align="center">Two Stream</h3>
<p align="center", color="red">TODO</p>


<p align="center">.......</p>
<h3 align="center">R(2+1)D</h3>
<p align="center", color="red">TODO</p>

<h2></h2>
<h2 align="center">Captioning</h2>
<p align="center", color="red">TODO</p>