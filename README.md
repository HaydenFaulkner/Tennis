<h1 align='center'>Tennis</h1>
<p align=center>
A Tennis dataset and models for event detection & commentary generation. Discussed in <a href="http://hf.id.au/papers/DICTA17_Tennis.pdf">"TenniSet: A Dataset for Dense Fine-Grained Event Recognition, Localisation and Description"</a>


<b>IMPORTANT NOTE</b>

The results in the paper were with outdated <a href="https://keras.io/">Keras</a> models, new results listed below are 
with updated <a href="https://mxnet.apache.org/">MXNet and Gluon</a> models.
</p>


## The Dataset

### Overview
The tennis dataset consists of five matches taken from YouTube and has manually annotated temporal events and commentary 
captions.

| Type | Attributes | # Events | # Frames | Frames per Event |
| :---:        |     :---:      |        :---: |     :---:      |        :---: |
| **`match`**   | `winner`     | 5    | 786455 | 157291 |
| **`set`**   | `winner`, `score`  | 11    | 765738 | 69613 |
| **`game`**   | `winner`, `score`, `server`     | 118    | 588759 | 4989 |
| **`point`**   | `winner`, `score`     | 746    | 159494 | 214 |
| **`serve`**   | `near/far`, `in/fault/let`    | 1017   | 68385 | 67 |
| **`hit`**   | `near/far`, `left/right`  | 2551    | 73564 | 29 |


Individual shots (`serve` and `hit`) are used to generate **11** temporal event categories:
<img src="img/tennis_cls_hier.svg" align=center>

More about the sample numbers for these individual classes can be seen below in the split information.



### The Annotator
The [annotator](https://github.com/HaydenFaulkner/TemporalEventAnnotator) was used to annotate the videos with
dense temporal events. See the README in the
[TemporalEventAnnotator](/TemporalEventAnnotator/) directory for more information.

### Data Downloading and Pre-processing
See [data](/data/) for download and organisation information.

Once you have `.json` annotation files with the annotator, you can run:
```
python utils/annotations/preprocess.py
```

This does pre-processing on the annotations, specifically:
1. Generates slice `.txt` files for each `.json` annotation file
2. Generalises the `.json` annotation files from player names and
forehand/backhand to near/far and left/right
3. Generates label `.txt` files for each generalised `.json` annotation
 file

Alternatively you can download our annotations `.tar.gz`
([see data](/data/))

### The Splits

## The Models
Coming Soon - More information on the models can be found in the README in the models directory.

### Feature extraction
After you have trained a model you can save the backbone output as `.npy` features by using `--save_feats`
```
--model_id 0006 --backbone DenseNet121 --num_gpus 1 --num_workers 16 --save_feats
```
Features will be saved in `\data\features\$model_id$\` with the same structure as `\data\frames\`.

### Evalutations
...

### Visualisations
...

### Captioning
Requires the [nlg-eval](https://github.com/Maluuba/nlg-eval) package. Install this first as per instructions on their 
[Github](https://github.com/Maluuba/nlg-eval).