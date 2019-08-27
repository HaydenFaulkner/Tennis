# Tennis (Work in progress)
Code supporting the paper:
["TenniSet: A Dataset for Dense Fine-Grained Event Recognition, Localisation and Description"](http://hf.id.au/papers/DICTA17_Tennis.pdf)

Code is being converted into [MXNet](https://mxnet.apache.org/) and
[GluonCV](https://gluon-cv.mxnet.io/). With newly trained models
available soon.

***PLEASE NOTE:*** The results in the paper were with outdated Keras
models, new results will be presented below in this readme.

## About
### The Annotator
The [annotator](/annotator/) can be used to annotate any video with
dense temporal events using a GUI. See the README in the
[annotator](/annotator/) directory for more information.

### Data Pre-processing
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

### The Models
Coming Soon - More information on the models can be found in the README in the models directory.

### Evalutations
...

### Visualisations
...