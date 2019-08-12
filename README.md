# Tennis (Work in progress)
Code supporting the paper:
["TenniSet: A Dataset for Dense Fine-Grained Event Recognition, Localisation and Description"](http://hf.id.au/papers/DICTA17_Tennis.pdf)

Code is being converted into [MXNet](https://mxnet.apache.org/) and
[GluonCV](https://gluon-cv.mxnet.io/). With newly trained models
available soon.

***PLEASE NOTE:*** The results in the paper were with outdated Keras
models, new results will be presented below in this readme.


### The Data
The main data can be downloaded from Google Drive, the directory structure should be organised as follows:
```bash
/Tennis/
/Tennis/data/ (first download below, but without video or frames dirs)
/Tennis/data/annotations/
/Tennis/data/videos/
...
/Tennis/models
etc.
```

You can download each of the sub-directories here:
- [``annotations``, ``commentary``, ``slices``, ``splits``](https://drive.google.com/open?id=1g8D2rS-6O9L0G540VlLeSe2iBj6S_hpA) \[2.8 MB\] - The annotation files
- [``videos``](https://drive.google.com/open?id=1O55GYUC93vIerrRQDxfI_e6ECoAVy03j) \[11.1 GB\] - The original videos
- [``frames``]() (coming soon) \[? GB\] (can be made using videos and code rather than downloaded)
- [``flow``](https://drive.google.com/open?id=1d587RcqnGSk4A5Tze7UpWMGgjHCvD5nx) \[40 GB\] - The cropped and resized optical flow frames


## About
### The Annotator
The annotator can be used to annotate any video with dense temporal events using a GUI. See the README in the annotator directory for more information.

### The Data Processing
There are a number of processing files to extract and convert data. Descriptions of each files purpose can be found in the file header.

### The Models
Coming Soon - More information on the models can be found in the README in the models directory.

### Evalutations
...

### Visualisations
...