# Tennis
Code supporting the paper: ["TenniSet: A Dataset for Dense Fine-Grained Event Recognition, Localisation and Description"](http://hf.id.au/papers/DICTA17_Tennis.pdf)

### Note: This repository isn't quite complete. Will be completed over next couple of weeks.
## Prerequisites
- Linux or macOS
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN
- Suggest the use of a [virtualenv](http://www.pythonforbeginners.com/basics/how-to-use-python-virtualenv)

## Getting Started
### Installation
1. Install required packages:
    ```bash
    pip install numpy
    pip install pillow==4.3
    pip install easydict==1.7
    
    sudo apt-get install python3-tk
    ```

2. Install OpenCV (will need to do from source to get video functionality)
    1. Install FFMPEG
    
    2. [Download OpenCV](https://github.com/opencv/opencv/archive/3.3.1.zip), unzip and change to dir. 
        ```bash
        cd opencv-X.x.x
        mkdir build
        cd build
        cmake -D CMAKE_BUILD_TYPE=RELEASE -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=OFF -D PYTHON_DEFAULT_EXECUTABLE=~/virtualenvs/tennis-ve/bin/python3 ..

        ```
    3. Make sure cmake can find FFMPEG, and if using a virtualenv has that interpreter listed

    4. Run make etc.
        ```bash
        make
        sudo make install
        ```
        
### Configuration Setup
See `config.py` for the configuration parameters for the project, including default directory structure and path settings.

### The Data
The main data can be downloaded from Google Drive, the directory structure should be organised as follows:
```bash
/Tennis/
/Tennis/data/
/Tennis/data/videos/
/Tennis/data/annotations/
...
/Tennis/models
etc.
```

You can download each of the sub-directories here:
- [``annotations``, ``commentary``, ``slices``, ``splits``](https://drive.google.com/open?id=1g8D2rS-6O9L0G540VlLeSe2iBj6S_hpA) \[2.8 MB\]
- [``videos``](https://drive.google.com/open?id=1O55GYUC93vIerrRQDxfI_e6ECoAVy03j) \[11.1 GB\]
- [``frames``]() (coming soon) \[? GB\] (can be made using videos and code rather than downloaded)


## About
### The Annotator
The annotator can be used to annotate any video with dense temporal events using a GUI. See the README in the annotator directory for more information.

### The Data Processing
There are a number of processing files to extract and convert data. Descriptions of each files purpose can be found in the file header.

### The Models
More information on the models can be found in the README in the models directory.

### Evalutations
...

### Visualisations
...