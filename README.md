# Tennis
Code supporting the paper: "TenniSet: A Dataset for Dense Fine-Grained Event Recognition, Localisation and Description"

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
See `config.py` for the configuration parameters for the project

### Running
#### Annotator
The annotator can be used to annotate any video with dense temporal events using a GUI.