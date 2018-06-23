# U-Net Model for segmenting medical data using a Transfer Learning approach on a pre-trained model.

This repository contains the implementation of a U-Net architecture using Keras with Tensorflow at its backened for segmenting any kind of medical data using a Transfer Learning approach. The neural network structure is derived from the *U-Net* architecture, described in this [paper](https://arxiv.org/pdf/1505.04597.pdf).  The transfer learning approach is used for fine tuning a pre trained U-Net model by using less number of samples since annotating the medical data is often a tedious job.

The following preprocessing is applied during the pre-training as well as for fine-tuning the pre-trained model:
- Gray-scale conversion
- Standardization
- Contrast-limited adaptive histogram equalization (CLAHE)
- Gamma adjustment

The training of the neural network is performed on sub-images (patches) of the pre-processed full images. Each patch, of dimension 48x48, is obtained by randomly selecting its center inside the full image. Also, the patches partially or completely outside the Field Of View (FOV) are selected, in this way the neural network learns how to discriminate the FOV border from the the segmentation region.

A set of 1,90,000 patches could be used for getting the pre-trained model and a set of 14,000 patches could be used for fie tuning the pre-trained model. These patches are obtained by randomly extracting them in each image for training. The first 90% of the dataset is used for training while the last 10% is used for validation.

Training is performed for 150 epochs, with a mini-batch size of 32 patches. Using an NVIDIA Tesla P100 GPU accelerator, the training of the pre-trained model lasts for about 4 hours while the fine tuning lasts for around 30 minutes.

For the testing purpose, only the pixels belonging to the FOV are considered. The FOV is identified with the masks included in the data. With a stride of 5 pixels in both height and width, multiple consecutive overlapping patches are extracted in each testing image. Then, for each pixel, the probability is obtained by averaging probabilities over all the predicted patches covering the pixel. This helps in improving the performance.


## Getting Started

Clone this repository to your local system. For implementing the pre trained model, DRIVE dataset is used which has both training samples as well as the annotations. The dataset can be obtained from the website :
[DRIVE](http://www.isi.uu.nl/Research/Databases/DRIVE/)

Download the dataset and extract the images into a folder and name it ‘DRIVE’. It will have the following tree:
```
DRIVE
│
└───test
|    ├───1st_manual
|    └───2nd_manual
|    └───images
|    └───mask
│
└───training
├───1st_manual
└───images
└───mask
```


The data which is used for fine tuning and segmentation should be copied into a folder named ‘Fine_tuning_data’ in the main directory with the same tree structure as that of the DRIVE dataset. Images represents the input samples, Mask represents the area where the cells are present and the manual represents the ground truth.


### Prerequisites
The following dependencies are required to run the U-Net model:
* Keras with Tensorflow at it's backend
* Pillow
* OpenCV
* h5py
* ConfigParser
* Scikit learn
* Matplotlib
* Graphviz
* Pydot

### Installing

* **Keras**
The installation procedures can be found  at [Keras](https://keras.io)
```
pip3 install keras
```
After installing Keras, make sure that the contents inside the keras.json file has **image_data_format": "channel_first"**

* **Pillow**
```
pip3 install pillow
```

* **OpenCV**
```
pip3 install opencv-contrib-python
```

* **h5py**
```
pip3 install h5py
```

* **ConfigParser**
```
pip3 install configparser
```


* **Scikit-learn**
```
pip3 install scikit-learn
```


* **Matplotlib**
```
“python3 –mpip install matplotlib” or “Pip3 install matplotlib”
```
* **Graphviz**
```
“pip3 install graphviz” or “brew install GraphViz”
```

* **Pydot**
```
“pip3 install pydot”
```

## Running the tests

For running the model for training and testing, make sure all the parameters in the configuration.txt file are set properly.

### For getting the Pre-trained weights

Make finetuning=False under [training settings] inside the configuration.txt file

Then run,
```
python prepare_datasets_DRIVE.py
```
This will will create the corresponding .hdf5 files for all the image sets which can be used for training

Now run,
```
python run_training.py
```
This will save the model weights in the particular folder with the [experiment name] mentioned in the configuration.txt file.


### Fine-tuning

Set the parameter **finetuning**  in the configuration.txt file to **True** for finetuning the model with the new dataset.
Again run,

```
python prepare_datasets_DRIVE.py
```
and followed by,
```
python run_training.py
```
This will update the pre-trained weights with the fine tuning data.

### Testing
Run
```
python run_testing.py
```
All the results will be saved in the folder with the name [experiment name] mentioned in the configuration.txt file.

