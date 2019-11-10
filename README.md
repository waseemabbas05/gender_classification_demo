# gender_classification_demo
This repository contains a a CNN classifier for gender classification on CelebA benchmark
The 'scripts' folder has one jupyter notebook and three python files.
To run the demo, either launch the jupyter notebook or run "gender_classification_demo.py"

### Requirements
Python 3+

Tensorflow 1.14+

pip install tensorflow-gpu==1.14 (for gpu version)

pip install tensorflow==1.14 (for CPU only version)

matplotlib

pandas (pip install pandas)

numpy (pip install numpy)

glob 

shutil (pip install pytest-shutil)


#### The model
The classification is a standard CNN model with a few skip connections added to the convolutional layers. The model has three convolutionl layers. Each convolutional layer has two 3 by 3 convolutions. The input to the convolution is copied and concatenated to the output of the convolutions to add a skip connection. This is followed by average pooling, batch norm and relu activation.

The convolutional layers are followed by 4 fully connected layers and then an output layer. The FC layers have 128, 64, 32 and 16 nodes and the output layer has 2 nodes, one for each class. The FC layers have relu activation while the output layer has softmax activation. 

A block diagram of the network is shown below 
![image](https://user-images.githubusercontent.com/5336269/68546976-b1450380-03dc-11ea-823d-3904793f2c26.png)

### The code hierarchy
This code is divided in 2 hierarchies. The top hierarchy has the code for training and inference. It includes the jupyter notebook "gender_classification_demo.ipynb" and pyhton file "gender_classification_demo.py". The notebook can be launched either on colab or any other jupyter capable platforms while the python file can be called in any python IDE or command line.

The second hierarchy has two files, "read_data.py" and "model.py".

"read_data.py" contains sub-routines for downloadibg the CelebA dataset, splitting it into training, validation and test subsets, extract features, do some pre-processing, get random batches etc. It can be called from main. 

"model.py" contains sub-routines for building the inference graphs, cost functions and supporting functions. At this point, it only accepts 2 arguments when called externally. One; a batch of images or a single image and a boolean variable which tells it if it should run in training mode or inference mode. The model is illustrated in the figure above.

### Training
