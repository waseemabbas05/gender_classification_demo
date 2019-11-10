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

### Reasoning

* The reason I chose a CNN based model is to let the model learn the features instead of giving it hand crafted features. We cannot objectively find out exactly which features help our visual system identify a gender. We might not even know the specific feature set our vosual cortex employs for gender identification. Therefore, instead of going for a features based classifier, I went for CNN based classifier.

* The reason why I added skip connections in the convolutional layers is to add complexity to the learned feature space. My intuotion is that if we let the input image of the CNN layer to guide the learned features, it can result in better feature space. This is why I added a skip connection after the two 3 by 3 convolutions. 

* I chose weighted cross entropy as the cost function because I observed that the classes are not well balanced.

* I chose 4 fully connected layer to bring down the number of features gradually instead of bringing them down from more than 300 to just 2 by a single output layer.

### The code hierarchy
This code is divided in 2 hierarchies. The top hierarchy has the code for training and inference. It includes the jupyter notebook "gender_classification_demo.ipynb" and pyhton file "gender_classification_demo.py". The notebook can be launched either on colab or any other jupyter capable platforms while the python file can be called in any python IDE or command line.

The second hierarchy has two files, "read_data.py" and "model.py".

"read_data.py" contains sub-routines for downloadibg the CelebA dataset, splitting it into training, validation and test subsets, extract features, do some pre-processing, get random batches etc. It can be called from main. 

"model.py" contains sub-routines for building the inference graphs, cost functions and supporting functions. At this point, it only accepts 2 arguments when called externally. One; a batch of images or a single image and a boolean variable which tells it if it should run in training mode or inference mode. The model is illustrated in the figure above.

### Training
Before starting the training, the user should specify a folder for downloading the data and where the kernel can be stored (not included at this point). Since the CelebA dataset contains a lot of images, the unzipping process can take a while so be patient.

Once the data is downloaded, the rest of the code can be run as it is. However, if the user wants to change model structure, they can do so by editing the "model.py" file. They can change the number of filters, change activation function, add or remove convolutional layers

At this time, the training code loops over all the batches available in training data once. The user can change how many epochs to run and how many batches to include. They can simply increase the number of epochs to any number they want. However, the number of batches should be less than the total number of batches available. The total number of batches depend on the batch size selected by the user

### Inference
Once the model is trained, the user can simply pass an image or batch of images to the "prediction" function/graph defined in the main and it will return a class label for the image or the batch of images.
