## Dog Breed Classification using InceptionV3, VGG19 CNN Model and by building the CNN architecture from scratch on Stanford Dogs Dataset
### Description
The <a href= "http://vision.stanford.edu/aditya86/ImageNetDogs/">Stanford Dogs Dataset</a> contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. There are 20,580 images, out of which 12,000 are used for training and 8580 for testing. Class labels and bounding box annotations are provided for all the 12,000 images.

-I have used the InceptionV3 CNN Model, which is pre-trained on the ImageNet dataset for classification. Data augementation has been used for making the model generalize better and also to avoid overfitting. The model achieved an accuracy of 80% on validation set, which is decent for this dataset.

-Then I have used the VGG19 CNN Model. which is pre-trained on the ImageNet dataset for classification. Data augementation hasn't been used for making the model generalize better. The model achieved an accuracy 23% on validation set. This is not a good situation, but I don't understand why.

-In last place,  I built vgg19 architecture from scratch via Tensorflow. I didn't use data augementation in this either for making the model generalize better. The model achieved an accuracy 1.25% for 10 epochs on validation set. This is a bad situation, but I don't understand why. I think there is not enough information for each class.

### Getting Started
The `standford_dog.ipynb` notebook can be directly run on Jupyter Notebook or others. Use GPU for faster training and evaluation.

#### Pre-Requisites
For running the notebook on your local machine, following pre-requisites must be satisfied:
- NumPy
- Pandas
- IPython
- Matplotlib
- Tensorflow 2.X
- Keras

#### Installation
**Dependencies:**
```
# With Tensorflow GPU
pip install -r requirements-gpu.txt
```
### Dataset
Contents of the dataset:
- Number of categories: 120
- Number of images: 20,580
- Annotations: Class labels, Bounding boxes

The dataset can be downloaded from <a href= "http://vision.stanford.edu/aditya86/ImageNetDogs/">here.</a>

Sample images of 9 different categories from the dataset:

![Images of Dogs](/images/dog_images.png)

  
### Steps
<br />
<b>Step 1.</b> Clone this repository: https://github.com/makhmudjumanazarov/CIFAR100.git via Terminal, cmd or PowerShell
<br/><br/>
<b>Step 2.</b> Create a new virtual environment 
<pre>
python -m venv cifar100
</pre> 
<br/>
<b>Step 3.</b> Activate your virtual environment
<pre>
source cifar100/bin/activate # Linux
.\cifar100\Scripts\activate # Windows 
</pre>
<br/>
<b>Step 4.</b> Install dependencies and add virtual environment to the Python Kernel
<pre>
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=cifar100
</pre>
<br/>
<b>Step 5.</b> Model to Train and Evaluate
<pre>
Open a CIFAR100.ipynb file via jupyter lab or jupyter notebook commands
</pre> 
<br/>


## CIFAR100 - Streamlit - Demo

CIFAR100 via Streamlit 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/makhmudjumanazarov/CIFAR100/main/app.py)
