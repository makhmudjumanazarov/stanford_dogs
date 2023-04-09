# Stanford dogs
<p>Description:
The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. There are 20,580 images, out of which 12,000 are used for training and 8580 for testing. Class labels and bounding box annotations are provided for all the 12,000 images.
  
## Steps
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


# CIFAR100 - Streamlit - Demo

CIFAR100 via Streamlit 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/makhmudjumanazarov/CIFAR100/main/app.py)
