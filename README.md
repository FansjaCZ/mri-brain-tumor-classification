# MRI brain tumor classification
## Introduction
The model can identify four types of brain tumor (glioma tumor, meningioma tumor, pituitary tumor and no tumor) from 2D MRI grayscale images. This project contains saved model parameters and trainings which can be loaded and used.
The whole project is written in Python and based on PyTorch. The CNN structure is based on the method proposed in an article by [https://www.mdpi.com/2076-3425/13/9/1320](Rasheed, Z. et al.), where it's stated that the method achieved an accuracy of 97.84%.

## Setup
Make sure you have python 3.14.x and git installed on your device.
Clone the repository to your target directory with 
```
git clone https://github.com/FansjaCZ/mri-brain-tumor-classification.git
```
in the terminal.
Install dependencies and setup project using 
```
pip install -e .
```
 or install the depenadncies manually from `pyproject.toml/`. A virtual environment is strongly recommended.

## How to use
There are two types of saves you can make: training saves and model saves. Training saves contain all data from the training, optimizer and model. Model saves contain only model parameters. Therefore model saves can be used during predictions and training saves can be used to save training progress and data.

A full training save and model save have already been created in `.\data\training_saves\default.tar` and `.\data\model_saves\default.pt`.
To start your own training run 
```
python3 .\scripts\train_model.py .\data\training_saves\<your_training_name>.tar
``` 
in the terminal on the main directory. It's good practice to use the `.tar` format to save trainings, as recommended by the official PyTorch documentation.
To save model parameters from a training execute 
```
python3 .\scripts\save_model.py .\data\training_saves\<your_training_name>.tar .\data\model_saves\<your_model_save_name>.pt
```
in the terminal from the main directory. Use the `.pt` extension for good practice.

Scripts to run predictions on images using a model save will be added soon.

## References
Rasheed, Z., Ma, Y.-K., Ullah, I., Ghadi, Y. Y., Khan, M. Z., Khan, M. A., Abdusalomov, A., Alqahtani, F., & Shehata, A. M. (2023). Brain Tumor Classification from MRI Using Image Enhancement and Convolutional Neural Network Techniques. Brain Sciences, 13(9), 1320. https://doi.org/10.3390/brainsci13091320 