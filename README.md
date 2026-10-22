# MRI brain tumor classification
## Introduction
The model can identify four types of brain tumor (glioma tumor, meningioma tumor, pituitary tumor and no tumor) from 2D MRI grayscale images. This project contains saved model parameters and trainings which can be loaded and used.
The whole project is written in Python and based on PyTorch. The CNN structure is based on the method proposed in an article by [Rasheed, Z. et al.](https://www.mdpi.com/2076-3425/13/9/1320), where it's stated that the method achieved an accuracy of 97.84%.

## Setup
First of all, make sure you have python 3.14.x and git installed on your device.

Clone the repository to your target directory with the following command in the terminal:
```
git clone https://github.com/FansjaCZ/mri-brain-tumor-classification.git
```

Before installing dependencies, make sure your GPU is compatible with cuda12.6. If your GPU is not NVIDIA, it may not be compatible by default. Install dependencies and setup the project using the following command:
```
pip install -e .
```
Or install the depenadncies manually from `pyproject.toml/`. A virtual environment is strongly recommended.

## How to use
Set your preferences in `config.toml/`. The initial configuration will function, however you may need to set `device` to `"cpu"` instead of `"cuda"` if your device does not have a GPU or your GPU is not compatible with PyTorch. This may slow down training drastically, so it is not recommended.

The model can be trained using the `train_model.py` script. This will create a training save in the target directory. This training save can also be used to load the model for predictions. A full training save save was already created in `.\data\saves\default.tar`.

To start your own training run the following command on the main directory.
```
python3 .\scripts\train_model.py .\data\saves\<your_training_name>.tar
``` 

To run a prediction, the `use_model.py` script can be applied. This takes two arguments: the path to the image and the path to a training save to load parameters from. The second parameter is optional. By default the default training file will be used (set in the config).

To run a prediction on a test image from the nickparvar dataset run the following command.
```
python3 .\scripts\use_model.py .\data\dataset\Testing\meningioma\Te-aug-me_1.jpg .\data\saves\<your_training_name>.tar
```

Evaluating (old) trainings can be done with the `plot_training.py` script. This script takes the target training save as its argument. This script plots accuracy, avarage loss and the learning rate recorded from the last training. A matplotlib tab will open after running the script.

Plotting a graph using the script can be done with the following command.
```
python3 .\scripts\plot_graph.py .\data\saves\<your_training_name>.tar
```

## Specs
The model was trained with the following specs:
- GPU: NVIDIA GeForce RTX 3050 8GB
- CPU: AMD Ryzen 5 5600x 6-Core Processor

It is reccomended to have similar or better specs to train the model.

## References
Rasheed, Z., Ma, Y.-K., Ullah, I., Ghadi, Y. Y., Khan, M. Z., Khan, M. A., Abdusalomov, A., Alqahtani, F., & Shehata, A. M. (2023). Brain Tumor Classification from MRI Using Image Enhancement and Convolutional Neural Network Techniques. Brain Sciences, 13(9), 1320. https://doi.org/10.3390/brainsci13091320 