# Emotion-Prediction

This project aims to predict emotions from facial expressions using deep learning. The model is trained on the FER 2013 dataset from Kaggle and the training weights are saved in an H5 file.
This was a group project with my dear friends Ammari Hiba and Lfellous Rim

## Project Structure

- `model0.ipynb`: This notebook contains the code for training the model. It will save the model weights to a file named `model.weights.h5`.
- `emojis.ipynb`: This notebook uses the trained model to predict emotions and display corresponding emojis.

## Setup Instructions

### Step 1: Download the Dataset

The dataset used in this project is FER 2013, which can be downloaded from Kaggle.

- Download the training and testing datasets from the following link: [FER 2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

### Step 2: Update the Dataset Paths

After downloading the datasets, update the paths in the `model0.ipynb` file to point to your local copies of the dataset files.

For example:
```python
train_dir = 'path/to/your/downloaded/train_data'
val_dir = 'path/to/your/downloaded/val_data'
```
### Step 3: Train the Model

Open and run the model0.ipynb notebook.
This will train the model on the FER 2013 dataset and save the trained weights to model.weights.h5.

### Step 4: Predict Emotions and Display Emojis

Open and run the emojis.ipynb notebook.
This notebook will use the trained model to predict emotions from facial images and display corresponding emojis.

### Requirements:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
