# Emoji Recognition Model

This project implements a deep learning model for emoji recognition using TensorFlow/Keras. The model is trained on preprocessed emoji data to classify different emoji types.

## Project Structure
- `data_training.py`: Main script for training the model
- `model.h5`: Trained model file (generated after training)
- `labels.npy`: Saved labels file (generated after training)
- `.npy` files: Training data files

## Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- OpenCV

## Setup
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Place your .npy training data files in the project directory

3. Run the training script:
```bash
python data_training.py
```

## Model Architecture
The model uses a simple neural network architecture with:
- Input layer
- Dense layer (512 units, ReLU activation)
- Dense layer (256 units, ReLU activation)
- Output layer (softmax activation)

## Training
The model is trained for 50 epochs with a batch size of 32 using the RMSprop optimizer and categorical crossentropy loss.

## Output
After training, the script will:
1. Save the trained model as `model.h5`
2. Save the labels as `labels.npy`
3. Print the label dictionary mapping 