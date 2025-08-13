# Dogs vs. Cats Image Classification

## 📌 Project Overview
This project classifies images of dogs and cats using a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras.  
The dataset is sourced from Kaggle's **Dogs vs. Cats** dataset and processed in Google Colab.

## 📂 Dataset
- **Source:** [Kaggle - Dogs vs. Cats](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- Downloaded via Kaggle API.
- Extracted and organized into training, validation, and test sets.

## 🛠️ Tools & Libraries
- **Python 3**
- **TensorFlow / Keras**
- **Kaggle API**
- **Google Colab**
- **NumPy, Matplotlib**

## ⚙️ Project Workflow
1. **Dataset Preparation**
   - Download and extract dataset.
   - Load images using `image_dataset_from_directory()`.
   - Split into training, validation, and test sets.
2. **Model Architecture**
   - Sequential CNN with Conv2D, MaxPooling, Batch Normalization, Dropout, Dense layers.
   - Sigmoid output layer for binary classification.
3. **Training**
   - Loss: Binary Crossentropy
   - Optimizer: Adam
   - Metrics: Accuracy
   - Early stopping to prevent overfitting (optional).
4. **Evaluation**
   - Validate model on unseen test data.
   - Track accuracy and loss metrics.

## 📊 Results
- The model achieved high accuracy on validation data.
- Overfitting risks can be reduced using data augmentation and dropout.

## 🚀 Future Improvements
- Implement **data augmentation** (rotation, flipping, zoom).
- Use **transfer learning** with pretrained models like VGG16, ResNet.
- Perform hyperparameter tuning.

## ▶️ How to Run
1. Clone the repository.
2. Install required libraries:
   ```bash
   pip install tensorflow keras kaggle matplotlib numpy
   ```
3. Set up Kaggle API credentials (`kaggle.json`).
4. Run the Jupyter Notebook in Google Colab or locally.

--- 
