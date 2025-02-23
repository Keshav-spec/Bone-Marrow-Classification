# Bone Marrow Cell Classification

This project focuses on the classification of bone marrow cells using deep learning techniques. Leveraging a comprehensive dataset of bone marrow cell images, the developed model achieves an accuracy of **93.94%**, aiding in the diagnosis of hematologic diseases such as leukemia and lymphomas.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Accurate classification of bone marrow cells is crucial for diagnosing various hematologic disorders. Traditional manual examination methods are time-consuming and prone to human error. This project presents an automated approach using a convolutional neural network (CNN) to classify bone marrow cells, streamlining the diagnostic process and reducing the potential for errors.

## Dataset Overview

The dataset used in this project is the **[Bone Marrow Cell Classification](https://www.kaggle.com/datasets/andrewmvd/bone-marrow-cell-classification)** dataset from Kaggle. It comprises over **170,000 annotated images** of bone marrow cells from **945 patients**, stained using the May-Grünwald-Giemsa/Pappenheim technique. The images were acquired with a brightfield microscope at **40x magnification and oil immersion**. This extensive dataset provides a robust foundation for training and evaluating the classification model.

**Dataset Sample Images:**

![Alt Text](https://github.com/Keshav-spec/Bone-Marrow-Classification/blob/main/Screenshot%202025-02-23%20173717.png?raw=true)


## Model Architecture

The model is built using TensorFlow and Keras, employing a **convolutional neural network (CNN)** architecture. The architecture includes:

- Multiple **convolutional layers** for feature extraction
- **Max pooling layers** to reduce dimensionality
- Fully connected **dense layers** for classification
- **Dropout layers** to prevent overfitting

**Model Architecture Diagram:**

*Insert a diagram of the CNN architecture here, illustrating the flow from input images through convolutional layers to the output classification layer.*

## Training Process

The training process involves:

- **Data Preprocessing:** Normalizing images and augmenting the dataset to enhance model generalization.
- **Compilation:** Using the Adam optimizer with a learning rate scheduler to adjust the learning rate during training.
- **Callbacks:** Implementing custom callbacks to monitor performance and adjust learning rates dynamically based on validation metrics.

**Training and Validation Accuracy/Loss Plots:**

*Include plots showing the training and validation accuracy and loss over epochs to visualize the model's learning progress.*

## Results

The model achieved an accuracy of **93.94%** on the validation set, demonstrating its effectiveness in classifying bone marrow cells. This high accuracy indicates the model's potential utility in assisting medical professionals with hematologic disease diagnosis.

**Confusion Matrix:**

*Insert a confusion matrix here to provide insight into the model's performance across different cell classes.*

## Usage

### 1. Clone the Repository:

```bash
git clone https://github.com/yourusername/bone-marrow-classification.git
cd bone-marrow-classification
```

### 2. Install Dependencies:

Ensure you have Python 3.x and install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset:

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/bone-marrow-cell-classification) and place it in the `data/` directory.

### 4. Train the Model:

Run the training script:

```bash
python train.py
```

This will preprocess the data, train the model, and save the trained model to the `models/` directory.

### 5. Make Predictions:

Use the trained model to make predictions on new images:

```bash
python predict.py --image_path path_to_image
```

Replace `path_to_image` with the path to your image file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

