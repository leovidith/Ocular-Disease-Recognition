# Ocular Disease Recognition

## Overview
This project utilizes a custom dataset designed for binary classification tasks. The dataset comprises labeled images belonging to two classes, "N" and "X," and is split into training, validation, and testing sets to facilitate model evaluation.

<img src="https://github.com/leovidith/Ocular-Disease-Recognition/blob/master/Sample%20data.png" alt="Centered Image" style="max-width: 100%; height: auto;">

## Agile Features (Sprints)
- **Sprint 1:** Data preprocessing and augmentation
  - Performed image resizing, normalization, and augmentation (rotation, flipping, and scaling).
  - Split data into training, validation, and test sets.

- **Sprint 2:** Model architecture design
  - Designed a deep convolutional neural network with layers for feature extraction and classification.

- **Sprint 3:** Model training and evaluation
  - Trained the model using the training set.
  - Evaluated performance on the validation and test sets.

- **Sprint 4:** Performance analysis
  - Analyzed metrics such as accuracy, loss, precision, recall, and F1-score.
 
## Principle Component Analysis (PCA):
- **Principal Component Analysis (PCA)** is a dimensionality reduction technique used to reduce the number of features in a dataset while retaining as much variance as possible.
- PCA transforms data into a set of orthogonal (uncorrelated) components, ordered by the amount of variance they explain.
- It helps in visualizing high-dimensional data, improving model performance by removing noise, and speeding up training times for machine learning algorithms.
- In the project, PCA was applied to reduce the complexity of features, enabling more efficient data processing and enhancing the performance of the model.
<img src="https://github.com/leovidith/Ocular-Disease-Recognition/blob/master/PCA.png" alt="Centered Image" width:700px>


## Model Specifications
| Layer (type)                     | Output Shape            | Param #    |
|----------------------------------|-------------------------|------------|
| conv2d_8 (Conv2D)                | (None, 224, 224, 256)  | 7,168      |
| conv2d_9 (Conv2D)                | (None, 224, 224, 256)  | 590,080    |
| conv2d_10 (Conv2D)               | (None, 224, 224, 256)  | 590,080    |
| max_pooling2d_4 (MaxPooling2D)   | (None, 112, 112, 256)  | 0          |
| conv2d_11 (Conv2D)               | (None, 112, 112, 128)  | 295,040    |
| conv2d_12 (Conv2D)               | (None, 112, 112, 128)  | 147,584    |
| max_pooling2d_5 (MaxPooling2D)   | (None, 56, 56, 128)    | 0          |
| dropout_3 (Dropout)              | (None, 56, 56, 128)    | 0          |
| conv2d_13 (Conv2D)               | (None, 56, 56, 64)     | 73,792     |
| conv2d_14 (Conv2D)               | (None, 56, 56, 64)     | 36,928     |
| max_pooling2d_6 (MaxPooling2D)   | (None, 28, 28, 64)     | 0          |
| dropout_4 (Dropout)              | (None, 28, 28, 64)     | 0          |
| conv2d_15 (Conv2D)               | (None, 28, 28, 32)     | 18,464     |
| max_pooling2d_7 (MaxPooling2D)   | (None, 14, 14, 32)     | 0          |
| flatten_1 (Flatten)              | (None, 6272)           | 0          |
| dense_2 (Dense)                  | (None, 256)            | 1,605,888  |
| dropout_5 (Dropout)              | (None, 256)            | 0          |
| dense_3 (Dense)                  | (None, 2)              | 514        |

**Total params:** 3,365,538 (12.84 MB)  
**Trainable params:** 3,365,538 (12.84 MB)  
**Non-trainable params:** 0

## Results
### Training and Validation
- Training Loss: 0.0762, Training Accuracy: 98.36%
- Validation Loss: 0.0977, Validation Accuracy: 97.11%
- Test Loss: 0.0930, Test Accuracy: 97.73%

### Classification Metrics
| Metric          | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| **N**           | 0.98      | 0.97   | 0.98     | 591     |
| **X**           | 0.97      | 0.98   | 0.98     | 687     |
| **Accuracy**    |           |        | 0.98     | 1278    |
| **Macro Avg**   | 0.98      | 0.98   | 0.98     | 1278    |
| **Weighted Avg**| 0.98      | 0.98   | 0.98     | 1278    |

### Accuracy and Loss Curves:
<img src="https://github.com/leovidith/Ocular-Disease-Recognition/blob/master/Loss%20curves.png" alt="Centered Image" style="max-width: 100%; height: auto;">

## Conclusion
The designed convolutional neural network achieved excellent performance on the binary classification task, with high accuracy, precision, recall, and F1-scores across all datasets. This demonstrates the model's robustness and effectiveness in handling the given dataset.

