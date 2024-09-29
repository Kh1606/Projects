# Rice Leaf Disease Classification

This repository contains a project for classifying diseases in rice leaves using machine learning techniques, specifically leveraging PyTorch and the ResNet18 model. The notebook walks through dataset preparation, model training, and evaluation processes.

## Project Overview

The goal of this project is to create a classifier that can accurately identify different diseases affecting rice leaves. The dataset used for training is sourced from Kaggle and contains multiple classes of rice leaf diseases.

### Features

- **Dataset Handling**: Custom dataset class using PyTorch, handling image loading, transformations, and labels.
- **Data Visualization**: Visualizes a subset of images from each class for better understanding.
- **Model Training**: Utilizes ResNet18 for feature extraction and classification.

## Dataset

The dataset is obtained from Kaggle and contains multiple classes of rice leaf diseases. You can use the provided function to automatically download the dataset and organize it into folders.

### Dataset Download

To download and prepare the dataset, run the following command:

```python
data_yuklab_olish(saqlash_uchun_papka = "data")
```

### Data Splitting
The dataset is split into training, validation, and test sets with an 80-10-10 ratio:
```python
total_len = len(ds)
tr_len = int(total_len * 0.8)
vl_len = int(total_len * 0.1)
ts_len = total_len - (tr_len + vl_len)
tr_ds, vl_ds, ts_ds = random_split(dataset=ds, lengths=[tr_len, vl_len, ts_len])
bs = 14
```
### Model Training
The model used for this project is ResNet18. This pre-trained model is leveraged for transfer learning to classify rice leaf diseases.

