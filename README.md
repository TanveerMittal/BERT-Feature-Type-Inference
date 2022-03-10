# BERT for Feature Type Inference

This repository contains the release of 2 BERT CNN models for Feature Type Inference produced by my senior data science capstone at UC San Diego. This source code contains torch hub entrypoints to allow for anyone to easily use our models as well as code to benchmark these models against the ML Data Prep Zoo benchmark data.

This work was completed as a part of Project SortingHat out of the [ADA lab](https://adalabucsd.github.io/index.html) at UCSD with the mentorship of Professor Arun Kumar. 

## Resources:
------------------
- [Model Training and Experiment Code](https://github.com/TanveerMittal/Feature_Type_Inference_Capstone)
- [ML Data Prep Zoo](https://github.com/pvn25/ML-Data-Prep-Zoo)
- [Project Sortinghat](https://adalabucsd.github.io/sortinghat.html)

## Quick Start:
------------------
### Loading models from Torch Hub
To import the models from torch hub, user's only need the [HuggingFace Transformers Library](https://huggingface.co/) and [PyTorch](https://pytorch.org/get-started/locally/) installed.

```
# Name of this repo
repo = "TanveerMittal/BERT-Feature-Type-Inference"

# Load bert model using stats
model = torch.hub.load(repo, 'BERT_fti_with_stats', pretrained=True)

# Load bert model not using stats
model = torch.hub.load(repo, 'BERT_fti_no_stats', pretrained=True)
```
### Benchmark
In the benchmark folder, I have provided code for benchmarking our 2 released models in `run.py`. There is documented code for preprocessing the benchmark data in `process_data.py` and model prediction in `evaluation.py`.

Steps to benchmark the models:
1. Clone the repo and cd into the benchmark folder
2. `pip install -r requirements.txt`
3. `python run.py`

### Inference
As of now, I have put together code for an end to end pipline for Feature Type Inference on new data. Thankfully, a lot of work has already been done in this area by Project SortingHat. Please refer to the [SortingHat library](https://github.com/pvn25/ML-Data-Prep-Zoo/tree/master/MLFeatureTypeInference/Library) to preprocess new data for inference.

## Background:
------------------
The first step for AutoML software is to identify the feature types of individual columns in input data. This information then allows the software to understand the data and then preprocess it to allow machine learning algorithms to run on it. Project Sortinghat frames this task of Feature Type Inference as a machine learning multiclass classification problem. As an extension of Project SortingHat, I worked on applying transformer models to produce state of the art performance on this task. Our models currently outperform all existing tools currently benchmarked against SortingHat's ML Data Prep Zoo.

### Label Vocabulary
For this task, we have been using the following label vocabulary for our model's predictions:

| Feature Type      | Label |
|-------------------|-------|
| numeric           | 0     |
| categorical       | 1     |
| datetime          | 2     |
| sentence          | 3     |
| url               | 4     |
| embedded-number   | 5     |
| list              | 6     |
| not-generalizable | 7     |
| context-specific  | 8     |


### Preprocessing
Our machine learning models use the following 3 sets of features:

1. The name of the given column
2. 5 not null sample values
3. Descriptive numeric statistics computed from the given column

The descriptive statistics the model uses are listed below:

| Descriptive Stats                                                        |
|--------------------------------------------------------------------------|
| Total number of values                                                   |
| Number of nans and % of nans                                             |
| Number of unique values and % of unique values                           |
| Mean and std deviation of the column values, word count, stopword count, |
| char count, whitespace count, and delimiter count                        |
| Min and max value of the column                                          |
| Regular expression check for the presence of url, email, sequence of  delimiters, and list on the 5 sample values                              |
| Pandas timestamp check on 5 sample values                                |

### Transformer Model Architecture
As a part of our capstone I ran many experiments to find the optimal CNN architecture for this task. The diagram below shows the optimal architecture I found.

<p align="center">
<img  src="https://github.com/TanveerMittal/BERT-Feature-Type-Inference/blob/main/img/Best%20Model.png?raw=True">
</p>

Full documentation of my model and experiments can be found in our [tech report](https://tanveermittal.github.io/capstone/).

We released 2 models in this repository; one uses all 3 sets of features, the other uses just the column name and sample values and doesn't have a concatenation operation for the statistics.

The results of the models can be seen below:

- BERT CNN with Descriptive Statistics:
    - 9 Class Test Accuracy: **0.934**

| Data Type | numeric | categorical | datetime | sentence | url   | embedded-number | list  | not-generalizable | context-specific |
|-----------|---------|-------------|----------|----------|-------|-----------------|-------|-------------------|------------------|
| **Accuracy**  |   0.983 |       0.972 |        1 |    0.986 | 0.999 |           0.997 | 0.994 |             0.968 |            0.967 |
| **Precision** |   0.959 |       0.935 |        1 |    0.849 | 0.969 |           0.989 |  0.96 |             0.848 |             0.87 |
| **Recall**    |   0.996 |       0.943 |        1 |    0.859 | 0.969 |           0.949 | 0.842 |             0.856 |            0.762 |

- BERT CNN without Descriptive Statistics:
    - 9 Class Test Accuracy: **0.929**

| Data Type | numeric | categorical | datetime | sentence | url   | embedded-number | list  | not-generalizable | context-specific |
|-----------|---------|-------------|----------|----------|-------|-----------------|-------|-------------------|------------------|
| Accuracy  |   0.981 |       0.967 |    0.999 |    0.987 | 0.999 |           0.997 | 0.994 |             0.966 |            0.968 |
| Precision |   0.958 |       0.917 |    0.993 |    0.853 | 0.969 |            0.99 | 0.959 |             0.869 |            0.854 |
| Recall    |   0.992 |       0.941 |        1 |     0.88 | 0.969 |            0.96 | 0.825 |             0.805 |            0.789 |
