# Contradictory, My Dear Watson

## Introduction
In this Kaggle competition, we aim to classify pairs of sentences into three categories: entailment, contradiction, or neutral. The dataset contains premise-hypothesis pairs in fifteen different languages. We will use BERT-based models to perform this task.

## Model Design and Implementation

We use the BERT (Bidirectional Encoder Representations from Transformers) model, which is a pre-trained deep learning model that has shown excellent performance in natural language processing (NLP) tasks, including sentence classification. Specifically, we use the BERT-base-multilingual-cased version, which is a BERT model that has been pre-trained on text in 104 languages.

We fine-tune the BERT model for our task of classifying pairs of sentences into three categories - entailment, contradiction, or neutral. We split the training set into training and validation sets to evaluate the performance of the model during training.

We use the transformers library from Hugging Face to load the pre-trained BERT model and tokenizer. We then use the tokenizer to tokenize the input text and convert it into input encodings that can be fed into the model. We create a custom PyTorch Dataset class to store the input encodings and labels.

We use the TrainingArguments and Trainer classes from the transformers library to set up the training and evaluation parameters and to train the model. We use a batch size of 16, train the model for three epochs, and use a learning rate scheduler with warmup steps and weight decay.

## Solution consists

+ train.py: This file contains the code to preprocess the data, build the model, train the model, and save the trained model.

+ predict.py: This file contains the code to load the saved model and predict the labels on the test set.

+ requirements.txt: This file lists the dependencies required to run the code.

## Development Environment and Dependencies
The project was developed using Python 3.8 and depends on the following libraries:

+ transformers 4.10.3
+ pandas 1.3.3
+ numpy 1.19.5
+ torch 1.9.0+cu111
