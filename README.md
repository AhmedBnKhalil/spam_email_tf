
# Spam Email Detection Project

This project implements a machine learning model to detect spam emails using TensorFlow and the Word2Vec model for generating word embeddings.

## Description
The Spam Email Detection project is designed to classify emails into spam or not spam based on their content. It utilizes a neural network built with TensorFlow, incorporating Word2Vec embeddings trained on the email dataset to capture semantic meanings of words.

## Features

- Tokenization and preprocessing of email data.
- Training a Word2Vec model on the email dataset for word embeddings.
- A neural network model for spam detection, implemented in TensorFlow.
- Custom callback for monitoring training time.
- Evaluation and logging of model performance.

## Getting Started

### Dependencies

- Python 3.8+
- TensorFlow 2.x
- Gensim
- NLTK
- Pandas
- Scikit-learn

You can install the necessary libraries using `pip`:

```bash
pip install tensorflow gensim nltk pandas scikit-learn
```

### Installing
1. Clone this repository to your local machine.
2. Ensure you have all the required dependencies installed.

### Executing Program
Prepare your dataset of emails and labels.
Run the main script to train the model:
```bash
python main.py
```

1. The script will preprocess the data, train the Word2Vec model, train the neural network, and log the performance metrics.

   ## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

