import pandas as pd

from data_loader import *
from fine_tune_evaluate import evaluate_and_log_model
from train_validate import compile_and_train
from train_word2vec import tokenize_emails, train_word2vec

data = pd.read_csv('./spam_ham_dataset.csv')
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)  # Use maximum width for displaying

emails = data['text']
labels = data['label_num']
tokenized_emails = tokenize_emails(emails)
w2v_model = train_word2vec(tokenized_emails)

X_train, X_val, X_test, y_train, y_val, y_test, word_index = load_data(tokenized_emails, labels)
print(f"Training set features shape: {X_train.shape}")
print(f"Training set labels shape: {y_train.shape}\n")

print(f"Validation set features shape: {X_val.shape}")
print(f"Validation set labels shape: {y_val.shape}\n")

print(f"Test set features shape: {X_test.shape}")
print(f"Test set labels shape: {y_test.shape}\n")

print(f"Size of word_index: {len(word_index)}\n")

vocab_size = len(word_index) + 1
embedding_dim = 1000  # Make sure this matches your Word2Vec training setting
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if word in w2v_model.wv:
        embedding_vector = w2v_model.wv[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

model, history, total_training_time, epoch_times = compile_and_train(X_train, y_train, X_val, y_val, vocab_size,
                                                                     embedding_matrix=embedding_matrix)
model.summary()
print(total_training_time, '\n', epoch_times)
evaluate_and_log_model(model, history, X_test, y_test)
