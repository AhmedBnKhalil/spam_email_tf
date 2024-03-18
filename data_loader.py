import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def load_data(emails, labels, test_size=0.2, val_size=0.25, max_words=2500, max_length=120):
    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(emails)
    sequences = tokenizer.texts_to_sequences(emails)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    # Split data into training and test set
    X_temp, X_test, y_temp, y_test = train_test_split(padded_sequences, np.array(labels), test_size=test_size,
                                                      random_state=42)

    # Split remaining data into training and validation set
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size,
                                                      random_state=42)  # val_size here is relative to X_temp size

    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer.word_index