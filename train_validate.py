import tensorflow as tf
from tensorflow.keras.models import Sequential
from time_callback import TimeHistory


def build_model(vocab_size, embedding_matrix, embedding_dim=1000):
    model = Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                  embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                  trainable=False),
        tf.keras.layers.LSTM(500, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64,return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def compile_and_train(X_train, y_train, X_val, y_val, vocab_size, embedding_matrix=None):
    embedding_dim = embedding_matrix.shape[1] if embedding_matrix is not None else 100  # Default to 100 if no matrix
    model = build_model(vocab_size, embedding_matrix, embedding_dim=embedding_dim)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    time_callback = TimeHistory()
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val),callbacks=time_callback)
    total_training_time = time_callback.total_time
    epoch_times = time_callback.times
    return model, history,total_training_time,epoch_times
