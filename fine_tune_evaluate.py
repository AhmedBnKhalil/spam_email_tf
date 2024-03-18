import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf

def evaluate_and_log_model(model, history, X_test, y_test, log_filename="model_logs.csv", learning_rate=None, batch_size=None):
    # Perform evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Generate predictions
    predictions = (model.predict(X_test) > 0.5).astype(int)
    predictions_proba = model.predict(X_test).ravel()  # Use probabilities for AUC

    # Calculate metrics
    report = classification_report(y_test, predictions, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    auc_score = roc_auc_score(y_test, predictions_proba)  # Calculate AUC

    # Extract model parameters for logging
    num_layers = len(model.layers)
    embedding_output_dim = model.layers[0].output_dim if isinstance(model.layers[0], tf.keras.layers.Embedding) else None
    optimizer_name = type(model.optimizer).__name__

    # Extract metrics from the training history
    final_epoch = len(history.history['loss'])

    # Create log entry with new parameters and AUC score
    log_entry = {
        "final_epoch": final_epoch,
        "final_accuracy": history.history['accuracy'][-1],
        "final_val_accuracy": history.history['val_accuracy'][-1],
        "final_loss": history.history['loss'][-1],
        "final_val_loss": history.history['val_loss'][-1],
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "auc_score": auc_score,
        "num_layers": num_layers,
        "embedding_output_dim": embedding_output_dim,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }

    # Convert the log entry to a DataFrame
    df_log = pd.DataFrame([log_entry])

    # Append the log entry to the CSV file
    with open(log_filename, 'a') as f:
        df_log.to_csv(f, header=f.tell() == 0, index=False)

    # Print out the classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print(f"AUC Score: {auc_score}")
