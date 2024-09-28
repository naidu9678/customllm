import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras import models
from data import load_data

def evaluate_model():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Create a combined vocabulary from both training and testing data
    combined_text = " ".join(X_train) + " " + " ".join(X_test)
    vocab = {word: idx + 1 for idx, word in enumerate(set(combined_text.split()))}

    # Function to convert text to numerical format with handling for unknown words
    def text_to_numerical(text):
        return [vocab.get(word.lower(), 0) for word in text.split()]

    # Convert strings to numerical format
    X_train_numerical = [text_to_numerical(question) for question in X_train]
    X_test_numerical = [text_to_numerical(question) for question in X_test]

    # Print the vocabulary to check
    print("Vocabulary:", vocab)

    # Pad sequences to ensure they have the same length
    max_length = max(len(seq) for seq in X_train_numerical + X_test_numerical)
    X_train_padded = pad_sequences(X_train_numerical, maxlen=max_length, padding='post')
    X_test_padded = pad_sequences(X_test_numerical, maxlen=max_length, padding='post')

    # Update y_test to match the size of X_test
    y_test = np.array([0] * len(X_test))  # Adjust based on your actual test labels

    # Load the model
    model = models.load_model('chatbot_model.h5')

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_padded, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate_model()
