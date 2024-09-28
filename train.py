import numpy as np
from data import load_data
from model import create_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def train_model():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Create a combined vocabulary from both training and testing data
    combined_text = " ".join(X_train) + " " + " ".join(X_test)
    vocab = {word: idx + 1 for idx, word in enumerate(set(combined_text.split()))}

    # Function to convert text to numerical format with handling for unknown words
    def text_to_numerical(text):
        return [vocab.get(word.lower(), 0) for word in text.split()]  # Use 0 for unknown words

    # Convert strings to numerical format
    X_train_numerical = [text_to_numerical(question) for question in X_train]
    X_test_numerical = [text_to_numerical(question) for question in X_test]

    # Pad sequences to ensure they have the same length
    max_length = max(len(seq) for seq in X_train_numerical + X_test_numerical)
    X_train_padded = pad_sequences(X_train_numerical, maxlen=max_length, padding='post')
    X_test_padded = pad_sequences(X_test_numerical, maxlen=max_length, padding='post')

    # Use appropriate labels for y_train (dummy labels for demonstration)
    y_train = np.array([0, 1, 2, 3])  # Ensure this matches the number of training samples

    # Create model
    model = create_model(input_dim=len(vocab) + 1, num_classes=len(set(y_train)))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_padded, y_train, epochs=10, batch_size=2)

    # Save the model
    model.save('chatbot_model.h5')

if __name__ == "__main__":
    train_model()
