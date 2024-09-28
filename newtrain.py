import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Sample training data (replace with your actual data)
training_sentences = ["Hello!", "How are you?", "Tell me a joke.", "Goodbye!", "What is your name?"]
training_labels = [0, 1, 2, 3, 4]  # Example labels corresponding to the responses

# Create and fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_sentences)

# Save the tokenizer
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer_json)

# Convert texts to sequences
training_sequences = tokenizer.texts_to_sequences(training_sentences)
max_length = max(len(seq) for seq in training_sequences)  # Find the maximum length

# Pad sequences
X_train = pad_sequences(training_sequences, maxlen=max_length, padding='post')
y_train = np.array(training_labels)

# Build and train the model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=max_length),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(len(set(training_labels)), activation='softmax')  # Adjust based on your labels
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

# Save the model
model.save('chatbot_model.h5')
