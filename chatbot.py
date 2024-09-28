import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Load the trained model
model = keras.models.load_model('chatbot_model.h5')

# Load the tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer_data = f.read()  # Read the file content as a string
tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_data)  # Now it should be a string

# Function to process user input and get a response
def get_response(user_input):
    # Preprocess the input
    sequences = tokenizer.texts_to_sequences([user_input])
    max_length = model.input_shape[1]  # Use the max_length from the model input
    padded_sequence = pad_sequences(sequences, maxlen=max_length, padding='post')

    # Predict the response
    prediction = model.predict(padded_sequence)
    response_index = np.argmax(prediction)  # Get the index of the predicted response
    return response_index

# Main loop for the chatbot
print("Chatbot is ready! Type 'exit' to stop the chat.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response_index = get_response(user_input)
    
    # You can define a mapping of response indices to responses
    responses = {
        0: "Hello! How can I help you?",
        1: "I'm doing well, thank you!",
        2: "Why did the chicken cross the road? To get to the other side!",
        3: "Goodbye! Have a nice day!",
        4: "I'm just a bot, but I love talking to you!"
    }
    
    print(f"Chatbot: {responses.get(response_index, 'I am not sure how to respond to that.')}")
