from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

def create_model(input_dim, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=None))  # Set input_length to None
    model.add(LSTM(64))
    model.add(Dense(num_classes, activation='softmax'))
    return model
