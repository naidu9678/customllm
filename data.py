import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    # Sample dataset of questions and answers
    data = [
        ("Hello!", "Hi there!"),
        ("How are you?", "I'm good, thank you!"),
        ("What is your name?", "I am a chatbot."),
        ("Tell me a joke.", "Why did the chicken cross the road? To get to the other side!"),
        ("Goodbye!", "See you later!")
    ]

    questions = [item[0] for item in data]
    answers = [item[1] for item in data]
    
    # Convert to arrays for easy manipulation
    questions = np.array(questions)
    answers = np.array(answers)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(questions, answers, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
