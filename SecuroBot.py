from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import gym
import numpy as np
import pandas as pd

def create_model(vocab_size, embedding_dim, input_length, num_classes):
    """
    Creates a keras Model with Embedding, LSTM, and Dense layers.
    """
    try:
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=input_length))
        model.add(LSTM(32))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def preprocess_text(dialogues, max_length=500):
    """
    Preprocesses the provided dialogues using Keras' Tokenizer and pad_sequences.
    """
    try:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dialogues)
        sequences = tokenizer.texts_to_sequences(dialogues)
        padded_sequences = pad_sequences(sequences, maxlen=max_length)
        return padded_sequences, tokenizer
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return None, None

if __name__ == "__main__":
    # Load your dialogues and labels from a data source
    dialogues = pd.read_csv("your_data_source.csv")  # replace with your data source
    labels = pd.read_csv("your_labels_source.csv")  # replace with your labels source
    
    # preprocess dialogues
    dialogues, tokenizer = preprocess_text(dialogues)
    vocab_size = len(tokenizer.word_index) + 1
    
    # create model
    embedding_dim = 50
    input_length = len(dialogues[0])
    num_classes = len(set(labels))
    model = create_model(vocab_size, embedding_dim, input_length, num_classes)
        
    if model is not None:
        # split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(dialogues, labels, test_size=0.2, random_state=42)

        # Train the model
        model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
        print(model.summary())
        
        # Save the model
        model.save("cybersecurity_chatbot_model.h5")
