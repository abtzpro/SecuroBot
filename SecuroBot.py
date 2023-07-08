from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import gym
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
import string
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

def create_model(input_shape, num_classes):
    """
    Creates a keras Model with Conv1D, LSTM, and Dense layers.
    """
    try:
        input_layer = Input(shape=(input_shape,))
        conv_layer = Conv1D(filters=32, kernel_size=5, activation='relu')(input_layer)
        flat_layer = Flatten()(conv_layer)
        rnn_layer = LSTM(32)(flat_layer)
        output_layer = Dense(num_classes, activation='softmax')(rnn_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def q_learning_with_table(env, num_episodes=50000, learning_rate=0.1, discount_factor=0.9):
    """
    Performs Q-learning on the provided Gym environment.
    """
    try:
        Q_table = np.zeros([env.observation_space.n, env.action_space.n])
        for i_episode in range(num_episodes):
            state = env.reset()
            for t in range(100):
                action = np.argmax(Q_table[state])
                next_state, reward, done, _ = env.step(action)
                old_value = Q_table[state, action]
                next_max = np.max(Q_table[next_state])
                new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
                Q_table[state, action] = new_value
                state = next_state
                if done:
                    break
        return Q_table
    except Exception as e:
        print(f"Error during Q-learning: {e}")
        return None

def scrape_website(url):
    """
    Scrapes the provided website URL using BeautifulSoup.
    """
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"Error scraping website: {e}")
        return None

def preprocess_text(text):
    """
    Preprocesses the provided text using NLTK.
    """
    try:
        words = word_tokenize(text)
        words = [word.lower() for word in words if word not in stopwords and word not in string.punctuation]
        words = [lemmatizer.lemmatize(word) for word in words]
        return words
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return None

if __name__ == "__main__":
    url = "https://www.nist.gov/topics/cybersecurity" 
    scraped_text = scrape_website(url)
    if scraped_text is not None:
        words = preprocess_text(scraped_text)
        if words is not None:
            print(words)

    # Assume that dialogues is a list of cybersecurity dialogues and labels is their corresponding labels
    dialogues = words  # Replace this with your list
    labels = []  # Replace this with your labels

    # Convert text to sequences for training
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dialogues)
    sequences = tokenizer.texts_to_sequences(dialogues)
    X = pad_sequences(sequences)
    y = to_categorical(labels)

    # Split the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the model
    model = create_model((X_train.shape[1],), y_train.shape[1])
    if model is not None:
        print(model.summary())
        # Train the model
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

        # Save the model
        model.save("chatbot_model.h5")

    # Q-learning
    env = gym.make('FrozenLake-v0') 
    q_table = q_learning_with_table(env, num_episodes=5000, learning_rate=0.1, discount_factor=0.9)
    if q_table is not None:
        print(q_table)
