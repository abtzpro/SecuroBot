# SecuroBot
A Python script to train an AI chatbot on cybersecurity topics using a neural network and reinforcement learning

## How it works

1.	Import necessary libraries: This script uses a variety of libraries to facilitate its functions. These include TensorFlow for machine learning, Keras for neural networks, Gym for reinforcement learning, NumPy for array manipulation, and Pandas for data handling.

2.	Define a function to create the chatbot model: The create_model function creates the structure of our chatbot - a neural network. This network is built in layers, including an Embedding layer that transforms words into numerical vectors, an LSTM layer (Long Short Term Memory) which helps the network understand the context of a conversation over time, and a Dense layer that makes the final decision on what the chatbot should say.

3.	Define a function to preprocess text: The preprocess_text function transforms the text data into a format that the model can understand and learn from. It uses a Keras’ Tokenizer to convert words into numerical representations (tokenization), and pad_sequences to ensure all input sequences have the same length.

4.	The main function: This function loads cybersecurity related dialogues and labels from a given data source. Then, it preprocesses these dialogues using the function we described above. After that, it creates the chatbot model with the create_model function.

5.	Splitting the data: The data is then divided into training and testing datasets. The training set is used to teach the model, while the testing set is used to evaluate how well the model has learned.

6.	Train the model: The chatbot model is then trained using the training dataset. This is where the chatbot learns from the dialogues about cybersecurity. It tries to understand the context, the meaning of different words, and how they are used in different situations.

7.	Save the model: After training, the model is saved as a .h5 file. This file represents the trained chatbot and can be loaded later to make predictions or continue training.

In a nutshell, this script is about creating a chatbot that can understand and participate in cybersecurity-related conversations. It does this by learning from a dataset of cybersecurity dialogues and their associated labels. The model it creates can be saved and used later for cybersecurity discussions.

## In simpler terms 

1.	Load Libraries: This script loads a bunch of software tools that do different tasks, like Google’s TensorFlow which helps to build and train artificial intelligence models, or Gym for making AI play games to learn new things.

2.	Creating the chatbot model: create_model is like the brain of the chatbot, where it’s structured to learn and understand things. It uses special layers like an LSTM, which is sort of like the chatbot’s memory. It remembers past words to understand the context of what’s being said.

3.	Preprocessing Text: The preprocess_text function cleans up the language that the chatbot will learn from. It’s like when you simplify a tough concept into easier words. This step makes sure the chatbot understands what it’s learning.

4.	Main function: This part of the script grabs the cybersecurity conversations from somewhere and gets them ready for the chatbot to learn from. Then, it uses create_model to create the chatbot’s brain.

5.	Splitting the Data: Just like in school, the chatbot has a “study phase” and a “test phase”. The cybersecurity conversations are split into a part for learning (training) and a part for testing how well it learned.

6.	Train the model: The chatbot now studies the training data. This is when it’s learning all about cybersecurity. It’s trying to understand the words and how they’re used in different situations.

7.	Save the model: After the chatbot has finished studying, we save its brain as a .h5 file. This way, we can use it later to chat about cybersecurity without having to learn everything again.

In simple words, this script is teaching a chatbot to talk and understand cybersecurity. It’s like a tutor, giving the chatbot lessons, then testing it, and finally saving what it learned for later.

## Notes & Requirements

To use this script, users need to have:

1.	Install the required Python libraries: TensorFlow, Keras, Scikit-learn, NLTK, Gym, Requests, BeautifulSoup, and NumPy. You can do this with pip by running the command: pip install tensorflow keras scikit-learn nltk gym requests beautifulsoup4 numpy.

2.	Replace the dialogues and labels placeholders in the script with your actual training data. dialogues should be a list of text dialogues related to cybersecurity, and labels should be a list of corresponding labels for each dialogue.
(For better results the datasets should
consist of sentences rather than words.)
If you choose to go this route you can
make slight modifications in the script
to do so with ease. 

4.	Run the script. It will scrape text from the provided URL, preprocess the text, and print the preprocessed words. It will also create a model with the provided input shape and number of classes, train the model with your data, and save the trained model as “chatbot_model.h5”. Additionally, it will perform Q-learning on the ‘FrozenLake-v0’ Gym environment and print the resulting Q-table.

5.	After running the script, you can use the trained model saved as “chatbot_model.h5” for your tasks.

## Active Development Disclaimer

This project is in active development
and as such, is prone to bugs, glitches
errors, and more. Much testing and 
development is yet required to consider
this a completed project. please report
errors and bugs if you are so inclined. 
I will address them forthright.

## Credits and thanks

Developed by 
- Adam Rivers
- Hello Security LLC
- ChatGPT

- Thanks to Microsoft for making
  Training and ethics material
  readily available. 
