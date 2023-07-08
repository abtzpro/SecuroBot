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

This script is essentially a smart program that can learn from examples and perform tasks, like how a kid learns from observing the world around them.

1.	Setting Things Up: At the beginning, the script prepares a few tools it will need. Just like you might need a dictionary to understand a new language, the script prepares a list of common words that it should ignore when processing text.

2.	Creating the Brain: Next, the script builds a “model”, which is like the brain of the operation. This model will later learn from examples and make decisions based on that learning. This is done in the create_model function.

3.	Learning from Games: The script also learns from playing a game. This is called Q-learning. It’s as if the program is playing a video game, and each time it plays, it learns how to get a better score.

4.	Reading the Internet: The script then visits a cybersecurity-related website and reads all the text there. This is a bit like how you might go to a library to research a topic.

5.	Understanding the Text: Once the script has the text from the website, it processes that text to understand it better. It does this by removing common words and punctuation, and breaking down the text into simpler words. This is similar to highlighting important points in a textbook.

6.	Learning from Examples: The script now takes the “dialogues” and “labels” (which you would have to provide) and feeds them into the “brain” it created earlier. The “brain” learns to associate dialogues with their corresponding labels. This is akin to a student learning by studying examples before an exam.

7.	Saving the Knowledge: Once the “brain” has learned from the examples, the script saves this knowledge in a file named “chatbot_model.h5”. This is like writing down what you have learned in a notebook so that you can refer to it later.

8.	Recalling the Learning: Finally, the script demonstrates its learning ability by playing the video game again, but this time it’s using the knowledge it gained from the previous plays. This is like replaying a level in a video game, but now you know where all the hidden treasures are.

And that’s it! The script ends with a trained “brain” that can be used to analyze new cybersecurity dialogues, and it has also learned how to play a game efficiently. Just like a student after a long day of studying and playing video games!

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
