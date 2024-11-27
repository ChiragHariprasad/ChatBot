##training code
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize the Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
intents = json.loads(open('intents.json').read())

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []

# Iterate through the intents to extract words and tags
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w.isalpha()]
words = sorted(set(words))

# Remove duplicate classes
classes = sorted(set(classes))

# Save words and classes for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training_sentences = []
training_labels = []
output_empty = [0] * len(classes)

# Create a bag of words for each sentence in the intents
for document in documents:
    bag = []
    sentence_words = document[0]
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]

    # Create a bag of words for each sentence
    for word in words:
        bag.append(1 if word in sentence_words else 0)

    # Set the output for the current sentence's tag
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training_sentences.append(bag)
    training_labels.append(output_row)

# Convert to numpy arrays
training_sentences = np.array(training_sentences)
training_labels = np.array(training_labels)

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(training_sentences[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(training_labels[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(training_sentences, training_labels, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5')
print("Model trained and saved.")
