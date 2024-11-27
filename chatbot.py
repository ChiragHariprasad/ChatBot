##Main code
import os
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Download NLTK dependencies
nltk.download('punkt')
nltk.download('wordnet')

# Load WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize data containers
words = []
classes = []
documents = []

# Check if the necessary files exist
if not os.path.exists('words.pkl') or not os.path.exists('classes.pkl') or not os.path.exists('chatbot_model.h5'):
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize and clean words
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha()]
    words = sorted(set(words))
    classes = sorted(set(classes))

    # Save words and classes
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    # Prepare training data
    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1 if word in word_patterns else 0)  # Ensure consistent bag length

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1  # Correct index assignment for output
        training.append([bag, output_row])

    # Ensure training is consistently shaped
    train_x = np.array([np.array(tr[0]) for tr in training])
    train_y = np.array([np.array(tr[1]) for tr in training])

    # Define the model
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile the model
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Train the model
    model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    # Save the model
    model.save('chatbot_model.h5')
else:
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.h5')

def cleanup_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = cleanup_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

def main():
    name = input("What is your name? ")
    print(f"Hi {name}, how can I help you today? (Type 'bye' to exit)")

    while True:
        user_input = input(f"{name}: ")
        if user_input.lower() in ['bye', 'exit']:
            print("AZ: Bye! Have a great day!")
            break

        ints = predict_class(user_input)
        if ints:
            response = get_response(ints, intents)
        else:
            response = "I'm sorry, I didn't understand that."
        print(f"AZ: {response}")

if __name__ == "__main__":
    main()
