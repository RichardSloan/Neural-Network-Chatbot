import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open(r'{FILE_PATH}', 'r') as f:
    data = json.load(f)

# Loading pre-processed data
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
# Process and transform data       
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Stem words and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    # List of zeros for the amount of words
    out_empty = [ 0 for _ in range(len(labels))]

    # Create bag of words from patterns in docs_x
    for x, doc in enumerate(docs_x):
        bag = []

        # Stem words 
        wrds = [stemmer.stem(w) for w in doc]

        # Encode if a word in the pattern is present in words list
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        # Encode pattern label
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        # Appended encoded lists  
        training.append(bag)
        output.append(output_row)

    # Change to numpy arrays for tflearn
    training = numpy.array(training)
    output = numpy.array(output)

    # Saving transformed data
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tf.reset_default_graph()

# Defines input shape of model
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
# Output layer
net = tflearn.fully_connected(net,len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Load existing model
try:
    model.load("model.tflearn")
# Fit model/Re-train data
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# Transforming user's input, into a bag of words
def bag_of_words(s, words):
    # List of zeros for the amount of words
    bag = [0 for _ in range(len(words))]

    # Tokenize and stem words 
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower())for word in s_words]

    # Generating bag of words from tokenized and stemmed words
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return numpy.array(bag)

# Chatbot for user input
def chat():
    print("Start talking with the bot! (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Predicting probability of each class
        results = model.predict([bag_of_words(inp, words)])[0]
        # Gets index of the greatest value
        results_index = numpy.argmax(results)
        # Assigns predicted index to tag
        tag = labels[results_index]
        
        # Probability threshold
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg ['tag'] == tag:
                    responses = tg['responses']
                    
            print(random.choice(responses))
        else:
            print("I didn't get that, please try again")
        

chat()