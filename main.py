import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
from hazm import *
import json
import pickle

stemmer = LancasterStemmer()


with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            normalizer = Normalizer()
            normalizer.normalize(pattern)
            word = nltk.word_tokenize(pattern)
            words.extend(word)
            docs_x.append(word)
            docs_y.append(intent["tags"])
        labels.append(intent["tags"])


    training = []
    output = []
    # print(docs_y, 'docs_y')
    output_empty = []

    # lemmatizer = Lemmatizer()
    print(words)
    # words = [lemmatizer.lemmatize(w) for w in words if w not in "؟"]
    print(docs_x)
    print(words)

    output_empty = [0 for _ in range(len(labels))]

    print(output_empty)

    for x, doc in enumerate(docs_x):
        bag = []
        print(x)
        print(doc)
        stemmer = Stemmer()
        wrds = [stemmer.stem(w) for w in doc]
        # print(wrds)
        for w in words:
            if w in doc:
                bag.append(1)
            else:
                bag.append(0)
        training.append(bag)
        output_row = output_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        output.append(output_row)
        # print(output_row)
    print(output, 'output')
    print(training, 'training')

    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
print(output, 'output')

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    # print(model, 'model')
    model.fit(training, output, n_epoch=100, batch_size=8, show_metric=True)

    model.save("model.tflearn")


def bag_of_words(inputs, wrd):
    bags = [0 for _ in wrd]
    input_words = nltk.word_tokenize(inputs)
    for s in input_words:
        for i, w in enumerate(wrd):
            if w == s:
                bags[i] = 1
    return numpy.array(bags)


def chat_box():
    print("با من حرف بزن لعنتی!")
    while True:
        inputss = input("شما: ")
        if inputss == "quit":
            break

        results = model.predict([bag_of_words(inputss, words)])
        print(results)
        results_index = numpy.argmax(results)


chat_box()



