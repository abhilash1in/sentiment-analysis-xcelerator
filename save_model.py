import os
from nltk.tokenize import word_tokenize
from nltk.classify.util import accuracy
from nltk.classify import NaiveBayesClassifier
import pickle

def format_sentence(sentence):
	return {word:True for word in word_tokenize(sentence)}

def save_classifier(classifier):
   f = open('my_classifier.pickle', 'wb')
   pickle.dump(classifier, f, -1)
   f.close()

def load_classifier():
   f = open('my_classifier.pickle', 'rb')
   classifier = pickle.load(f)
   f.close()
   return classifier


train_pos_path = "data/aclImdb/train/pos"
train_neg_path = "data/aclImdb/train/neg"
test_pos_path = "data/aclImdb/test/pos"
test_neg_path = "data/aclImdb/test/neg"

print('Training model, please wait...')

training_data = []
for file in os.listdir(train_pos_path):
        current = os.path.join(train_pos_path, file)
        if os.path.isfile(current):
            data = open(current, "r").read()
            training_data.append([format_sentence(data),'pos'])

for file in os.listdir(train_neg_path):
        current = os.path.join(train_neg_path, file)
        if os.path.isfile(current):
            data = open(current, "r").read()
            training_data.append([format_sentence(data),'neg'])

testing_data = []
for file in os.listdir(test_pos_path):
        current = os.path.join(test_pos_path, file)
        if os.path.isfile(current):
            data = open(current, "r").read()
            testing_data.append([format_sentence(data),'pos'])

for file in os.listdir(test_neg_path):
        current = os.path.join(test_neg_path, file)
        if os.path.isfile(current):
            data = open(current, "r").read()
            testing_data.append([format_sentence(data),'neg'])

bayesModel = NaiveBayesClassifier.train(training_data)
print('Training completed!')
print('Saving model, please wait...')
save_classifier(bayesModel)
print('Model Saved! \n')
print('Testing Accuracy, please wait...')
print('Accuracy: ' + str(accuracy(bayesModel,testing_data)))