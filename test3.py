import os
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.classify.util import accuracy
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from nltk.classify import NaiveBayesClassifier

def format_sentence(sentence):
	return {word:True for word in word_tokenize(sentence)}

def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names


pos_path = "data/aclImdb/train/pos"
neg_path = "data/aclImdb/train/neg"

pos_data = []
for file in os.listdir(pos_path):
        current = os.path.join(pos_path, file)
        if os.path.isfile(current):
            data = open(current, "r").read()
            pos_data.append([format_sentence(data),'pos'])
print(pos_data[100])