from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.classify.util import accuracy
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from nltk.classify import NaiveBayesClassifier
import pickle

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

def save_classifier(classifier):
   f = open('my_classifier.pickle', 'wb')
   pickle.dump(classifier, f, -1)
   f.close()

def load_classifier():
   f = open('my_classifier.pickle', 'rb')
   classifier = pickle.load(f)
   f.close()
   return classifier

print('\n')

bayesModel = load_classifier()
print('Classifier Loaded!')

print('\n')

entity_names = []
review = input('Enter your review: ')
sentences = sent_tokenize(review)
for sentence in sentences:
	tokenized = word_tokenize(sentence)
	tagged = pos_tag(tokenized)
	namedEnt = ne_chunk(tagged,binary=True)
	entity_names.extend(extract_entity_names(namedEnt))

print('\n')
print('The review is : '+ str(bayesModel.classify(format_sentence(review))))

if(len(entity_names) > 0):
	result_named_entities = 'Named Entities are: '+''.join(str(word)+', ' for word in set(entity_names))
	new_result = list(result_named_entities)
	new_result[len(result_named_entities) - 2	] = '.'
	print(''.join(new_result))
else:
	print('No named entities!')
print('\n')