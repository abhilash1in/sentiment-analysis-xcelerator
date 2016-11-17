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


pos_data = []
with open('data/rt-polaritydata/rt-polarity-pos.txt', encoding = 'latin-1') as f:
	for line in f:
		pos_data.append([format_sentence(line),'pos'])

neg_data = []
with open('data/rt-polaritydata/rt-polarity-neg.txt', encoding = 'latin-1') as f:
	for line in f:
		neg_data.append([format_sentence(line),'neg'])

training_data = pos_data[:5000] + neg_data[:5000]
testing_data = pos_data[5000:] + neg_data[5000:]

print('\n')

bayesModel = NaiveBayesClassifier.train(training_data)
# print(bayesModel.classify(format_sentence('this is a nice article')))
# print(bayesModel.classify(format_sentence('this article is horrible!!')))
print('Training completed!')
print('Accuracy: ' + str(accuracy(bayesModel,testing_data)))


# bernoulliNBModel = SklearnClassifier(BernoulliNB()).train(training_data)
# print(bernoulliNBModel.classify(format_sentence('this is a nice article')))
# print(bernoulliNBModel.classify(format_sentence('this article is horrible!!')))
# print('Training completed!')
# print('Accuracy: ' + str(accuracy(bernoulliNBModel,testing_data)))


# SVCModel = SklearnClassifier(SVC(), sparse=False).train(training_data)
# print(SVCModel.classify(format_sentence('this is a nice article')))
# print(SVCModel.classify(format_sentence('this article is horrible!!')))
# print(accuracy(SVCModel,testing_data))

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
# print('The review is : '+ str(bernoulliNBModel.classify(format_sentence(review))))
# print('The review is : '+ str(SVCModel.classify(format_sentence(review))))

if(len(entity_names) > 0):
	result_named_entities = 'Named Entities are: '+''.join(str(word)+', ' for word in entity_names)
	new_result = list(result_named_entities)
	new_result[len(result_named_entities) - 2	] = '.'
	print(''.join(new_result))
else:
	print('No named entities!')
print('\n')