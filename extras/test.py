from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk_sents
from nltk.chunk import ne_chunk
from nltk.classify.util import accuracy
from nltk.classify import NaiveBayesClassifier

def format_sentence(sentence):
	return {word:True for word in word_tokenize(sentence)}

def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
            # print('case 1')
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
            # print('case 2')

    return entity_names

entity_names = []
sentences = sent_tokenize('Desierto was released in the US way back in 2015 (it was Mexico’s entry to the Oscars for the Best Foreign Language Film category), but its Indian release could not have come at a more appropriate time. Using the worn-out premise of a lone killer and his hapless victims who have to fight for survival, Jonás Cuarón, the son of Oscar-winner Alfonso Cuarón, gives us a tense thriller that throbs with topical urgency following Donald Trump’s victory in the recent US elections.')
for sentence in sentences:
	tokenized = word_tokenize(sentence)
	tagged = pos_tag(tokenized)
	namedEnt = ne_chunk(tagged,binary=True)
	# namedEnt.draw()
	entity_names.extend(extract_entity_names(namedEnt))
print(entity_names)
 
# entity_names = []
# with open('data/rt-polaritydata/rt-polarity-pos.txt', encoding = 'latin-1') as f:
# 	for line in f:
# 		# print(sent_tokenize(line))
# 		sentences = sent_tokenize(line)
# 		tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
# 		tagged_sentences = [pos_tag(sentence) for sentence in tokenized_sentences]
# 		chunked_sentences = ne_chunk_sents(tagged_sentences, binary=False)
# 		for tree in chunked_sentences:
# 		    # Print results per sentence
# 		    print(extract_entity_names(tree))
# 		    # entity_names.extend(extract_entity_names(tree))

# # Print all entity names
# # print(entity_names)

# # Print unique entity names
# # print(set(entity_names))
