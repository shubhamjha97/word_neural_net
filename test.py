import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

words=[]
with open('aliceinwonderland.txt', 'r') as f:
	full_text=f.read().lower().decode('utf-8')
	sentences=list(sent_tokenize(full_text))

	for sentence in sentences:
		for w in word_tokenize(sentence):
			words.append(w)

unique_words=set(words)
word_dict=dict((w,i) for i,w in enumerate(unique_words))
no_total_words=len(words)
no_unique_words=len(word_dict)
#print words

print "Total words=" + str(no_total_words)
print "Unique words=" + str(no_unique_words)

#one-hot encoding
def one_hot(word, word_dict):
	temp=np.zeros([len(word_dict), 1])
	i=word_dict[word]
	print i
	temp[i]=1
	return temp

print word_dict['is']
print one_hot('is', word_dict)