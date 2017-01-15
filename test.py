import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import neural_net
from tqdm import tqdm
import gc
import sys

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

print "Total words=" + str(no_total_words)
print "Unique words=" + str(no_unique_words)

del sentences #!!!!!!!!!

#one-hot encoding
def one_hot(word, word_dict):
	temp=np.zeros(len(word_dict), dtype=np.int8)
	i=word_dict[word]
	temp[i]=1
	return temp

#print word_dict['is']
#print one_hot('alice', word_dict)

#Generate train data-- list of numpy arrays
net_structure={'distributed_encoding':10, 'learn_features':30, 'unique_words':len(unique_words)}
net=neural_net.Net(len(unique_words), net_structure)

train_X_1=[]
train_X_2=[]
train_y=[]


'''for w in tqdm(words[0:no_total_words-3]):
	train_X_1.append(one_hot(w, word_dict))
for w in tqdm(words[1:no_total_words-2]):
	train_X_2.append(one_hot(w, word_dict))
for w in tqdm(words[2:no_total_words-1]):
	train_y.append(one_hot(w, word_dict))

del words
gc.collect()
train_X_1=np.array(train_X_1)
train_X_2=np.array(train_X_2)
train_y=np.array(train_y)'''
#net.train(train_X_1, train_X_2, train_y)

#print train_X_1
#print len(train_X_1)
#print sys.getsizeof(train_X_1.shape)
#print len(train_X_2)
#print len(train_y)

word1=one_hot('alice', word_dict)
word2=one_hot('will', word_dict)

def decode(word, word_dict):
	idx=np.where(word==1)[0]
	print idx
	#return


'''for i in xrange(100):
	word3=net.predict([word1, word2])
	print decode(word3, word_dict)
	word1=word2
	word2=word3'''

if __name__=='__main__':
	enc=one_hot('alice', word_dict)
	decode(enc, word_dict)
