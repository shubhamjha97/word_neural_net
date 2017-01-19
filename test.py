import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import neural_net
from tqdm import tqdm


no_to_predict=10
words=[]
with open('sherlock_holmes.txt', 'r') as f:
	full_text=f.read().lower().decode('utf-8')
	sentences=list(sent_tokenize(full_text))

	for sentence in sentences:
		for w in word_tokenize(sentence):
			if len(w)>1:
				words.append(w)
words=[x for x in tqdm(words) if (words.count(x)<400 and words.count(x)>100)]


unique_words=set(words)
word_dict=dict((w,i) for i,w in enumerate(unique_words))
no_total_words=len(words)
no_unique_words=len(word_dict)

for x in set(words):
	print x.encode('utf-8')+' '+ str(words.count(x))

print "Total words=" + str(no_total_words)
print "Unique words=" + str(no_unique_words)

del sentences #!!!!!!!!!

#one-hot encoding
def one_hot(word, word_dict): #reqd shape=(no_unique_words)
	temp=np.zeros(len(word_dict), dtype=np.int8)
	i=word_dict[word]
	temp[i]=1
	return temp

#Generate train data
net_structure={'distributed_encoding':10, 'learn_features':30, 'unique_words':len(unique_words)}
net=neural_net.Net(len(unique_words), net_structure)

train_X_1=[]
train_X_2=[]
train_y=[]

for w in words[0:no_total_words-3]:
	train_X_1.append(one_hot(w, word_dict))
for w in words[1:no_total_words-2]:
	train_X_2.append(one_hot(w, word_dict))
for w in words[2:no_total_words-1]:
	train_y.append(one_hot(w, word_dict))

#del words
#gc.collect()
train_X_1=np.array(train_X_1)
train_X_2=np.array(train_X_2)
train_y=np.array(train_y)

#net.train(train_X_1, train_X_2, train_y)

#word1=one_hot('is', word_dict)
#word2=one_hot('turkey', word_dict)

def decode(word, word_dict): #reqd shape=(no_unique_words)
	idx=np.where(word==1)[0]
	return str(word_dict.keys()[word_dict.values().index(idx)])

def find_word(word, no_unique_words):
	retarray=[]
	retarray=np.zeros(no_unique_words)
	temp=np.sort(word)
	second_max_val=temp[1]
	second_max_idx=np.where(word==second_max_val)
	idx=np.argmax(word)
	retarray[idx]=1
	second_max_array=np.zeros(no_unique_words)
	second_max_array[second_max_idx]=1
	return list(retarray, second_max_array)

#print decode(word1, word_dict)
#print decode(word2, word_dict)

#print net.pred(one_hot('is', word_dict).reshape((-1, no_unique_words)), one_hot('turkey', word_dict).reshape((-1, no_unique_words)))
#print word_dict
'''print decode(net.pred(one_hot('different', word_dict).reshape((-1, no_unique_words)), one_hot('discontinue', word_dict).reshape((-1, no_unique_words))), word_dict)
print decode(net.pred(one_hot('merchantibility', word_dict).reshape((-1, no_unique_words)), one_hot('neighbour', word_dict).reshape((-1, no_unique_words))), word_dict)
print decode(net.pred(one_hot('seaography', word_dict).reshape((-1, no_unique_words)), one_hot('patiently', word_dict).reshape((-1, no_unique_words))), word_dict)
print decode(net.pred(one_hot('shoulders', word_dict).reshape((-1, no_unique_words)), one_hot('accusation', word_dict).reshape((-1, no_unique_words))), word_dict)'''
'''for i in xrange(no_to_predict):
	print 'words going into net='+str(decode(word1.reshape(no_unique_words), word_dict))+', '+str(decode(word2.reshape(no_unique_words),word_dict))
	word3=net.pred(word1.reshape((-1, no_unique_words)), word2.reshape((-1, no_unique_words)))
	print word3
	#ip shape=(no_unique_words)
	#op shape=(1,no_unique_words)
	word3=find_word(word3, no_unique_words)
	print decode(word3.reshape(no_unique_words), word_dict)
	word1=word2
	word2=word3.reshape(no_unique_words)'''

'''if __name__=='__main__':
	enc=one_hot('alice', word_dict)
	decode(enc, word_dict)'''
word_final='he'
while(1):
	while word_final in set(words):
		word_final=raw_input('1st word: ')
		if word_final not in set(words):
			word_final='he'
		word1=one_hot(word_final, word_dict)
		word_final=raw_input('2nd word: ')
		if word_final not in set(words):
			word_final='he'
		word2=one_hot(word_final, word_dict)
		word3=net.pred(word1.reshape((-1, no_unique_words)), word2.reshape((-1, no_unique_words)))
		print 'Predicted word: '+ decode(find_word(word3, no_unique_words)[0].reshape(no_unique_words), word_dict)
		print 'Seond Prediction: '+ decode(find_word(word3, no_unique_words)[1].reshape(no_unique_words), word_dict)

	