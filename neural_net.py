import theano
from keras.models import Sequential
from keras.layers import Dense, Activation, Merge

class Net:
	"""Sequential model
		hidden
		'distributed_encoding'
		'learn_features'
		'unique_words'
		"""
	self.model_a=Sequential()
	self.model_b=Sequential()
	self.final_model=Sequential()

	def __init__(input_dimension, act, hidden): #!!!! FIX THIS
		"""hidden= no of hidden units
			input dimension-> tuple of input dim Eg:(784,1)
			act-> string Eg: 'relu' """
		self.model_a.add(Dense(hidden['distributed_encoding'], input_dim=input_dimension, activation='sigmoid'))
		self.model_b.add(Dense(hidden['distributed_encoding'], input_dim=input_dimension, activation='sigmoid'))
		self.merged=Merge([self.model_a, self.model_b], mode='concat')
		self.final_model.add(merged)
		self.final_model.add(Dense(hidden['learn_features'], activation='sigmoid'))
		self.final_model.add(Dense(hidden['unique_words'], activation='softmax'))
		self.final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
					metrics=['accuracy'])

	def train(train_X, train_y):
		self.final_model.fit(train_X, train_y, nb_epoch= 10,  batch_size=50, validation_split=0.33)

	#Create an evaluate function