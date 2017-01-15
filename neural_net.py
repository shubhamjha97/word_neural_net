import theano
from keras.models import Sequential
from keras.layers import Dense, Activation, Merge
from keras.optimizers import SGD
from keras.metrics import top_k_categorical_accuracy

class Net:
	"""Sequential model
		hidden
		'distributed_encoding'
		'learn_features'
		'unique_words'
		"""


	def __init__(self, input_dimension, structure_dict): #!!!! FIX THIS
		"""hidden= no of hidden units
			input dimension-> tuple of input dim Eg:(784,1)
			act-> string Eg: 'relu' """
		self.model_a=Sequential()
		self.model_b=Sequential()
		self.final_model=Sequential()
		self.model_a.add(Dense(structure_dict['distributed_encoding'], input_dim=input_dimension, activation='sigmoid'))
		self.model_b.add(Dense(structure_dict['distributed_encoding'], input_dim=input_dimension, activation='sigmoid'))
		self.merged=Merge([self.model_a, self.model_b], mode='concat')
		self.final_model.add(self.merged)
		self.final_model.add(Dense(structure_dict['learn_features'], activation='sigmoid'))
		self.final_model.add(Dense(structure_dict['unique_words'], activation='softmax'))
		self.sgd=SGD(lr=0.01)
		def top_k(y_true, y_pred):
			return top_k_categorical_accuracy(y_true, y_pred, k=10)
		self.final_model.compile(optimizer=self.sgd, loss='categorical_crossentropy',
					metrics=['accuracy',top_k ])

	def train(self, train_X_1, train_X_2, train_y):
		self.final_model.fit([train_X_1, train_X_2], train_y, nb_epoch= 10,  batch_size=100, validation_split=0.33)

	