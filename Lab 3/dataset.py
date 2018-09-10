import numpy as np
import codecs
import collections

# ...
# Code is nested in class definition, indentation is not representative.
# "np" stands for numpy.
class Dataset:
	def __init__(self,input_file,batch_size=3, sequence_size=3):
		"""Arguments:
		 input_file - path to data (txt file)
		 batch_size - self explanatory
		 sequence_size - size of data array (number of steps in which reccurent network untangles)"""
		self.batch_size = batch_size
		self.sequence_size = sequence_size
		self.sorted_chars=[]
		self.char2id = {}
		self.id2char = {}
		self.x = []
		self.batch_pointer = 0
	
	def preprocess(self, input_file):
		with open(input_file, "r") as f:
			with codecs.open(input_file,'r',encoding='utf8') as f:
				data = f.read()
			# count and sort most frequent characters
			self.sorted_chars = list(dict(collections.Counter(data).most_common(None)).keys())
			print(self.sorted_chars)
			# self.sorted chars contains just the characters ordered descending by frequency
			
			self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars)))) 
			#print(self.char2id)
			# reverse the mapping
			self.id2char = {k:v for v,k in self.char2id.items()}
			#print(self.id2char)
			# convert the data to ids
			self.x = np.array(list(map(self.char2id.get, data)))
			print(self.x)
			
	def encode(self, sequence):
		encoded = []
		for s in list(sequence):
			encoded.append(self.char2id[s])
		return encoded
		
	def decode(self, encoded_sequence):
		decoded = []
		for es in (encoded_sequence):
			decoded.append(self.id2char[es])
		return decoded
		
	def create_minibatches(self):
		self.batches = []
		batch_x, batch_y = [],[]	
		shifted_x = np.delete(self.x,-1)
		shifted_y = self.x[1:]
		step = self.batch_size * self.sequence_size
		self.num_batches = int(len(self.x) / (self.batch_size * self.sequence_size))
		batch = []		
		for batch_index in range(0,self.num_batches):
			batch_start = batch_index * step
			batch_x = np.array(shifted_x[batch_start:batch_start + step])
			batch_y = np.array(shifted_y[batch_start:batch_start + step])
			self.batches.append((batch_x.reshape(self.batch_size, self.sequence_size), batch_y.reshape(self.batch_size, self.sequence_size)))
		self.batches = np.array(self.batches)
	def next_minibatch(self):
		new_epoch = False
		if (self.batch_pointer == 0):
			new_epoch = True
		batch_x, batch_y = self.batches[self.batch_pointer]
		self.batch_pointer = self.batch_pointer + 1

		if self.num_batches <= self.batch_pointer:
			self.batch_pointer = 0
		return new_epoch, batch_x,batch_y

        
if __name__=="__main__":
	dataset = Dataset("a")
	dataset.preprocess("C:\\Users\\Ana\\Desktop\\Duboko Ucenje\\3\\data\\test.txt") 
	#print(dataset.encode("hello"))
	#print(dataset.decode([3,4,5]))
	dataset.create_minibatches()
	print(dataset.encode("are"))
	print(np.array(dataset.batches).shape)
	print(dataset.next_minibatch())
	print(dataset.next_minibatch())
	#print(dataset.batches)
