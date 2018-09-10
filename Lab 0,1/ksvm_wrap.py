import sklearn
from sklearn import svm
import numpy as np
import data
from data import *
class KSVMWrap:
	def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
		"""
		Konstruira omotac i uci RBF SVM klasifikator
		X,Y_: podaci i točni indeksi razreda
		param_svm_c: relativni značaj podatkovne cijene
		param_svm_gamma: sirina RBF jezgre
		"""
		self.clf = svm.SVC(C=1, kernel='rbf', gamma=param_svm_gamma)
		self.clf.fit(X,Y_)
		
	def predict(self,X):
		"""
		Predvida i vraca indekse razreda podataka X
		"""
		return self.clf.predict(X)
		
	def scores(self,X):
		"""
		Vraca klasifikacijske mjere podataka X
		"""
		return self.clf.decision_function(X)
	
	def support(self):
		return self.clf.support_

if __name__=="__main__":
	# inicijaliziraj generatore slucajnih brojeva
	np.random.seed(100)

	# instanciraj podatke X i labele Yoh_
	
	X,Y_ = sample_gmm_2d(6,2,10)
	#print(Y_)
	ksvm = KSVMWrap(X,Y_)

	# nauci parametre
	scores = ksvm.scores(X)
	Y = ksvm.predict(X)

	accuracy, pr, M= data.eval_perf_multi(Y, Y_)
	print("Accuracy: ",accuracy)
	print("Precision / Recall: ",pr)
	print("Confussion Matrix: ",M)

	#graph the decision surface
	decfun = lambda x: ksvm.scores(x)
	bbox=(np.min(X, axis=0), np.max(X, axis=0))
	data.graph_surface(decfun, bbox, offset=0.5)
	data.graph_data(X, Y_, Y, special=[ksvm.support()])
	
	# show the plot
	plt.show()

		
