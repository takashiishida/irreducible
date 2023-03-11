import numpy as np
import torchvision
import scipy
from scipy import stats

# Clone https://github.com/jcpeterson/cifar-10h in the current directory.
counts = np.load('cifar-10h/data/cifar10h-counts.npy')
probs = np.load('cifar-10h/data/cifar10h-probs.npy')

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

class BinaryLabels:
	def __init__(self, testset, counts):
		self.testset = testset
		self.counts = counts
		self.targets = np.array(testset.targets)
		self.c2i = testset.class_to_idx

	def _preparePositiveIdx(self, setup):
		if setup == 'firstfive':
			'''the first five classes are positive and the last five classes are negative'''
			positive_idx = [0,1,2,3,4]
		elif setup == 'odd':
			'''the odd classes are positive and the even classes are negative'''
			positive_idx = [0,2,4,6,8]
		elif setup == 'land':
			'''the land-related classes are positive and the rest (e.g., water-related, sky-related) are negative'''
			land = ['automobile', 'cat', 'deer', 'dog', 'horse', 'truck']
			positive_idx = [self.c2i[j] for j in land]
		elif setup == 'animals':
			'''animals are positive and the rest (e.g., vehicles) are negative'''
			animals = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
			positive_idx = [self.c2i[j] for j in animals]
		return positive_idx

	def getSoftBinaryLabels(self, setup):
		positive_idx = self._preparePositiveIdx(setup)
		return np.sum(self.counts[:, positive_idx] ,1) / np.sum(self.counts, 1)
	
	def getTrueBinaryLabels(self, setup):
		positive_idx = self._preparePositiveIdx(setup)
		return np.logical_or.reduce(([self.targets == i for i in positive_idx]))

if __name__ == '__main__':

	bl = BinaryLabels(testset, counts)

	for setup in ['animals', 'land', 'odd', 'firstfive']:
		soft_binary_labels = bl.getSoftBinaryLabels(setup)
		true_binary_labels = bl.getTrueBinaryLabels(setup)
		min_soft = np.min(np.stack((soft_binary_labels, 1-soft_binary_labels)), axis=0)
		hatBER = np.mean(min_soft)

		conf_level = 0.95
		df = len(soft_binary_labels) - 1
		hatBERse = stats.sem(min_soft)
		conf_interval = scipy.stats.t.interval(conf_level, df, hatBER, hatBERse)
		print(f'Bayes error of {setup} vs other: {hatBER*100:.3f}% ({conf_interval[0]*100:.3f}%, {conf_interval[1]*100:.3f}%)')
