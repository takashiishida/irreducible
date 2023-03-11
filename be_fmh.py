import numpy as np
import scipy
from scipy import stats

class_labels = np.array(['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
probs = np.loadtxt('data/fmh_probs.csv', delimiter = ',')

tops= ['Shirt', 'Dress', 'Coat', 'Pullover', 'T-shirt/top']
positive_class = np.logical_or.reduce(([class_labels == tops[i] for i in range(len(tops))]))

positive_idx = list(np.where(positive_class)[0])
positive = np.sum(probs[:, positive_idx], axis=1)
negative = 1 - positive
pn = np.vstack((positive, negative))
hatBER = np.mean(np.min(pn, axis=0))

confidence_level = 0.95
df = pn.shape[1] - 1
hatBERse = stats.sem(np.min(pn, axis=0))
conf_interval = scipy.stats.t.interval(confidence_level, df, hatBER, hatBERse)
print(f'Bayes error of tops vs. other: {hatBER*100:.3f}% ({conf_interval[0]*100:.3f}%, {conf_interval[1]*100:.3f}%)')
