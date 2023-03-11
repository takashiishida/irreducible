import numpy as np
import argparse
import scipy
from gaussian import GaussianDataNumpy

parser = argparse.ArgumentParser(prog='Synthetic experiments', usage='Experiments for Fig 1, 2, and 3.', description='description', epilog='end', add_help=True)
parser.add_argument('-su', '--setup', help='4 choices for Gaussian distributions', choices=[0, 1, 2, 3], type=int, required=True)
parser.add_argument('-n', '--noise', help='confidence noise', action='store_true')
parser.add_argument('-std', '--standard_deviation', help='std of confidence noise', default=0.4, type=float)
parser.add_argument('-ss', '--seed_size', help='number of trials with different random seeds', default=10, type=int)
args = parser.parse_args()

data_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1028] # this is size for each class

setup_0 = {'mu1': np.array([0,  0]),
           'mu2': np.array([-2, 5]),
           'cov1': np.array([[7, -6], [-6, 7]]),
           'cov2': np.array([[2, 0], [0, 2]])}
setup_1 = {'mu1': np.array([0, 0]),
           'mu2': np.array([0, 4]),
           'cov1': np.array([[5, 3], [3, 5]]),
           'cov2': np.array([[5, -3], [-3, 5]])}
setup_2 = {'mu1': np.zeros(10),
           'mu2': np.ones(10),
           'cov1': np.identity(10),
           'cov2': np.identity(10)}
setup_3 = {'mu1': np.zeros(20),
           'mu2': np.ones(20),
           'cov1': np.identity(20),
           'cov2': np.identity(20)}
setups = {0: setup_0,
          1: setup_1,
          2: setup_2,
          3: setup_3}
mu1, mu2, cov1, cov2 = list(setups[args.setup].values())

for data_size in data_sizes:
	print(f'setup: {args.setup} | noise: {args.noise} | data size: {data_size} | seed:', end=' ', flush=True)
	PN_BERs, Pc_BERs, Modified_BERs = [], [], []
	for random_seed in range(args.seed_size):
		print(random_seed, end=' ', flush=True)
		np.random.seed(random_seed)
		dataset = GaussianDataNumpy(mu_pos=mu1, mu_neg=mu2, cov_pos=cov1, cov_neg=cov2, n_pos=data_size, n_neg=data_size, std=args.standard_deviation)
		x, r, r_noise, y = dataset.makeData()
		if args.noise:
			r_noise_pos = r_noise[y == 0, 0]
			inter = 2 - 1/r_noise_pos
			PN_BER = np.mean(np.min(r_noise, 1))
			Modified_BER = (np.sum(r_noise[r[:, 0] >= 0.5, 1]) + np.sum(r_noise[r[:, 0] < 0.5, 0])) / len(y)
			Modified_BERs.append(Modified_BER)
		else:
			r_pos = r[y == 0, 0]
			inter = 2 - 1/r_pos
			Pc_BER = 0.5 * (1 - np.sum(inter[inter>0])/len(inter)) # pi=0.5 because size of P and N are the same
			Pc_BERs.append(Pc_BER)
			PN_BER = np.mean(np.min(r, 1))
		PN_BERs.append(PN_BER)
	if args.noise:
		print('\n naive in Eq.5 mean (ste): {:.3g} ({:.3g}) modified in Eq.6 mean (ste): {:.3g} ({:.3g})'.format(np.mean(PN_BERs), scipy.stats.sem(PN_BERs), np.mean(Modified_BERs), scipy.stats.sem(Modified_BERs)))
	else:
		print('\nPN in Eq.4 mean (ste): {:.3g} ({:.3g}) Pconf in Eq.8 mean (ste): {:.3g} ({:.3g})'.format(np.mean(PN_BERs), scipy.stats.sem(PN_BERs), np.mean(Pc_BERs), scipy.stats.sem(Pc_BERs)))
