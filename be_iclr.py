import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import scipy
import torch.optim as optim

def main(year):

    def sig_scaling(logits, params):
        sig = nn.Sigmoid()
        return sig(logits*params[0] + params[1])

    def eval():
        loss = criterion(sig_scaling(logits, params), y.float())
        loss.backward()
        return loss

    def getBayesError(logits_list, params):
        calib_probs = sig_scaling(logits_list, params)
        torch.histogram(calib_probs, bins=20, range=(0.,1.))
        minProbs = torch.min(torch.stack((calib_probs, 1-calib_probs)).T, dim=1)[0]
        be = torch.mean(minProbs)
        confidence_level = 0.95
        df = len(calib_probs) - 1
        minProbsNp = minProbs.detach().numpy()
        hatBERse = scipy.stats.sem(minProbsNp)
        confidence_interval = scipy.stats.t.interval(confidence_level, df, be.detach().numpy(), hatBERse)
        print('{}: {:.1f} \pm {:.1f}'.format(year, be.detach().item()*100, be.detach().item()*100-confidence_interval[0]*100))

    fullDataFrame = pd.read_csv(f'data/iclr_{year}.csv')
    logits = np.array(fullDataFrame['weightedAverage'])
    logits = logits - np.mean(logits)
    logits = torch.from_numpy(logits).float()
    y = np.array([1 if d == 'Accept' else 0 for d in fullDataFrame['decision']])
    y = torch.from_numpy(y)
    
    params = nn.Parameter(torch.ones(2))
    criterion = nn.BCELoss()
    optimizer = optim.LBFGS([params], lr=0.0001, max_iter=10000, line_search_fn='strong_wolfe')

    optimizer.step(eval)
    getBayesError(logits, params)

if __name__ == "__main__":
    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
    for year in years:
        main(str(year))