import numpy as np
from sklearn.metrics import roc_auc_score

"""
INSTRUCTION: Just change these two lines: Put the predictions (softmax vectors) of the in-distribution data in preds_in,
while put the predictions of OOD data in preds_out.
"""
preds_in = np.load('in_dist_conf.npy')
preds_out = np.load('out_dist_conf.npy')

preds_in = np.nan_to_num(preds_in)
preds_out = np.nan_to_num(preds_out)

# Compute MMC
mmc_in = preds_in.max(1).mean()
mmc_out = preds_out.max(1).mean()

print(f'MMC in-distribution: {mmc_in:.3f}; out-distribution: {mmc_out:.3f}')

# Compute AUROC
labels = np.zeros(len(preds_in)+len(preds_out), dtype='int32')
labels[:len(preds_in)] = 1
examples = np.concatenate([preds_in.max(1), preds_out.max(1)])
auroc = roc_auc_score(labels, examples)

print(f'AUROC: {auroc:.3f}')
