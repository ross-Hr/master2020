import torch
from torch import nn, optim, autograd
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import input_data
from sklearn.utils import shuffle as skshuffle
from math import *
from backpack import backpack, extend
from backpack.extensions import KFAC
from sklearn.metrics import roc_auc_score
import scipy
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

np.random.seed(123)
torch.manual_seed(123)


def get_batches(X, y, shuffle=False):
    if shuffle:
        X_s, y_s = skshuffle(X, y)
    else:
        X_s, y_s = X, y

    for i in range(0, len(X_s), batch_size):
        yield (X_s[i:i+batch_size], y_s[i:i+batch_size])


@torch.no_grad()
def predict(X_test, y_test, model, sample=1, T=1):
    py = []
    for x, y in get_batches(X_test, y_test):
        x, y = torch.from_numpy(x).float().cuda(), torch.from_numpy(y).long().cuda()
        #x, y = torch.from_numpy(x).float().cpu(), torch.from_numpy(y).long().cpu()

        py_ = 0
        for _ in range(sample):
            py_ += torch.softmax(model(x)/T, 1)
        py_ /= sample

        py.append(py_)
    return torch.cat(py, dim=0)


# MNIST
mnist = input_data.read_data_sets('./MNIST_data', one_hot=False)

X_train_mnist, y_train_mnist = mnist.train.images, mnist.train.labels
X_val_mnist, y_val_mnist = mnist.validation.images, mnist.validation.labels
X_test_mnist, y_test_mnist = mnist.test.images, mnist.test.labels

X_train_mnist = X_train_mnist.reshape(-1, 1, 28, 28)
X_val_mnist = X_val_mnist.reshape(-1, 1, 28, 28)
X_test_mnist = X_test_mnist.reshape(-1, 1, 28, 28)


# FMNIST
fmnist = input_data.read_data_sets('./FMNIST_data', one_hot=False)

X_train_fmnist, y_train_fmnist = fmnist.train.images, fmnist.train.labels
X_test_fmnist, y_test_fmnist = fmnist.test.images, fmnist.test.labels

X_train_fmnist = X_train_fmnist.reshape(-1, 1, 28, 28)
X_test_fmnist = X_test_fmnist.reshape(-1, 1, 28, 28)

#plt.imshow(X_train_fmnist[1].reshape(28,28))
#plt.show()

M = len(X_train_mnist)


class CNN(nn.Module):
    """ CNN from Hein et al., 2019 """

    def __init__(self, use_dropout=False):
        super(CNN, self).__init__()

        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc_h = nn.Linear(1024, 1024)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        phi = self.features(x)
        out = self.fc(phi)
        return out

    def features(self, x):
        h1 = F.max_pool2d(F.relu(self.conv1(x)), 2)
        h2 = F.max_pool2d(F.relu(self.conv2(h1)), 2)
        # h2 = h2.permute(0, 2, 3, 1)
        h2 = h2.reshape(h2.shape[0], -1)
        out = self.fc_h(h2)

        return out



"""
=============================================================================================
MAP
=============================================================================================
"""

model = CNN()
model = model.cuda()

opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[50, 75, 90])

# model.load_state_dict(torch.load(f'./lenet_mnist.th'))

batch_size = 200
pbar = trange(3)

for it in pbar:
    for x, y in get_batches(X_train_mnist, y_train_mnist, shuffle=True):
        x, y = torch.from_numpy(x).float().cuda(), torch.from_numpy(y).long().cuda()
        #x, y = torch.from_numpy(x).float().cpu(), torch.from_numpy(y).long().cpu()

        loss = F.cross_entropy(model(x), y)

        loss.backward()
        opt.step()
        opt.zero_grad()

    lr_scheduler.step()
    current_lr = opt.param_groups[0]['lr']
    pbar.set_description(f'[Loss: {loss.item():.5f}, lr: {current_lr}]')

torch.save(model.state_dict(), f'./lenet_mnist.th')


model.eval()


# In-distribution
py_in = predict(X_test_mnist, y_test_mnist, model).cpu().numpy()
targets = y_test_mnist
acc_in = np.mean(np.argmax(py_in, 1) == targets)
ents_in_map = -np.sum(py_in*np.log(py_in+1e-8), axis=1)
print(f'[In, MAP] Accuracy: {acc_in:.3f}; average entropy: {ents_in_map.mean():.3f}; MMC: {py_in.max(1).mean():.3f}')


# Out-distribution - FMNIST
py_out = predict(X_test_fmnist, y_test_fmnist, model).cpu().numpy()
ents_out_fmnist_map = -np.sum(py_out*np.log(py_out+1e-8), axis=1)

labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
labels[:len(py_in)] = 1
examples = np.concatenate([py_in.max(1), py_out.max(1)])
auroc = roc_auc_score(labels, examples)

print(f'[Out-FMNIST, MAP] Average entropy: {ents_out_fmnist_map.mean():.3f}; MMC: {py_out.max(1).mean():.3f}; AUROC: {auroc:.3f}')

print()


"""
=============================================================================================
Laplace-KF
=============================================================================================
"""

W = list(model.parameters())[-2]
b = list(model.parameters())[-1]
m, n = W.shape
lossfunc = torch.nn.CrossEntropyLoss()

var0 = 10
tau = 1/var0

extend(lossfunc, debug=False)
extend(model.fc, debug=False)

with backpack(KFAC()):
    U, V = torch.zeros(m, m, device='cuda'), torch.zeros(n, n, device='cuda')
    B = torch.zeros(m, m, device='cuda')
    #U, V = torch.zeros(m, m, device='cpu'), torch.zeros(n, n, device='cpu')
    #B = torch.zeros(m, m, device='cpu')

    for i, (x, y) in tqdm(enumerate(get_batches(X_train_mnist, y_train_mnist, shuffle=True))):
        x, y = torch.from_numpy(x).float().cuda(), torch.from_numpy(y).long().cuda()
        #x, y = torch.from_numpy(x).float().cpu(), torch.from_numpy(y).long().cpu()

        model.zero_grad()
        lossfunc(model(x), y).backward()

        with torch.no_grad():
            # Hessian of weight
            U_, V_ = W.kfac
            B_ = b.kfac[0]

            U_ = sqrt(batch_size)*U_ + sqrt(tau)*torch.eye(m, device='cuda')
            V_ = sqrt(batch_size)*V_ + sqrt(tau)*torch.eye(n, device='cuda')
            B_ = batch_size*B_ + tau*torch.eye(m, device='cuda')
            #U_ = sqrt(batch_size)*U_ + sqrt(tau)*torch.eye(m, device='cpu')
            #V_ = sqrt(batch_size)*V_ + sqrt(tau)*torch.eye(n, device='cpu')
            #B_ = batch_size*B_ + tau*torch.eye(m, device='cpu')

            rho = min(1-1/(i+1), 0.95)

            U = rho*U + (1-rho)*U_
            V = rho*V + (1-rho)*V_
            B = rho*B + (1-rho)*B_


# Predictive distribution
with torch.no_grad():
    M_W_post = W.t()
    M_b_post = b

    # Covariances for Laplace
    U_post = torch.inverse(V)  # Interchanged since W is transposed
    V_post = torch.inverse(U)
    B_post = torch.inverse(B)


@torch.no_grad()
def predict_laplace(X_test, y_test, n_samples=200):
    py = []

    for x, y in tqdm(get_batches(X_test, y_test)):
        x, y = torch.from_numpy(x).float().cuda(), torch.from_numpy(y).long().cuda()
        #x, y = torch.from_numpy(x).float().cpu(), torch.from_numpy(y).long().cpu()

        phi = model.features(x)

        mu_pred = phi @ M_W_post + M_b_post
        Cov_pred = torch.diag(phi @ U_post @ phi.t()).reshape(-1, 1, 1) * V_post.unsqueeze(0) + B_post.unsqueeze(0)

        post_pred = MultivariateNormal(mu_pred, Cov_pred)

        # MC-integral
        py_ = 0

        for _ in range(n_samples):
            f_s = post_pred.rsample()
            py_ += torch.softmax(f_s, 1)

        py_ /= n_samples

        py.append(py_)

    return torch.cat(py, dim=0)


# In-distribution
py_in = predict_laplace(X_test_mnist, y_test_mnist).cpu().numpy()
targets = y_test_mnist
acc_in = np.mean(np.argmax(py_in, 1) == targets)
ents_in_laplace_kf = -np.sum(py_in*np.log(py_in+1e-8), axis=1)
print(f'[In, LLLA-KF] Accuracy: {acc_in:.3f}; average entropy: {ents_in_laplace_kf.mean():.3f}; MMC: {py_in.max(1).mean():.3f}')


# Out-distribution - FMNIST
py_out = predict_laplace(X_test_fmnist, y_test_fmnist).cpu().numpy()
ents_out_fmnist_laplace_kf = -np.sum(py_out*np.log(py_out+1e-8), axis=1)

labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
labels[:len(py_in)] = 1
examples = np.concatenate([py_in.max(1), py_out.max(1)])
auroc = roc_auc_score(labels, examples)

print(f'[Out-FMNIST, LLLA-KF] Average entropy: {ents_out_fmnist_laplace_kf.mean():.3f}; MMC: {py_out.max(1).mean():.3f}; AUROC: {auroc:.3f}')
