import torch
import torchvision
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from backpack import backpack, extend
from backpack.extensions import KFAC, DiagHessian, DiagGGNMC
from sklearn.metrics import roc_auc_score


"""####### general utils #########"""

def get_accuracy(output, targets):
    """Helper function to print the accuracy"""
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()

"""####### Functions related to the Dirichlet Laplace transform ###########"""

def get_mu_from_Dirichlet(alpha):
    K = len(alpha)
    mu = torch.zeros(K)
    for i in range(K):
        mu_i = torch.log(alpha[i]) - 1/K * torch.sum(torch.log(alpha))
        mu[i] = mu_i

    return(torch.Tensor(mu))

def get_Sigma_from_Dirichlet(alpha):
    K = len(alpha)
    sum_of_inv = 1/K * torch.sum(1/alpha)
    Sigma = torch.zeros((K,K))
    for k in range(K):
        for l in range(K):
            delta = 1 if k==l else 0
            Sigma[k][l] = delta * 1/alpha[k] - 1/K*(1/alpha[k] + 1/alpha[l] - sum_of_inv)

    return(torch.Tensor(Sigma))


def get_alpha_from_Normal(mu, Sigma):
    batch_size, K = mu.size(0), mu.size(-1)
    alpha = torch.zeros(batch_size, K)
    sum_exp = torch.sum(torch.exp(-1*torch.Tensor(mu)), dim=1)
    for j in range(batch_size):
        for k in range(K):
            alpha[j][k] = 1/(Sigma[j][k][k] + 10e-8)*(1 - 2/K + torch.exp(mu[j][k])/K**2 * sum_exp[j])

    return(torch.Tensor(alpha))


def get_Gaussian_output_old(x, mu_w, mu_b, sigma_w, sigma_b):
    #get the distributions per class
    batch_size = x.size(0)
    num_classes = len(mu_b)
    #print("batch_size, num_classes: ", batch_size, num_classes)
    mu_batch = torch.zeros(batch_size, num_classes)
    sigma_batch = torch.zeros(batch_size, num_classes, num_classes)
    for i in range(batch_size):
        per_class_sigmas = torch.zeros(num_classes)
        for j in range(num_classes):
            #create a diagonal Hessian
            hess = torch.diag(sigma_w[j])
            #b = x[i] @ hess @ x[i].t()
            #a = sigma_b[i]
            per_class_sigmas[j] = x[i] @ hess @ x[i].t() + sigma_b[j]

        #print("sizes: ", mu_w.size(), x[i].size(), mu_b.size())
        per_class_mus = x[i] @ mu_w + mu_b
        mu_batch[i] = per_class_mus
        sigma_batch[i] = torch.diag(per_class_sigmas)

    return(mu_batch, sigma_batch)

def get_Gaussian_output(x, mu_w, mu_b, sigma_w, sigma_b):
    #get the distributions per class
    batch_size = x.size(0)
    num_classes = mu_b.size(0)
    
    # get mu batch
    mu_w_batch = mu_w.repeat(batch_size, 1, 1)
    mu_b_batch = mu_b.repeat(batch_size, 1)
    mu_batch = torch.bmm(x.view(batch_size, 1, -1), mu_w_batch).view(batch_size, -1) + mu_b_batch
    
    #get sigma batch
    sigma_w_batch = sigma_w.repeat(batch_size, 1, 1)
    sigma_b_batch = sigma_b.repeat(batch_size, 1)
    sigmas_diag = torch.zeros(batch_size, num_classes, device='cuda')
    for j in range(num_classes):
        h1 = x * sigma_w_batch[:, j]
        helper = torch.matmul(h1.view(batch_size, 1, -1), x.view(batch_size, -1, 1))
        helper = helper.view(-1) + sigma_b_batch[:,j]
        sigmas_diag[:,j] = helper
        
    sigma_batch = torch.stack([torch.diag(x) for x in sigmas_diag])

    
    return(mu_batch, sigma_batch)



"""########## Functions related to the acquisition of second order information ###########"""

###########Laplace-KF
#=============================================================================================

def KFLP_second_order(model, batch_size, train_loader, var0 = 10, device='cpu'):

    W = list(model.parameters())[-2]
    b = list(model.parameters())[-1]
    m, n = W.shape
    lossfunc = torch.nn.CrossEntropyLoss()

    tau = 1/var0

    extend(lossfunc, debug=False)
    extend(model.fc, debug=False)

    with backpack(KFAC()):
        U, V = torch.zeros(m, m, device=device), torch.zeros(n, n, device=device)
        B = torch.zeros(m, m, device=device)

        max_len = int(np.ceil(len(train_loader.dataset)/batch_size))
        for batch_idx, (x, y) in enumerate(train_loader):

            if device == 'cuda':
                x, y = x.cuda(), y.cuda()

            model.zero_grad()
            lossfunc(model(x), y).backward()

            with torch.no_grad():
                # Hessian of weight
                U_, V_ = W.kfac
                B_ = b.kfac[0]

                U_ = np.sqrt(batch_size)*U_ + np.sqrt(tau)*torch.eye(m, device=device)
                V_ = np.sqrt(batch_size)*V_ + np.sqrt(tau)*torch.eye(n, device=device)
                B_ = batch_size*B_ + tau*torch.eye(m, device=device)

                rho = min(1-1/(batch_idx+1), 0.95)

                U = rho*U + (1-rho)*U_
                V = rho*V + (1-rho)*V_
                B = rho*B + (1-rho)*B_

            print("Batch: {}/{}".format(batch_idx, max_len))


    # Predictive distribution
    with torch.no_grad():
        M_W_post = W.t()
        M_b_post = b

        # Covariances for Laplace
        U_post = torch.inverse(V)  # Interchanged since W is transposed
        V_post = torch.inverse(U)
        B_post = torch.inverse(B)

    return(M_W_post, M_b_post, U_post, V_post, B_post)


###### Different way of getting the normal distribution over the outputs
###### we just assume everything to be independent everywhere
###########Laplace-KF
#=============================================================================================

def Diag_second_order(model, batch_size, train_loader, var0 = 10, device='cpu'):

    W = list(model.parameters())[-2]
    b = list(model.parameters())[-1]
    m, n = W.shape
    print("n: {} inputs to linear layer with m: {} classes".format(n, m))
    lossfunc = torch.nn.CrossEntropyLoss()

    tau = 1/var0

    extend(lossfunc, debug=False)
    extend(model.fc, debug=False)

    with backpack(DiagHessian()):

        max_len = int(np.ceil(len(train_loader.dataset)/batch_size))
        weights_cov = torch.zeros(max_len, m, n, device=device)
        biases_cov = torch.zeros(max_len, m, device=device)

        for batch_idx, (x, y) in enumerate(train_loader):

            if device == 'cuda':
                x, y = x.cuda(), y.cuda()

            model.zero_grad()
            lossfunc(model(x), y).backward()

            with torch.no_grad():
                # Hessian of weight
                W_ = W.diag_h
                b_ = b.diag_h

                #add_prior: since it will be flattened later we can just add the prior like that
                W_ += tau * torch.ones(W_.size(), device=device)
                b_ += tau * torch.ones(b_.size(), device=device)


            weights_cov[batch_idx] = W_
            biases_cov[batch_idx] = b_

            print("Batch: {}/{}".format(batch_idx, max_len))

        print(len(weights_cov))
        C_W = torch.mean(weights_cov, dim=0)
        C_b = torch.mean(biases_cov, dim=0)

    # Predictive distribution
    with torch.no_grad():
        M_W_post = W.t()
        M_b_post = b

        C_W_post = C_W
        C_b_post = C_b

    return(M_W_post, M_b_post, C_W_post, C_b_post)



"""########## Functions related to predicting the distribution over the outputs ############"""



@torch.no_grad()
def predict_MAP(model, test_loader, num_samples=1, cuda=False):
    py = []

    max_len = int(np.ceil(len(test_loader.dataset)/len(test_loader)))
    for batch_idx, (x, y) in enumerate(test_loader):

        if cuda:
            x, y = x.cuda(), y.cuda()

        py_ = 0
        for _ in range(num_samples):
            py_ += torch.softmax(model(x), 1)
        py_ /= num_samples

        py.append(py_)
    return torch.cat(py, dim=0)


@torch.no_grad()
def predict_KFAC_sampling(model, test_loader, M_W_post, M_b_post, U_post, V_post, B_post, n_samples, verbose=False, cuda=False):
    py = []
    max_len = int(np.ceil(len(test_loader.dataset)/len(test_loader)))

    for batch_idx, (x, y) in enumerate(test_loader):

        if cuda:
            x, y = x.cuda(), y.cuda()

        phi = model.phi(x)

        mu_pred = phi @ M_W_post + M_b_post
        Cov_pred = torch.diag(phi @ U_post @ phi.t()).reshape(-1, 1, 1) * V_post.unsqueeze(0) + B_post.unsqueeze(0)

        post_pred = MultivariateNormal(mu_pred, Cov_pred)

        # MC-integral
        py_ = 0

        for _ in range(n_samples):
            f_s = post_pred.rsample()
            py_ += torch.softmax(f_s, 1)


        py_ /= n_samples
        py_ = py_.detach()

        py.append(py_)

        if verbose:
            print("Batch: {}/{}".format(batch_idx, max_len))

    return torch.cat(py, dim=0)


@torch.no_grad()
def predict_diagonal_sampling(model, test_loader, M_W_post, M_b_post, C_W_post, C_b_post, n_samples, verbose=False, cuda=False):
    py = []
    max_len = len(test_loader)

    for batch_idx, (x, y) in enumerate(test_loader):

        if cuda:
            x, y = x.cuda(), y.cuda()

        phi = model.phi(x)

        mu, Sigma = get_Gaussian_output(phi, M_W_post, M_b_post, C_W_post, C_b_post)

        post_pred = MultivariateNormal(mu, Sigma)

        # MC-integral
        py_ = 0

        for _ in range(n_samples):
            f_s = post_pred.rsample()
            py_ += torch.softmax(f_s, 1)

        py_ /= n_samples
        py_ = py_.detach()

        py.append(py_)

        if verbose:
            print("Batch: {}/{}".format(batch_idx, max_len))

    return torch.cat(py, dim=0)


@torch.no_grad()
def predict_DIR_LPA(model, test_loader, M_W_post, M_b_post, U_post, V_post, B_post, verbose=False, cuda=False):
    alphas = []

    max_len = len(test_loader)

    for batch_idx, (x, y) in enumerate(test_loader):
        
        if cuda:
            x, y = x.cuda(), y.cuda()

        phi = model.phi(x)

        mu_pred = phi @ M_W_post + M_b_post
        Cov_pred = torch.diag(phi @ U_post @ phi.t()).reshape(-1, 1, 1) * V_post.unsqueeze(0) + B_post.unsqueeze(0)

        alpha = get_alpha_from_Normal(mu_pred, Cov_pred)
        #print(alpha)
        #alpha = get_alpha_from_Normal(mu_pred.view(10), Cov_pred.view(10,10))
        alpha /= alpha.sum(dim=1).view(-1,1).detach()
        alpha = alpha.detach()

        #assert(torch.sum(alpha.sum(dim=1) - torch.ones(len(x))). == pytest.approx(0, 10e-5))

        alphas.append(alpha)


        if verbose:
            print("Batch: {}/{}".format(batch_idx, max_len))

    return(torch.cat(alphas, dim = 0))


@torch.no_grad()
def predict_DIR_LPA_diag(model, test_loader, M_W_post, M_b_post, C_W_post, C_b_post, verbose=False):
    #num_classes = M_b_post.size(0)
    #alphas = torch.zeros(10000, num_classes) #i need to automate the 10000
    alphas = []

    max_len = len(test_loader)
    for batch_idx, (x, y) in enumerate(test_loader):

        phi = model.phi(x)

        mu, Sigma = get_Gaussian_output(phi, M_W_post, M_b_post, C_W_post, C_b_post)
        #print("mu, sigma size: ", mu.size(), Sigma.size())

        alpha = get_alpha_from_Normal(mu, Sigma)
        #print(alpha)
        #alpha = get_alpha_from_Normal(mu_pred.view(10), Cov_pred.view(10,10))
        alpha /= alpha.sum(dim=1).view(-1,1).detach()
        alpha = alpha.detach()
        #print(alpha)


        alphas.append(alpha)

        if verbose:
            print("Batch: {}/{}".format(batch_idx, max_len))

    return(torch.cat(alphas, dim = 0))


"""############ Functions related to calculating and printing the results ############"""

######## get all the data for in and out of dist samples

def get_in_dist_values(py_in, targets):
    acc_in = np.mean(np.argmax(py_in, 1) == targets)
    prob_correct = np.choose(targets, py_in.T).mean()
    average_entropy = -np.sum(py_in*np.log(py_in+1e-8), axis=1).mean()
    MMC = py_in.max(1).mean()
    return(acc_in, prob_correct, average_entropy, MMC)

def get_out_dist_values(py_in, py_out, targets):
    average_entropy = -np.sum(py_out*np.log(py_out+1e-8), axis=1).mean()
    acc_out = np.mean(np.argmax(py_out, 1) == targets)
    if max(targets) > len(py_in[0]):
        targets = np.array(targets)
        targets[targets >= len(py_in[0])] = 0
    prob_correct = np.choose(targets, py_out.T).mean()
    labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
    labels[:len(py_in)] = 1
    examples = np.concatenate([py_in.max(1), py_out.max(1)])
    auroc = roc_auc_score(labels, examples)
    MMC = py_out.max(1).mean()
    return(acc_out, prob_correct, average_entropy, MMC, auroc)

def print_in_dist_values(acc_in, prob_correct, average_entropy, MMC, train='mnist', method='KFAC'):

    print(f'[In, {method}, {train}] Accuracy: {acc_in:.3f}; average entropy: {average_entropy:.3f}; \
    MMC: {MMC:.3f}; Prob @ correct: {prob_correct:.3f}')


def print_out_dist_values(acc_out, prob_correct, average_entropy, MMC, auroc, train='mnist', test='FMNIST', method='KFAC'):

    print(f'[Out-{test}, {method}, {train}] Accuracy: {acc_out:.3f}; Average entropy: {average_entropy:.3f};\
    MMC: {MMC:.3f}; AUROC: {auroc:.3f}; Prob @ correct: {prob_correct:.3f}')
