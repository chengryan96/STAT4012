import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

jpg_path = os.path.abspath(os.path.join('__file__', '..', '..', 'jpg'))
# mnist is a dataset of 28x28 images of handwritten digits and their labels
mnist = tf.keras.datasets.mnist
# X for features , y for labels
# 28x28 numbers of 0-9
# unpacks images to x_train / x_test and labels to y_train / y_test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Limit the pixel value between 0 and 1 to avoid compoutational explode

x_train[x_train != 0] = 1
x_test[x_test != 0] = 1

# set index
M = len(set(y_train))  # M = 10
img_dim = x_train.shape[1:3]  # 28x28
N_train = len(y_train)  # 60000
N_test = len(y_test)  # 10000

# compute posteriot prob where post: posterior prob and pi : $\pi$


def loglike_func(post, pi, mu, x, y):
    loglike = 0
    mu[mu == 0] = 1e-323
    mu[mu == 1] = 1-(1e-323)
    for m in range(0, M):
        id = np.where(y == m)[0]
        for i in id:
            log_sums = np.sum(x[i, :, :]*np.log(mu[m, :, :]) +
                              (1-x[i, :, :])*np.log(1-mu[m, :, :]))
            if np.isnan(log_sums) == True:
                break
            loglike = loglike + post[i]*(np.log(pi[m])+log_sums)
    return loglike


# ???

pi0 = np.random.uniform(0, 1, 10)
pi0 = pi0/np.sum(pi0)
mu0 = np.zeros((M, img_dim[0], img_dim[1]))
for i in range(1, M):
    digit = x_train[np.where(y_train == i)[0]]
    mu0[i, :, :] = np.mean(digit, axis=0)
mu_new = mu0
pi_new = pi0

#
iter = 10
pi_record = np.zeros((M, iter))
loglike_record = np.zeros((iter, 1))

for it in range(0, iter):
    # ard 40s for each iteration
    print('no. iteration', it)
    mu_old = mu_new
    pi_old = pi_new
    print(pi_old)
    posterior = np.zeros((N_train, 1))
    for m in range(0, M):
        id = np.where(y_train == m)[0]
        for i in id:
            numerator = np.zeros((M, 1))
            for r in range(0, M):
                binm = (mu_old[r, :, :]**x_train[i, :, :]) * \
                    (1-mu_old[r, :, :])**(1-x_train[i, :, :])
                # avoid 0
                binm[np.argwhere(binm == 0)] = 1
                numerator[r] = pi_old[r]*np.prod(binm)

            #denominator = sum(numerator)
            # replace nan to 0
            posterior[np.argwhere(np.isnan(posterior))] = 0
            posterior[i] = numerator[m]/np.sum(numerator)
            # convert all 0 to 1
            # if posterior[i] == 0:
            #     posterior[i] = 1

# compute new mu and pi using 4.2.22
    mu_new = np.zeros((M, img_dim[0], img_dim[1]))
    pi_new = np.zeros((M, 1))
    for m in range(0, M):
        pm_old = np.copy(posterior)
        pm_old[np.where(y_train != m)[0]] = 0
        px = np.zeros((img_dim[0], img_dim[1]))
        for j in range(0, N_train):
            px = px + pm_old[j]*x_train[j, :, :]
        # mu_new
        mu_new[m, :, :] = px/np.sum(pm_old.ravel())
        pi_new[m] = np.sum(pm_old)/N_train
        pi_new = pi_new/np.sum(pi_new)
        pi_record[:, it] = pi_new.ravel()
        loglike_record[it] = loglike_func(
            posterior, pi_old, mu_old, x_train, y_train)
        print(loglike_record)
#
post, pi, mu, x, y = posterior, pi_old, mu_old, x_train, y_train

# test
y_hat = np.zeros((N_test, 1))
for i in range(0, N_test):
    y_prob = np.zeros((M, 1))
    for m in range(0, M):
        binm = (mu_new[m, :, :]**x_test[i, :, :]) * \
            (1-mu_new[m, :, :])**(1-x_test[i, :, :])
        binm[np.argwhere(binm == 0)] = 1
        y_prob[m] = np.prod(binm)
    y_hat[i] = np.argmax(y_prob)

pd.crosstab(y_hat.ravel(), y_test, rownames=['x'], colnames=['y'])

sum(np.diag(pd.crosstab(y_hat.ravel(), y_test)))/N_test

fig = plt.figure()
plt.style.use('ggplot')
plt.plot(loglike_record, 'o', label="log-likelihood")
plt.xlabel("Iterations")
plt.ylabel("Log-likelihood")
plt.title('Log-likelihood record')
plt.legend(loc="best")
fig.savefig(os.path.join(jpg_path, 'q3.jpg'))
