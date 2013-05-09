import numpy as np
# from numpy.normal import
from scipy.optimize import fmin_bfgs, minimize
from scipy.stats import norm
from munging import read_csv
from time import time


def probit(x):
    return norm.cdf(x)


def logit(x):
    return np.log(x) - np.log(1 - x)


def logistic(x):
    # i.e., inverse-logit
    # e_xs_sum = np.e**(xs)
    # return e_xs_sum / (1. + e_xs_sum)
    return 1. / (1. + np.e**(-x))


def loglogistic(x):
    # i.e., inverse-logit
    # e_xs_sum = np.e**(xs)
    # return e_xs_sum / (1. + e_xs_sum)
    return np.log(1.) - np.log(1. + np.e**(-x))


rows = read_csv('/Users/chbrown/Downloads/05/Logistic Regression/logistic/ex2data1.txt')
dt = np.array(rows)

# return negmloglik(betas, X, Y, m, reflevel=reflevel)
rows, cols = np.shape(dt)
xss = np.column_stack((
    np.repeat([1], rows).reshape(100, 1),
    dt[:, 0:2]
    # dt[:, 1] * dt[:, 2]
))
ys = dt[:, 2]


def loglikelihood(betas):
    # the dot product computes the sums, leave us with a vector
    dotted = xss.dot(betas)
    logps = np.log(1.) - np.log(1. + np.e**(-dotted))
    invlogps = np.log(1.) - np.log(1. + np.e**(dotted))
    # logls = np.log(ps**ys) + np.log((1 - ps) ** (1 - ys))
    logls = ys * logps + (1 - ys) * invlogps
    # print 'logls', logls
    nll = -np.sum(logls)
    # print
    # print betas
    # print ps
    # print '-log(likelihood):', nll
    return nll

    # ls = (ps**ys) * ((1 - ps) ** (1 - ys))
    # return -np.log(np.prod(ls))


# we want to maximize log likelihood, we we minimize log likelihood
#  epsilon=1.0e-8


def test(betas):
    # print 'test:', xss.dot(betas)
    predicted = logistic(xss.dot(betas)) > 0.5
    gold = ys > 0.5
    return float(sum(predicted == gold)) / len(gold)


def learn_with_method(x0, method):
    print '\n# %s' % method

    started = time()
    x0 = np.array(x0)
    result = minimize(loglikelihood, x0, method=method)
    # result_full = fmin_bfgs(loglikelihood, prior, full_output=True, retall=False, disp=True)
    elapsed = time() - started

    print 'argmin = ', result.x
    print 'Time: %.3f s.' % elapsed
    if result.x.any():
        print '(overfit) Accuracy: %0.3f%%' % (test(result.x) * 100)
    else:
        print 'Failed'


# 'Newton-CG', requires Jacobian
# 'Anneal', , fails
# 'COBYLA', hangs
x0 = np.array([-1., 2., 2.])
print 'x0 = ', x0

methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']
for method in methods:
    # x0 = np.array([-25.0, 0.2, 0.2])
    learn_with_method(x0, method)

# print result
