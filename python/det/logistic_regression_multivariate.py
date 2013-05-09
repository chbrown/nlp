import numpy as np
from scipy.optimize import fmin_bfgs
from munging import read_csv
# minimize(fun, x0, args=(), method='BFGS', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
# fmin_bfgs(f, x0, fprime=None, args=(), gtol=1e-05, norm=inf, epsilon=1.4901161193847656e-08, maxiter=None, full_output=0, disp=1, retall=0, callback=None)

# def estmlogit(Betas, X,Y, m=None, maxiter=10, full_output=True, reflevel= 0, disp=False):


rows = read_csv('/Users/chbrown/Downloads/05/logistic/ex2data1.txt')
dt = np.array(rows)

# return negmloglik(betas, X, Y, m, reflevel=reflevel)
xss = dt[:, 0:2]
ys = dt[:, 2]

betas = [0.5, 0.5]

def fun(betas):

    return -loglikelihood()



# we want to maximize log likelihood, we we minimize log likelihood
#  epsilon=1.0e-8
prior = betas
result = fmin_bfgs(fun, prior, maxiter=maxiter, full_output=True, retall=False, disp=True)



def negmloglik(Betas, X, Y,  m, reflevel=0):
    """
    log likelihood for polytomous regression or mlogit.
    Betas - estimated coefficients, as a SINGLE array!
    Y values are coded from 0 to ncategories - 1

    Beta matrix
            b[0][0] + b[0][1]+ b[0][2]+ ... + b[[0][D-1]
            b[1][0] + b[1][1]+ b[1][2]+ ... + b[[1][D-1]
                        ...
            b[ncategories-1][0] + b[ncategories-1][1]+ b[ncategories-1][2]
             .... + ... + b[[ncategories - 1][D-1]

            Stored in one array! The beta   coefficients for each level
            are stored with indices in range(level*D , level *D + D)
    X,Y   data X matrix and integer response Y vector with values
            from 0 to maxlevel=ncategories-1
    m - number of categories in Y vector. each value of  ylevel in Y must be in the
            interval [0, ncategories) or 0 <= ylevel < m
    reflevel - reference level, default code: 0
    """

    n  = len(X[0]) # number of coefficients per level.
    L  = 0
    for row, (xrow, ylevel) in enumerate(zip(X,Y)):
        h   = [0.0] * m
        denom = 0.0
        for k in range(m):
                 if k == reflevel:
                    h[k] = 0
                    denom += 1
                 else:
                    sa = k * n
                    v = sum([(x * b) for (x,b) in zip(xrow, Betas[sa: sa + n])])
                    h[k] = v
                    denom += exp(v)
        deltaL = h[ylevel] - log(denom)
        L += deltaL
    return -2 * L