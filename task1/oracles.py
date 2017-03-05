import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        logadd = np.logaddexp(0, - self.b * self.matvec_Ax(x))
        res = np.linalg.norm(logadd, 1) / self.b.size +\
              np.linalg.norm(x, 2) ** 2 * self.regcoef / 2
        return res

    def grad(self, x):
        return self.regcoef * x - self.matvec_ATx(self.b * (expit(-self.b * self.matvec_Ax(x)))) / self.b.size

    def hess(self, x):
        tmp = expit(self.b * self.matvec_Ax(x))
        return self.matmat_ATsA(tmp * (1 - tmp)) / self.b.size + self.regcoef * np.identity(x.shape[0])


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self.prev_x = None
        self.prev_Ax = None
        self.prev_new_x = None
        self.prev_d = None
        self.prev_Ad = None
        self.prev_new_Ax = None

    def check_Av(self, v, key=''):
        x_to_res = dict((('prev_x', 'prev_Ax'), ('prev_d', 'prev_Ad'), ('prev_new_x', 'prev_new_Ax')))
        for u, k in x_to_res.items():
            tmp = getattr(self, u, None)
            if tmp is not None and np.all(tmp == v):
                return getattr(self, k)

        if key:
            Av = self.matvec_Ax(v)
            setattr(self, x_to_res[key], Av)
            setattr(self, key, v)
            return Av

    def func(self, x):
        Ax = self.check_Av(x, 'prev_x')
        return np.linalg.norm(np.logaddexp(0, -self.b * Ax), 1) / self.b.shape[0] + \
               (self.regcoef * np.linalg.norm(x, 2) ** 2) / 2

    def grad(self, x):
        Ax = self.check_Av(x, 'prev_x')
        return self.regcoef * x - self.matvec_ATx(self.b * (expit(- self.b * Ax))) / self.b.shape[0]

    def hess(self, x):
        Ax = self.check_Av(x, 'prev_x')
        tmp = expit(self.b * Ax)
        return self.matmat_ATsA(tmp * (1 - tmp)) / self.b.shape[0] + self.regcoef * np.identity(x.shape[0])

    def func_directional(self, x, d, alpha):
        new_x = x + alpha * d

        new_Ax = self.check_Av(new_x)
        if new_Ax is None:
            Ax = self.check_Av(x, 'prev_x')
            Ad = self.check_Av(d, 'prev_d')
            new_Ax = Ax + alpha * Ad
            self.prev_new_x = new_x
            self.prev_new_Ax = new_Ax

        return np.linalg.norm(np.logaddexp(0, -self.b * new_Ax), 1) / self.b.shape[0] + \
               (self.regcoef * np.linalg.norm(new_x, 2) ** 2) / 2

    def grad_directional(self, x, d, alpha):
        new_x = x + alpha * d

        new_Ax = self.check_Av(new_x)
        Ad = self.check_Av(d, 'prev_d')
        if new_Ax is None:
            Ax = self.check_Av(x, 'prev_x')
            new_Ax = Ax + alpha * Ad
            self.prev_new_x = new_x
            self.prev_new_Ax = new_Ax

        return self.regcoef * np.dot(np.transpose(new_x), d) - \
               np.dot(self.b * (expit(- self.b * new_Ax)), Ad) / self.b.shape[0]


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    if scipy.sparse.issparse(A):
        A = scipy.sparse.csr_matrix(A)
        matmat_ATsA = lambda x: matvec_ATx(matvec_ATx(scipy.sparse.diags(x)).T)
    else:
        matmat_ATsA = lambda x: np.dot(matvec_ATx(np.diag(x)), A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.array(x)
    f = func(x)
    e, res = np.identity(x.size), np.zeros_like(x)

    for i in range(x.size):
        res[i] = func(x + e[i] * eps) - f

    return res / eps


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i)
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.array(x)
    f = func(x)
    res, tmp, e = np.zeros((x.size, x.size)), np.zeros_like(x), np.identity(x.size)

    for i in range(x.size):
        tmp[i] = func(x + e[i] * eps)

    for i in range(x.size):
        for j in range(x.size):
            res[i][j] = func(x + eps * (e[i] + e[j]))- tmp[j] - tmp[i] + f

    return res / (eps ** 2)
