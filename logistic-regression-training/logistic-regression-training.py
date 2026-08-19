import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asarray(X, float)
    y = np.asarray(y,float)
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    for i in range(steps):
        logits = X @ w + b
        predict = _sigmoid(logits)
        Deri_w = X.T @ ( predict - y) / N
        Deri_b = np.mean(predict - y)
        w = w - lr * Deri_w
        b = b - lr * Deri_b
    return w, b
    pass 

   