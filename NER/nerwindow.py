from numpy import *
from nn.base import NNBase
from nn.math import softmax, make_onehot, sigmoid
from misc import random_weight_matrix


##
# Evaluation code; do not change this
##
from sklearn import metrics
def full_report(y_true, y_pred, tagnames):
    cr = metrics.classification_report(y_true, y_pred,
                                       target_names=tagnames)
    print cr

def eval_performance(y_true, y_pred, tagnames):
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    print "=== Performance (omitting 'O' class) ==="
    print "Mean precision:  %.02f%%" % (100*sum(pre[1:] * support[1:])/sum(support[1:]))
    print "Mean recall:     %.02f%%" % (100*sum(rec[1:] * support[1:])/sum(support[1:]))
    print "Mean F1:         %.02f%%" % (100*sum(f1[1:] * support[1:])/sum(support[1:]))


# def softmax(x):
#         if not hasattr(x[0], "__iter__"):
#             x = [x]

#         probs = exp(x - amax(x, axis=1, keepdims=True))
#         probs /= sum(probs, axis=1, keepdims=True)

#         return probs

##
# Implement this!
##
class WindowMLP(NNBase):
    """Single hidden layer, plus representation learning."""

    def __init__(self, wv, windowsize=3,
                 dims=[None, 100, 5],
                 reg=0.001, alpha=0.01, rseed=10):
        """
        Initialize classifier model.

        Arguments:
        wv : initial word vectors (array |V| x n)
            note that this is the transpose of the n x |V| matrix L
            described in the handout; you'll want to keep it in
            this |V| x n form for efficiency reasons, since numpy
            stores matrix rows continguously.
        windowsize : int, size of context window
        dims : dimensions of [input, hidden, output]
            input dimension can be computed from wv.shape
        reg : regularization strength (lambda)
        alpha : default learning rate
        rseed : random initialization seed
        """

        # Set regularization
        self.lreg = float(reg)
        self.alpha = alpha # default training rate

        dims[0] = windowsize * wv.shape[1] # input dimension
        param_dims = dict(W=(dims[1], dims[0]),
                          b1=(dims[1],),
                          U=(dims[2], dims[1]),
                          b2=(dims[2],),
                          )
        param_dims_sparse = dict(L=wv.shape)

        # initialize parameters: don't change this line
        NNBase.__init__(self, param_dims, param_dims_sparse)

        random.seed(rseed) # be sure to seed this for repeatability!
        #### YOUR CODE HERE ####
        self.nclass = dims[-1] # number of output classes
        self.sparams.L = wv.copy() # store own representations
        self.params.W = random_weight_matrix(*self.params.W.shape)
        self.params.U = random_weight_matrix(*self.params.U.shape)
        #### END YOUR CODE ####

    def _acc_grads(self, window, label):
        """
        Accumulate gradients, given a training point
        (window, label) of the format

        window = [x_{i-1} x_{i} x_{i+1}] # three ints
        label = {0,1,2,3,4} # single int, gives class

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.U += (your gradient dJ/dU)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # this adds an update for that index
        """
        #### YOUR CODE HERE ####

        ##
        # Forward propagation
        N = len(window)
        x = self.sparams.L[window]
        d = x.shape[1]
        x = x.reshape((x.shape[0]*x.shape[1]))
        z = self.params.W.dot(x) + self.params.b1
        h = tanh(z)
        y_hat = softmax(self.params.U.dot(h) + self.params.b2)

        ##
        # Backpropagation
        y = make_onehot(label, len(y_hat))
        delta = y_hat - y

        # dJ/dU, dJ/db2, dJ/dW, dJ/db1, dJ/dL
        self.grads.U += outer(delta, h) + self.lreg * self.params.U
        self.grads.b2 += delta

        tanh_prime_z = 1-tanh(z)**2
        self.grads.W += outer(tanh_prime_z*delta.dot(self.params.U), x) + self.lreg * self.params.W
        self.grads.b1 += tanh_prime_z*delta.dot(self.params.U)

        temp = (tanh_prime_z*delta.dot(self.params.U)).dot(self.params.W)
        for n in xrange(N):
            self.sgrads.L[window[n]] = temp[n*d:(n+1)*d]
        #### END YOUR CODE ####


    def predict_proba(self, windows):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.

        windows = array (n x windowsize),
            each row is a window of indices
        """
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]

        #### YOUR CODE HERE ####
        N = len(windows)
        P = zeros((N, self.params.b2.shape[0]))

        for n in xrange(N):
            x = self.sparams.L[windows[n]]
            x = x.reshape((x.shape[0]*x.shape[1]))
            z = self.params.W.dot(x) + self.params.b1
            h = tanh(z)
            P[n,:] = softmax(self.params.U.dot(h) + self.params.b2)

        #### END YOUR CODE ####

        return P # rows are output for each input


    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """

        #### YOUR CODE HERE ####
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]

        P = self.predict_proba(windows)
        c = argmax(P, axis=1)

        #### END YOUR CODE ####
        return c # list of predicted classes

    def compute_loss(self, windows, labels):
        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """

        #### YOUR CODE HERE ####
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]
            labels = [labels]

        N = len(windows)

        # x = self.sparams.L[windows]
        # x = x.reshape((N,x.shape[-2]*x.shape[-1]))
        # z = x.dot(self.params.W.T) + self.params.b1
        # h = tanh(z)
        # z2 = h.dot(self.params.U.T) + self.params.b2
        # p = softmax(z2)
        # J -= sum(log(p[0][labels])
        # J += (self.lreg / 2.0) * (sum(self.params.W**2.0) + sum(self.params.U**2.0))

        J = 0
        for n in xrange(N):
            x = self.sparams.L[windows[n]]
            x = reshape(x, x.shape[0]*x.shape[1])
            h = tanh(self.params.W.dot(x) + self.params.b1)
            y_hat = softmax(self.params.U.dot(h) + self.params.b2)
            y = make_onehot(labels[n], len(y_hat))
            J -= sum(y*log(y_hat))
        J += (self.lreg / 2.0) * (sum(self.params.W**2.0) + sum(self.params.U**2.0))
        #### END YOUR CODE ####
        return J
