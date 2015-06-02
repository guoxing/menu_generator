from numpy import *
from nn.base import NNBase
from nn.math import make_onehot
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

        # any other initialization you need
        self.nclass = dims[2]
        self.sparams.L = wv.copy()
        self.params.W = random_weight_matrix(*self.params.W.shape)
        self.params.U = random_weight_matrix(*self.params.U.shape)


        #### END YOUR CODE ####

    # def tanh_grad(h):
    #     return 1 - h*h


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
        x = self.sparams.L[window]
        x = reshape(x,(-1,))

        W = self.params.W
        U = self.params.U
        b1 = self.params.b1
        b2 = self.params.b2
        lreg = self.lreg

        h = tanh(W.dot(x) + b1)
        p = softmax(U.dot(h) + b2)

        y = make_onehot(label, len(p))
        delta2 = p - y
        delta1 = multiply(transpose(U).dot(delta2), 1-h*h)

        ##
        # Backpropagation
        self.grads.b2 += delta2
        self.grads.U += outer(delta2, h) + lreg * U
        self.grads.b1 += delta1
        self.grads.W += outer(delta1, x) + lreg * W
        C = window.shape[0]
        gradx = reshape(transpose(W).dot(delta1),(C,-1))
        # for i in range(C):
            # self.sgrads.L[window[i]] = gradx[i]
        self.sgrads.L[window] = gradx
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
            x = self.sparams.L[windows]
            x = reshape(x,(-1,))
        else:
            x = self.sparams.L[windows]
            x = reshape(x,(x.shape[0],-1,))

        #### YOUR CODE HERE ####

        h = tanh(x.dot(transpose(self.params.W)) + self.params.b1)
        P = softmax(h.dot(transpose(self.params.U)) + self.params.b2)

        #### END YOUR CODE ####

        return P # rows are output for each input
    
    def predict(self, windows):

        #### YOUR CODE HERE ####
        P = self.predict_proba(windows)
        return argmax(P, axis=1)

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
            x = self.sparams.L[windows]
            x = reshape(x,(-1,))

        else:
            x = self.sparams.L[windows]
            x = reshape(x,(x.shape[0],-1,))

        N = len(windows)        

        h = tanh(x.dot(transpose(self.params.W)) + self.params.b1)
        p = softmax(h.dot(transpose(self.params.U)) + self.params.b2)
        reg = (self.lreg / 2.0) * (sum(self.params.W**2.0) + sum(self.params.U**2.0))
        if N == 1:
            J = sum(-1 * log(p[labels]) + reg)
        else:
            J = sum(-1 * log(p[range(N), labels]) + reg)

        #### END YOUR CODE ####
        return J

def softmax(x):
    """ Softmax function """
    ### YOUR CODE HERE
    if x.ndim == 1:
        e = exp(x - max(x))
        return e / sum(e)
    else:
        e = exp(x - x.max(axis=1).reshape(-1,1))
        row_sums = sum(e, axis=1).reshape(-1, 1)
        return e / row_sums

        ### END YOUR CODE
