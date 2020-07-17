from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            # First compute gradient of the regularization term
            dW[i,j] = reg*2*W[i,j]
            # Now compute gradient of the loss term
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if j == y[i]:
                for k in range(num_classes):
                    if k != j:
                        margin_k = scores[k] - scores[j] + 1
                        if margin_k > 0:
                            dW[:,j] -= X[i]
            else:
                if margin > 0:
                    dW[:,j] += X[i]

    dW *= 1/num_train
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]

    scores = X.dot(W)
    correct_class_weights = W[:,y]
    # Compute diagonal entries of X.dot(correct_class_weights)
    correct_class_scores = (X*correct_class_weights.T).sum(-1)
    margins = np.transpose(np.transpose(scores) - correct_class_scores)+1
    margins = np.maximum(0,margins)

    # For each training sample i,
    # zero the entries loss_matrix[i,j],
    # which doesn't appear in L_i
    margins[np.arange(num_train), y] = 0

    loss = 1/num_train * np.sum(margins)
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # construct matrix with 1 at entries with positive margin
    # and zero otherwise
    margins_met = margins
    margins_met[margins_met > 0] = 1
    margin_met_count = margins_met.sum(axis = 1)

    # Subtract for term corresponding to correct class
    margins_met[np.arange(num_train), y] -= margin_met_count
    dW = (X.T).dot(margins_met)
    dW /= num_train

    # Add regularization term
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
