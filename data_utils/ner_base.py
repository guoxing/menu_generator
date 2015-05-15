##
# Utility functions for NER baseline model
##

from utils import invert_dict
from numpy import *
from sklearn import metrics

def load_wv(vocabfile):
    with open(vocabfile) as fd:
        words = [line.strip() for line in fd if line.strip() != '']
    words = list(set(words))
    num_to_word = dict(enumerate(words))
    word_to_num = invert_dict(num_to_word)
    return word_to_num, num_to_word

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