import sys, os
from numpy import *
from matplotlib.pyplot import *
matplotlib.rcParams['savefig.dpi'] = 100
import data_utils.utils as du
import data_utils.ner_base as nerbase
from sklearn import linear_model


def isCapital(x):
    if len(x) > 0:
        return x[0].isupper()
    return false

def extractFeatures(X, words, windowsize, word_to_num):
    fv1 = contextFeature(X, windowsize, word_to_num)
    fv2 = capitalFeature(words)
    fv = column_stack((fv1,fv2))
    return fv

def contextFeature(X, windowsize, word_to_num):
    m = len(word_to_num)
    onehot_x = identity(m)

    wdim = m * windowsize
    N = X.shape[0]
    fvec = onehot_x[X].reshape(N, wdim)
    return fvec

def capitalFeature(words):
    fvec = array([isCapital(ww) for ww in words]).astype(int)
    return fvec

def dataSubset(size, X, y, words):
    return X[:size], y[:size], words[:size]

word_to_num, num_to_word = nerbase.load_wv('data/vocab.txt')
tagnames = ["O", "B", "I", "L", "U"]
num_to_tag = dict(enumerate(tagnames))
tag_to_num = du.invert_dict(num_to_tag)
wsize = 3
X_train, y_train, words_train = du.generateData('data/train_small', word_to_num, tag_to_num, wsize)

print "Training..."

TRAIN_SIZE = len(X_train) 
X_s, y_s, w_s = dataSubset(TRAIN_SIZE, X_train, y_train, words_train)

fv = extractFeatures(X_s, w_s, wsize, word_to_num)

lr = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')
lr.fit(fv, y_s)
y_pred = lr.predict(fv)
print nerbase.full_report(y_s, y_pred, tagnames)

X_dev, y_dev, words_dev = du.generateData('data/dev_small', word_to_num, tag_to_num, wsize)


DEV_SIZE = len(X_dev)
Xd_s, yd_s, wd_s = dataSubset(DEV_SIZE, X_dev, y_dev, words_dev)

fvdev = extractFeatures(Xd_s, wd_s, wsize, word_to_num)
yp = lr.predict(fvdev)

nerbase.full_report(yd_s, yp, tagnames)
