import optparse
import sys, os
from numpy import *
from matplotlib.pyplot import *
from misc import random_weight_matrix
import data_utils.utils as du
import data_utils.ner as ner
from nerwindow import full_report, eval_performance
from nerwindow import WindowMLP
#from keras.models import Sequential
#from keras.utils import np_utils
#from keras.layers.core import Dense, Activation
#from keras.optimizers import SGD
#from keras.regularizers import l2

matplotlib.rcParams['savefig.dpi'] = 100
random.seed(10)
tagnames = ["O", "B", "I", "L", "U"]

def load_data():
  wsize = opts.wsize
  print "load data..."
  wv, word_to_num, num_to_word = ner.load_wv('data/vocab.txt',
                                             'data/wvec.txt')
  num_to_tag = dict(enumerate(tagnames))
  tag_to_num = du.invert_dict(num_to_tag)
  pad = (wsize - 1)/2
  X_train, y_train, words_train = du.generateData('data/train_small', word_to_num, tag_to_num, wsize)
  X_dev, y_dev, words_dev = du.generateData('data/dev_small', word_to_num, tag_to_num, wsize)
  return X_train, y_train, X_dev, y_dev, wv

def train(X_train, y_train, X_dev, y_dev, wv):
  wsize = opts.wsize
  batchSize = opts.batchSize
  N = opts.nepoch * len(y_train)
  l = len(y_train)
  idxiter = [list(random.choice(l, batchSize)) for i in xrange(N/batchSize)]
  clf = WindowMLP(wv, windowsize=wsize, dims=[None, opts.hiddenDim, 5],
                  reg=opts.reg, alpha=opts.alpha)
  print "training..."
  clf.train_sgd(X_train, y_train,
                idxiter, 
                alphaiter=None,
                printevery=10000, 
                costevery=10000,
                devidx=None)
  y_pred = clf.predict(X_train)
  eval_performance(y_train, y_pred, tagnames)
  yp = clf.predict(X_dev)
  ner.save_predictions(yp, "dev.predicted")
  full_report(y_dev, yp, tagnames)
  eval_performance(y_dev, yp, tagnames) 


if __name__=='__main__':

    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--wsize",dest="wsize",type="int",default=3)
    parser.add_option("--hiddenDim",dest="hiddenDim",type="int",default=100)
    parser.add_option("--reg",dest="reg",type="float",default=0.001)
    parser.add_option("--alpha",dest="alpha",type="float",default=0.01)
    parser.add_option("--nepoch",dest="nepoch",type="int",default=5)
    parser.add_option("--batchSize",dest="batchSize",type="int",default=5)
    (opts,args)=parser.parse_args()

    X_train_ori, y_train_ori, X_dev_ori, y_dev_ori, wv = load_data()
    train(X_train_ori, y_train_ori, X_dev_ori, y_dev_ori, wv)
#    evaluate(X_dev_ori, y_dev_ori)

def keras():
    Xt = wv[X_train_ori]
    Xd = wv[X_dev_ori]
    X_train = Xt.reshape(Xt.shape[0], Xt.shape[1]*Xt.shape[2])
    X_dev = Xd.reshape(Xd.shape[0], Xd.shape[1]*Xd.shape[2])
    Y_train = np_utils.to_categorical(y_train_ori, 5)
    Y_dev = np_utils.to_categorical(y_dev_ori, 5)

    model = Sequential()
    inputDim = opts.wsize * 50
    model.add(Dense(input_dim=inputDim, output_dim=opts.hiddenDim, init="uniform"))
    model.add(Activation("tanh"))
    model.add(Dense(input_dim=opts.hiddenDim, output_dim=5, init="uniform"))
    model.add(Activation("softmax"))
    model.add(Dense(64, 64, W_regularizer = l2(opts.reg)))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=opts.alpha, momentum=0, nesterov=False))

    model.fit(X_train, Y_train, nb_epoch=opts.nepoch, batch_size=opts.batchSize)
