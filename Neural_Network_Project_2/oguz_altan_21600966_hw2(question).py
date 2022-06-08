# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:06:42 2019

@author: Oguz
"""


#*************************************************************Q1 PYTHON CODE BELOW*********************************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.manifold import TSNE
import h5py
import os
import sys

def reader(batch_size):

    data1 = {}

    with h5py.File('assign2_data2.mat', 'r') as file:

        testd = file['testd'][:,:].T
        testx = file['testx'][:,:].T
        traind = file['traind'][:,:].T
        trainx = file['trainx'][:].T
        vald = file['vald'][:].T
        valx = file['valx'][:].T
        words = []
        
        for i in range(250):
            L = file[file['words'][i].item()][:].reshape(-1)
            string = ''.join(map(chr,L))
            words.append(string)

        data1['words'] = words
        data1['valx'] = valx
        data1['vald'] = vald
        data1['trainx'] = trainx
        data1['traind'] = traind
        data1['testx'] = testx
        data1['testd'] = testd

        #number of batches
        num_batches = np.int(data1['trainx'].shape[1] / batch_size )

        #subtract one for array index
        X_train = data1['trainx'][:,:num_batches * batch_size].reshape(3, batch_size, num_batches ) -1
        y_train = data1['traind'][:,:num_batches * batch_size].reshape(1, batch_size, num_batches ) - 1
        X_val = data1['valx'] -1
        y_val = data1['vald'].reshape(1,-1) -1
        X_test = data1['testx'] - 1
        y_test = data1['testd'].reshape(1,-1) -1

        words = np.array(data1['words']).reshape(1,-1)

    return X_train, y_train, X_val, y_val, X_test, y_test, words

def weight paramt_initializer( w_size , b_size  , loc = 0.0, scale = 0.01):

    np.random.seed(0)
    #initialize weights
    W = np.random.normal(size = w_size, loc = loc, scale = scale ).astype(np.float64)
    #initialize biases
    b = np.random.normal(size = b_size, loc = loc , scale = scale ).astype(np.float64)
    return W,  b

def random_shuffle(X, y):

    #random shuffle data
    batch_idx = np.random.permutation(y.shape[1])
    X,y  = X[:,batch_idx, :], y[:, batch_idx, :]
    sec_idx = np.random.permutation(y.shape[2])
    return  X[:,:, sec_idx], y[:, :, sec_idx]

def sigmoid_activation(z, derive = False ):

    if( derive ):
        #derivative of sigmoid
        return z * ( 1  - z )
    #sigmoid function
    return 1 / ( 1 + np.exp( -z ))
def cross_entropy(y, y_):

    return -np.sum(np.sum(y* np.log(y_+np.exp(-32))))/y_.shape[1]


def backpropagation_q1(X, y, momentum_rate, lr_rate, model):

    #find batch size to find mean
    batch_size = model['pred'].shape[1]

    #number of hidden unit 1
    num_hidden_unit1 = model['embed'].shape[1]
    #number of hidden unit in second hidden layer
    num_hidden_unit2 = model['w1'].shape[1]

    #error delta
    dz2 = model['pred'] - y

    #gradient of second hidden weights and biases
    w2_grad = np.dot(model['a1'],dz2.T)
    b2_grad = np.sum(dz2, axis=1)

    #local gradient delta
    dz1 = np.dot(model['w2'], dz2)* sigmoid_activation(model['a1'], derive = True)

    #gradient of first hidden weights and biases
    w1_grad = np.dot(model['a0'],dz1.T)
    b1_grad = np.sum(dz1, axis=1).reshape(num_hidden_unit2,-1)

    #embed delta
    dz0 = np.dot(model['w1'],dz1)

    #empty matrix for embed grads
    x_grad = np.zeros((250, num_hidden_unit1))

    #embed grad
    for i in range(3):
        x_matrix =  np.eye(250)[:,X[i,:]]
        dz0_channel = dz0[i * num_hidden_unit1 : (i+1) * num_hidden_unit1, :]
        x_grad = x_grad + np.dot(x_matrix,dz0_channel.T)


    #update deltas
    model['dx'] = momentum_rate * model['dx'] + x_grad / batch_size
    model['dw1'] = momentum_rate * model['dw1'] + w1_grad / batch_size
    model['db1'] = momentum_rate * model['db1'] + b1_grad / batch_size
    model['dw2'] = momentum_rate * model['dw2'] + w2_grad / batch_size
    model['db2'] = momentum_rate * model['db2'] + b2_grad.reshape( model['db2'].shape[0],-1) / batch_size

    #update weights and biases
    model['embed']=  model['embed'] - lr_rate * model['dx']
    model['w1'] = model['w1'] - lr_rate * model['dw1']
    model['b1'] = model['b1'] - lr_rate * model['db1']
    model['w2'] = model['w2'] - lr_rate * model['dw2']
    model['b2'] = model['b2'] - lr_rate * model['db2']

def  predict(pred,y ):

    #true values
    c = np.argmax(pred,axis=0) - y
    return len(c[c==0]) / c.shape[1]

def q1_forward(X, model):

    num_hidden_unit1 = model['embed'].shape[1]


    #flat input
    X_flat = np.reshape(X, (1,-1), order="F").ravel()

    #flat embed
    embed_flat = model['embed'][X_flat].T

    #triagram embed values
    a0 = np.reshape(embed_flat, (num_hidden_unit1 * 3,-1), order="F")

    #forwards propagation
    z1 = np.dot(model['w1'].T,a0) + model['b1']
    a1 = sigmoid_activation(z1)
    z2 = np.dot(model['w2'].T,a1) + model['b2']
    pred = np.exp(z2 -np.amax(z2,axis=0))


    pred = pred / np.sum(pred,axis=0)

    return a0, a1, pred


def q1( lr_rate = 0.15, momentum_rate = 0.85, num_hidden_unit1 = 8, num_hidden_unit2 = 64, batch_size = 200, epochs = 50 ):

    #read all data
    X_train, y_train, X_dev, y_dev, X_test, y_test, words = read_data(batch_size)

    #create model to hold everything related to network
    model = {}
    model['testx'] = X_test
    model['testy'] = y_test
    model['words'] = words

    num_batches = X_train.shape[2]

    # initialize weights and biases
    model['embed'] = normal_weight_and_bias_initializer((250,num_hidden_unit1), (num_hidden_unit1, 1))[0]
    model['w1'], model['b1'] = normal_weight_and_bias_initializer((3 * num_hidden_unit1, num_hidden_unit2), (num_hidden_unit2, 1))
    model['w2'], model['b2'] = normal_weight_and_bias_initializer((num_hidden_unit2, 250), (250, 1))

    #initialize deltas for momentum
    model['dx'] = np.zeros((250, num_hidden_unit1))
    model['dw1'] = np.zeros((3 * num_hidden_unit1, num_hidden_unit2))
    model['dw2'] = np.zeros((num_hidden_unit2, 250))
    model['db1'] = np.zeros((num_hidden_unit2, 1))
    model['db2'] = np.zeros((250, 1))

    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        print("Epoch", epoch+1)
        train_batch_loss = []
        train_batch_acc = []

        #random shuffle get randomize result
        X_train, y_train = random_shuffle(X_train, y_train)
        for batch_no in range(num_batches):
            X_train_batch = X_train[:,:,batch_no]
            y_train_batch = y_train[:,:,batch_no]

            #forward propagation
            model['a0'], model['a1'], model['pred'] = q1_forward(X_train_batch, model )

            #create matrix with targets adjust size with predictions shape
            y_matrix =  np.eye(250)[:,y_train_batch.ravel()]

            #finding loss
            model['loss'] = cross_entropy(y_matrix, model['pred'])

            #hold loss
            train_batch_loss.append(model['loss'])
            train_batch_acc.append(predict(model['pred'], y_train_batch))
            #backpropagation
            backpropagation_q1(X_train_batch, y_matrix, momentum_rate, lr_rate, model)


        train_loss.append(np.mean(train_batch_loss))
        print("Train loss      :", np.mean(train_batch_loss), "------Train acc : ", np.mean(train_batch_acc))
        #find validation loss
        model['a0'], model['a1'],model['pred']= q1_forward(X_dev,model)


        y_dev_matrix =  np.eye(250)[:, y_dev.ravel()]

        model['loss'] = cross_entropy(y_dev_matrix, model['pred'] )
        val_loss.append(model['loss'])
        if( epoch == 0 ):
            print("Validation loss :", model['loss'],  '-----Validation accuracy : {}'.format(predict(model['pred'], y_dev)))
            model['val_loss'] = model['loss']

        elif( np.abs(model['val_loss'] - model['loss']) <= 0.0025 ):
            print("Validation loss :", model['loss'],  '-----Validation accuracy : {}'.format(predict(model['pred'], y_dev)))
            print('\nEarly stop due to insufficient reduction of validation loss\n')
            break
        else :
            print("Validation loss :", model['loss'],  '-----Validation accuracy : {}'.format(predict(model['pred'], y_dev)))
            model['val_loss'] = model['loss']



    if epoch == (epochs - 1):
        print("\nTraining Completed\n")


    print("\nTest Results\n")

    model['a0'], model['a1'], model['pred'] = q1_forward(X_test, model)
    #find test loss
    y_test_matrix =  np.eye(250)[:, y_test.ravel()]
    model['loss'] = cross_entropy(y_test_matrix, model['pred'])
    print("Test loss : ", model['loss'] ,'-----Test accuracy : {}'.format(predict(model['pred'], y_test)))

    plt.plot(range(np.array(val_loss).shape[0]), np.array(val_loss)[:], label = 'Validation Loss')
    plt.plot(range(np.array(train_loss).shape[0]), np.array(train_loss)[:], label = 'Train Loss')
    plt.rcParams['figure.figsize'] = (10,10)
    plt.title('(D,P) = [{},{}]'.format(num_hidden_unit1, num_hidden_unit2))
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    return model, np.array(train_loss), np.array(val_loss)


def plot_words(model):

    print('Word Plot is Loading...')
    #TSNE transforms
    points = TSNE().fit_transform(model['embed'])
    texts = []
    plt.rcParams['font.weight'] = 'bold'
    #plt.rcParams['figure.figsize'] = (20, 30)
    for i in range(points.shape[0]):
        x = points[i,0]
        y = points[i,1]
        plt.plot(x, y, 'bo')
        plt.text(x * (1 + 0.01), y * (1 + 0.01) , str(model['words'].T[i][0]), fontsize=12)
        #texts.append(plt.text(x * (1 + 0.01), y * (1 + 0.01) , str(model['words'].T[i][0]), fontsize=17))
    #adjust_text(texts)

    #plt.savefig('textscatter.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def close_words(model):

    #get same values
    np.random.seed(19)

    #select samples
    random_samples = np.random.randint(1,model['pred'].shape[1], size=(5))

    #get prediction values
    random_samplesx = model['pred'][:,random_samples].T
    #sort to find 10 max
    sorted_values = np.argsort(random_samplesx, axis=1)
    #find realted words
    words_found = sorted_values[:,-10:][:,::-1]

    #dimension reduction
    words_related = np.squeeze(model['words'][:,words_found])

    #tragrams
    triagrams = model['testx'][:,random_samples].T
    #triagram words
    triagram_words = np.squeeze(model['words'][:,triagrams])
    #one string triagram
    triagram_words = np.array([ i[0] +" "+ i[1] +" "+ i[2] for i in triagram_words]).reshape(-1,1)
    #word table
    word_table = np.hstack([triagram_words, words_related])
    #print
    for i in range(len(word_table)):
        string = str(word_table[i,0])+ ' : '
        for j in range(1, len(word_table[i,:])):
            string += str(word_table[i,j]) + '-->'
        print('\n'+string)

def q1_helper():
    model, train_loss, val_loss = q1(epochs=50, num_hidden_unit1=16, num_hidden_unit2=128)
    plot_words(model)
    close_words(model)

question = sys.argv[1]

def oguz_altan_21600966_hw2(question):
    if question == '3' :
        print (question)
        q1_helper()

oguz_altan_21600966_hw2(question)