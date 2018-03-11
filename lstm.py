from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd 
import numpy as np
import random
import time
import os
import datetime
from tensorflow.python.client import timeline
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import mods
#from . import ProductionPipe
if __name__ == "__main__":
    # Training Parameters
    device="/gpu:0"
    learning_rate = .0001

    batch_size = (20*5)
    display_step = 10


    timesteps = 1 # timesteps

    num_classes = 2 # MNIST total classes (0-9 digits)

    def splitDataFrameIntoSmaller(df, chunkSize = batch_size): 
        listOfDf = list()
        numberChunks = len(df) // chunkSize + 1
        for i in range(numberChunks):
            listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
        return listOfDf


    to_drop = ['Date','Time','Open','High','Low','Close','Volume',"Period_4_Lookforward_p1","Period_4_Lookforward_p2"]
    to_bool = []
    create = True
    if(create):
        #df = pd.read_csv('dhy.us.txt',names=["Date","Time","Open","High","Low","Close","Volume","Ope"])
        df = pd.read_csv('bac.us.txt')
        origs = df[['Close']].copy()
        #df = pd.read_csv('combined.csv',header=None,names=["Open","High","Low","Close","Volume"])
        prod_pipe = mods.CreatePipeline() #Pipeline([('ft1',ft1),('ft2',ft2),('ft3',ft3)])
        
        df['Period_4_Lookforward_p1'] = (df['Close'].shift(-4).astype(float) > df['Close'].astype(float))
        df['Period_4_Lookforward_p2'] =(np.min([df['Low'].shift(-n) for n in range(1,4)],axis=0) > (df['Close'].astype(float)*.99)) #np.amax([df['Close'].shift(-n).astype(float) for n in range(1,4)],axis=0) > df['Close']
        df['Period_4_Lookforward'] = (df['Period_4_Lookforward_p1'].astype(bool) == True) & (df['Period_4_Lookforward_p2'].astype(bool) == True)
        df = prod_pipe.transform(df)
        #to_bool.append('Period_4_Lookforward')
        #for b in to_bool:
        #    df[b] = df[b].astype(bool)
        df = df.dropna()
        df.drop(df.index[:32], inplace=True,errors="ignore")
        #df.to_csv('Hours.csv',index=False)

        #df = df.drop(to_drop,axis=1)
        df = df.astype(float)
        df.to_csv('Hours.csv',index=False,float_format="%.8g")
    else:
        df = pd.read_csv('Hours.csv')
    print(sum(df['Period_4_Lookforward'].values),'total ups',sum(df['Period_4_Lookforward'].values)/len(df))
    num_input = len(df.columns)-1
    num_hidden =num_input*20 # int((num_input*num_input)//2) # hidden layer num of features
    train = df[:int(len(df)*0.9)]
    test = df[len(train):]
    dfs = splitDataFrameIntoSmaller(train)
    training_steps = len(dfs)*100
    
    del dfs[-1]
    random.shuffle(dfs)
    dfs.append(test)
    print(num_input,'inputs')
    print(num_hidden,'nodes per layer')
    print(len(dfs),'batches')
    print(len(test),'test parts')
    ind = 0
    # Network Parameters

    with tf.device(device):
        # tf Graph input
        X = tf.placeholder("float", [None, timesteps, num_input])
        Y = tf.placeholder("float", [None, num_classes])
        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        def RNN(x, weights, biases):
            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, timesteps, n_input)
            # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

            # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
            x = tf.unstack(x, timesteps, 1)

            # Define a lstm cell with tensorflow
            lstm_cell1 = rnn.LSTMBlockCell(num_hidden, forget_bias=1.0)
            #lstm_cell = rnn.BasicRNNCell(num_hidden)
            #lstm_cell = rnn.PhasedLSTMCell(num_hidden)
            #lstm_cell2 = rnn.PhasedLSTMCell(num_hidden)
            lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell1, output_keep_prob=0.75)
            lstm_cell2 = rnn.LSTMBlockCell(num_hidden, forget_bias=1.0,use_peephole=True)
            lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell1, output_keep_prob=0.75)
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1,lstm_cell2]*4)  
            # Get lstm cell output
            outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)


            # Linear activation, using rnn inner loop last output
            return tf.matmul(outputs[-1], weights['out']) + biases['out']


        logits = RNN(X, weights, biases)
        with tf.device("/cpu:0"):
            prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=Y))
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        #optimizer = tf.train.FtrlOptimizer(learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        y_p = tf.argmax(prediction, 1)
        init = tf.global_variables_initializer()
    # Initialize the variables (i.e. assign their default value)
    test_len = len(dfs[-1])
    test_data = dfs[-1]
    closes = origs.loc[test_data.index].values
    test_data.reset_index(drop=True,inplace=True)
    test_true = test_data['Period_4_Lookforward'].values
    test_label = np.array([test_data['Period_4_Lookforward'] == 0,test_data['Period_4_Lookforward'] == 1]).reshape((-1,2))
    test_data = test_data.drop(['Period_4_Lookforward'],axis=1)
    test_data = test_data.as_matrix()
    test_data = test_data.reshape((-1, timesteps, num_input))


    max_fails = 10
    fails = 0
    min_improv = .00000001
    min_loss = 99
    max_f1 = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.intra_op_parallelism_threads=4
    config.inter_op_parallelism_threads=4
    with tf.Session(config=config) as sess:
        # Run the initializer
        sess.run(init)
        start_time = time.time()
        for step in range(1, training_steps+1):
            batch_x= dfs[ind]
            #print(len(batch_x),'rows')
            ind = (ind + 1)%(len(dfs)-1)
            y_true = batch_x['Period_4_Lookforward'].values

            batch_y =np.array([batch_x['Period_4_Lookforward'] == 0,batch_x['Period_4_Lookforward'] == 1]).reshape((-1,2))
            batch_x = batch_x.drop(['Period_4_Lookforward'],axis=1)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.as_matrix()
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
           
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            # if(learning_rate > .0001):
            #     learning_rate = learning_rate/10
            #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            #     train_op = optimizer.minimize(loss_op)

            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc,y_pred = sess.run([loss_op, accuracy,y_p], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                loss2, acc2,test_pred = sess.run([loss_op, accuracy,y_p], feed_dict={X: test_data,
                                                                     Y: test_label})
                test_f1 = f1_score(test_true,test_pred)
                print("Step " + "{:07d}".format(step) + ",L= " + \
                      "{:.6f}".format(loss) + ",Tr=" + \
                      "{:.3f}".format(acc) + ",Te=" + \
                      "{:.3f}".format(acc2) + ",F1(Tr)={:.3f}".format(f1_score(y_true,y_pred)) + \
                      ",F1(Te)={:.8f}".format(test_f1))
                if(loss < (min_loss-min_improv) or test_f1 > max_f1):
                    min_loss = loss
                    fails = 0
                    max_f1 = max(max_f1,test_f1)
                else:
                    fails = fails + 1
                    if(fails > max_fails):
                        print('Ended early due to failure to improve')
                        break
        duration = time.time() - start_time
        print("\nOptimization Finished in {:.4f}s ({:0.8f} per step)\n\tMax of {:.4f}".format(duration,duration/step,max_f1))

        # Calculate accuracy for 128 mnist test images

        print("Final Testing Accuracy {:0.4f}%".format(f1_score(test_true,sess.run(y_p, feed_dict={X: test_data, Y: test_label}))))
        last_price = 0
        gain = 1
        ind = 0
        min_gain = 1
        max_gain = 1
        for row in test_data:
            output = sess.run(y_p,feed_dict={X:[row]})[0]
            if(output == 1):
                if(last_price == 0):
                    last_price = closes[ind]
                if(closes[ind] < last_price):
                    gain = gain * (1+((last_price - closes[ind]))*20)
                    min_gain = min(gain,min_gain)
                    max_gain = max(gain,max_gain)
                    last_price = 0
            else:
                if(last_price != 0):
                    gain = gain * (1+((last_price - closes[ind]))*20)
                    min_gain = min(gain,min_gain)
                    max_gain = max(gain,max_gain)
                    last_price = 0
            ind = ind + 1
        print(ind,"rows gives",gain)
        print(min_gain," | ",max_gain)
        #saver = tf.train.Saver()
        #saver.save(sess, "D:\\dev\\forex_17\\model.ckpt")

