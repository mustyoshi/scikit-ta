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
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import mods
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.client import random_forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
#from . import ProductionPipe
if __name__ == "__main__":
    # Training Parameters
    device="/cpu:0"
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
        #df = pd.read_csv('EURUSD_M5_UTC-5_00.csv',header=None,names=["Date","Time","Open","High","Low","Close","Volume"])
        df = pd.read_csv('amd.csv')
        origs = df[['Close']].copy()

        #df = pd.read_csv('combined.csv',header=None,names=["Open","High","Low","Close","Volume"])
        prod_pipe = mods.CreatePipeline() #Pipeline([('ft1',ft1),('ft2',ft2),('ft3',ft3)])
        
        df['Period_4_Lookforward_p1'] = (df['Close'].shift(-20).astype(float) > df['Close'].astype(float))
        df['Period_4_Lookforward_p2'] =True #(np.min([df['Low'].shift(-n) for n in range(1,4)],axis=0) > (df['Close'].astype(float)*.99)) #np.amax([df['Close'].shift(-n).astype(float) for n in range(1,4)],axis=0) > df['Close']
        #df['Period_4_Lookforward'] = (df['Period_4_Lookforward_p1'].astype(bool) == True) & (df['Period_4_Lookforward_p2'].astype(bool) == True)
        df['Period_4_Lookforward'] = df['Close'].shift(-1).astype(float)/df['Close'].astype(float)
        df = prod_pipe.transform(df)
        #to_bool.append('Period_4_Lookforward')
        #for b in to_bool:
        #    df[b] = df[b].astype(bool)
        df = df.dropna()
        df.drop(df.index[:32], inplace=True,errors="ignore")
        #df.to_csv('Hours.csv',index=False)

        #df = df.drop(to_drop,axis=1)
        df = df.astype(float)
        df.to_csv('H1.csv',index=False,float_format="%.8g")
    else:
        df = pd.read_csv('H1.csv')
    #print(sum(df['Period_4_Lookforward'].values),'total ups',sum(df['Period_4_Lookforward'].values)/len(df))
    num_features = len(df.columns)-1
    num_classes = 2 # The 10 digits
    num_trees = 200
    max_nodes = 1000
    train = df[:int(len(df)*0.9)]
    train_y = train['Period_4_Lookforward']

    test = df[len(train):]
    #closes = origs.loc[test.index].values
    test_y = test['Period_4_Lookforward']
    #dfs = splitDataFrameIntoSmaller(train)
    #training_steps = len(dfs)*100
    train.drop('Period_4_Lookforward',axis=1,inplace=True)
    test.drop('Period_4_Lookforward',axis=1,inplace=True)
    #del dfs[-1]
    #random.shuffle(dfs)
    #dfs.append(test)
    print(num_features,'inputs')
    print(len(train),'rows')
    print(len(test),'test parts')
    ind = 0
    # Network Parameters

    #closes = origs.loc[test_data.index].values





    # Calculate accuracy for 128 mnist test images

    #print("Final Testing Accuracy {:0.4f}%".format(f1_score(test_true,sess.run(y_p, feed_dict={X: test_data, Y: test_label}))))
    clf = RandomForestRegressor(n_estimators=num_trees,n_jobs=-2,max_features=10)
    clf.fit(train,train_y)
    t_out = clf.predict(train)
    print(clf.score(train,train_y))
    t2_out = clf.predict(test)
    print(clf.score(test,test_y))
    print(clf)
    last_price = 0
    gain = 1
    ind = 0
    min_gain = 1
    max_gain = 1
    # for row in test:
    #     output = clf.predict(row)[0]
    #     if(output == 1):
    #         if(last_price == 0):
    #             last_price = closes[ind]
    #         if(closes[ind] < last_price):
    #             gain = gain * (1+((last_price - closes[ind]))*20)
    #             min_gain = min(gain,min_gain)
    #             max_gain = max(gain,max_gain)
    #             last_price = 0
    #     else:
    #         if(last_price != 0):
    #             gain = gain * (1+((last_price - closes[ind]))*20)
    #             min_gain = min(gain,min_gain)
    #             max_gain = max(gain,max_gain)
    #             last_price = 0
    #     ind = ind + 1
    print(ind,"rows gives",gain)
    print(min_gain," | ",max_gain)
    #saver = tf.train.Saver()
    #saver.save(sess, "D:\\dev\\forex_17\\model.ckpt")

