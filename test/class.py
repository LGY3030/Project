
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[4]:


def readData(name):
    train = pd.read_csv(name+".csv")
    train=train.drop(["date"], axis=1)
    train=train.drop(["Unnamed: 0"], axis=1)
    return train

def sta(train):
    train=train.convert_objects(convert_numeric=True)
    train= train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return train

def buildTrain(train, pastDay, futureDay):
    X_train, Y_train ,Z_train= [], [],[]
    for i in range(train.shape[0]-futureDay-pastDay+1):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        a=np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["high"])
        b=np.array(train.iloc[i+pastDay-1:i+pastDay+futureDay-1]["high"])
        if a>=b:
            Y_train.append(1)
        else:
            Y_train.append(0)                                                       
    return np.array(X_train), np.array(Y_train)

def shuffle(X,Y):
    np.random.seed()
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

def splitData(X,Y,rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val

def buildModel(train_x,train_y,bs):
    model = Sequential()
    model.add(LSTM(32, input_length=train_x.shape[1],input_dim= train_x.shape[2],return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(train_x,train_y, epochs=1000, batch_size=bs, validation_split=0.1, callbacks=[callback])
    return model,[32,32,0,0,0]


# In[ ]:


lookback=7
batch_size=32
col_name=["1","2","3","4","5","acc","lookback","batch_size","name","num"]
place_name=["train+oil+weather+國定假日+拜拜日 雲林","train+oil+weather+國定假日+拜拜日 嘉義","train+oil+weather+國定假日+拜拜日 彰化","train+oil+weather+國定假日+拜拜日 台南","train+oil+weather+國定假日+拜拜日 高雄","train+oil+weather+國定假日+拜拜日 屏東","train+oil+weather+國定假日+拜拜日 台中","train+oil+weather+國定假日+拜拜日 苗栗","train+oil+weather+國定假日+拜拜日 桃園","train+oil+weather+國定假日+拜拜日 台北"]
for i in place_name:
    df=pd.DataFrame(columns=col_name)
    data=[]
    for j in range(30):
        train=readData(i)
        temp=train
        train=sta(train)
        train_x1,train_y1=buildTrain(train,lookback,1)
        train_x2,train_y2=buildTrain(temp,lookback,1)
        train_x,train_y=train_x1,train_y2
        train_x,train_y= shuffle(train_x,train_y)
        train_x,train_y, val_x, val_y= splitData(train_x,train_y, 0.05)
        model,layer=buildModel(train_x,train_y,batch_size)
        pre=model.evaluate(val_x, val_y)
        data.append({"1":layer[0],"2":layer[1],"3":layer[2],"4":layer[3],"5":layer[4],"acc":pre,"lookback":lookback,"batch_size":batch_size,"name":i,"num":j})
        print(pre)
    df=pd.concat([pd.DataFrame(data), df], ignore_index=True,sort=True)
    df.to_csv('class'+i+'_data'+'(lookback  '+str(lookback)+')'+'(bs  '+str(batch_size)+')'+'.csv', encoding='utf_8_sig')

