
# coding: utf-8

# In[1]:


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


# In[3]:


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
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["high"]))
        Z_train.append(np.array(train.iloc[i+pastDay-1:i+pastDay+futureDay-1]["high"]))
    return np.array(X_train), np.array(Y_train), np.array(Z_train)

def shuffle(X,Y,Z):
    np.random.seed()
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList],Z[randomList]

def splitData(X,Y,Z,rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    Z_val = Z[:int(Z.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val,Z_val

def buildModel(train_x,train_y,bs):
    model = Sequential()
    model.add(LSTM(32, input_length=train_x.shape[1],input_dim= train_x.shape[2],return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(train_x,train_y, epochs=1000, batch_size=bs, validation_split=0.1, callbacks=[callback])
    return model,[32,32,0,0,0]

def predict(model,layer,val_x,val_y,val_z,x,y,name,num):
    a=range(0,val_y.shape[0])
    val_y=val_y.reshape(-1)
    val_z=val_z.reshape(-1)
    plt.plot(a,val_y)
    b=[]
    c=[]
    co=0
    totalss = 0
    for i in range(0,val_x.shape[0]):
        temp=val_x[i]
        temp=temp.reshape(1,x,28)
        z=int(model.predict(temp, verbose=0))
        if val_y[i]>=val_z[i] and z>=val_z[i]:
            co=co+1
        if val_y[i]<val_z[i] and z<val_z[i]:
            co=co+1
        if val_y[i] >= z:
            sub = val_y[i] - z
        if val_y[i] < z:
            sub = z - val_y[i]
        b.append(z)
        c.append(sub)
    b=np.array(b)
    b=b.reshape(-1)
    plt.plot(a,b)
    acc=100*(co/val_x.shape[0])
    for i in c:
        totalss = totalss + i * i
    StandardDeviation=float((totalss / len(c))**0.5)
    AverageDeviation=sum(c)/len(c)
    plt.savefig('img/'+name+'('+str(num)+')'+str(layer[0])+'+'+str(layer[1])+'+'+str(layer[2])+'+'+str(layer[3])+'+'+str(layer[4])+'+'+'ac,'+str(acc)+"%"+'+'+'lb,'+str(x)+'+'+'bs,'+str(y)+'.jpg')
    plt.clf()
    return str(acc),StandardDeviation,AverageDeviation


# In[ ]:


lookbacks=[30,60,90,120]
batch_size=32
col_name=["1","2","3","4","5","acc","lookback","batch_size","name","StandardDeviation","AverageDeviation","num"]
place_name=["train+oil+weather+國定假日+拜拜日+高麗菜trend 雲林","train+oil+weather+國定假日+拜拜日+高麗菜trend 嘉義","train+oil+weather+國定假日+拜拜日+高麗菜trend 彰化","train+oil+weather+國定假日+拜拜日+高麗菜trend 台南","train+oil+weather+國定假日+拜拜日+高麗菜trend 高雄","train+oil+weather+國定假日+拜拜日+高麗菜trend 屏東","train+oil+weather+國定假日+拜拜日+高麗菜trend 台中","train+oil+weather+國定假日+拜拜日+高麗菜trend 苗栗","train+oil+weather+國定假日+拜拜日+高麗菜trend 桃園","train+oil+weather+國定假日+拜拜日+高麗菜trend 台北"]
for lookback in lookbacks:
    for i in place_name:
        df=pd.DataFrame(columns=col_name)
        data=[]
        for j in range(5):
            train=readData(i)
            temp=train
            train=sta(train)
            train_x1,train_y1,train_z1=buildTrain(train,lookback,1)
            train_x2,train_y2,train_z2=buildTrain(temp,lookback,1)
            train_x,train_y,train_z=train_x1,train_y2,train_z2
            train_x,train_y,train_z= shuffle(train_x,train_y,train_z)
            train_x,train_y, val_x, val_y ,val_z= splitData(train_x,train_y,train_z, 0.05)
            model,layer=buildModel(train_x,train_y,batch_size)
            pre,StandardDeviation,AverageDeviation=predict(model,layer,val_x,val_y,val_z,lookback,batch_size,i,j)
            data.append({"1":layer[0],"2":layer[1],"3":layer[2],"4":layer[3],"5":layer[4],"acc":pre,"lookback":lookback,"batch_size":batch_size,"name":i,"StandardDeviation":StandardDeviation,"AverageDeviation":AverageDeviation,"num":j})
        df=pd.concat([pd.DataFrame(data), df], ignore_index=True,sort=True)
        df.to_csv(i+'_data'+'(lookback  '+str(lookback)+')'+'(bs  '+str(batch_size)+')'+'.csv', encoding='utf_8_sig')

