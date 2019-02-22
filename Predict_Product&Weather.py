
# coding: utf-8

# In[32]:


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


# In[33]:


train=pd.read_csv(r"C:\Users\admin\Desktop\Project\data\台北一\蔬果\奇異果-進口.csv")
weather=pd.read_csv(r"C:\Users\admin\Desktop\Project\data\台北一\氣象資料\氣象資料.csv")
weather=weather.drop(["Unnamed: 0"], axis=1)
weather=weather.drop(["一小時最大降水量(mm)"], axis=1)
weather=weather.drop(["10分鐘最大降水起始時間(LST)"], axis=1)
weather=weather.drop(["10分鐘最大降水量(mm)"], axis=1)
weather=weather.drop([" 一小時最大降水量(mm)"], axis=1)
weather=weather.drop(["一小時最大降水量起始時間(LST)"], axis=1)
weather=weather.drop(["日最高紫外線指數時間(LST)"], axis=1)
weather=weather.drop(["最低氣溫時間(LST)"], axis=1)
weather=weather.drop(["最大陣風風速時間(LST)"], axis=1)
weather=weather.drop(["最小相對溼度時間(LST)"], axis=1)
weather=weather.drop(["最高氣溫時間(LST)"], axis=1)
weather=weather.drop(["測站最低氣壓時間(LST)"], axis=1)
weather=weather.drop(["測站最高氣壓時間(LST)"], axis=1)
weather=weather.drop(["降水量(mm)"], axis=1)
weather=weather.drop(["日最高紫外線指數"], axis=1)
weather=weather.drop(["A型蒸發量(mm)"], axis=1)

a=0
b=0
c=weather.shape[0]
d=0
for i in range(0,train.shape[0]):
    for j in range(a,c-d):
        if all([train["year"][i]==weather["year"][j] , train["month"][i]==weather["month"][j] , train["day"][i]==weather["day"][j]]):
            weather=weather.drop(weather.index[a:b+a])
            weather.reset_index(inplace=True)
            weather=weather.drop(["index"], axis=1)
            a=a+1
            b=0
            break
        else:
            b=b+1
            d=d+1
weather=weather.drop(weather.index[train.shape[0]:])
weather.reset_index(inplace=True)
weather=weather.drop(["index"], axis=1)
train=pd.concat([train,weather],axis=1)


# In[34]:


print(train)
train.to_csv("test1.csv", encoding='utf_8_sig')


# In[35]:


def buildTrain(train, pastDay=1, futureDay=1):
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


# In[36]:


train=train.drop(["Unnamed: 0"], axis=1)
train=train.drop(["crop_name"], axis=1)
train=train.drop(["crop_num"], axis=1)
train=train.drop(["market_name"], axis=1)
train=train.drop(["market_num"], axis=1)
train=train.convert_objects(convert_numeric=True)
temp=train


# In[37]:


print(train)


# In[38]:


train= train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
train_x1,train_y1,train_z1=buildTrain(train)
train_x2,train_y2,train_z2=buildTrain(temp)
train_x,train_y,train_z=train_x1,train_y2,train_z2
train_x,train_y,train_z= shuffle(train_x,train_y,train_z)
train_x,train_y, val_x, val_y ,val_z= splitData(train_x,train_y,train_z, 0.1)
print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)
print(val_z.shape)


# In[41]:


model = Sequential()
model.add(LSTM(100, input_length=train_x.shape[1],input_dim= train_x.shape[2],return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
model.fit(train_x,train_y, epochs=300, batch_size=8, validation_split=0.1, callbacks=[callback])


# In[42]:


a=range(0,val_y.shape[0])
val_y=val_y.reshape(-1)
val_z=val_z.reshape(-1)
plt.plot(a,val_y)
b=[]
co=0
for i in range(0,val_x.shape[0]):
    temp=val_x[i]
    temp=temp.reshape(1,1,32)
    z=model.predict(temp, verbose=0)
    if val_y[i]>=val_z[i] and z>=val_z[i]:
        co=co+1
    if val_y[i]<val_z[i] and z<val_z[i]:
        co=co+1
    b.append(z)
b=np.array(b)
b=b.reshape(-1)
plt.plot(a,b)
acc=100*(co/val_x.shape[0])
print("accuracy:"+str(acc)+"%")
plt.show()

