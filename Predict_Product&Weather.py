
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


# In[5]:


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
result=pd.concat([train,weather],axis=1)


# In[6]:


print(result)

