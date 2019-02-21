
# coding: utf-8

# In[17]:


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


# In[ ]:


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
for i in range(0,train.shape[0]):
    for j in range(0,c):
        if all([train["year"][i]==weather["year"][j] , train["month"][i]==weather["month"][j] , train["day"][i]==weather["day"][j]]):
            a=a+1
            b=b+1
            #weather=weather.drop(weather.index[:b])
            break
        else:
            b=b+1
print(a)

