from django.shortcuts import render
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from urllib.request import urlopen
from bs4 import BeautifulSoup
import json
import requests
import urllib
import time
import os

posts = [
    {
        'author': 'Jimmy',
        'title': 'first',
        'content': 'first content',
        'date': 'May 23,2018',
    },
    {
        'author': 'Amy',
        'title': 'second',
        'content': 'second content',
        'date': 'May 24,2018',
    }
]


def home(request):
    context = {'posts': posts}
    return render(request, 'main/home.html', context)


def price(request):
    data = crawler()
    train_x, train_y, val_x, val_y, val_z = manageData(data)
    model = buildModel(train_x, train_y)
    result, average, ss = getResult(model, val_x, val_y, val_z)
    context = {'title': 'price', 'result': result, 'average': average, 'ss': ss}
    return render(request, 'main/price.html', context)


def volume(request):
    return render(request, 'main/volume.html', {'title': 'volume'})


def crawler():
    place = "台北一"
    crop_num = "G49"
    crop_name = "奇異果-進口"
    col_name = ["date", "crop_num", "crop_name", "market_num", "market_name", "high", "medium", "low", "mean", "volume"]
    df = pd.DataFrame(columns=col_name)
    skip_num = 0
    for j in range(0, 1000):
        url = "https://data.coa.gov.tw/Service/OpenData/FromM/FarmTransData.aspx?$top=1000&$skip=" + str(skip_num) + "&Crop=" + urllib.parse.quote(crop_name) + "&StartDate=101.01.01&EndDate=108.02.08&Market=" + urllib.parse.quote(place)
        get = json.loads(urlopen(url).read())
        data = []
        for i in range(0, len(get)):
            if get[i]['作物代號'] == crop_num:
                Date = get[i]['交易日期'].split('.')
                year, month, date = Date[0], Date[1], Date[2]
                year = str(int(year) + 1911)
                get[i]['交易日期'] = year + '/' + month + '/' + date
                data.append({"date": get[i]['交易日期'], "crop_num": get[i]['作物代號'], "crop_name": get[i]['作物名稱'], "market_num": get[i]['市場代號'], "market_name": get[i]['市場名稱'], "high": get[i]['上價'], "medium": get[i]['中價'], "low": get[i]['下價'], "mean": get[i]['平均價'], "volume": get[i]['交易量']})
        data.reverse()
        df = pd.concat([pd.DataFrame(data), df], ignore_index=True, sort=True)
        if len(get) < 1000:
            break
        skip_num = skip_num + 1000
        time.sleep(2000.0 / 1000.0)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df = df.drop(["date"], axis=1)

    return df


def buildTrain(train, pastDay=1, futureDay=1):
    X_train, Y_train, Z_train = [], [], []
    for i in range(train.shape[0] - futureDay - pastDay + 1):
        X_train.append(np.array(train.iloc[i:i + pastDay]))
        Y_train.append(np.array(train.iloc[i + pastDay:i + pastDay + futureDay]["high"]))
        Z_train.append(np.array(train.iloc[i + pastDay - 1:i + pastDay + futureDay - 1]["high"]))
    return np.array(X_train), np.array(Y_train), np.array(Z_train)


def shuffle(X, Y, Z):
    np.random.seed()
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList], Z[randomList]


def splitData(X, Y, Z, rate):
    X_train = X[int(X.shape[0] * rate):]
    Y_train = Y[int(Y.shape[0] * rate):]
    X_val = X[:int(X.shape[0] * rate)]
    Y_val = Y[:int(Y.shape[0] * rate)]
    Z_val = Z[:int(Z.shape[0] * rate)]
    return X_train, Y_train, X_val, Y_val, Z_val


def manageData(train):
    #train = train.drop(["Unnamed: 0"], axis=1)
    train = train.drop(["crop_name"], axis=1)
    train = train.drop(["crop_num"], axis=1)
    train = train.drop(["market_name"], axis=1)
    train = train.drop(["market_num"], axis=1)
    temp = train
    train = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    train_x1, train_y1, train_z1 = buildTrain(train)
    train_x2, train_y2, train_z2 = buildTrain(temp)
    train_x, train_y, train_z = train_x1, train_y2, train_z2
    train_x, train_y, train_z = shuffle(train_x, train_y, train_z)
    train_x, train_y, val_x, val_y, val_z = splitData(train_x, train_y, train_z, 0.05)
    return train_x, train_y, val_x, val_y, val_z


def buildModel(train_x, train_y):
    model = Sequential()
    #model.add(LSTM(200, input_length=train_x.shape[1],input_dim= train_x.shape[2],return_sequences=True))
    # model.add(Dropout(0.6))
    model.add(LSTM(10, input_length=train_x.shape[1], input_dim=train_x.shape[2]))
    # model.add(Dropout(0.6))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(train_x, train_y, epochs=300, batch_size=32, validation_split=0.1, callbacks=[callback])
    return model


def getResult(model, val_x, val_y, val_z):
    a = range(0, val_y.shape[0])
    val_y = val_y.reshape(-1)
    val_z = val_z.reshape(-1)
    # plt.plot(a,val_y)
    b = []
    c = []
    co = 0
    coo = 0
    for i in range(0, val_x.shape[0]):
        temp = val_x[i]
        temp = temp.reshape(1, 1, 9)
        z = model.predict(temp, verbose=0)
        if val_y[i] >= val_z[i] and z >= val_z[i]:
            co = co + 1
        if val_y[i] < val_z[i] and z < val_z[i]:
            co = co + 1
        if val_y[i] >= z:
            sub = val_y[i] - z
        if val_y[i] < z:
            sub = z - val_y[i]
        '''
        sub = val_y[i] - val_z[i]
        sub_a = 0.5 * sub + val_z[i]
        sub_b = 1.5 * sub + val_z[i]
        sub2 = val_z[i] - val_y[i]
        sub2_a = val_z[i] - 0.5 * sub2
        sub2_b = val_z[i] - 1.5 * sub2
        if val_y[i] >= val_z[i] and z >= val_z[i]:
            if z >= sub_a and z <= sub_b:
                coo = coo + 1
        if val_y[i] < val_z[i] and z < val_z[i]:
            if z <= sub2_a and z >= sub2_b:
                coo = coo + 1
        '''
        b.append(z)
        c.append(sub)
    b = np.array(b)
    b = b.reshape(-1)
    # plt.plot(a,b)
    acc = 100 * (co / val_x.shape[0])
    # print("accuracy:"+str(acc)+"%")
    #acc2 = 100 * (coo / val_x.shape[0])
    # print("accuracy(50%):"+str(acc2)+"%")
    # plt.show()
    total = 0
    totalss = 0
    for i in c:
        total = total + i
        totalss = totalss + i * i
    average = float(total / len(c))
    ss = float((totalss / len(c))**0.5)
    return acc, average, ss
