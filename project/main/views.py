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

def Information(request):
    if request.method == "POST":
        Crop_market = request.POST['Crop market']
        Crop_name = request.POST['Crop name']
        data=pd.read_csv(r"C:\Users\admin\Desktop\train\data\蔬果\\"+Crop_market+"\\"+Crop_name+".csv")
        data=data.drop(["Unnamed: 0"], axis=1)
        data=data.drop(["day"], axis=1)
        data=data.drop(["month"], axis=1)
        data=data.drop(["year"], axis=1)
        data=data.drop(["dayofweek"], axis=1)
        html_table = data.to_html(index=False)
    return render(request, 'main/Information.html',locals())

def price(request):
    if request.method == "POST":
        try:
            Crop_market = request.POST['Crop market']
            Crop_name = request.POST['Crop name']
            Crop_num = request.POST['Crop num']
            W_station_name = request.POST['Wstation name']
            W_station_num = request.POST['Wstation num']
            Today_date = request.POST['Today date']
            data ,valid= crawler(Crop_market, Crop_name, Crop_num,W_station_name,W_station_num,Today_date,"2012-01-01")
            train_x, train_y,train_w, val_x, val_y, val_z ,mul= manageData(data,'high',7,1,"2012-01-01",valid,0)
            model = buildModel(train_x, train_y,1)
            result, average, ss, a, b, c = getResult(model,train_w, val_x, val_y, val_z,mul,7,0,0,0)
            title = 'Original price & Price predicted by model'
            accuracy = 'Accuracy of prediction: ' + str(result) + ' %'
            average = 'Average deviation of prediction: ' + str(average) + ' TWD'
            sd = 'Standard deviation of prediction: ' + str(ss)
            context = {'title': title, 'accuracy': accuracy, 'average': average, 'sd': sd, 'a': a, 'b': b, 'c': c}
        except:
            context = {"wrong": wrong}
    return render(request, "main/price.html", locals())


def volume(request):
    if request.method == "POST":
        try:
            Crop_market = request.POST['Crop market']
            Crop_name = request.POST['Crop name']
            Crop_num = request.POST['Crop num']
            W_station_name = request.POST['Wstation name']
            W_station_num = request.POST['Wstation num']
            Today_date = request.POST['Today date']
            data ,valid= crawler(Crop_market, Crop_name, Crop_num,W_station_name,W_station_num,Today_date,"2012-01-01")
            train_x, train_y,train_w, val_x, val_y, val_z ,mul= manageData(data,'volume',7,"2012-01-01",valid,0)
            model = buildModel(train_x, train_y,1)
            result, average, ss, a, b, c = getResult(model,train_w, val_x, val_y, val_z,mul,7,0,0,0)
            title = 'Original volume & Volume predicted by model'
            accuracy = 'Accuracy of prediction: ' + str(result) + ' %'
            average = 'Average deviation of prediction: ' + str(average) + ' volume'
            sd = 'Standard deviation of prediction: ' + str(ss)
            context = {'title': title, 'accuracy': accuracy, 'average': average, 'sd': sd, 'a': a, 'b': b, 'c': c}
        except:
            context = {"wrong": wrong}
    return render(request, "main/volume.html", locals())
def price_trend(request):
    if request.method == "POST":
        try:
            Crop_market = request.POST['Crop market']
            Crop_name = request.POST['Crop name']
            Crop_num = request.POST['Crop num']
            W_station_name = request.POST['Wstation name']
            W_station_num = request.POST['Wstation num']
            Today_date = request.POST['Today date']
            Validation_date = request.POST['Validation date']
            Predict_days = int(request.POST['Predict days'])
            data,valid= crawler(Crop_market, Crop_name, Crop_num,W_station_name,W_station_num,Today_date,Validation_date)
            train_x, train_y,train_w, val_x, val_y, val_z ,mul,valid_x,valid_y= manageData(data,'high',Predict_days,Predict_days,Validation_date,valid,1)
            model = buildModel(train_x, train_y,Predict_days)
            a, b,c= getResult(model,train_w, val_x, val_y, val_z,mul,Predict_days,valid_x,valid_y,1)
            title1 = 'Show the model perfoemance - Predict price and original price (60 days from validation date)'
            title2 = 'Price trend (in 60 days)'
            context = {'title1': title1,'title2': title2, 'a': a, 'b': b,'c':c,'d':list(valid_y)}
        except:
            context = {"wrong": wrong}
    return render(request, "main/price_trend.html", locals())

def price_trend_compare(request):
    if request.method == "POST":
        try:
            Crop_market_1= request.POST['Crop market 1']
            Crop_market_2= request.POST['Crop market 2']
            Crop_name = request.POST['Crop name']
            Crop_num = request.POST['Crop num']
            W_station_name_1 = request.POST['Wstation name 1']
            W_station_num_1 = request.POST['Wstation num 1']
            W_station_name_2 = request.POST['Wstation name 2']
            W_station_num_2 = request.POST['Wstation num 2']
            Today_date = request.POST['Today date']
            Validation_date = request.POST['Validation date']
            Predict_days = int(request.POST['Predict days'])
            data,valid = crawler(Crop_market_1, Crop_name, Crop_num,W_station_name_1,W_station_num_1,Today_date,Validation_date)
            train_x, train_y,train_w, val_x, val_y, val_z ,mul,valid_x,valid_y1= manageData(data,'high',Predict_days,Predict_days,Validation_date,valid,1)
            model = buildModel(train_x, train_y,Predict_days)
            a, b,c= getResult(model,train_w, val_x, val_y, val_z,mul,Predict_days,valid_x,valid_y1,1)
            data ,valid= crawler(Crop_market_2, Crop_name, Crop_num,W_station_name_2,W_station_num_2,Today_date,Validation_date)
            train_x, train_y,train_w, val_x, val_y, val_z ,mul,valid_x,valid_y2= manageData(data,'high',Predict_days,Predict_days,Validation_date,valid,1)
            model = buildModel(train_x, train_y,Predict_days)
            a, d,e= getResult(model,train_w, val_x, val_y, val_z,mul,Predict_days,valid_x,valid_y2,1)
            title1 = 'Show the model perfoemance - Predict price and original price (30 days from validation date)'
            title2 = 'Price trend (in 30 days)'
            context = {'title1': title1,'title2': title2, 'a': a, 'b': b,'c':c,'d':d,'e':e,'f':list(valid_y1),'g':list(valid_y2)}
        except:
            context = {"wrong": wrong}
    return render(request, "main/price_trend_compare.html", locals())

def crawler(Crop_market, Crop_name, Crop_num,W_station_name,W_station_num,Today_date,Validation_date):
    place=Crop_market
    crop_num=Crop_num
    crop_name=Crop_name
    col_name=["date","crop_num","crop_name","market_num","market_name","high","medium","low","mean","volume"]
    crop=pd.DataFrame(columns=col_name)
    skip_num=0

    for j in range(0,1000):
        url="https://data.coa.gov.tw/Service/OpenData/FromM/FarmTransData.aspx?$top=1000&$skip="+str(skip_num)+"&Crop="+urllib.parse.quote(crop_name)+"&StartDate=101.01.01&EndDate=108.03.29&Market="+urllib.parse.quote(place)
        get = json.loads(urlopen(url).read().decode('utf-8'))
        data = []
        for i in range(0,len(get)):
            temp_str=get[0]['交易日期']
            Date=get[i]['交易日期'].split('.')
            year,month,date=Date[0],Date[1],Date[2]
            year=str(int(year)+1911)
            get[i]['交易日期']=year+'/'+month+'/'+date
            if get[i]['作物代號']==crop_num:
                data.append({"date":get[i]['交易日期'],"crop_num":get[i]['作物代號'],"crop_name":get[i]['作物名稱'],"market_num":get[i]['市場代號'],"market_name":get[i]['市場名稱'],"high":get[i]['上價'],"medium":get[i]['中價'],"low":get[i]['下價'],"mean":get[i]['平均價'],"volume":get[i]['交易量']})
            else:
                data.append({"date":get[i]['交易日期'],"crop_num":get[i]['作物代號'],"crop_name":get[i]['作物名稱'],"market_num":get[i]['市場代號'],"market_name":get[i]['市場名稱'],"high":0,"medium":0,"low":0,"mean":0,"volume":0})
        data.reverse()
        crop=pd.concat([pd.DataFrame(data), crop], ignore_index=True,sort=True)
        if len(get)<1000:
            break
        skip_num=skip_num+1000
        time.sleep(2000.0/1000.0)
    crop["date"] = pd.to_datetime(crop["date"])

    place_name=W_station_name
    place_num=W_station_num
    today_date=Today_date
    date=[]
    for year in ['2012','2013','2014','2015','2016','2017','2018']:
        for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
            date.append('-'.join([year,month]))
    date.append('2019-01')
    date.append('2019-02')
    date.append('2019-03')
    date.append('2019-04')
    flag=0
    name=urllib.parse.quote(urllib.parse.quote(place_name))
    for dd in date:
        url="http://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain"+"&station="+place_num+"&stname="+name+"&datepicker="+str(dd)
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text)
        trs = soup.findAll('tr')
        ths = trs[2].findAll('th')
        day=1
        data=[]
        fdata=[]
        if flag==0:
            title=[]
            for th in ths:
                title.append(th.text)
            title.pop(0)
            title.append("year")
            title.append("month")
            title.append("day")
            title.append("date")
            weather=pd.DataFrame(columns=title)
            flag=1
        for tr in trs[4:]:
            t=tr.findAll('td')
            for x in t:
                data.append(x.text.strip())
            sp=dd.split('-')
            data.pop(0)
            data.append(sp[0])
            data.append(sp[1])
            data.append(str(day))
            data.append(sp[0]+'/'+sp[1]+'/'+str(day))
            fdata.append({"測站氣壓(hPa)":data[0],"海平面氣壓(hPa)":data[1],"測站最高氣壓(hPa)":data[2],"測站最高氣壓時間(LST)":data[3],"測站最低氣壓(hPa)":data[4],"測站最低氣壓時間(LST)":data[5],"氣溫(℃)":data[6],"最高氣溫(℃)":data[7],"最高氣溫時間(LST)":data[8],"最低氣溫(℃)":data[9],"最低氣溫時間(LST)":data[10],"露點溫度(℃)":data[11],"相對溼度(%)":data[12],"最小相對溼度(%)":data[13],"最小相對溼度時間(LST)":data[14],"風速(m/s)":data[15],"風向(360degree)":data[16],"最大陣風(m/s)":data[17],"最大陣風風向(360degree)":data[18],"最大陣風風速時間(LST)":data[19],"降水量(mm)":data[20],"降水時數(hr)":data[21],"10分鐘最大降水量(mm)":data[22],"10分鐘最大降水起始時間(LST)":data[23]," 一小時最大降水量(mm)":data[24],"一小時最大降水量起始時間(LST)":data[25],"日照時數(hr)":data[26],"日照率(%)":data[27],"全天空日射量(MJ/㎡)":data[28],"能見度(km)":data[29],"A型蒸發量(mm)":data[30],"日最高紫外線指數":data[31],"日最高紫外線指數時間(LST)":data[32],"總雲量(0~10)":data[33],"year":data[34],"month":data[35],"day":data[36],"date":data[37]})
            day=day+1
            data=[]
        weather=pd.concat([weather,pd.DataFrame(fdata)], ignore_index=True,sort=True)
    weather["date"] = pd.to_datetime(weather["date"])


    weather=weather[['最低氣溫(℃)','最大陣風(m/s)','最大陣風風向(360degree)','最小相對溼度(%)','最高氣溫(℃)','氣溫(℃)','測站最低氣壓(hPa)','測站最高氣壓(hPa)','相對溼度(%)','風向(360degree)','測站氣壓(hPa)','降水量(mm)','風速(m/s)','year','month','day','date']]
    a=weather[weather['date']==today_date].index.item()
    weather=weather[:a+1]
    column_list=list(weather.columns.values)
    column_list.pop(-1)
    for i in column_list:
        for j in range(weather.shape[0]):
            if str(weather[i][j]).replace(".", "", 1).isdigit()==False:
                if j-1<0:
                    weather[i][j]=0
                else:
                    weather[i][j]=weather[i][j-1]


    train=weather.merge(crop, on='date', how='left')
    train=train.fillna(0)
    train=train.drop(["crop_name"], axis=1)
    train=train.drop(["crop_num"], axis=1)
    train=train.drop(["market_name"], axis=1)
    train=train.drop(["market_num"], axis=1)
    valid=train[train['date']==Validation_date].index.item()
    train=train.drop(["date"], axis=1)
    return train,valid


def buildTrain(train, pastDay, futureDay,type):
    X_train, Y_train, Z_train,M_train = [], [], [],[]
    for i in range(train.shape[0] - futureDay - pastDay + 1):
        M_train.append(np.array(train.iloc[i:i + 1][type]))
        X_train.append(np.array(train.iloc[i:i + pastDay]))
        Y_train.append(np.array(train.iloc[i + pastDay:i + pastDay + futureDay][type]))
        Z_train.append(np.array(train.iloc[i + pastDay - 1:i + pastDay + futureDay - 1][type]))
    W_train=train.iloc[train.shape[0] -pastDay - 1:train.shape[0]-1]
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    Z_train=np.array(Z_train)
    W_train=np.array(W_train)
    M_train=np.array(M_train)
    mul=np.mean(M_train)
    Y_train=Y_train/mul
    Z_train=Z_train/mul
    return X_train, Y_train, Z_train,W_train,mul


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


def manageData(train,type,past,future,Validation_date,valid,flag):
    if flag==0:
        train=train.convert_objects(convert_numeric=True)
        temp = train
        train = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
        train_x1, train_y1, train_z1,train_w1,mul_1 = buildTrain(train, past, future,type)
        train_x2, train_y2, train_z2,train_w2,mul_2 = buildTrain(temp, past, future,type)
        train_x, train_y, train_z ,train_w= train_x1, train_y2, train_z2,train_w1
        train_x, train_y, train_z = shuffle(train_x, train_y, train_z)
        train_x, train_y, val_x, val_y, val_z = splitData(train_x, train_y, train_z, 0.05)
        return train_x, train_y,train_w, val_x, val_y, val_z,mul_2
    else:
        train=train.convert_objects(convert_numeric=True)
        temp = train
        train = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
        train_x1, train_y1, train_z1,train_w1,mul_1 = buildTrain(train, past, future,type)
        train_x2, train_y2, train_z2,train_w2,mul_2 = buildTrain(temp, past, future,type)
        train_x, train_y, train_z ,train_w= train_x1, train_y2, train_z2,train_w1
        train_x, train_y, train_z = shuffle(train_x, train_y, train_z)
        train_x, train_y, val_x, val_y, val_z = splitData(train_x, train_y, train_z, 0.05)
        valid_x=np.array(train.iloc[valid-future:valid])
        valid_y=np.array(temp.iloc[valid:valid+future][type])
        return train_x, train_y,train_w, val_x, val_y, val_z,mul_2,valid_x,valid_y


def buildModel(train_x, train_y,future):
    model = Sequential()
    model.add(LSTM(32, input_length=train_x.shape[1],input_dim= train_x.shape[2],return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(future))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(train_x,train_y, epochs=300, batch_size=64, validation_split=0.1, callbacks=[callback])
    return model


def getResult(model,train_w, val_x, val_y, val_z,mul,past,valid_x,valid_y,flag):
    if flag==0:
        a = range(0, val_y.shape[0])
        val_y = val_y.reshape(-1)
        val_z = val_z.reshape(-1)
        b = []
        c = []
        co = 0
        coo = 0
        for i in range(0, val_x.shape[0]):
            temp = val_x[i]
            temp = temp.reshape(1, past, 21)
            z = model.predict(temp, verbose=0)
            if val_y[i]*mul >= val_z[i]*mul and z*mul >= val_z[i]*mul:
                co = co + 1
            if val_y[i]*mul < val_z[i]*mul and z*mul < val_z[i]*mul:
                co = co + 1
            if val_y[i] >= z:
                sub = val_y[i]*mul - z*mul
            if val_y[i] < z:
                sub = z*mul - val_y[i]*mul
            b.append(z)
            c.append(sub)
        b = np.array(b)
        b = b.reshape(-1)
        #plt.plot(a, b)
        acc = 100 * (co / val_x.shape[0])
        total = 0
        totalss = 0
        for i in c:
            total = total + i
            totalss = totalss + i * i
        average = float(total / len(c))
        ss = float((totalss / len(c))**0.5)
        return acc, average, ss, list(a), list(b*mul), list(val_y*mul)
    else:
        a = range(past)
        temp=train_w
        temp = temp.reshape(1, past, 21)
        z = model.predict(temp, verbose=0)
        z = z.reshape(-1)
        z=np.absolute(z)
        temp1=train_w
        temp1=temp1.reshape(1, past, 21)
        get = model.predict(temp1, verbose=0)
        get = get.reshape(-1)
        get=np.absolute(get)
        return list(a), list(z*mul),list(get*mul)
