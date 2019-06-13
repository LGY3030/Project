from django.shortcuts import render
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model
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
import random
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def home(request):
    return render(request, 'main/home.html')

def Information(request):
    if request.method == "POST":
        Crop_market = request.POST['Crop market']
        Crop_name = request.POST['Crop name']
        crop_info=pd.read_csv("../data/蔬果/crop_info.csv")
        index=crop_info[crop_info['Crop_type']==Crop_name].index.item()
        data=pd.read_csv("../data/蔬果/"+Crop_market+"/"+crop_info["Crop_name"][index]+".csv")
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
            weather_name=["雲林","嘉義","彰化","台南","高雄","屏東","台中","苗栗","桃園","台北","新北","基隆","新竹","南投","宜蘭","花蓮","台東"]
            crop_info=pd.read_csv("../data/蔬果/crop_info.csv")
            index=crop_info[crop_info['Crop_type']==Crop_name].index.item()
            if not os.path.exists("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Price(7-1)"+"/"+"data.csv"):
                context={"message":"該農產品資料量過少,無法預測結果!!"}
                return render(request, "main/mistake.html", locals())
            model = load_model("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Price(7-1)"+"/"+"model.h5")
            for weather in weather_name:
                if weather=="雲林":
                    train=pd.read_csv("../data/天氣/"+weather+".csv")
                    train=train.drop(["Unnamed: 0"], axis=1)
                else:
                    temp=pd.read_csv("../data/天氣/"+weather+".csv")
                    temp=temp.drop(["Unnamed: 0"], axis=1)
                    temp=temp.drop(["year"], axis=1)
                    temp=temp.drop(["month"], axis=1)
                    temp=temp.drop(["day"], axis=1)
                    train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/油價/中油.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')

            price_title=" "

            check_1=pd.read_csv("../data/date.csv")
            check_2=pd.read_csv("../data/蔬果/date.csv")
            if check_1["date"][check_1.shape[0]-1] != check_2["date"][check_2.shape[0]-1]:
                new_data = pd.DataFrame(train[-1:].values, index=[train.shape[0]], columns=train.columns)
                new_data["date"]=check_2["date"][check_2.shape[0]-1]
                train=train.append(new_data)

                oneday=datetime.timedelta(days=1)
                price_title=str(datetime.datetime.strptime(check_2["date"][check_2.shape[0]-1], '%Y-%m-%d').date()+oneday)
            else:
                oneday=datetime.timedelta(days=1)
                price_title=str(datetime.datetime.strptime(check_2["date"][check_2.shape[0]-1], '%Y-%m-%d').date()+oneday)

            temp=pd.read_csv("../data/國定假日/國定假日.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/拜拜日/拜拜日.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/蔬果/"+Crop_market+"/"+crop_info["Crop_name"][index]+".csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            temp=temp.drop(["year"], axis=1)
            temp=temp.drop(["month"], axis=1)
            temp=temp.drop(["day"], axis=1)
            train=train.merge(temp, on='date', how='left')

            train=train.fillna(0)

            scale=np.mean(train["high"])

            train=train.drop(["date"], axis=1)
            train=train.drop(["crop_name"], axis=1)
            train=train.drop(["crop_num"], axis=1)
            train=train.drop(["market_name"], axis=1)
            train=train.drop(["market_num"], axis=1)
            train=train.apply(pd.to_numeric)
            train= train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
            train=train[:7]
            train=np.array(train)
            train=train.reshape(1,7,236)
            result=(model.predict(train, verbose=0)*scale)
            result=result.reshape(-1)
            result=result[0]

            unit1="筆"
            unit2="元"
            unit3="%"

            graph_title="測試資料的預測價格&測試資料的原始價格:"
            bi_title="訓練模型使用的資料筆數:"
            bi_info1="非休市日:"
            bi_info2="休市日:"
            accuracy_title="測試資料的預測資訊(所有測試資料):"
            accuracy_info1="Number Of Data:"
            accuracy_info2="Accuracy:"
            accuracy_info3="Average Deviation:"
            accuracy_info4="Standard Deviation:"
            accuracy_info5="Coefficient Of Variation:"
            accuracy_0_title="測試資料的預測資訊(扣除休市日的資料):"
            graph = pd.read_csv("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Price(7-1)"+"/"+"graph.csv")
            predict_graph=list(graph["Predict"])
            origin_graph=list(graph["Origin"])
            graph_index=list(range(0, graph.shape[0]))
            test_bi=len(predict_graph)
            test_bi_0=len([i for i in origin_graph if i != 0])
            data = pd.read_csv("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Price(7-1)"+"/"+"data.csv")
            get_bi=pd.read_csv("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"data.csv")
            data_bi=get_bi["non-rest"][0]
            data_bi_rest=get_bi["rest"][0]
            accuracy=data["Accuracy"][0]
            accuracy_0=data["Accuracy_0"][0]
            average_deviation=data["AverageDeviation"][0]
            average_deviation_0=data["AverageDeviation_0"][0]
            coefficient_of_variation=data["CoefficientOfVariation"][0]
            coefficient_of_variation_0=data["CoefficientOfVariation_0"][0]
            standard_deviation=data["StandardDeviation"][0]
            standard_deviation_0=data["StandardDeviation_0"][0]
            context={"result":result,"predict_graph":predict_graph,"origin_graph":origin_graph,"graph_index":graph_index,"price_title":price_title,"graph_title":graph_title,"bi_title":bi_title,"bi_info1":bi_info1,"bi_info2":bi_info2,"data_bi":data_bi,"data_bi_rest":data_bi_rest,"accuracy_title":accuracy_title,"accuracy_0_title":accuracy_0_title,"accuracy_info1":accuracy_info1,"accuracy_info2":accuracy_info2,"accuracy_info3":accuracy_info3,"accuracy_info4":accuracy_info4,"accuracy_info5":accuracy_info5,"accuracy":accuracy,"accuracy_0":accuracy_0,"average_deviation":average_deviation,"average_deviation_0":average_deviation_0,"standard_deviation":standard_deviation,"standard_deviation_0":standard_deviation_0,"coefficient_of_variation":coefficient_of_variation,"coefficient_of_variation_0":coefficient_of_variation_0,"test_bi":test_bi,"test_bi_0":test_bi_0,"unit1":unit1,"unit2":unit2,"unit3":unit3}
            return render(request, "main/price.html", locals())
        except:
            context={"message":"請注意搜尋字是否有打錯,或是稍後再試!!"}
            return render(request, "main/mistake.html", locals())
    return render(request, "main/price.html", locals())

def recommendation(request):
    if request.method == "POST":
        try:
            Crop_market = request.POST['Crop market']
            Crop_name = request.POST['Crop name']
            weather_name=["雲林","嘉義","彰化","台南","高雄","屏東","台中","苗栗","桃園","台北","新北","基隆","新竹","南投","宜蘭","花蓮","台東"]
            crop_info=pd.read_csv("../data/蔬果/crop_info.csv")
            index=crop_info[crop_info['Crop_type']==Crop_name].index.item()
            if not os.path.exists("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Price(7-1)"+"/"+"data.csv"):
                context={"message":"該農產品資料量過少,無法預測結果!!"}
                return render(request, "main/mistake.html", locals())
            model = load_model("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Price(7-1)"+"/"+"model.h5")
            for weather in weather_name:
                if weather=="雲林":
                    train=pd.read_csv("../data/天氣/"+weather+".csv")
                    train=train.drop(["Unnamed: 0"], axis=1)
                else:
                    temp=pd.read_csv("../data/天氣/"+weather+".csv")
                    temp=temp.drop(["Unnamed: 0"], axis=1)
                    temp=temp.drop(["year"], axis=1)
                    temp=temp.drop(["month"], axis=1)
                    temp=temp.drop(["day"], axis=1)
                    train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/油價/中油.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')

            check_1=pd.read_csv("../data/date.csv")
            check_2=pd.read_csv("../data/蔬果/date.csv")
            if check_1["date"][check_1.shape[0]-1] != check_2["date"][check_2.shape[0]-1]:
                new_data = pd.DataFrame(train[-1:].values, index=[train.shape[0]], columns=train.columns)
                new_data["date"]=check_2["date"][check_2.shape[0]-1]
                train=train.append(new_data)

                oneday=datetime.timedelta(days=1)
                recommend_title="預測 "+str(datetime.datetime.strptime(check_2["date"][check_2.shape[0]-1], '%Y-%m-%d').date()+oneday)+" 價格和推薦結果:"
            else:
                oneday=datetime.timedelta(days=1)
                recommend_title="預測 "+str(datetime.datetime.strptime(check_2["date"][check_2.shape[0]-1], '%Y-%m-%d').date()+oneday)+" 價格和推薦結果:"

            temp=pd.read_csv("../data/國定假日/國定假日.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/拜拜日/拜拜日.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/蔬果/"+Crop_market+"/"+crop_info["Crop_name"][index]+".csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            temp=temp.drop(["year"], axis=1)
            temp=temp.drop(["month"], axis=1)
            temp=temp.drop(["day"], axis=1)
            train=train.merge(temp, on='date', how='left')

            train=train.fillna(0)

            scale=np.mean(train["high"])

            train=train.drop(["date"], axis=1)
            train=train.drop(["crop_name"], axis=1)
            train=train.drop(["crop_num"], axis=1)
            train=train.drop(["market_name"], axis=1)
            train=train.drop(["market_num"], axis=1)
            train=train.apply(pd.to_numeric)
            train= train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
            train=train[:7]
            train=np.array(train)
            train=train.reshape(1,7,236)
            result=(model.predict(train)*scale)
            result=result.reshape(-1)
            result=result[0]
            market_name=["台北一","台北二","板橋區","三重區","桃農","台中","豐原","東勢","嘉義","高雄","鳳山區","台東","宜蘭","南投"]
            random.shuffle(market_name)
            market_name.remove(Crop_market)
            ano_crop_market=""
            while True:
                if(len(market_name)>0):
                    if os.path.exists("../model/"+market_name[0]+"/"+crop_info["Crop_name"][index]+"/"+"Price(7-1)"+"/"+"model.h5"):
                        ano_model = load_model("../model/"+market_name[0]+"/"+crop_info["Crop_name"][index]+"/"+"Price(7-1)"+"/"+"model.h5")
                        for weather in weather_name:
                            if weather=="雲林":
                                ano_train=pd.read_csv("../data/天氣/"+weather+".csv")
                                ano_train=ano_train.drop(["Unnamed: 0"], axis=1)
                            else:
                                ano_temp=pd.read_csv("../data/天氣/"+weather+".csv")
                                ano_temp=ano_temp.drop(["Unnamed: 0"], axis=1)
                                ano_temp=ano_temp.drop(["year"], axis=1)
                                ano_temp=ano_temp.drop(["month"], axis=1)
                                ano_temp=ano_temp.drop(["day"], axis=1)
                                ano_train=ano_train.merge(ano_temp, on='date', how='left')
                        ano_temp=pd.read_csv("../data/油價/中油.csv")
                        ano_temp=ano_temp.drop(["Unnamed: 0"], axis=1)
                        ano_train=ano_train.merge(ano_temp, on='date', how='left')

                        check_1=pd.read_csv("../data/date.csv")
                        check_2=pd.read_csv("../data/蔬果/date.csv")
                        if check_1["date"][check_1.shape[0]-1] != check_2["date"][check_2.shape[0]-1]:
                        	new_data = pd.DataFrame(ano_train[-1:].values, index=[ano_train.shape[0]], columns=ano_train.columns)
                        	new_data["date"]=check_2["date"][check_2.shape[0]-1]
                        	ano_train=ano_train.append(new_data)


                        ano_temp=pd.read_csv("../data/國定假日/國定假日.csv")
                        ano_temp=ano_temp.drop(["Unnamed: 0"], axis=1)
                        ano_train=ano_train.merge(ano_temp, on='date', how='left')
                        ano_temp=pd.read_csv("../data/拜拜日/拜拜日.csv")
                        ano_temp=ano_temp.drop(["Unnamed: 0"], axis=1)
                        ano_train=ano_train.merge(ano_temp, on='date', how='left')
                        ano_temp=pd.read_csv("../data/蔬果/"+market_name[0]+"/"+crop_info["Crop_name"][index]+".csv")
                        ano_temp=ano_temp.drop(["Unnamed: 0"], axis=1)
                        ano_temp=ano_temp.drop(["year"], axis=1)
                        ano_temp=ano_temp.drop(["month"], axis=1)
                        ano_temp=ano_temp.drop(["day"], axis=1)
                        ano_train=ano_train.merge(ano_temp, on='date', how='left')

                        ano_train=ano_train.fillna(0)

                        ano_scale=np.mean(ano_train["high"])

                        ano_train=ano_train.drop(["date"], axis=1)
                        ano_train=ano_train.drop(["crop_name"], axis=1)
                        ano_train=ano_train.drop(["crop_num"], axis=1)
                        ano_train=ano_train.drop(["market_name"], axis=1)
                        ano_train=ano_train.drop(["market_num"], axis=1)
                        ano_train=ano_train.apply(pd.to_numeric)
                        ano_train= ano_train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
                        ano_train=ano_train[:7]
                        ano_train=np.array(ano_train)
                        ano_train=ano_train.reshape(1,7,236)
                        ano_result=(ano_model.predict(ano_train)*ano_scale)
                        ano_result=ano_result.reshape(-1)
                        ano_result=ano_result[0]
                        if ano_result>result:
                            ano_crop_market=market_name[0]
                            break
                        else:
                            market_name.pop(0)
                    else:
                        market_name.pop(0)
                else:
                    break
            Crop_market_f=""
            if ano_crop_market=="":
                Crop_market_f=Crop_market
                ano_result=""
                recommendate=" 的該蔬果預測價格大於其他市場"
            else:
                Crop_market_f=""
                ano_result=str(ano_result)+"元"
                recommendate=" 該蔬果可以出售給"

            ano_data_bi=ano_temp.shape[0]
            ano_data_bi_rest=2677-ano_data_bi
            unit1="筆"
            unit2="元"
            unit3="%"
            result=str(result)+"元"
            graph_title="測試資料的預測價格&測試資料的原始價格:"
            bi_title="訓練模型使用的資料筆數:"
            bi_info1="非休市日:"
            bi_info2="休市日:"
            accuracy_title="測試資料的預測資訊(所有測試資料):"
            accuracy_info1="Number Of Data:"
            accuracy_info2="Accuracy:"
            accuracy_info3="Average Deviation:"
            accuracy_info4="Standard Deviation:"
            accuracy_info5="Coefficient Of Variation:"
            accuracy_0_title="測試資料的預測資訊(扣除休市日的資料):"
            graph = pd.read_csv("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Price(7-1)"+"/"+"graph.csv")
            predict_graph=list(graph["Predict"])
            origin_graph=list(graph["Origin"])
            graph_index=list(range(0, graph.shape[0]))
            test_bi=len(predict_graph)
            test_bi_0=len([i for i in origin_graph if i != 0])
            data = pd.read_csv("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Price(7-1)"+"/"+"data.csv")
            data_bi=temp.shape[0]
            data_bi_rest=2677-data_bi
            accuracy=data["Accuracy"][0]
            accuracy_0=data["Accuracy_0"][0]
            average_deviation=data["AverageDeviation"][0]
            average_deviation_0=data["AverageDeviation_0"][0]
            coefficient_of_variation=data["CoefficientOfVariation"][0]
            coefficient_of_variation_0=data["CoefficientOfVariation_0"][0]
            standard_deviation=data["StandardDeviation"][0]
            standard_deviation_0=data["StandardDeviation_0"][0]

            context={"result":result,"predict_graph":predict_graph,"origin_graph":origin_graph,"graph_index":graph_index,"recommend_title":recommend_title,"graph_title":graph_title,"bi_title":bi_title,"bi_info1":bi_info1,"bi_info2":bi_info2,"data_bi":data_bi,"data_bi_rest":data_bi_rest,"accuracy_title":accuracy_title,"accuracy_0_title":accuracy_0_title,"accuracy_info1":accuracy_info1,"accuracy_info2":accuracy_info2,"accuracy_info3":accuracy_info3,"accuracy_info4":accuracy_info4,"accuracy_info5":accuracy_info5,"accuracy":accuracy,"accuracy_0":accuracy_0,"average_deviation":average_deviation,"average_deviation_0":average_deviation_0,"standard_deviation":standard_deviation,"standard_deviation_0":standard_deviation_0,"coefficient_of_variation":coefficient_of_variation,"coefficient_of_variation_0":coefficient_of_variation_0,"test_bi":test_bi,"test_bi_0":test_bi_0,"unit1":unit1,"unit2":unit2,"unit3":unit3,"recommendate":recommendate,"ano_crop_market":ano_crop_market,"crop_market":Crop_market,"crop_market_f":Crop_market_f,"ano_result":ano_result,"ano_data_bi":ano_data_bi,"ano_data_bi_rest":ano_data_bi_rest}
            return render(request, "main/recommendation.html", locals())
        except:
            context={"message":"請注意搜尋字是否有打錯,或是稍後再試!!"}
            return render(request, "main/mistake.html", locals())

    return render(request, "main/recommendation.html", locals())

def volume(request):
    if request.method == "POST":
        try:
            Crop_market = request.POST['Crop market']
            Crop_name = request.POST['Crop name']
            weather_name=["雲林","嘉義","彰化","台南","高雄","屏東","台中","苗栗","桃園","台北","新北","基隆","新竹","南投","宜蘭","花蓮","台東"]
            crop_info=pd.read_csv("../data/蔬果/crop_info.csv")
            index=crop_info[crop_info['Crop_type']==Crop_name].index.item()
            if not os.path.exists("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Volume(7-1)"+"/"+"data.csv"):
                context={"message":"該農產品資料量過少,無法預測結果!!"}
                return render(request, "main/mistake.html", locals())
            model = load_model("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Volume(7-1)"+"/"+"model.h5")
            for weather in weather_name:
                if weather=="雲林":
                    train=pd.read_csv("../data/天氣/"+weather+".csv")
                    train=train.drop(["Unnamed: 0"], axis=1)
                else:
                    temp=pd.read_csv("../data/天氣/"+weather+".csv")
                    temp=temp.drop(["Unnamed: 0"], axis=1)
                    temp=temp.drop(["year"], axis=1)
                    temp=temp.drop(["month"], axis=1)
                    temp=temp.drop(["day"], axis=1)
                    train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/油價/中油.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')

            check_1=pd.read_csv("../data/date.csv")
            check_2=pd.read_csv("../data/蔬果/date.csv")
            if check_1["date"][check_1.shape[0]-1] != check_2["date"][check_2.shape[0]-1]:
                new_data = pd.DataFrame(train[-1:].values, index=[train.shape[0]], columns=train.columns)
                new_data["date"]=check_2["date"][check_2.shape[0]-1]
                train=train.append(new_data)

                oneday=datetime.timedelta(days=1)
                price_title=str(datetime.datetime.strptime(check_2["date"][check_2.shape[0]-1], '%Y-%m-%d').date()+oneday)
            else:
                oneday=datetime.timedelta(days=1)
                price_title=str(datetime.datetime.strptime(check_2["date"][check_2.shape[0]-1], '%Y-%m-%d').date()+oneday)


            temp=pd.read_csv("../data/國定假日/國定假日.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/拜拜日/拜拜日.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/蔬果/"+Crop_market+"/"+crop_info["Crop_name"][index]+".csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            temp=temp.drop(["year"], axis=1)
            temp=temp.drop(["month"], axis=1)
            temp=temp.drop(["day"], axis=1)
            train=train.merge(temp, on='date', how='left')

            train=train.fillna(0)

            scale=np.mean(train["volume"])

            train=train.drop(["date"], axis=1)
            train=train.drop(["crop_name"], axis=1)
            train=train.drop(["crop_num"], axis=1)
            train=train.drop(["market_name"], axis=1)
            train=train.drop(["market_num"], axis=1)
            train=train.apply(pd.to_numeric)
            train= train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
            train=train[:7]
            train=np.array(train)
            train=train.reshape(1,7,236)
            result=(model.predict(train)*scale)
            result=result.reshape(-1)
            result=result[0]

            unit1="筆"
            unit2="元"
            unit3="%"
            graph_title="測試資料的預測交易量&測試資料的原始交易量:"
            bi_title="訓練模型使用的資料筆數:"
            bi_info1="非休市日:"
            bi_info2="休市日:"
            accuracy_title="測試資料的預測資訊(所有測試資料):"
            accuracy_info1="Number Of Data:"
            accuracy_info2="Accuracy:"
            accuracy_info3="Average Deviation:"
            accuracy_info4="Standard Deviation:"
            accuracy_info5="Coefficient Of Variation:"
            accuracy_0_title="測試資料的預測資訊(扣除休市日的資料):"
            graph = pd.read_csv("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Volume(7-1)"+"/"+"graph.csv")
            predict_graph=list(graph["Predict"])
            origin_graph=list(graph["Origin"])
            graph_index=list(range(0, graph.shape[0]))
            test_bi=len(predict_graph)
            test_bi_0=len([i for i in origin_graph if i != 0])
            data = pd.read_csv("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Volume(7-1)"+"/"+"data.csv")
            get_bi=pd.read_csv("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"data.csv")
            data_bi=get_bi["non-rest"][0]
            data_bi_rest=get_bi["rest"][0]
            accuracy=data["Accuracy"][0]
            accuracy_0=data["Accuracy_0"][0]
            average_deviation=data["AverageDeviation"][0]
            average_deviation_0=data["AverageDeviation_0"][0]
            coefficient_of_variation=data["CoefficientOfVariation"][0]
            coefficient_of_variation_0=data["CoefficientOfVariation_0"][0]
            standard_deviation=data["StandardDeviation"][0]
            standard_deviation_0=data["StandardDeviation_0"][0]

            context={"result":result,"predict_graph":predict_graph,"origin_graph":origin_graph,"graph_index":graph_index,"price_title":price_title,"graph_title":graph_title,"bi_title":bi_title,"bi_info1":bi_info1,"bi_info2":bi_info2,"data_bi":data_bi,"data_bi_rest":data_bi_rest,"accuracy_title":accuracy_title,"accuracy_0_title":accuracy_0_title,"accuracy_info1":accuracy_info1,"accuracy_info2":accuracy_info2,"accuracy_info3":accuracy_info3,"accuracy_info4":accuracy_info4,"accuracy_info5":accuracy_info5,"accuracy":accuracy,"accuracy_0":accuracy_0,"average_deviation":average_deviation,"average_deviation_0":average_deviation_0,"standard_deviation":standard_deviation,"standard_deviation_0":standard_deviation_0,"coefficient_of_variation":coefficient_of_variation,"coefficient_of_variation_0":coefficient_of_variation_0,"test_bi":test_bi,"test_bi_0":test_bi_0,"unit1":unit1,"unit2":unit2,"unit3":unit3}
            return render(request, "main/volume.html", locals())
        except:
            context={"message":"請注意搜尋字是否有打錯,或是稍後再試!!"}
            return render(request, "main/mistake.html", locals())
    return render(request, "main/volume.html", locals())
def price_trend(request):
    if request.method == "POST":
        try:
            Crop_market = request.POST['Crop market']
            Crop_name = request.POST['Crop name']
            weather_name=["雲林","嘉義","彰化","台南","高雄","屏東","台中","苗栗","桃園","台北","新北","基隆","新竹","南投","宜蘭","花蓮","台東"]
            crop_info=pd.read_csv("../data/蔬果/crop_info.csv")
            index=crop_info[crop_info['Crop_type']==Crop_name].index.item()
            if not os.path.exists("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Price(30-30)"+"/"+"data.csv"):
                context={"message":"該農產品資料量過少,無法預測結果!!"}
                return render(request, "main/mistake.html", locals())
            model = load_model("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Price(30-30)"+"/"+"model.h5")
            for weather in weather_name:
                if weather=="雲林":
                    train=pd.read_csv("../data/天氣/"+weather+".csv")
                    train=train.drop(["Unnamed: 0"], axis=1)
                else:
                    temp=pd.read_csv("../data/天氣/"+weather+".csv")
                    temp=temp.drop(["Unnamed: 0"], axis=1)
                    temp=temp.drop(["year"], axis=1)
                    temp=temp.drop(["month"], axis=1)
                    temp=temp.drop(["day"], axis=1)
                    train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/油價/中油.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')
            check_1=pd.read_csv("../data/date.csv")
            check_2=pd.read_csv("../data/蔬果/date.csv")
            if check_1["date"][check_1.shape[0]-1] != check_2["date"][check_2.shape[0]-1]:
                new_data = pd.DataFrame(train[-1:].values, index=[train.shape[0]], columns=train.columns)
                new_data["date"]=check_2["date"][check_2.shape[0]-1]
                train=train.append(new_data)


                oneday=datetime.timedelta(days=1)
                for30day=datetime.timedelta(days=30)
                price_title=str(datetime.datetime.strptime(check_2["date"][check_2.shape[0]-1], '%Y-%m-%d').date()+oneday)+" 到 "+str(datetime.datetime.strptime(check_2["date"][check_2.shape[0]-1], '%Y-%m-%d').date()+for30day)

            else:
                oneday=datetime.timedelta(days=1)
                for30day=datetime.timedelta(days=30)
                price_title=str(datetime.datetime.strptime(check_2["date"][check_2.shape[0]-1], '%Y-%m-%d').date()+oneday)+" 到 "+str(datetime.datetime.strptime(check_2["date"][check_2.shape[0]-1], '%Y-%m-%d').date()+for30day)

            temp=pd.read_csv("../data/國定假日/國定假日.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/拜拜日/拜拜日.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/蔬果/"+Crop_market+"/"+crop_info["Crop_name"][index]+".csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            temp=temp.drop(["year"], axis=1)
            temp=temp.drop(["month"], axis=1)
            temp=temp.drop(["day"], axis=1)
            train=train.merge(temp, on='date', how='left')

            train=train.fillna(0)

            scale=np.mean(train["high"])

            train=train.drop(["date"], axis=1)
            train=train.drop(["crop_name"], axis=1)
            train=train.drop(["crop_num"], axis=1)
            train=train.drop(["market_name"], axis=1)
            train=train.drop(["market_num"], axis=1)
            train=train.apply(pd.to_numeric)
            train= train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
            train=train[:30]
            train=np.array(train)
            train=train.reshape(1,30,236)
            result=model.predict(train)*scale
            result=list(result.reshape(-1))
            result_index=list(range(30))

            unit1="筆"
            unit2="元"
            unit3="%"
            graph_1_title="測試資料的預測價格&測試資料的原始價格:"
            graph_2_title="測試資料的預測價格&測試資料的原始價格:"
            bi_title="訓練模型使用的資料筆數:"
            bi_info1="非休市日:"
            bi_info2="休市日:"
            accuracy_title="測試資料的預測資訊(所有測試資料):"
            accuracy_info1="Number Of Data:"
            accuracy_info2="Accuracy:"
            accuracy_info3="Average Deviation:"
            accuracy_info4="Standard Deviation:"
            accuracy_info5="Coefficient Of Variation:"
            accuracy_0_title="測試資料的預測資訊(扣除休市日的資料):"
            graph_1 = pd.read_csv("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Price(30-30)"+"/"+"graph_1.csv")
            predict_graph_1=list(graph_1["Predict"])
            origin_graph_1=list(graph_1["Origin"])
            graph_1_index=list(range(0, graph_1.shape[0]))
            test_bi=len(predict_graph_1)
            test_bi_0=len([i for i in origin_graph_1 if i != 0])

            graph_2 = pd.read_csv("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Price(30-30)"+"/"+"graph_2.csv")
            predict_graph_2=list(graph_2["Predict"])
            origin_graph_2=list(graph_2["Origin"])
            graph_2_index=list(range(0, graph_2.shape[0]))


            data = pd.read_csv("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"Price(30-30)"+"/"+"data.csv")
            get_bi=pd.read_csv("../model/"+Crop_market+"/"+crop_info["Crop_name"][index]+"/"+"data.csv")
            data_bi=get_bi["non-rest"][0]
            data_bi_rest=get_bi["rest"][0]
            accuracy=data["Accuracy"][0]
            accuracy_0=data["Accuracy_0"][0]
            average_deviation=data["AverageDeviation"][0]
            average_deviation_0=data["AverageDeviation_0"][0]
            coefficient_of_variation=data["CoefficientOfVariation"][0]
            coefficient_of_variation_0=data["CoefficientOfVariation_0"][0]
            standard_deviation=data["StandardDeviation"][0]
            standard_deviation_0=data["StandardDeviation_0"][0]

            context={"result":result,"result_index":result_index,"predict_graph_1":predict_graph_1,"origin_graph_1":origin_graph_1,"graph_1_index":graph_1_index,"predict_graph_2":predict_graph_2,"origin_graph_2":origin_graph_2,"graph_2_index":graph_2_index,"price_title":price_title,"graph_1_title":graph_1_title,"graph_2_title":graph_2_title,"bi_title":bi_title,"bi_info1":bi_info1,"bi_info2":bi_info2,"data_bi":data_bi,"data_bi_rest":data_bi_rest,"accuracy_title":accuracy_title,"accuracy_0_title":accuracy_0_title,"accuracy_info1":accuracy_info1,"accuracy_info2":accuracy_info2,"accuracy_info3":accuracy_info3,"accuracy_info4":accuracy_info4,"accuracy_info5":accuracy_info5,"accuracy":accuracy,"accuracy_0":accuracy_0,"average_deviation":average_deviation,"average_deviation_0":average_deviation_0,"standard_deviation":standard_deviation,"standard_deviation_0":standard_deviation_0,"coefficient_of_variation":coefficient_of_variation,"coefficient_of_variation_0":coefficient_of_variation_0,"test_bi":test_bi,"test_bi_0":test_bi_0,"unit1":unit1,"unit2":unit2,"unit3":unit3}
            return render(request, "main/price_trend.html", locals())
        except:
            context={"message":"請注意搜尋字是否有打錯,或是稍後再試!!"}
            return render(request, "main/mistake.html", locals())
    return render(request, "main/price_trend.html", locals())

def price_trend_compare(request):
    if request.method == "POST":
        try:
            Crop_market_1 = request.POST['Crop market 1']
            Crop_market_2 = request.POST['Crop market 2']
            Crop_name = request.POST['Crop name']
            weather_name=["雲林","嘉義","彰化","台南","高雄","屏東","台中","苗栗","桃園","台北","新北","基隆","新竹","南投","宜蘭","花蓮","台東"]
            crop_info=pd.read_csv("../data/蔬果/crop_info.csv")
            index=crop_info[crop_info['Crop_type']==Crop_name].index.item()
            if not os.path.exists("../model/"+Crop_market_1+"/"+crop_info["Crop_name"][index]+"/"+"Price(30-30)"+"/"+"data.csv"):
                context={"message":"該農產品資料量過少,無法預測結果!!"}
                return render(request, "main/mistake.html", locals())
            if not os.path.exists("../model/"+Crop_market_2+"/"+crop_info["Crop_name"][index]+"/"+"Price(30-30)"+"/"+"data.csv"):
                context={"message":"該農產品資料量過少,無法預測結果!!"}
                return render(request, "main/mistake.html", locals())
            model = load_model("../model/"+Crop_market_1+"/"+crop_info["Crop_name"][index]+"/"+"Price(30-30)"+"/"+"model.h5")
            for weather in weather_name:
                if weather=="雲林":
                    train=pd.read_csv("../data/天氣/"+weather+".csv")
                    train=train.drop(["Unnamed: 0"], axis=1)
                else:
                    temp=pd.read_csv("../data/天氣/"+weather+".csv")
                    temp=temp.drop(["Unnamed: 0"], axis=1)
                    temp=temp.drop(["year"], axis=1)
                    temp=temp.drop(["month"], axis=1)
                    temp=temp.drop(["day"], axis=1)
                    train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/油價/中油.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')

            check_1=pd.read_csv("../data/date.csv")
            check_2=pd.read_csv("../data/蔬果/date.csv")
            if check_1["date"][check_1.shape[0]-1] != check_2["date"][check_2.shape[0]-1]:
                new_data = pd.DataFrame(train[-1:].values, index=[train.shape[0]], columns=train.columns)
                new_data["date"]=check_2["date"][check_2.shape[0]-1]
                train=train.append(new_data)

                oneday=datetime.timedelta(days=1)
                for30day=datetime.timedelta(days=30)
                price_title=str(datetime.datetime.strptime(check_2["date"][check_2.shape[0]-1], '%Y-%m-%d').date()+oneday)+" 到 "+str(datetime.datetime.strptime(check_2["date"][check_2.shape[0]-1], '%Y-%m-%d').date()+for30day)

            else:
                oneday=datetime.timedelta(days=1)
                for30day=datetime.timedelta(days=30)
                price_title=str(datetime.datetime.strptime(check_2["date"][check_2.shape[0]-1], '%Y-%m-%d').date()+oneday)+" 到 "+str(datetime.datetime.strptime(check_2["date"][check_2.shape[0]-1], '%Y-%m-%d').date()+for30day)

            temp=pd.read_csv("../data/國定假日/國定假日.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/拜拜日/拜拜日.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/蔬果/"+Crop_market_1+"/"+crop_info["Crop_name"][index]+".csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            temp=temp.drop(["year"], axis=1)
            temp=temp.drop(["month"], axis=1)
            temp=temp.drop(["day"], axis=1)
            train=train.merge(temp, on='date', how='left')

            train=train.fillna(0)

            scale=np.mean(train["high"])

            train=train.drop(["date"], axis=1)
            train=train.drop(["crop_name"], axis=1)
            train=train.drop(["crop_num"], axis=1)
            train=train.drop(["market_name"], axis=1)
            train=train.drop(["market_num"], axis=1)
            train=train.apply(pd.to_numeric)
            train= train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
            train=train[:30]
            train=np.array(train)
            train=train.reshape(1,30,236)
            result_1=model.predict(train)*scale
            result_1=list(result_1.reshape(-1))
            result_index=list(range(30))

            model = load_model("../model/"+Crop_market_2+"/"+crop_info["Crop_name"][index]+"/"+"Price(30-30)"+"/"+"model.h5")
            for weather in weather_name:
                if weather=="雲林":
                    train=pd.read_csv("../data/天氣/"+weather+".csv")
                    train=train.drop(["Unnamed: 0"], axis=1)
                else:
                    temp=pd.read_csv("../data/天氣/"+weather+".csv")
                    temp=temp.drop(["Unnamed: 0"], axis=1)
                    temp=temp.drop(["year"], axis=1)
                    temp=temp.drop(["month"], axis=1)
                    temp=temp.drop(["day"], axis=1)
                    train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/油價/中油.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')

            check_1=pd.read_csv("../data/date.csv")
            check_2=pd.read_csv("../data/蔬果/date.csv")
            if check_1["date"][check_1.shape[0]-1] != check_2["date"][check_2.shape[0]-1]:
                new_data = pd.DataFrame(train[-1:].values, index=[train.shape[0]], columns=train.columns)
                new_data["date"]=check_2["date"][check_2.shape[0]-1]
                train=train.append(new_data)


            temp=pd.read_csv("../data/國定假日/國定假日.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/拜拜日/拜拜日.csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            train=train.merge(temp, on='date', how='left')
            temp=pd.read_csv("../data/蔬果/"+Crop_market_2+"/"+crop_info["Crop_name"][index]+".csv")
            temp=temp.drop(["Unnamed: 0"], axis=1)
            temp=temp.drop(["year"], axis=1)
            temp=temp.drop(["month"], axis=1)
            temp=temp.drop(["day"], axis=1)
            train=train.merge(temp, on='date', how='left')

            train=train.fillna(0)

            scale=np.mean(train["high"])

            train=train.drop(["date"], axis=1)
            train=train.drop(["crop_name"], axis=1)
            train=train.drop(["crop_num"], axis=1)
            train=train.drop(["market_name"], axis=1)
            train=train.drop(["market_num"], axis=1)
            train=train.apply(pd.to_numeric)
            train= train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
            train=train[:30]
            train=np.array(train)
            train=train.reshape(1,30,236)
            result_2=model.predict(train)*scale
            result_2=list(result_2.reshape(-1))

            context={"result_1":result_1,"result_2":result_2,"result_index":result_index,"price_title":price_title}
            return render(request, "main/price_trend_compare.html", locals())
        except:
            context={"message":"請注意搜尋字是否有打錯,或是稍後再試!!"}
            return render(request, "main/mistake.html", locals())
    return render(request, "main/price_trend_compare.html", locals())


def mistake(request):
    return render(request, 'main/mistake.html')
