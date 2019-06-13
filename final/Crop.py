


from urllib.request import urlopen
import pandas as pd
import json
import requests
import urllib
import time
import csv
import os
import datetime
import ssl
import time


def set_time():
    day_flag=0
    flag=0
    while True:
        today_date=datetime.datetime.now().day
        if day_flag==today_date:
            flag=1
        else:
            flag=0
        now=datetime.datetime.now()
        if now.hour==13 and flag==0:
            print(now)
            crawl()
            day_flag=today_date
            flag=1
        print("Waiting!!!")
        time.sleep(60)


def crawl():
    ssl._create_default_https_context = ssl._create_unverified_context
    place=["台北一","台北二","板橋區","三重區","桃農","台中","豐原","東勢","嘉義","高雄","鳳山區","台東","宜蘭","南投"]
    col_name=["date","crop_num","crop_name","market_num","market_name","high","medium","low","mean","volume"] 
    crop_info=pd.read_csv("data/蔬果/crop_info.csv")
    now = datetime.datetime.now()
    today_date=str(now.year)+"."+str(now.month)+"."+str(now.day)
    for a in place:
        q=0
        for b in range(crop_info.shape[0]):
            print(a)
            print(crop_info["Crop_name"][b])
            q=q+1
            print(q)
            print()
            print()
            get_csv=pd.read_csv("data/蔬果/"+a+"/"+crop_info["Crop_name"][b]+".csv")
            get_csv=get_csv.drop(["Unnamed: 0"], axis=1)
            if get_csv.shape[0]==0:
                continue
            csv_date=get_csv["date"][get_csv.shape[0]-1]
            temp=csv_date.split('-')
            last_date=temp[0]+"."+temp[1]+"."+temp[2]
            df=pd.DataFrame(columns=col_name)
            url="https://data.coa.gov.tw/Service/OpenData/FromM/FarmTransData.aspx?$top=1000&$skip=0&Crop="+urllib.parse.quote(crop_info["Crop_name"][b])+"&StartDate=108.04.01&EndDate="+today_date+"&Market="+urllib.parse.quote(a)
            get = json.loads(urlopen(url).read().decode("utf-8")) 
            data = [] 
            for i in range(0,len(get)):
                if last_date==get[i]['交易日期']:
                    break
                else:
                    if get[i]['作物代號']==crop_info["Crop_num"][b]:
                        Date=get[i]['交易日期'].split('.')
                        year,month,date=Date[0],Date[1],Date[2]
                        year=str(int(year)+1911)
                        get[i]['交易日期']=year+'/'+month+'/'+date
                        data.append({"date":get[i]['交易日期'],"crop_num":get[i]['作物代號'],"crop_name":get[i]['作物名稱'],"market_num":get[i]['市場代號'],"market_name":get[i]['市場名稱'],"high":get[i]['上價'],"medium":get[i]['中價'],"low":get[i]['下價'],"mean":get[i]['平均價'],"volume":get[i]['交易量']}) 
            data.reverse() 
            df=pd.concat([get_csv ,pd.DataFrame(data)], ignore_index=True,sort=True)
            path="data/蔬果/"+a+"/"+crop_info["Crop_name"][b]+".csv"
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day"] = df["date"].dt.day
            df["dayofweek"] = df["date"].dt.dayofweek
            df=df[["crop_name","crop_num","date","high","low","market_name","market_num","mean","medium","volume","year","month","day","dayofweek"]]
            df.to_csv(path, encoding='utf_8_sig')
            
            bi_col=["non-rest","rest"]
            bi_data=[]
            date_csv=pd.read_csv("data/date.csv")
            bi_df=pd.DataFrame(columns=bi_col)
            bi_data.append({"non-rest":df.shape[0],"rest":date_csv.shape[0]+1-df.shape[0]})
            bi_df=pd.concat([pd.DataFrame(bi_data), bi_df], ignore_index=True,sort=True)
            bi_df.to_csv("model/"+a+"/"+crop_info["Crop_name"][b]+"/"+"data.csv", encoding='utf_8_sig')
            time.sleep(3000.0/1000.0)
    
    
    date_csv=pd.read_csv("data/date.csv")
    date_csv=date_csv.drop(["Unnamed: 0"], axis=1)
    data=[]
    get_year=str(now.year)
    get_month=str(now.month)
    get_day=str(now.day)
    if int(now.month)<10:
        get_month="0"+get_month
    if int(now.day)<10:
        get_day="0"+get_day
    data.append({"date":get_year+"-"+get_month+"-"+get_day})
    date_csv=pd.concat([date_csv,pd.DataFrame(data)], ignore_index=True,sort=True)
    date_csv.to_csv("data/蔬果/date.csv", encoding='utf_8_sig')
    
    
set_time()
