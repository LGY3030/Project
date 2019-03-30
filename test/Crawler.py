
# coding: utf-8

# In[2]:


from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import json
import requests
import urllib
import time
import csv
import os


# In[123]:


place=["台北一"]
crop_num=["LA1"]
crop_name=["甘藍-初秋"]
col_name=["date","crop_num","crop_name","market_num","market_name","high","medium","low","mean","volume"] 
for a in place:
    for b in range(0,len(crop_name)):
        df=pd.DataFrame(columns=col_name)
        skip_num=0
        temp_str=""
        for j in range(0,1000): 
            url="https://data.coa.gov.tw/Service/OpenData/FromM/FarmTransData.aspx?$top=1000&$skip="+str(skip_num)+"&Crop="+urllib.parse.quote(crop_name[b])+"&StartDate=101.01.01&EndDate=108.03.29&Market="+urllib.parse.quote(a) 
            get = json.loads(urlopen(url).read().decode('utf-8')) 
            data = [] 
            for i in range(0,len(get)):
                temp_str=get[0]['交易日期']
                Date=get[i]['交易日期'].split('.')
                year,month,date=Date[0],Date[1],Date[2]
                year=str(int(year)+1911)
                get[i]['交易日期']=year+'/'+month+'/'+date
                if get[i]['作物代號']==crop_num[b]:
                    data.append({"date":get[i]['交易日期'],"crop_num":get[i]['作物代號'],"crop_name":get[i]['作物名稱'],"market_num":get[i]['市場代號'],"market_name":get[i]['市場名稱'],"high":get[i]['上價'],"medium":get[i]['中價'],"low":get[i]['下價'],"mean":get[i]['平均價'],"volume":get[i]['交易量']}) 
                else:
                    data.append({"date":get[i]['交易日期'],"crop_num":get[i]['作物代號'],"crop_name":get[i]['作物名稱'],"market_num":get[i]['市場代號'],"market_name":get[i]['市場名稱'],"high":0,"medium":0,"low":0,"mean":0,"volume":0}) 
            data.reverse() 
            df=pd.concat([pd.DataFrame(data), df], ignore_index=True,sort=True)
            if len(get)<1000: 
                break 
            skip_num=skip_num+1000 
            time.sleep(2000.0/1000.0)
        path=r"C:\Users\admin\Desktop\Project\data"+"\\"+a+"\\"+"蔬果"+"\\"+crop_name[b]+".csv"
        df["date"] = pd.to_datetime(df["date"])
        #df=df.drop(["date"], axis=1)
        df.to_csv("甘藍-初秋.csv", encoding='utf_8_sig')
        time.sleep(3000.0/1000.0)


# In[3]:


place_name=["雲林","嘉義","彰化","台南","高雄","屏東","台中","苗栗","桃園","台北"]
place_num=["C0K240","467480","C0G620","467410","467440","467590","467490","C0E420","C0C480","466920"]
date=[]
for year in ['2012','2013','2014','2015','2016','2017','2018']:
    for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
        date.append('-'.join([year,month]))
date.append('2019-01')
date.append('2019-02')
date.append('2019-03')
for a in range(0,len(place_name)):
    flag=0
    name=urllib.parse.quote(urllib.parse.quote(place_name[a]))
    for dd in date:
        url="http://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain"+"&station="+place_num[a]+"&stname="+name+"&datepicker="+str(dd)
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
            df=pd.DataFrame(columns=title)
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
        df=pd.concat([df,pd.DataFrame(fdata)], ignore_index=True,sort=True)
        df["date"] = pd.to_datetime(df["date"])
    path=r"C:\Users\admin\Desktop\Project\test"+"\\"+place_name[a]+".csv"
    df.to_csv(path, encoding='utf_8_sig')


# In[4]:


place=["雲林","嘉義","彰化","台南","高雄","屏東","台中","苗栗","桃園","台北"]
for pl in place:
    weather=pd.read_csv(pl+".csv")
    weather=weather[['最低氣溫(℃)','最大陣風(m/s)','最大陣風風向(360degree)','最小相對溼度(%)','最高氣溫(℃)','氣溫(℃)','測站最低氣壓(hPa)','測站最高氣壓(hPa)','相對溼度(%)','風向(360degree)','測站氣壓(hPa)','降水量(mm)','風速(m/s)','year','month','day','date']]
    weather=weather.drop(weather.index[[-1]])
    column_list=list(weather.columns.values)
    column_list.pop(-1)
    for i in column_list:
        for j in range(weather.shape[0]):
            if str(weather[i][j]).replace(".", "", 1).isdigit()==False:
                if j-1<0:
                    weather[i][j]=0
                else:
                    weather[i][j]=weather[i][j-1]
    weather.to_csv("changed"+pl+".csv", encoding='utf_8_sig')  


# In[6]:


place=["雲林","嘉義","彰化","台南","高雄","屏東","台中","苗栗","桃園","台北"]
for i in place:
    crop=pd.read_csv("甘藍-初秋.csv")
    weather=pd.read_csv("changed"+i+".csv")
    crop=crop.drop(["Unnamed: 0"], axis=1)
    weather=weather.drop(["Unnamed: 0"], axis=1)
    train=weather.merge(crop, on='date', how='left')
    train=train.fillna(0)
    train=train.drop(["crop_name"], axis=1)
    train=train.drop(["crop_num"], axis=1)
    train=train.drop(["market_name"], axis=1)
    train=train.drop(["market_num"], axis=1)
    train.to_csv("train"+i+".csv", encoding='utf_8_sig')  

