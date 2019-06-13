

from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import json
import requests
import urllib
import time
import csv
import os
import datetime

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
        if now.hour==1 and flag==0:
            print(now)
            crawl_weather()
            crawl_oil()
            day_flag=today_date
            flag=1
        print("Waiting!!!")
        time.sleep(60)

def crawl_weather():
    place_name=["雲林","嘉義","彰化","台南","高雄","屏東","台中","苗栗","桃園","台北","新北","基隆","新竹","南投","宜蘭","花蓮","台東"]
    place_num=["C0K240","467480","C0G620","467410","467440","467590","467490","C0E420","C0C480","466920","466880","466940","C0D570","467650","467060","466990","467540"]


    get_csv=pd.read_csv("data/天氣/台北.csv")
    csv_date=get_csv["date"][get_csv.shape[0]-1]
    split_csv_date=str(csv_date).split("-")
    csv_date_year=split_csv_date[0]
    csv_date_month=split_csv_date[1]
    csv_date_day=split_csv_date[2]

    today_date=datetime.date.today() 
    oneday=datetime.timedelta(days=1) 
    yesterday_date=today_date-oneday
    split_yesterday_date=str(yesterday_date).split("-")
    yesterday_date_year=split_yesterday_date[0]
    yesterday_date_month=split_yesterday_date[1]
    yesterday_date_day=split_yesterday_date[2]

    date=[]

    if csv_date_month==yesterday_date_month:
        if int(yesterday_date_month)<10:
            date.append(csv_date_year+"-0"+str(int(csv_date_month)))
        else:
            date.append(csv_date_year+"-"+csv_date_month)
    else:
        for dev in range(int(yesterday_date_month)-int(csv_date_month)+1):
            if int(csv_date_month)+dev<10:
                date.append(csv_date_year+"-0"+str(int(csv_date_month)+dev))
            else:
                date.append(csv_date_year+"-"+str(int(csv_date_month)+dev))
    print(date)

    for a in range(0,len(place_name)):
        print(place_name[a])
        each_csv=pd.read_csv("data/天氣/"+place_name[a]+".csv")
        each_csv=each_csv.drop(["Unnamed: 0"], axis=1)
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
                if day<10:
                    data.append("0"+str(day))
                    data.append(sp[0]+'-'+sp[1]+'-0'+str(day))
                else:
                    data.append(str(day))
                    data.append(sp[0]+'-'+sp[1]+'-'+str(day))
                fdata.append({"測站氣壓(hPa)":data[0],"海平面氣壓(hPa)":data[1],"測站最高氣壓(hPa)":data[2],"測站最高氣壓時間(LST)":data[3],"測站最低氣壓(hPa)":data[4],"測站最低氣壓時間(LST)":data[5],"氣溫(℃)":data[6],"最高氣溫(℃)":data[7],"最高氣溫時間(LST)":data[8],"最低氣溫(℃)":data[9],"最低氣溫時間(LST)":data[10],"露點溫度(℃)":data[11],"相對溼度(%)":data[12],"最小相對溼度(%)":data[13],"最小相對溼度時間(LST)":data[14],"風速(m/s)":data[15],"風向(360degree)":data[16],"最大陣風(m/s)":data[17],"最大陣風風向(360degree)":data[18],"最大陣風風速時間(LST)":data[19],"降水量(mm)":data[20],"降水時數(hr)":data[21],"10分鐘最大降水量(mm)":data[22],"10分鐘最大降水起始時間(LST)":data[23]," 一小時最大降水量(mm)":data[24],"一小時最大降水量起始時間(LST)":data[25],"日照時數(hr)":data[26],"日照率(%)":data[27],"全天空日射量(MJ/㎡)":data[28],"能見度(km)":data[29],"A型蒸發量(mm)":data[30],"日最高紫外線指數":data[31],"日最高紫外線指數時間(LST)":data[32],"總雲量(0~10)":data[33],"year":data[34],"month":data[35],"day":data[36],"date":data[37]})
                day=day+1
                data=[]
            df=pd.concat([df,pd.DataFrame(fdata)], ignore_index=True,sort=True)
            time.sleep(2000.0/1000.0)
        weather=df[['最低氣溫(℃)','最大陣風(m/s)','最大陣風風向(360degree)','最小相對溼度(%)','最高氣溫(℃)','氣溫(℃)','測站最低氣壓(hPa)','測站最高氣壓(hPa)','相對溼度(%)','風向(360degree)','測站氣壓(hPa)','降水量(mm)','風速(m/s)','year','month','day','date']]
        column_list=list(weather.columns.values)
        column_list.pop(-1)
        for i in column_list:
            for j in range(weather.shape[0]):
                if str(weather[i][j]).replace(".", "", 1).isdigit()==False:
                    if j-1<0:
                        weather[i][j]=each_csv[i][each_csv.shape[0]-1]
                    else:
                        weather[i][j]=weather[i][j-1]
        index_last=weather[weather['date']==str(yesterday_date)].index.item()
        index_first=weather[weather['date']==str(csv_date)].index.item()

        weather=weather[index_first+1:index_last+1]

        each_csv=pd.concat([each_csv,pd.DataFrame(weather)], ignore_index=True,sort=True)

        path="data/天氣/"+place_name[a]+".csv"
        each_csv=each_csv[['最低氣溫(℃)','最大陣風(m/s)','最大陣風風向(360degree)','最小相對溼度(%)','最高氣溫(℃)','氣溫(℃)','測站最低氣壓(hPa)','測站最高氣壓(hPa)','相對溼度(%)','風向(360degree)','測站氣壓(hPa)','降水量(mm)','風速(m/s)','year','month','day','date']]
        each_csv.to_csv(path, encoding='utf_8_sig')

        date_csv=each_csv[["date"]]
        date_csv.to_csv("data/date.csv", encoding='utf_8_sig')
        time.sleep(4000.0/1000.0)

def crawl_oil():
    head_Html_lotto='https://web.cpc.com.tw/division/mb/oil-more4.aspx'
    res = requests.get(head_Html_lotto, timeout=30)
    soup = BeautifulSoup(res.text,'html.parser')
    a = soup.findAll(id="Showtd")
    b=a[0].findAll("tr")
    alldata=[]
    for i in b:
        c=i.findAll("td")
        data=[]
        for j in c:   
            data.append(j.text.strip())
        data=data[:5]
        alldata.append(data)
    column_name=["date","無鉛汽油92","無鉛汽油95","無鉛汽油98","超級/高級柴油"]
    df=pd.DataFrame(columns=column_name)
    data=[]
    for i in alldata:
        split_temp=i[0].split("/")
        i[0]=split_temp[0]+"-"+split_temp[1]+"-"+split_temp[2]
        data.append({"date":i[0],"無鉛汽油92":i[1],"無鉛汽油95":i[2],"無鉛汽油98":i[3],"超級/高級柴油":i[4]})

    data.reverse() 
    data=data[8:]

    df=pd.concat([pd.DataFrame(data), df], ignore_index=True,sort=True)


    origin_oil=pd.read_csv("data/油價/中油.csv")
    origin_oil=origin_oil.drop(["Unnamed: 0"], axis=1)
    date_csv=pd.read_csv("data/date.csv")
    date_csv=date_csv.drop(["Unnamed: 0"], axis=1)
    oil_csv=date_csv.merge(df, on='date', how='left')
    oil_csv=oil_csv.convert_objects(convert_numeric=True)
    oil_csv=oil_csv.fillna(0)
    if oil_csv["無鉛汽油92"][origin_oil.shape[0]]==0:
        oil_csv["無鉛汽油92"][origin_oil.shape[0]]=origin_oil["無鉛汽油92"][origin_oil.shape[0]-1]
    if oil_csv["無鉛汽油95"][origin_oil.shape[0]]==0:
        oil_csv["無鉛汽油95"][origin_oil.shape[0]]=origin_oil["無鉛汽油95"][origin_oil.shape[0]-1]
    if oil_csv["無鉛汽油98"][origin_oil.shape[0]]==0:
        oil_csv["無鉛汽油98"][origin_oil.shape[0]]=origin_oil["無鉛汽油98"][origin_oil.shape[0]-1]
    if oil_csv["超級/高級柴油"][origin_oil.shape[0]]==0:
        oil_csv["超級/高級柴油"][origin_oil.shape[0]]=origin_oil["超級/高級柴油"][origin_oil.shape[0]-1]

    for i in range(origin_oil.shape[0],oil_csv.shape[0]):
        if oil_csv["無鉛汽油92"][i]==0:
            oil_csv["無鉛汽油92"][i]=oil_csv["無鉛汽油92"][i-1]
        if oil_csv["無鉛汽油95"][i]==0:
            oil_csv["無鉛汽油95"][i]=oil_csv["無鉛汽油95"][i-1]
        if oil_csv["無鉛汽油98"][i]==0:
            oil_csv["無鉛汽油98"][i]=oil_csv["無鉛汽油98"][i-1]
        if oil_csv["超級/高級柴油"][i]==0:
            oil_csv["超級/高級柴油"][i]=oil_csv["超級/高級柴油"][i-1]

    oil_csv=oil_csv[origin_oil.shape[0]:]
    oil_csv=pd.concat([pd.DataFrame(origin_oil), oil_csv], ignore_index=True,sort=True)
    oil_csv.to_csv("data/油價/中油.csv", encoding='utf_8_sig')

set_time()
