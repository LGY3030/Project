#!/usr/bin/env python
# coding: utf-8

# In[28]:


from urllib.request import urlopen
import json
import pandas as pd
import urllib
import time
import csv


# In[7]:


place=["台北一","台北二","板橋區","三重區","桃農","台中","豐原","東勢","嘉義","高雄","鳳山區","台東","宜蘭","南投"]
crop_num=["A0","A1","A2","A3","B0","B1","B2","B4","B5","B6","B7","B9","C0","C1","C4","C6","C7","C9","E0","E1","E3","E9","F0","F1","F2","F4","F5","F9","G1","G2","G3","G6","G7","G8","G9","G49","H2","H3","H4","H41","H49","H5","H6","I0","I1","I2","I3","I4","J0","J1","J2","J3","J4","J5","K0","K2","K3","K4","K41","L1","L9","M0","M2","M3","N0","N1","N2","N3","N4","N5","N6","N9","O0","O1","O2","O3","O4","O5","O7","O8","OV","OW","O9","O99","P0","P1","P2","P3","P4","P5","Q0","Q1","Q2","Q3","Q4","R0","R1","R2","R3","R4","R5","R6","R7","R8","S1","S2","S4","S9","S49","T0","T1","T2","T3","T4","T5","T6","T7","T9","V1","V2","W0","W1","W2","W4","W5","W6","W7","W9","X0","X1","X2","X3","X09","X19","X29","X39","X49","X59","X69","Y0","Y1","Y2","Y3","Y4","Y5","Y9","Z0","Z1","Z2","Z3","Z4","Z5","Z6","Z9","11","119","12","129","22","31","32","41","42","43","45","459","469","50","51","61","70","71","72","73","74","811","812","819","839","849","859"]
crop_name=["香蕉-其他","香蕉","香蕉-芭蕉","香蕉-旦蕉","鳳梨-其他","鳳梨-開英","鳳梨-金鑽鳳梨","鳳梨-鳳梨花","鳳梨-蘋果鳳梨","鳳梨-甜蜜蜜","鳳梨-牛奶鳳梨","鳳梨-進口","椪柑(其他)-其他","椪柑","紅柑-美女柑","佛利檬-佛利檬","豔陽柑-豔陽柑","椪柑-進口","甜橙-其他","甜橙-柳橙","甜橙-紅肉柳橙","甜橙-進口","雜柑-其他","雜柑-檸檬","雜柑-金棗","雜柑-桔子","雜柑-無子檸檬","雜柑-進口","蛋黃果-仙桃","黃金果","酪梨","香櫞-佛手","橄欖","栗子","波羅蜜","奇異果-進口","柚子-白柚","柚子-紅柚","葡萄柚-紅肉","葡萄柚-紅寶石","葡萄柚-進口紅肉","葡萄柚-白肉","西施柚-西施柚","木瓜-其他","木瓜-網室紅肉","木瓜-一般紅肉","木瓜-日昇種","木瓜-青木瓜","荔枝-其他","荔枝-玉荷包","荔枝-黑葉","荔枝-糯米","荔枝-竹葉黑","荔枝-桂味","龍眼-其他","龍眼-十月眼","龍眼-粉殼","龍眼-龍眼乾帶殼","龍眼-龍眼肉","枇杷-茂木","枇杷-進口","楊桃-其他水晶","楊桃-紅龍","楊桃-馬來亞種","李-其他","李-沙蓮李","李-桃接李","李-紅肉李","李-黃肉李","李-加州李","李-泰安李","李-進口","梨-其他","梨-橫山梨","梨-秋水梨","梨-世紀梨","梨-新興梨","梨-豐水梨","梨-鳥梨","梨-4029梨","梨-蜜梨","梨-雪梨","梨-進口","西洋梨-西洋梨進口","番石榴-其他","番石榴-珍珠芭","番石榴-紅心","番石榴-帝王芭","番石榴-世紀芭","番石榴-水晶無仔","蓮霧-其他","蓮霧-紅蓮霧","蓮霧-子彈型","蓮霧-翠玉","蓮霧-巴掌蓮霧","芒果-其他","芒果-愛文","芒果-紅龍玉文","芒果-本島","芒果-凱特","芒果-黑香金興","芒果-金煌","芒果-聖心","芒果-芒果青","葡萄-巨峰","葡萄-意大利","葡萄-進口無子","葡萄-進口","葡萄-進口無子","西瓜-其他","西瓜-大西瓜","西瓜-無子西瓜","西瓜-黑美人","西瓜-鳳光、英妙","西瓜-黃肉","西瓜-紅肉","西瓜-秀鈴","西瓜-進口","甜瓜-美濃","甜瓜-溫室吊瓜","洋香瓜-其他","洋香瓜-網狀紅肉","洋香瓜-網狀綠肉","洋香瓜-新疆","洋香瓜-光面紅肉","洋香瓜-光面綠肉","洋香瓜-光面白肉","洋香瓜-進口","蘋果-其他","蘋果-五爪","蘋果-秋香","蘋果-惠","蘋果-其他(進口)","蘋果-五爪(進口)","蘋果-秋香(進口)","蘋果-惠(進口)","蘋果-金冠(進口)","蘋果-紅玉(進口)","蘋果-富士(進口)","桃子-其他","桃子-水蜜桃","桃子-鶯歌桃","桃子-甜桃","桃子-早桃","桃子-福壽桃","桃子-進口","柿子-其他","柿子-紅柿","柿子-水柿","柿子-柿餅","柿子-甜柿","柿子-筆柿","柿子-秋柿","柿子-進口","椰子","椰子-進口","椰子-剝殼","椰子-進口剝殼","棗子","釋迦","釋迦-鳳梨釋迦","梅","楊梅","桑椹","草莓","草莓-進口","藍莓-進口藍莓","百香果-其他","百香果-改良種","甘蔗-帶皮","小番茄-其他","小番茄-一般","小番茄-聖女","小番茄-嬌女","小番茄-玉女","火龍果-白肉","火龍果-紅肉","火龍果-進口","櫻桃-進口","石榴-進口","榴槤-進口",] 
col_name=["date","crop_num","crop_name","market_num","market_name","high","medium","low","mean","volume"] 
for a in place:
    for b in range(0,len(crop_name)):
        df=pd.DataFrame(columns=col_name)
        skip_num=0
        temp_str=""
        for j in range(0,1000): 
            url="https://data.coa.gov.tw/Service/OpenData/FromM/FarmTransData.aspx?$top=1000&$skip="+str(skip_num)+"&Crop="+urllib.parse.quote(crop_name[b])+"&StartDate=101.01.01&EndDate=108.02.08&Market="+urllib.parse.quote(a) 
            get = json.loads(urlopen(url).read()) 
            data = [] 
            for i in range(0,len(get)):
                temp_str=get[0]['交易日期']
                if get[i]['作物代號']==crop_num[b]:
                    Date=get[i]['交易日期'].split('.')
                    year,month,date=Date[0],Date[1],Date[2]
                    year=str(int(year)+1911)
                    get[i]['交易日期']=year+'/'+month+'/'+date
                    data.append({"date":get[i]['交易日期'],"crop_num":get[i]['作物代號'],"crop_name":get[i]['作物名稱'],"market_num":get[i]['市場代號'],"market_name":get[i]['市場名稱'],"high":get[i]['上價'],"medium":get[i]['中價'],"low":get[i]['下價'],"mean":get[i]['平均價'],"volume":get[i]['交易量']}) 
            data.reverse() 
            df=pd.concat([pd.DataFrame(data), df], ignore_index=True,sort=True)
            if len(get)<1000: 
                break 
            skip_num=skip_num+1000 
            time.sleep(2000.0/1000.0)
        path=r"C:\Users\admin\Desktop\Project\data"+"\\"+a+"\\"+crop_name[b]+".csv"
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["dayofweek"] = df["date"].dt.dayofweek
        df=df.drop(["date"], axis=1)
        df.to_csv(path, encoding='utf_8_sig')
        time.sleep(3000.0/1000.0)


# In[70]:


import requests
from time import sleep
from bs4 import BeautifulSoup
import csv
import json
import os

date=[]
for year in ['2012']:#,'2013','2014','2015','2016','2017','2018']:
    for month in ['01']:#, '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
        date.append('-'.join([year,month]))
a=urllib.parse.quote(urllib.parse.quote("臺北"))
for dd in date:
    url="http://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain"+"&station="+"466920"+"&stname="+a+"&datepicker="+str(dd)
    json_data = {}

    resp = requests.get(url)
    soup = BeautifulSoup(resp.text)
    trs = soup.findAll('tr')
    ths = trs[2].findAll('th')
    for th in ths:
        print(th.text)
    '''
    title = [th.text.split(')')[1] for th in ths]

    for tr in trs[3:]:
        tds = tr.findAll('td')

        row = [td.text.strip() for td in tds]

        dictionary = mapping_two_list_to_dict(title, row)

        json_data[dictionary['ObsTime']] = dictionary
    json_data.to_csv('path', encoding='utf_8_sig')  
    '''


# In[ ]:




