#!/usr/bin/env python
# coding: utf-8

# In[18]:


from urllib.request import urlopen
import json
import pandas as pd


# In[17]:


#def getdata():
name=["date","crop_num","crop_name","market_num","market_name","high","medium","low","mean","volume"]
url="https://data.coa.gov.tw/Service/OpenData/FromM/FarmTransData.aspx?$top=1000&$skip=0&Crop=%E6%A4%B0%E5%AD%90&StartDate=107.02.01&EndDate=107.11.29&Market=%E5%8F%B0%E5%8C%97%E4%BA%8C"
data = json.loads(urlopen(url).read())
a=data[20]['交易日期']
print(a)


# In[ ]:


"交易日期": "108.01.25",
"作物代號": "IC408",
"作物名稱": "進口大菊-粉乒乓",
"市場代號": "105",
"市場名稱": "台北市場",
"上價": 185.0,
"中價": 185.0,
"下價": 185.0,
"平均價": 185.0,
"交易量": 24.0

