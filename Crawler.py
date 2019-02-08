#!/usr/bin/env python
# coding: utf-8

# In[45]:


from urllib.request import urlopen
import json
import pandas as pd
import urllib


# In[51]:


#def getdata():
crop_num=["A0","A1","A2","A3","B0","B1","B2","B4","B5","B6","B7","B9","C0","C1","C4","C6","C7","C9","E0","E1","E3","E9","F0","F1","F2","F4","F5","F9","G1","G2","G3","G6","G7","G8","G9","G49","H2","H3","H4","H41","H49","H5","H6",...."H0","H9","I0","I1","I2","I3","I4","J0","J1","J2","J3","J4","J5","J9","K0","K2","K3","K4","K9","K41","L1","L9","M0","M1","M2","M3","N0","N1","N2","N3","N4","N5","N6","N9","O0","O1","O2","O3","O4","O5,""O6","O7","O8","OV","OW","O9","O99","P0","P1","P2","P3","P4","P5","Q0","Q1","Q2","Q3","Q4","R0","R1","R2","R3","R4","R5","R6","R7","R8","S0","S1","S2","S4","S5","S9","S49","T0","T1","T2","T3","T4","T5","T6","T7","T9","V0","V1","V2","W0","W1","W2","W3","W4","W5","W6","W7","W9","X0","X1","X2","X3","X4","X09","X19","X29","X39","X49","X59","X69","X79","Y0","Y1","Y2","Y3","Y4","Y5","Y9","Y39","Z0","Z1","Z2","Z3","Z4","Z5","Z6","Z9","Z39","11","119","12","129","22","229","30","31","32","41","42","43","45","459","46","469","50","51","61","62","70","71","72","73","74","811","812","819","829","839","849","859","869","879","91","919"]
crop_name=["香蕉-其他","香蕉","香蕉-芭蕉","香蕉-旦蕉","鳳梨-其他","鳳梨-開英","鳳梨-金鑽鳳梨","鳳梨-鳳梨花","鳳梨-蘋果鳳梨","鳳梨-甜蜜蜜","鳳梨-牛奶鳳梨","鳳梨-進口","椪柑(其他)-其他","椪柑","紅柑-美女柑","佛利檬-佛利檬","豔陽柑-豔陽柑","椪柑-進口","甜橙-其他","甜橙-柳橙","甜橙-紅肉柳橙","甜橙-進口","雜柑-其他","雜柑-檸檬","雜柑-金棗","雜柑-桔子","雜柑-無子檸檬","雜柑-進口","蛋黃果-仙桃","黃金果","酪梨","香櫞-佛手","橄欖","栗子","波羅蜜","奇異果-進口","柚子-白柚","柚子-紅柚","葡萄柚-紅肉","葡萄柚-紅寶石","葡萄柚-進口紅肉","葡萄柚-白肉","西施柚-西施柚"]
col_name=["date","crop_num","crop_name","market_num","market_name","high","medium","low","mean","volume"]
df=pd.DataFrame(columns=name)
stop_flag=0
skip_num=0
for j in range(0,999):
    url="https://data.coa.gov.tw/Service/OpenData/FromM/FarmTransData.aspx?$top=1000&$skip="+str(skip_num)+"&Crop=%E6%A4%B0%E5%AD%90&StartDate=101.01.01&EndDate=108.02.08&Market=%E5%8F%B0%E5%8C%97%E4%BA%8C"
    get = json.loads(urlopen(url).read())
    data = []
    for i in range(0,999):
        if get[i]['作物代號']=="11":
            data.append({"date":get[i]['交易日期'],"crop_num":get[i]['作物代號'],"crop_name":get[i]['作物名稱'],"market_num":get[i]['市場代號'],"market_name":get[i]['市場名稱'],"high":get[i]['上價'],"medium":get[i]['中價'],"low":get[i]['下價'],"mean":get[i]['平均價'],"volume":get[i]['交易量']})
        if get[i]['交易日期']=="101.01.01":
            stop_flag=1
            break
    data.reverse()
    df=pd.concat([pd.DataFrame(data), df], ignore_index=True)
    skip_num=skip_num+1000
    if stop_flag==1:
        break
df.to_csv('test.csv', encoding='utf_8_sig')


# In[52]:


print(urllib.parse.quote('豔陽柑-豔陽柑'))


# In[ ]:




