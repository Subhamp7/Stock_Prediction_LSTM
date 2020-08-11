# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 23:11:25 2020

@author: subham
"""
#importing libraries
try:
    from requests import get
    from bs4 import BeautifulSoup
    import pandas as pd
except Exception as e:
    print("Error while loading the libraies",e)

#initilizing the list to store data
infy_stock=[]
present_date="2020-08-11"

#iterating through pages
for index in range(1,11,1):
    URL="https://www.moneycontrol.com/stocks/hist_stock_result.php?sc_id=IT&pno={}&hdn=daily&fdt=2001-01-01&todt={}".format(str(index),present_date)
    page=get(URL)
    print("Connected page {} : {}  ".format(index,page.status_code==200))
    
    soup=BeautifulSoup(page.text, 'html.parser')
    stocks=(soup.find('div',class_="MT12")).find_all('tr')
    
    for index_1 in stocks:
        td = index_1.find_all('td')
        rows = [index_2.text.replace('\n','') for index_2 in td]
        infy_stock.append(rows)
        
#storing the data to a Dataframe and converting to csv
dataset=pd.DataFrame(infy_stock, columns=['Date',"Open","High","Low","Close","Volume","High-Low","Open-Close"])
dataset.to_csv("infy_stock.csv")