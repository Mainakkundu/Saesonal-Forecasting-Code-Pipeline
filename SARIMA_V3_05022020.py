# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:47:29 2020

@author: mainak.kundu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:13:04 2020

@author: mainak.kundu
"""

import pandas as pd 
import numpy as np 
import os
from statsmodels.tsa.stattools import adfuller
#from pyramid.arima import auto_arima
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
#from pmdarima import auto_arima
from pyramid.arima import auto_arima ## comment 
import seaborn as sns
import statsmodels.api as sm
import sys
from pyramid.arima.utils import ndiffs,nsdiffs
#from pmdarima.arima.utils import ndiffs,nsdiffs


#p1 = sys.argv[1]  ## input path 
#p2 = sys.argv[2] ## output path 
#p3 = sys.argv[3] ## usr
#p4 = sys.argv[4] ## cncpt
#p5 = sys.argv[5] ## dept



def StationarityTest(df, ts):
    """
    Test stationarity using moving average statistics and Dickey-Fuller test
    """
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(df[ts])
                      
    dfoutput = pd.Series(dftest[0:4], 
                         index = ['Test Statistic',
                                  'p-value',
                                  '# Lags Used',
                                  'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    p_value = dfoutput[1]
    return p_value


def SeasonalityChk_Modf(h):
    data = pd.DataFrame()
    #snl_chk = []
    for obs in range(1,105):
        data["T_" + str(obs)] = h.RTL_QTY.shift(obs)
    data.fillna(0)
    c = data.corr()
    k = []
    for i in c.iloc[0]:
        if i >= 0.5:
            k.append('correlated')
        else:
            k.append('non-correlated')
    seasonal_index =[]
    for index,i in enumerate(k):
        if i == 'correlated':
            seasonal_index.append(index)
    print(seasonal_index)
    s_idx = [51,52,53,54,55,56,57,58,101,102,103,104,105,106,107,108] ## hardcoded list matches with the correlated lags 
    retn = list(set(seasonal_index).intersection(s_idx)) ## matches 
    perecentage = len(retn)/len(s_idx)*100
    print(perecentage)
    #df['SEASONALITY_EXIST'] = k
    #df['SEASONAL'] = np.where(df['SEASONALITY_EXIST']>0.75,'yes','no')
    return perecentage


def SeasonalityChk_Modf1(h):
    data = pd.DataFrame()
    #snl_chk = []
    for obs in range(1,105):
        data["T_" + str(obs)] = h.RTL_QTY.shift(obs)
    data.fillna(0)
    c = data.corr()
    k = []
    for i in c.iloc[0]:
        if i >= 0.5:
            k.append('correlated')
        else:
            k.append('non-correlated')
    seasonal_index =[]
    for index,i in enumerate(k):
        if i == 'correlated':
            seasonal_index.append(index)
    print(seasonal_index)
    s_idx = [51,52,53,54,55,56,57,58,101,102,103,104,105,106,107,108] ## hardcoded list matches with the correlated lags 
    retn = list(set(seasonal_index).intersection(s_idx)) ## matches 
    perecentage = len(retn)/len(s_idx)*100
    print(perecentage)
    #df['SEASONALITY_EXIST'] = k
    #df['SEASONAL'] = np.where(df['SEASONALITY_EXIST']>0.75,'yes','no')
    return perecentage


def St_tagger(df,ts):
    tagg =[]
    p_value = StationarityTest(df,ts)
    if p_value > 0.05:
            sales = histData['RTL_QTY']
            sales =  np.where(sales==0,1,sales)
            sales = np.log(sales)
            histData['RTL_QTY_tr'] = sales
            p_v = StationarityTest(histData,'RTL_QTY_tr')
            if p_v > 0.05:
                tagg.append('DIFF')
            else:
                tagg.append('LOG_TRS')
    else:
        tagg.append('ST')
    return tagg


def SeasonalSimulation(sales):
    stepwise_seasonal_model = auto_arima(sales, max_p=2, max_q=2,
                           m=52,
                           start_P=0, seasonal=True,max_P=2,max_Q=2,start_Q=0,
                           d=0, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           )
    print(stepwise_seasonal_model.aic())
    
    seasonal_order = stepwise_seasonal_model.seasonal_order
    order = stepwise_seasonal_model.order
    #test_prd = stepwise_seasonal_model.predict(n_periods=52)
    return seasonal_order,order

def split_s_order(s_order):
    P=int(s_order.split(',')[0].split('(')[1])
    D=int(s_order.split(',')[1])
    Q=int(s_order.split(',')[2])
    m=int(s_order.split(',')[3].split(')')[0])
    return P,D,Q,m

def split_order(order):
    p=int(order.split(',')[0].split('(')[1])
    d=int(order.split(',')[1])
    q=int(order.split(',')[2].split(')')[0])
    return(p,d,q)

def NonSeasonalSimulation(sales):
    stepwise_seasonal_model = auto_arima(sales, max_p=2, max_q=2,
                           m=52,
                           seasonal=False,
                           d=1,trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           )
    print(stepwise_seasonal_model.aic())
    order = stepwise_seasonal_model.order
    #test_prd = stepwise_seasonal_model.predict(n_periods=52)
    return order






if __name__== '__main__':
    os.chdir('D:\PROJECTS\IM\SM_FORECASTING_12022020')
    #hads = pd.read_csv('BA_HADS_31MAR2019.txt')
    #lads = pd.read_csv('BA_LADS_31MAR2019.txt')
    #text_file = open('/shared/sasusers/mainakk/file_ckeck.txt','w')
    #text_file.writelines(p1)
    #text_file.writelines(p2)
    #text_file.writelines(p3)
    #text_file.writelines(p4)
    #text_file.writelines(p5)
    #text_file.close()
    import time
    import sys
    start = time.time()
    #input_path = p1
    #output_path = p2 
    #print(input_path)
    #path = str(input_path)
    #print(input_path)
    #path = str(input_path)
    #USR = p3 ## args
    #CNCPT = p4 ## args
    #DEPT = p5 ## args
    #logs = open(USR+'_'+CNCPT+'_'+DEPT+'_'+'SARIMA_LOGS.txt','w')
    #sys.stdout = logs
    print('---Path Declaration---')
    #INPUT_PATH = p1+'/'+USR+'_'+CNCPT+'_'+DEPT
    #OUTPUT_PATH = p2+'/'+USR+'_'+CNCPT+'_'+DEPT
    #hads_path = 
    #lads_path = OUTPUT_PATH+'_FCST_LADS.txt'
    #print(hads_path)
    #print(lads_path)
    #exit()
    h = pd.read_csv('PYL_SM_LDS_FCST_HADS.txt')
    l = pd.read_csv('PYL_SM_LDS_FCST_LADS.txt')
    
    
    #h = hads
    #l = lads

    print('=== Stationarity Tagging ===')
    resultF    = pd.DataFrame()
    terr = [v for v in h['STND_TRRTRY_NM'].unique()]
    for ter in terr:
        terrData   = h[h.STND_TRRTRY_NM==ter]
        optionList = list(set(terrData.KEY))
        #optionList = optionList[0:4]
        print(len(optionList))
        for ind,option in enumerate(optionList):
            
            result   = pd.DataFrame()
            print('In progress' +'\n'+ter + '-----'+option )
            histData = h[(h.STND_TRRTRY_NM==ter)& (h.KEY==option)]
            if len(histData) > 52:
                t = St_tagger(histData,'RTL_QTY')
                result['ST_TGGR'] = t
                result['STND_TRRTRY_NM'] = ter
                result['KEY'] = option
            else:
                result['ST_TGGR'] ='NOTH'
                result['STND_TRRTRY_NM'] = ter
                result['KEY'] = option
            resultF = pd.concat([result,resultF])
            print('== Done Stationarity Tagging ==')   
        
    print(h.shape,resultF.shape)
    h1 = h.merge(resultF,how='left',on=['STND_TRRTRY_NM','KEY'])
    print(h1.shape) ### Join (1)

    print('==== Option Level Seasonal Tagging  ====')
    seasF    = pd.DataFrame()
    terr = [v for v in h1['STND_TRRTRY_NM'].unique()]
    for ter in terr:
            terrData   = h1[h1.STND_TRRTRY_NM==ter]
            optionList = list(set(terrData.KEY))
            #optionList = optionList[0:5]
            print(len(optionList))
            for ind,option in enumerate(optionList):
                result   = pd.DataFrame()
                print('In progress' +'\n'+ter + '-----'+option )
                histData = h1[(h1.STND_TRRTRY_NM==ter)& (h1.KEY==option)]
                if len(histData) > 104:
                    per = SeasonalityChk_Modf(histData)
                    histData['SEASONAL_OVERLAP'] = per                         
                    result = histData
                    seasF = pd.concat([result,seasF])
    print('===SEASONAL_COMPONENTS TAGGING COMPLETE ===')

    k = seasF.groupby(['STND_TRRTRY_NM','SUB_CLSS_NM'])['SEASONAL_OVERLAP'].agg(pd.Series.max).reset_index()
    k['IS_SNL'] = np.where(k['SEASONAL_OVERLAP']>0,'NO','YES') ### this will help to run auto.arima 
    h1 = h1.merge(k,how='left',on=['STND_TRRTRY_NM','SUB_CLSS_NM'])
    print(h1.shape)
    h1.to_csv('USR_CNCPT_DEPT_HADS.txt') ### this will consume by auto.arima code 
    
    print('== Seasonal Data Bifarcated ==')
    h_snl = h1[h1['IS_SNL']=='YES'] ### roll up that data and do sub-class level simulation 
    print(h_snl.shape)

    print('== Seasonal Simulation starts on Sub-class-Level ==')
    order_temp    = []
    new_h = h_snl.groupby(['STND_TRRTRY_NM','SUB_CLSS_NM','TRDNG_WK_END_DT'])['RTL_QTY'].mean().reset_index() ## sub class level rollup
    terr = [v for v in new_h['STND_TRRTRY_NM'].unique()]
    for ter in terr:
        terrData   = new_h[new_h.STND_TRRTRY_NM==ter]
        clss_lst = list(set(terrData.SUB_CLSS_NM))
        for i in clss_lst:
            print('In progress' +'\n'+ter + '-----'+str(i) )
            histData = new_h[(new_h.STND_TRRTRY_NM==ter)& (new_h.SUB_CLSS_NM==i)]
            if len(histData) >= 104:
                sales = histData['RTL_QTY']
                sales = np.where(sales==0,1,sales)
                sales = np.log(sales)
                seasonal_order,order= SeasonalSimulation(sales)
                order_temp.append(ter +"_"+str(i)+"_"+str(seasonal_order)+"_"+str(order)) 
                print('===SEASONAL SIMULATIONS TAGGING COMPLETE ===')
    orderdf=pd.DataFrame()
    orderdf['STND_TRRTRY_NM']=[f.split('_')[0] for f in order_temp]
    orderdf['SUB_CLSS_NM']=[f.split('_')[1] for f in order_temp]
    orderdf['S_ORDER']=[f.split('_')[2] for f in order_temp]
    orderdf['ORDER']=[f.split('_')[3] for f in order_temp]
    print(orderdf.dtypes)
    print(orderdf.head(2))
    orderdf['P'],orderdf['D'],orderdf['Q'],orderdf['m']=zip(*orderdf['S_ORDER'].map(split_s_order))
    
    h_snl = h_snl.merge(orderdf,how='left',on=['STND_TRRTRY_NM','SUB_CLSS_NM']) #### S_ORDER join 
    h_snl.shape

    print('==-Option Level Seasonal Simulation Starts for (p,d,q) ===')
    order_temp    = []
    terr = [v for v in h_snl['STND_TRRTRY_NM'].unique()]
    for ter in terr:
        terrData   = h_snl[h_snl.STND_TRRTRY_NM==ter]
        optionList = list(set(terrData.KEY))
        for i in optionList:
            print('In progress' +'\n'+ter + '-----'+str(i) )
            histData = h_snl[(h_snl.STND_TRRTRY_NM==ter)& (h_snl.KEY==i)]
            if len(histData) >= 104:
                sales = histData['RTL_QTY']
                sales = np.where(sales==0,1,sales)
                sales = np.log(sales)
                order= NonSeasonalSimulation(sales)
                order_temp.append(ter +"_"+str(i)+"_"+str(order)) 
                print('===SEASONAL SIMULATIONS TAGGING COMPLETE ===')
    orderdf=pd.DataFrame()
    orderdf['STND_TRRTRY_NM']=[f.split('_')[0] for f in order_temp]
    orderdf['KEY']=[f.split('_')[1] for f in order_temp]
    orderdf['ORDER']=[f.split('_')[2] for f in order_temp]
    print(orderdf.dtypes)
    print(orderdf.head(2))
    orderdf['p'],orderdf['d'],orderdf['q']=zip(*orderdf['ORDER'].map(split_order))
    h_snl = h_snl.merge(orderdf,how='left',on=['STND_TRRTRY_NM','KEY'])
    print(h_snl.shape) ### last join 
    del h_snl['d'] 
    del h_snl['D'] 
    #h_snl.to_csv('h_snl_BA.csv')
    #----------------------------------
    #exit()
    print('==-Order of (d,D) ===')
    r    = []
    terr = [v for v in h_snl['STND_TRRTRY_NM'].unique()]
    for ter in terr:
        terrData   = h_snl[h_snl.STND_TRRTRY_NM==ter]
        optionList = list(set(terrData.KEY))
        for i in optionList:
            print('In progress' +'\n'+ter + '-----'+str(i) )
            r1 = pd.DataFrame()
            histData = h_snl[(h_snl.STND_TRRTRY_NM==ter)& (h_snl.KEY==i)]
            if len(histData) >= 52:
                sales = histData['RTL_QTY']
                try:
                    d = int(ndiffs(sales, test='adf')) 
                    D = int(nsdiffs(sales,m=52,max_D=5))
                    r.append(ter +"_"+str(i)+"_"+str(d)+"_"+str(D))
                    print('== Difference algo works ==')
                except:
                    d =1
                    D =0
                    r.append(ter +"_"+str(i)+"_"+str(d)+"_"+str(D))
                    print('== Difference algo fails  ==')
                
            else:
                histData['d']= 1
                histData['D'] = 0
                r.append(ter +"_"+str(i)+"_"+str(d)+"_"+str(D))
                print('== Less history ==')

    orderdf=pd.DataFrame()
    orderdf['STND_TRRTRY_NM']=[f.split('_')[0] for f in r]
    orderdf['KEY']=[f.split('_')[1] for f in r]
    orderdf['d']=[f.split('_')[2] for f in r]
    orderdf['D']=[f.split('_')[3] for f in r]
    print(orderdf.dtypes)
    print(orderdf.head(2))            
                
    h_snl = h_snl.merge(orderdf,how='left',on=['STND_TRRTRY_NM','KEY'])  ### last join 
    
    #---------------------------------------------------------------------
            
                
                
                
    
    
    
    
    print('== Parameters found run S-ARIMA execution ==')
    resultF    = pd.DataFrame()
    terrData   = h_snl[h_snl.STND_TRRTRY_NM==ter]
    terr = [v for v in h_snl['STND_TRRTRY_NM'].unique()]
    for ter in terr:
        histDataset1   = h_snl[h_snl.STND_TRRTRY_NM==ter]
        leadDataset    = l[l.STND_TRRTRY_NM==ter]
        optionList = list(set(terrData.KEY))
        print(len(optionList))
        #optionList= optionList[0]
        for ind,option in enumerate(optionList):
            result   = pd.DataFrame()
            print('In progress' +'\n'+ter + '-----'+option )
            histData = histDataset1[histDataset1.KEY==option]
            leadData = leadDataset[leadDataset.KEY==option]            
            sales = histData['RTL_QTY']
            #d = int(ndiffs(sales, test='adf')) 
            #D = int(nsdiffs(sales,m=52,max_D=1))
            sales = np.where(sales==0,1,sales)
            sales = np.log(sales)
            if (len(histData) >= 104):
                
                #d = int(ndiffs(sales, test='adf')) 
                #D = int(nsdiffs(sales,m=52,max_D=1))
                print(len(histData),len(leadData))
                p = int(histData['p'].iloc[ind])
                d = int(histData['d'].iloc[ind])
                q = int(histData['q'].iloc[ind])
                P = int(histData['P'].iloc[ind])
                D = int(histData['D'].iloc[ind])
                Q = int(histData['Q'].iloc[ind])
                m = int(histData['m'].iloc[ind])
                print(p,d,q)
                print(P,D,Q,m)
                mod = sm.tsa.statespace.SARIMAX(sales,order=(p,d,q),
                                                enforce_invertibility = False,
                                                enforce_stationarity = False,
                                seasonal_order=(P,D,Q,m))
                print(mod.seasonal_order)
                print(mod.order)
                results = mod.fit()
                pred = results.get_forecast(steps=52)
                sarima_mean =pred.predicted_mean
                leadData['S_ARIMA_FCST'] = sarima_mean
                leadData['S_ARIMA_FCST'] = np.exp(leadData['S_ARIMA_FCST'])
                print(leadData['S_ARIMA_FCST'].max(),leadData['S_ARIMA_FCST'].min())
            else:
                leadData['S_ARIMA_FCST'] =0 
            result = leadData
            resultF = pd.concat([resultF,result])
            cols = ['STND_TRRTRY_NM','KEY','TRDNG_WK_END_DT','RTL_QTY','S_ARIMA_FCST']        
            resultF = resultF[cols]
            #resultF.to_csv(USR+'_'+CNCPT+'_'+DEPT+'_'+'SARIMA_PYOUT2020.txt')
            resultF.to_csv('sarima_sm.csv')
            print('==DONE==')
    print('==Complete==')



