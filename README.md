# tip1
通用能力测试
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:35:17 2021

@author: 86132
"""


import pandas as pd
import numpy as np
import csv
from pandas.core.frame import DataFrame
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer as Imputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#1.加载数据集
filepath = 'D:\\PSBC11\\数分组工作\\Python学习资料\\Tips\\建模大赛\\data031417.csv'
df = pd.read_csv(filepath,delimiter=',',encoding = 'gb2312') #将建模数据集和预测数据集合并，方便后续统一处理


#2.特征（指标）唯一值、空值占比分析
def unqiue_element(x):
    a = len(np.unique(x))
    return a
df_desc= pd.DataFrame( [df.dtypes,df.astype(str).apply(unqiue_element),100*df.isnull().sum()/df.shape[0]]).T
df_desc.columns = ['数据类型', '数据集唯一值','数据集空值占比']
print(df_desc)
df_desc.to_csv('D:\\PSBC11\\数分组工作\\Python学习资料\\Tips\\建模大赛\\df_desc.csv') #导出特征（指标）唯一值、空值占比分析


#3.数据清洗
df.dtypes
print(df.head())
#应用Pandas和sklearn的OneHotEncoder和LabelEncoder对分类和顺序变量做转换
#3.1 第一步：下面几类变量需要先基于其中任意一个变量每类增加一个字符串变量，最后统一将字符串变量进行读热编码。
'''
第一类：{zc_15_max,jy_cnt_99,jy_amt_99,jy_cnt_15_3m,jy_cnt_15_2m,jy_cnt_15
,jy_cnt_01_3m,jy_cnt_01_2m,jy_cnt_01,jy_amt_15_3m,jy_amt_15_2m,jy_amt_15,jy_amt_01_3m
,jy_amt_01_2m,jy_amt_01}

第二类：
nucc_DBC_XF_MAX3m,nucc_DBC_XF_MAX2m,nucc_DBC_XF_MAX1m,nucc_DBC_XF_CNT1m,nucc_DBC_XF_AMT1m

第三类
p35_4_amt,p35_3_amt,p35_2_amt

第四类
p24_amt,p24_1_amt,p23_amt,p23,p22_amt,p22

第五类
p20_amt,p20,p19_amt,p19

第六类
p11_amt,p11,p10_amt,p10

第七类
OVRD_FLG,OVRD_AMT_USD,OVRD_AMT_RMB,CRDT_LMT_USD,CRDT_LMT_RMB,CRD_CARD_FLG

第八类
NTM_tranout_CNT3M,NTM_tranout_CNT2M,NTM_tranout_CNT1M,NTM_tranout_AMT3M
,NTM_tranout_AMT2M,NTM_tranout_AMT1M

第九类
NTM_TRANIN_CNT3M,NTM_TRANIN_CNT2M,NTM_TRANIN_CNT1M
,NTM_TRANIN_AMT3M,NTM_TRANIN_AMT2M,NTM_TRANIN_AMT1M

第十类
TM_TRANIN_CNT3M,TM_TRANIN_CNT2M,TM_TRANIN_CNT1M
,TM_TRANIN_AMT3M,TM_TRANIN_AMT2M,TM_TRANIN_AMT1M
'''


#增加字段 part1
df['dumm1']=df['jy_cnt_99'].isnull().apply(str)#根据zc_15_max是否为空增加一个字段，并转换为字符串格式
df['dumm2']=df['nucc_DBC_XF_MAX2m'].isnull().apply(str)
df['dumm3']=df['p35_2_amt'].isnull().apply(str)
df['dumm4']=df['p24_amt'].isnull().apply(str)
df['dumm5']=df['p20_amt'].isnull().apply(str)
df['dumm6']=df['p11_amt'].isnull().apply(str)
df['dumm7']=df['OVRD_FLG'].isnull().apply(str)
df['dumm8']=df['NTM_tranout_CNT3M'].isnull().apply(str)
df['dumm9']=df['NTM_TRANIN_CNT3M'].isnull().apply(str)
df['dumm10']=df['TM_TRANIN_CNT3M'].isnull().apply(str)

df_dumm=pd.concat((df['dumm1'],df['dumm2'],df['dumm3'],df['dumm4'],df['dumm5'],df['dumm6']
                   ,df['dumm7'],df['dumm8'],df['dumm9'],df['dumm10']), axis=1) #part1


#3.2 缺失值处理


#part2:数值型变量的缺失值处理(53)
#数值型变量包括：
'''
zc_15_max,jy_cnt_99,jy_amt_99,jy_cnt_15_3m,jy_cnt_15_2m,jy_cnt_15
,jy_cnt_01_3m,jy_cnt_01_2m,jy_cnt_01,jy_amt_15_3m,jy_amt_15_2m
,jy_amt_15,jy_amt_01_3m,jy_amt_01_2m,jy_amt_01,nucc_DBC_XF_MAX3m
,nucc_DBC_XF_MAX2m,nucc_DBC_XF_MAX1m,nucc_DBC_XF_CNT1m,nucc_DBC_XF_AMT1m
,p35_4_amt,p35_3_amt,p35_2_amt,p24_amt,p24_1_amt,p23_amt,p22_amt
,p20_amt,p19_amt,p11_amt,p10_amt,ACCT_AGE,OVRD_AMT_RMB,CRDT_LMT_USD
,CRDT_LMT_RMB,NTM_tranout_CNT3M,NTM_tranout_CNT2M,NTM_tranout_CNT1M,NTM_tranout_AMT3M
,NTM_tranout_AMT2M,NTM_tranout_AMT1M,NTM_TRANIN_CNT3M,NTM_TRANIN_CNT2M,NTM_TRANIN_CNT1M
,NTM_TRANIN_AMT3M,NTM_TRANIN_AMT2M,NTM_TRANIN_AMT1M,TM_TRANIN_CNT3M
,TM_TRANIN_CNT2M,TM_TRANIN_CNT1M,TM_TRANIN_AMT3M,TM_TRANIN_AMT2M,TM_TRANIN_AMT1M
'''
#以上变量全部用0填充缺失值,先拆分出来，便于填充，最后拼接
df_data1 = pd.concat((df.iloc[:,1:16], df.iloc[:,18:29],df.iloc[:,30:31]
                      , df.iloc[:,32:33], df.iloc[:,34:35], df.iloc[:,36:37], df.iloc[:,38:39]
                      , df.iloc[:,63:64], df.iloc[:,66:69], df.iloc[:,70:88]), axis=1)
df_data1_fill = df_data1.fillna(0) #part2
#df_data1_fill.dtypes
#print(df_data1_fill.head())


#part3:数值型变量先用特殊值填充，再转换为字符串型变量，待后续读热编码处理(22)
#需转换格式的变量包括：
'''
HIGH_SALARY_FLG,HIGH_PENSIONB_FLG,p23,p22,p20,p19,p11,p10
,PTY_RAT_RSLT,ACTV_IND,Pros_VIP_Ind_ZS2,Pros_VIP_Ind_ZS1
,Pros_VIP_Ind_JK1,Pros_VIP_Ind_BJ2,Pros_VIP_Ind_BJ1
,PROD_RISK_LVL_CD_L,PROD_HOLD_TERM_label,EDU_DEGR_CD2
,CHREM_PROD_DUE,OVRD_FLG,OVRD_AMT_USD,CRD_CARD_FLG
'''
df_data2=pd.concat((df.iloc[:,16:18],df.iloc[:,29:30],df.iloc[:,31:32],df.iloc[:,33:34],
                    df.iloc[:,35:36],df.iloc[:,37:38],df.iloc[:,39:41],df.iloc[:,48:56],
                    df.iloc[:,58:59],df.iloc[:,60:61],df.iloc[:,64:66],df.iloc[:,69:70]), axis=1)
df_data2_fill = df_data2.fillna(170314) #part3
#print(df_data2_fill.head())
df_data2_fill=df_data2_fill.astype(str)#全部dataframe的float格式转换为字符串格式str
#print(df_data2_fill.dtypes)


#part4:原字符串型变量的缺失值处理(7)
#字符串型变量包括
'''
JY_CNT_FLG_CNL17,JY_CNT_FLG_CNL15,JY_CNT_FLG_CNL12,JY_CNT_FLG_CNL01,JY_CNT_FLG
,AGE_OPEN_GROUP,AGE_GROUP
'''
df_data3=df.iloc[:,41:48]
df_data3_fill = df_data3.fillna('未知类型') #part4    
#df_data3.dtypes
#print(df_data3_fill.head())


#part5:字符串型,无缺失值，变量读热编码(4)
'''
MARRG_STAT_CD,GENDER_CD,CUST_LVL_CD,CARR_CD
'''
df_data4_fill=pd.concat((df.iloc[:,56:58],df.iloc[:,59:60]), axis=1) #part5
#df_data4_fill.dtypes
#part6 & part7
AUM_BAL=df[['AUM_BAL']]
fund_fix=df[['fund_fix']] #结果变量


#3.3 字符串型变量合并
df_str=pd.concat((pd.DataFrame(df_dumm),pd.DataFrame(df_data2_fill)
                   ,pd.DataFrame(df_data3_fill),pd.DataFrame(df_data4_fill)), axis=1)
#print(df_str.head())
#df_str.dtypes


#3.4 读热编码
#model_enc=OneHotEncoder()
#df_str_onehot1=model_enc.fit_transform(df_str).toarray()
df_str_onehot2=pd.get_dummies(df_str)
'''
将下面四部分数据合并
df_str_onehot2
df_data1_fill 
AUM_BAL
fund_fix #结果变量
'''
#3.5 合并数据
df_append=pd.concat((pd.DataFrame(df_str_onehot2),pd.DataFrame(df_data1_fill),AUM_BAL,fund_fix), axis=1)
#type(fund_fix)
#print(df_append.shape)
#df_append.head().to_csv('D:\\PSBC11\\数分组工作\\Python学习资料\\Tips\\建模大赛\\df_head.csv')
#
df_append.dtypes.to_csv('D:\\PSBC11\\数分组工作\\Python学习资料\\Tips\\建模大赛\\df_head_typenew.csv')
#3.6 异常值处理（保留）
#3.7 降维（保留）
#3.8 特征选择（保留）
#3.9 过采样（保留）
#3.10 数据标准化


#4.模型训练（随机森林）
#准备数据
#减小内存占用
df_append1=df_append.iloc[:,:154]
df_append2=df_append.iloc[:,154:]
df_append2 = df_append2.astype(np.float32)#float64转换为float32
df_append=pd.concat((pd.DataFrame(df_append1),pd.DataFrame(df_append2)), axis=1)
#样本、模型设置
df1=df_append.iloc[:138464,:] #测试集
#f1=df1.dropna() dumm1-3仍有缺失值？
df2=df_append.iloc[138463:,:] #预测数据集
text_y = df1['fund_fix'] #结果变量
text_x = df1.drop('fund_fix',axis=1)  #特征变量
x_train, x_test, y_train, y_test = train_test_split(text_x, text_y, test_size=0.3, random_state=101)
model = RandomForestClassifier(n_estimators=10,max_features='auto',random_state=101)
model.fit(x_train,y_train)#虚拟内存小，报错
preds = model.predict(x_test)
accuracy_score(y_test,preds) #建模数据集的acc
#nan_col=df_append.isnull().any()
#print(nan_col.head())

analyse_x = df2.drop('fund_fix',axis=1)  #预测数据集的特征变量
preds_result = model.predict(analyse_x)  #根据预测数据集特征变量做预测
preds_result = DataFrame(preds_result) #numpy.ndarray更改格式为DataFrame
preds_result.to_csv('D:\\PSBC11\\数分组工作\\Python学习资料\\Tips\\建模大赛\\df_result.csv') #输出结果
