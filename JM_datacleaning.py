import numpy as np
import pandas as pd
import os
import gc
import myfunc
import re
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from fuzzywuzzy import process

######################################导入源数据
data = pd.read_excel('e:\\jm\\附件1.xlsx')
data.to_csv('e:\\jm\\data1.csv')

data1 = pd.read_csv('e:\\jm\\data1.csv')
data2 = pd.read_sas('e:\\jm\\data.sas7bdat')

myfunc.search('map')
#######################################导入数据字典
lineData = []
with open('e:\\jm\\数据字典.txt') as txtData:
    lines = txtData.readlines()
    for line in lines:
        lineData = lineData + [line.strip()]


def var_dic(n):
    var_lx = lineData[n].replace(')', '）').replace('(', '（').replace(' ', '').split('、')
    t = [re.split('[（）]', x) for x in var_lx]
    var_lx = dict()
    for x in t:
        var_lx[x[1]] = x[0]
    return var_lx


var_lx = var_dic(0)  # 连续变量字典
var_str = var_dic(1)  # 字符变量字典
var_ls = var_dic(2)  # 离散变量字典
var_dictionary = dict(var_lx, **var_str, **var_ls)
var_dictionary['guncertain1']
#########################各个指标的缺失值比率统计
var = data.columns
var_N = len(var)
length = len(data)


def find_miss(data):
    var_missRate = []
    var = data.columns
    for x in var:
        t = sum(pd.isnull(data[x])) / length
        var_missRate = var_missRate + [t]
    return pd.Series(var_missRate, index=var)


var_missRate = find_miss(data)

#########################将缺失值比率大于80%的变量进行删除
print(len(var[np.where(var_missRate > 0.8)]))
var_del = var[np.where(var_missRate > 0.8)]
var = var[np.where(var_missRate <= 0.8)]

data1 = data[var]

#########################描述字符型变量的赋值
data1.eventid = data1.eventid.astype(str)
summary = data1.pop('summary')
target1 = data1.pop('target1')

data2 = data1.drop(['location', 'corp1', 'natlty1_txt', 'motive', 'weapdetail',
                    'propcomment', 'addnotes', 'scite1', 'scite2', 'scite3', 'dbsource'], axis=1)

# 记录字符型变量的对映值
# 国家名称：美国、英国等
country = data2[['country', 'country_txt']].sort_values('country').drop_duplicates('country')
country.to_csv('e:\\jm\\country.csv',encoding='gbk')
# 区域名称：北美、东亚等
region = data2[['region', 'region_txt']].sort_values('region').drop_duplicates('region')
print(region)
# 攻击类型：暗杀、劫持等
attacktype1 = data2[['attacktype1', 'attacktype1_txt']].sort_values('attacktype1').drop_duplicates('attacktype1')
# 目标类型：商业、政府等
targtype1 = data2[['targtype1', 'targtype1_txt']].sort_values('targtype1').drop_duplicates('targtype1')
# 目标子类型：银行、酒店等
targsubtype1 = data2[['targsubtype1', 'targsubtype1_txt']].sort_values('targsubtype1').drop_duplicates('targsubtype1')
# 武器类型：轻武器、爆炸物等
weaptype1 = data2[['weaptype1', 'weaptype1_txt']].sort_values('weaptype1').drop_duplicates('weaptype1')
# 武器子类型：炸药、车辆等
weapsubtype1 = data2[['weapsubtype1', 'weapsubtype1_txt']].sort_values('weapsubtype1').drop_duplicates('weapsubtype1')
# 财产损失：轻微损失、损失不明等
propextent = data2[['propextent', 'propextent_txt']].sort_values('propextent').drop_duplicates('propextent').dropna()

####对上述有映射的字符说明指标删除
data2 = data2.drop(['country_txt', 'region_txt', 'attacktype1_txt', 'targtype1_txt',
                    'targsubtype1_txt', 'weaptype1_txt', 'weapsubtype1_txt', 'propextent_txt'], axis=1)


######################进一步考察缺失值状况
var_missRate = find_miss(data2)
# 考虑到武器子类型、目标子类型并不重要且缺失值6~7%予以删除
data2 = data2.drop(['targsubtype1', 'weapsubtype1'], axis=1)
# 财产损失用0值填充
data2[['propextent', 'propvalue']] = data2[['propextent', 'propvalue']].fillna(0)
# 相关人数按字典要求用-99填充
data2[['nperps', 'nperpcap']] = data2[['nperps', 'nperpcap']].fillna(-99)
# 当前缺失值大于1%，仅伤亡人数4个指标
print(find_miss(data2)[find_miss(data2) > 0.01])
# 导出经纬度
# t = data2[['latitude', 'longitude','nkill']].dropna()
# t=t.groupby(['latitude', 'longitude']).sum()
# t = t.reset_index()
# t=t[t.nkill>0]
# t.to_csv('e:\\jm\\jwd.csv',index=False)

# 相关性，填充 受伤人数
def fill_var(data2, var1, var2):
    # var1 是自变量 var2 是因变量，对缺失部分的因变量进行填充
    print(data2[[var1, var2]].corr())
    # 回归
    t = data2[[var1, var2]].dropna()
    lrModel = LinearRegression()
    lrModel.fit(t[var1].values.reshape(-1, 1), t[var2])
    a, b = lrModel.coef_, lrModel.intercept_
    # 填充nwound缺失部分
    t1 = data2[(~data2[var1].isnull()) & (data2[var2].isnull())][[var1, var2]]
    data2.loc[t1.index, var2] = lrModel.predict(t1[var1].values.reshape(-1, 1))
    return data2


data2 = fill_var(data2, 'nkill', 'nwound')
data2 = fill_var(data2, 'nkillus', 'nwoundus')

# 填充
data3 = data2.fillna(-1)
len(data3)
data2.to_csv('e:\\jm\\data2.csv',encoding='utf-8')

data3 = data2.dropna()
print('清洗前数据量：', len(data2))
print('清洗后数据量：', len(data3))

#######################先不删除缺失值

###将数据集中多个分类数字修正为int型
var_int = list(set(data2.columns) - {'eventid', 'provstate', 'city', 'latitude', 'longitude', 'gname'})
data3[var_int] = data3[var_int].astype(int)
# 保存数据集
data3.to_csv('e:\\jm\\data3.csv')

#########################导入各国经济自由度数据集
eco = pd.read_excel('e:\\jm\\全球各国经济自由度指数.xlsx')
eco.columns = ['国家', '年份', '总体评分'] + eco.iloc[1, 3:].values.tolist()
eco = eco[2:]
#填充
a=eco.pivot(index='年份',columns='国家')
for column in list(a.columns[a.isnull().sum() > 0]):
    mean_val = a[column].mean()
    a[column].fillna(mean_val, inplace=True)
eco=a.stack(dropna=False).reset_index()


help(pd.DataFrame.stack)

# 对国家进行匹配
cou1 = eco[['国家']].sort_values('国家').drop_duplicates('国家')
cou2 = country.sort_values('country_txt')
# 原数据中名称字符分隔符删除
cou2['country_txt'] = cou2['country_txt'].str.replace(' ', '')
cou = pd.merge(cou1, cou2, left_on=['国家'], right_on=['country_txt'], how='outer')
# 将不能匹配的进行模糊匹配
a = cou[cou.isnull().any(axis=1)]
a1 = a['国家'].dropna().values.tolist()
a2 = a['country_txt'].dropna().values.tolist()

dic = dict()
for t in a2:
    x = process.extract(t, a1, limit=2)
    if (x[0][1] >= 68 or x[0][1] == 50) and (x[0][1] != 72):
        dic[t] = x[0][0]
t = [a for a, b in enumerate(cou['country_txt']) if b in dic]
cou.iloc[t, 0] = list(dic.values())
cou = cou.iloc[:,:2].dropna()
cou['country']=cou['country'].astype(int)
# 将国家编号并入eco经济数据集
eco1 = pd.merge(eco, cou, on=['国家'], how='right')
eco1.rename(columns={'年份':'iyear'},inplace=True)
eco1 = eco1.drop('国家',axis=1)

#将经济数据按照 国家编号、年份汇入 恐怖袭击 表格
eco1 = eco1.sort_values(['iyear', 'country'])
data4 = pd.merge(data3, eco1, on=['country', 'iyear'], how='left')

############################################
find_miss(data4)
data4=data4.drop(['司法效率','财政健康'],axis=1)
print(len(data4.dropna()))
len(data4)

data4.to_csv('e:\\jm\\data4.csv')
data4.to_csv('e:\\jm\\data4_1.csv')

####################第二问：对未知的分类
g = pd.read_excel('e:\\jm\\nogroup(1).xlsx',head=False)
g.eventid = g.eventid.astype('str')
data4 = pd.merge(g,data4,on='eventid',how='inner')
len(g)

np.shape(data3)

t = pd.Series(var_dictionary)

t911 = data2[data2.eventid.str.startswith('20010911')]
t911 = data3[data3.eventid.str.startswith('20010911')]

t = t[set(t.index) & set(a)]

del con2

t1 = data1[data1.country_txt == 'China']

data1.discribe()

myfunc.search('rename')
help(pd.merge)

data2.to_excel('e:\\jm\\data2.xlsx', 'sheet1')

data2.eventid[1]
myfunc.search('to_excel')

data.shape