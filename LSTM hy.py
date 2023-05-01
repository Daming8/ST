#!/usr/bin/env python
# coding: utf-8

# # 项目需求：通过XGboost来实现预测zclose值

# # 导入各python包

# In[52]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')
import joblib


# # 导入数据

# In[53]:


all_data_set_path = r'https://raw.githubusercontent.com/Daming8/ST/main/CSI300ALL.csv'
all_data_set = pd.read_csv(all_data_set_path)


# In[54]:


print(all_data_set.head())


# In[55]:


print(all_data_set.info()) #查看有多少数据及特征


# In[56]:


print(all_data_set.isnull().sum()) #检查是否有空数据


# # 研究数据

# In[57]:


# 特征热力图 相关性分析
list_columns = all_data_set.columns
plt.figure(figsize=(30,20))
sns.set(font_scale=1.2)
sns.heatmap(all_data_set[list_columns].corr(), annot=True, fmt=".2f")
plt.show()


# In[58]:


# 对特征重要性进行排序
corr_1 = all_data_set.corr()
corr_1["Close"].sort_values(ascending=False)


# # 数据预处理

# In[59]:


len_ = len(['Open','High','Low','Close'])*3
col_numbers_drop = []
for i in range(3):
    col_numbers_drop.append(len_+i)
print(col_numbers_drop)


# In[60]:


all_data_set.info()


# In[61]:


# 依据特征重要性，选择zlow zhigh zopen来进行预测zclose
# 数据选择t-n, ...., t-2 t-1 与 t 来预测未来 t+1
# 转换原始数据为新的特征列来进行预测,time_window可以用来调试用前几次的数据来预测
def series_to_supervised(data,time_window=3):
    data_columns = ['Open','High','Low','Close']
    data = data[data_columns]  # Note this is important to the important feature choice
    cols, names = list(), list()
    for i in range(time_window, -1, -1):
        # get the data
        cols.append(data.shift(i)) #数据偏移量
        
        # get the column name
        if ((i-1)<=0):
            suffix = '(t+%d)'%abs(i-1)
        else:
            suffix = '(t-%d)'%(i-1)
        names += [(colname + suffix) for colname in data_columns]
        
    # concat the cols into one dataframe
    agg = pd.concat(cols,axis=1)
    agg.columns = names
    agg.index = data.index.copy()
    # remove the nan value which is caused by pandas.shift
    agg = agg.dropna(inplace=False)

    # remove unused col (only keep the "close" fied for the t+1 period)
    # Note col "close" place in the columns

    len_ = len(data_columns)*time_window
    col_numbers_drop = []
    for i in range(len(data_columns)-1):
        col_numbers_drop.append(len_+i)

    agg.drop(agg.columns[col_numbers_drop],axis=1,inplace = True)
       
    return agg
    


# In[62]:


all_data_set2 = all_data_set.copy()
all_data_set2["Date"] = pd.to_datetime(all_data_set2["Date"])       # 日期object: to datetime
all_data_set2.set_index("Date", inplace=True, drop=True) # 把index设为索引


# In[63]:


all_data_set2 = all_data_set2[116:] # 这里把7月28日的数据全部删掉了，主要是数据缺失较多


# In[64]:


data_set_process = series_to_supervised(all_data_set2,10) #取近10分钟的数据
print(data_set_process.columns.values)


# In[65]:


print(data_set_process.info())


# In[66]:


print(data_set_process.head())


# # 搭建模型XGboost

# In[67]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_set_process)


# In[68]:


train_size = int(len(data_set_process)*0.8)
test_size = len(data_set_process) - train_size
train_XGB, test_XGB = scaled_data[0:train_size,:],scaled_data[train_size:len(data_set_process),:]

train_XGB_X, train_XGB_Y = train_XGB[:,:(len(data_set_process.columns)-1)],train_XGB[:,(len(data_set_process.columns)-1)]
test_XGB_X, test_XGB_Y = test_XGB[:,:(len(data_set_process.columns)-1)],test_XGB[:,(len(data_set_process.columns)-1)]


# In[69]:


test_XGB_Y[0:10]


# In[70]:


# 算法参数
params = {
    'booster':'gbtree',
    'objective':'binary:logistic',  # 此处为回归预测，这里如果改成multi:softmax 则可以进行多分类
    'gamma':0.1,
    'max_depth':5,
    'lambda':3,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':3,
    'slient':1,
    'eta':0.1,
    'seed':1000,
    'nthread':4,
}


# In[71]:


#生成数据集格式
xgb_train = xgb.DMatrix(train_XGB_X,label = train_XGB_Y)
xgb_test = xgb.DMatrix(test_XGB_X,label = test_XGB_Y)
num_rounds = 300
watchlist = [(xgb_test,'eval'),(xgb_train,'train')]


# In[72]:


#xgboost模型训练
model_xgb = xgb.train(params,xgb_train,num_rounds,watchlist)


# In[73]:


#对测试集进行预测
y_pred_xgb = model_xgb.predict(xgb_test)


# In[74]:


y_pred_xgb.shape


# In[75]:


y_pred_xgb


# In[76]:


test_XGB_Y


# In[77]:


test_XGB_Y.shape


# In[78]:


mape_xgb = np.mean(np.abs(y_pred_xgb-test_XGB_Y)/test_XGB_Y)*100
print('XGBoost平均误差率为：{}%'.format(mape_xgb))  


# In[79]:


joblib.dump(model_xgb, "XGBoost.pkl")


# # 搭建模型LSTM网络

# In[80]:


# 注意这里要安装Tensorflow 和 Keras才能使用
from keras.models import Sequential
from keras.layers import Dense,LSTM


# In[81]:


data_set_process.info()


# In[82]:


len(data_set_process.columns)


# In[83]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_set_process)


# In[84]:


scaled_data.shape


# In[85]:


train_size = int(len(data_set_process)*0.8)
test_size = len(data_set_process) - train_size
train_LSTM, test_LSTM = scaled_data[0:train_size,:],scaled_data[train_size:len(data_set_process),:]

train_LSTM_X, train_LSTM_Y = train_LSTM[:,:(len(data_set_process.columns)-1)],train_LSTM[:,(len(data_set_process.columns)-1)]
test_LSTM_X, test_LSTM_Y = test_LSTM[:,:(len(data_set_process.columns)-1)],test_LSTM[:,(len(data_set_process.columns)-1)]
# reshape input to be [samples, time steps, features]
train_LSTM_X2 = np.reshape(train_LSTM_X, (train_LSTM_X.shape[0], 1, train_LSTM_X.shape[1]))
test_LSTM_X2 = np.reshape(test_LSTM_X, (test_LSTM_X.shape[0], 1, test_LSTM_X.shape[1]))

# reshape input to be [samples, time steps, features]
train_LSTM_X = np.reshape(train_LSTM_X, (train_LSTM_X.shape[0],1,train_LSTM_X.shape[1]))
test_LSTM_X = np.reshape(test_LSTM_X, (test_LSTM_X.shape[0],1,test_LSTM_X.shape[1]))

print(train_LSTM_X.shape,train_LSTM_Y.shape,test_LSTM_X.shape,test_LSTM_Y.shape)


# In[86]:


type(scaler)


# In[87]:


test_LSTM_X[:,:,-4:].shape


# In[88]:


np.array(data_set_process[-284:]['Close(t+1)']).shape


# In[89]:


test_LSTM_X.shape


# In[90]:


# creat and fit the LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(train_LSTM_X2.shape[1], train_LSTM_X2.shape[2])))
# model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss="mae", optimizer="Adam")
print(model.summary())


# # 结果可视化及评估

# In[91]:


type(train_LSTM_Y)


# In[92]:


print("start to fit the model")
history = model.fit(train_LSTM_X2, train_LSTM_Y, epochs=50, batch_size=50, validation_data=(test_LSTM_X2, test_LSTM_Y),
                    verbose=2, shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[93]:


model.save('LSTM_model.h5') 


# In[94]:


yPredict = model.predict(test_LSTM_X2)
print(yPredict.shape)
print(test_LSTM_Y.shape)
print(test_LSTM_X.shape)


# In[95]:


# Reshape yPredict to have shape (606,)
yPredict_reshaped = np.reshape(yPredict, (606,))

# Reshape test_LSTM_X to have shape (606, 40)
test_LSTM_X_reshaped = np.reshape(test_LSTM_X, (606, 40))

# Concatenate test_LSTM_X_reshaped and yPredict_reshaped along the second axis
concatenated = np.concatenate((test_LSTM_X_reshaped, yPredict_reshaped.reshape(606, 1)), axis=1)

# Use the scaler to invert the transformation
testPredict = scaler.inverse_transform(concatenated)[:, -1:]
test_LSTM_Y2 = scaler.inverse_transform(np.concatenate((test_LSTM_X_reshaped, test_LSTM_Y.reshape(len(test_LSTM_Y), 1)), axis=1))[:, -1]


# In[96]:


mape = np.mean(np.abs(test_LSTM_Y2.flatten()-testPredict.flatten())/test_LSTM_Y2.flatten())*100  # 这里计算测试集预测结果与真实结果的误差率
print('Test LSTM for test set Score:%.6f MAPE' %(mape)) 


# In[97]:


yPredict_train = model.predict(train_LSTM_X2)
print(yPredict_train.shape)
print(train_LSTM_X.shape)
print(train_LSTM_Y.shape)


# In[98]:


train_LSTM_X_reshaped = train_LSTM_X.reshape(train_LSTM_X.shape[0], train_LSTM_X.shape[2])
trainPredict = scaler.inverse_transform(np.concatenate((train_LSTM_X_reshaped, yPredict_train), axis=1))[:, -1:]
train_LSTM_Y2 = scaler.inverse_transform(np.concatenate((train_LSTM_X_reshaped, train_LSTM_Y.reshape(len(train_LSTM_Y),1)), axis=1))[:, -1:]


# In[99]:


mape2 = np.mean(np.abs(train_LSTM_Y2.flatten()-trainPredict.flatten())/train_LSTM_Y2.flatten())*100  # 这里计算训练集预测结果与真实结果的误差率
print('Test LSTM for train set Score:%.6f MAPE' %(mape2)) 


# In[100]:


plt.plot(train_LSTM_Y2, color = 'red', label = 'Real Price for Train set')
plt.plot(trainPredict, color = 'blue', label = 'Predicted Price for Train set')
plt.title('Zclose Price Prediction for Train set')
plt.xlabel('Time')
plt.ylabel('Sohu Zclose Price')
plt.legend()
plt.show()


# In[101]:


plt.plot(test_LSTM_Y2, color = 'red', label = 'Real Price for Test set')
plt.plot(testPredict, color = 'blue', label = 'Predicted Price for Test set')
plt.title('Zclose Price Prediction for Test set')
plt.xlabel('Time')
plt.ylabel('Sohu Zclose Price')
plt.legend()
plt.show()


# # 模型调优

# In[102]:


def predictions(mae_lstm, mae_xgboost, prediction_xgb, prediction_lstm):
    
    """Returns the prediction at t+1 weighted by the respective mae. Giving a higher weight to the one which is lower"""
    
    prediction = (1-(mae_xgboost/(mae_lstm+mae_xgboost)))*prediction_xgb+(1-(mae_lstm/(mae_lstm+mae_xgboost)))*prediction_lstm
    return prediction


# In[103]:


#COMBINATION LSTM-XGBoost


# In[104]:


mae_xgboost = mae


# In[ ]:


xgboost_model = joblib.load("XGBoost.pkl")


# In[ ]:


scope = predictions(mae_lstm, mae_xgboost, pred_test_xgb, pred_test)


# In[ ]:


avg_mae = (mae_lstm + mae_xgboost)/2


# In[ ]:


plotting(y_val_save, y_test, scope, avg_mae, WINDOW, PREDICTION_SCOPE)


# In[ ]:




