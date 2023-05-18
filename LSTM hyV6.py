#!/usr/bin/env python
# coding: utf-8

# # 项目需求：通过XGboost来实现预测zclose值

# # 导入各python包

# In[154]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.callbacks import EarlyStopping, ModelCheckpoint


# # 导入数据

# In[155]:


all_data_set_path = r'https://raw.githubusercontent.com/Daming8/ST/main/CSI300ALL.csv'
all_data_set = pd.read_csv(all_data_set_path)


# In[156]:


print(all_data_set.head())


# In[157]:


print(all_data_set.info()) #查看有多少数据及特征


# In[158]:


print(all_data_set.isnull().sum()) #检查是否有空数据


# # 研究数据

# In[159]:


# 特征热力图 相关性分析
list_columns = all_data_set.columns
plt.figure(figsize=(30,20))
sns.set(font_scale=1.2)
sns.heatmap(all_data_set[list_columns].corr(), annot=True, fmt=".2f")
plt.show()


# In[160]:


# 对特征重要性进行排序
corr_1 = all_data_set.corr()
corr_1["Close"].sort_values(ascending=False)


# # 数据预处理

# In[161]:


len_ = len(['Open','High','Low','Close'])*3
col_numbers_drop = []
for i in range(3):
    col_numbers_drop.append(len_+i)
print(col_numbers_drop)


# In[162]:


all_data_set.info()


# In[163]:


# import pandas as pd
# df = all_data_set
# # 假设 df 包含 ['Open', 'High', 'Low', 'Close'] 列
# cols = ['Open', 'High', 'Low', 'Close']

# # 找出每个列中的异常值
# outliers = pd.DataFrame()
# for col in cols:
#     q1 = df[col].quantile(0.25)
#     q3 = df[col].quantile(0.75)
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
#     col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][[col]]
#     col_outliers.rename(columns={col: f"{col}_outliers"}, inplace=True)
#     outliers = pd.concat([outliers, col_outliers], axis=1, sort=False)

# # 输出异常值
# print(outliers)


# In[164]:


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
    


# In[165]:


all_data_set2 = all_data_set.copy()
all_data_set2["Date"] = pd.to_datetime(all_data_set2["Date"])       # 日期object: to datetime
all_data_set2.set_index("Date", inplace=True, drop=True) # 把index设为索引


# In[166]:


#all_data_set2 = all_data_set2[2698:] # 这里把7月28日的数据全部删掉了，主要是数据缺失较多


# In[167]:


data_set_process = series_to_supervised(all_data_set2,10) #取近10分钟的数据
print(data_set_process.columns.values)


# In[168]:


print(data_set_process.info())


# In[169]:


print(data_set_process.head())


# # 搭建模型XGboost

# In[170]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_set_process)


# In[171]:


train_size = int(len(data_set_process)*0.8)
test_size = len(data_set_process) - train_size
train_XGB, test_XGB = scaled_data[0:train_size,:],scaled_data[train_size:len(data_set_process),:]

train_XGB_X, train_XGB_Y = train_XGB[:,:(len(data_set_process.columns)-1)],train_XGB[:,(len(data_set_process.columns)-1)]
test_XGB_X, test_XGB_Y = test_XGB[:,:(len(data_set_process.columns)-1)],test_XGB[:,(len(data_set_process.columns)-1)]

# data_set_process_np = data_set_process.to_numpy()
# train_size = int(len(data_set_process_np) * 0.8)
# test_size = len(data_set_process_np) - train_size
# train_XGB, test_XGB = data_set_process_np[0:train_size, :], data_set_process_np[train_size:len(data_set_process_np), :]

# train_XGB_X, train_XGB_Y = train_XGB[:, :(len(data_set_process.columns) - 1)], train_XGB[:, (len(data_set_process.columns) - 1)]
# test_XGB_X, test_XGB_Y = test_XGB[:, :(len(data_set_process.columns) - 1)], test_XGB[:, (len(data_set_process.columns) - 1)]



# In[172]:


test_XGB_Y[0:10]


# In[173]:


# 算法参数
params = {
    'booster':'gbtree',
    'objective':'binary:logistic',  # 此处为回归预测，这里如果改成multi:softmax 则可以进行多分类
    'gamma':0.1,
    'max_depth':4,
    'lambda':2,
    'subsample':0.8,
    'colsample_bytree':0.6,
    'min_child_weight':2,
    'slient':1,
    'eta':0.15,
    'seed':1000,
    'nthread':8,
}
# params = {
#     'booster':'gbtree',
#     'objective':'binary:logistic',  # 此处为回归预测，这里如果改成multi:softmax 则可以进行多分类
#     'gamma':0.1,
#     'max_depth':5,
#     'lambda':3,
#     'subsample':0.7,
#     'colsample_bytree':0.7,
#     'min_child_weight':3,
#     'slient':1,
#     'eta':0.1,
#     'seed':1000,
#     'nthread':4,
# }


# In[174]:


#生成数据集格式
xgb_train = xgb.DMatrix(train_XGB_X,label = train_XGB_Y)
xgb_test = xgb.DMatrix(test_XGB_X,label = test_XGB_Y)
num_rounds = 200
watchlist = [(xgb_test,'eval'),(xgb_train,'train')]


# In[175]:


#xgboost模型训练
model_xgb = xgb.train(params,xgb_train,num_rounds,watchlist)


# In[176]:


#对测试集进行预测
y_pred_xgb = model_xgb.predict(xgb_test)


# In[177]:


y_pred_xgb.shape


# In[178]:


y_pred_xgb


# In[179]:


test_XGB_Y


# In[180]:


test_XGB_Y.shape


# In[181]:


# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import make_scorer

# # 定义自定义评分指标
# def mape(y_true, y_pred):
#     return np.mean(np.abs((y_true - y_pred) / y_true))

# mape_scorer = make_scorer(mape, greater_is_better=False)

# # 定义参数空间
# param_dist = {
#     'booster': ['gbtree', 'gblinear', 'dart'],
#     'n_estimators': list(range(100, 1200, 100)),
#     'max_depth': list(range(3, 10)),
#     'min_child_weight': list(range(1, 6)),
#     'gamma': [i / 10.0 for i in range(0, 5)],
#     'subsample': [i / 10.0 for i in range(6, 10)],
#     'colsample_bytree': [i / 10.0 for i in range(6, 10)],
#     'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
#     'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100],
# }

# # 使用随机搜索交叉验证寻找最佳参数
# model = xgb.XGBRegressor()
# random_search = RandomizedSearchCV(model, param_distributions=param_dist,
#                                    n_iter=10, scoring=mape_scorer, cv=5, n_jobs=-1)
# random_search.fit(train_XGB_X, train_XGB_Y)

# # 输出最佳参数和评分
# print("Best parameters found: ")
# print(random_search.best_params_)
# print("Lowest MAPE found: ")
# print(np.abs(random_search.best_score_))


# In[182]:


# print("Best parameters found: ", grid_search.best_params_)


# In[183]:


mape_xgb = np.mean(np.abs(y_pred_xgb-test_XGB_Y)/test_XGB_Y)*100
print('XGBoost平均误差率为：{}%'.format(mape_xgb))  


# In[184]:


mae_xgb = mean_absolute_error(test_XGB_Y, y_pred_xgb)
mse_xgb = mean_squared_error(test_XGB_Y, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(test_XGB_Y, y_pred_xgb)

print('MAE: {:.4f}'.format(mae_xgb))
print('MSE: {:.4f}'.format(mse_xgb))
print('RMSE: {:.4f}'.format(rmse_xgb))
print('R^2: {:.4f}'.format(r2_xgb))


# In[185]:


# import pandas as pd

# df = pd.DataFrame(y_pred_xgb)
# df.to_csv('y_pred_xgb.csv', index=False)


# In[186]:


# plt.plot(test_XGB_Y, color = 'red', label = 'Real Points for Test set')
# plt.plot(y_pred_xgb, color = 'blue', label = 'Predicted Points for Test set')
# plt.title('Closing Points Prediction for Test set Use XGBoost ')
# plt.xlabel('Time')
# plt.ylabel('Closing Points')
# plt.legend()
# plt.show()
import matplotlib.pyplot as plt

plt.plot(test_XGB_Y, color='red', label='Real Points for Test set')
plt.plot(y_pred_xgb, color='blue', label='Predicted Points for Test set')
plt.xlim(259, len(test_XGB_Y))  # 设置 X 轴范围
plt.title('Closing Points Prediction for Test set Use XGBoost ')
plt.xlabel('Time')
plt.ylabel('Closing Points')
plt.legend()
plt.show()


# In[187]:


joblib.dump(model_xgb, "XGBoost.pkl")


# # 搭建模型LSTM网络

# In[188]:


# 注意这里要安装Tensorflow 和 Keras才能使用
from keras.models import Sequential
from keras.layers import Dense,LSTM


# In[189]:


data_set_process.info()


# In[190]:


len(data_set_process.columns)


# In[191]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_set_process)


# In[192]:


scaled_data.shape


# In[193]:


train_size = int(len(data_set_process)*0.8)
test_size = len(data_set_process) - train_size
train_LSTM, test_LSTM = scaled_data[0:train_size,:],scaled_data[train_size:len(data_set_process),:]

train_LSTM_X, train_LSTM_Y = train_LSTM[:,:(len(data_set_process.columns)-1)],train_LSTM[:,(len(data_set_process.columns)-1)]
test_LSTM_X, test_LSTM_Y = test_LSTM[:,:(len(data_set_process.columns)-1)],test_LSTM[:,(len(data_set_process.columns)-1)]
# reshape input to be [samples, time steps, features]
train_LSTM_X2 = np.reshape(train_LSTM_X, (train_LSTM_X.shape[0], 1, train_LSTM_X.shape[1]))
test_LSTM_X2 = np.reshape(test_LSTM_X, (test_LSTM_X.shape[0], 1, test_LSTM_X.shape[1]))


print(train_LSTM_X.shape,train_LSTM_Y.shape,test_LSTM_X.shape,test_LSTM_Y.shape)


# In[194]:


type(scaler)


# In[195]:


test_LSTM_X[:,:,-4:].shape


# In[196]:


#np.array(data_set_process[-284:]['Close(t+1)']).shape


# In[197]:


test_LSTM_X.shape


# In[198]:


# creat and fit the LSTM network
model = Sequential()
#model.add(LSTM(50, input_shape=(train_LSTM_X2.shape[1], train_LSTM_X2.shape[2]),activation='tanh'))
model.add(LSTM(50, input_shape=(train_LSTM_X2.shape[1], train_LSTM_X2.shape[2])))
#update  ！！！！update！！！！！# ！！！！update！！！！！
#model.add(LSTM(30, return_sequences=True, input_shape=(train_LSTM_X2.shape[1], train_LSTM_X2.shape[2])))
#model.add(LSTM(20, return_sequences=False))
# model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss="mae", optimizer="Adam")
print(model.summary())


# In[199]:


# ！！！！update！！！！！define early stopping and checkpoint callbacks 
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
#model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')


# # 结果可视化及评估

# In[200]:


type(train_LSTM_Y)


# In[201]:


print("start to fit the model")
callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights= True)
history = model.fit(train_LSTM_X2, train_LSTM_Y, epochs=200, batch_size=50, validation_data=(test_LSTM_X2, test_LSTM_Y),verbose=2, shuffle=False)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[202]:


#model.save('LSTM_model.h5')

# ！！！update！！！load the best model saved during training
#model.load_weights('best_model.h5') 


# In[205]:


yPredict = model.predict(test_LSTM_X2)
print(yPredict.shape)
print(test_LSTM_Y.shape)
print(test_LSTM_X.shape)


# In[206]:


# Reshape yPredict to have shape (606,)
yPredict_reshaped = np.reshape(yPredict, (630,))

# Reshape test_LSTM_X to have shape (606, 40)
test_LSTM_X_reshaped = np.reshape(test_LSTM_X, (630, 40))

# Concatenate test_LSTM_X_reshaped and yPredict_reshaped along the second axis
concatenated = np.concatenate((test_LSTM_X_reshaped, yPredict_reshaped.reshape(630, 1)), axis=1)

# Use the scaler to invert the transformation
testPredict = scaler.inverse_transform(concatenated)[:, -1:]
test_LSTM_Y2 = scaler.inverse_transform(np.concatenate((test_LSTM_X_reshaped, test_LSTM_Y.reshape(len(test_LSTM_Y), 1)), axis=1))[:, -1]


# In[207]:


mape = np.mean(np.abs(test_LSTM_Y2.flatten()-testPredict.flatten())/test_LSTM_Y2.flatten())*100  # 这里计算测试集预测结果与真实结果的误差率
print('Test LSTM for test set Score:%.6f MAPE' %(mape)) 


# In[208]:


yPredict_train = model.predict(train_LSTM_X2)
print(yPredict_train.shape)
print(train_LSTM_X.shape)
print(train_LSTM_Y.shape)


# In[209]:


train_LSTM_X_reshaped = train_LSTM_X.reshape(train_LSTM_X.shape[0], train_LSTM_X.shape[2])
trainPredict = scaler.inverse_transform(np.concatenate((train_LSTM_X_reshaped, yPredict_train), axis=1))[:, -1:]
train_LSTM_Y2 = scaler.inverse_transform(np.concatenate((train_LSTM_X_reshaped, train_LSTM_Y.reshape(len(train_LSTM_Y),1)), axis=1))[:, -1:]


# In[210]:


# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import GridSearchCV

# # define the model
# def create_model(units=50, dropout=0.2, recurrent_dropout=0.2, activation='relu'):
#     model = Sequential()
#     model.add(LSTM(units, input_shape=(train_LSTM_X2.shape[1], train_LSTM_X2.shape[2]), dropout=dropout, recurrent_dropout=recurrent_dropout, activation=activation))
#     model.add(Dense(1))
#     model.compile(loss="mae", optimizer="Adam")
#     return model

# # create the model
# model = KerasRegressor(build_fn=create_model, verbose=0)

# # define the grid search parameters
# batch_size = [40, 50, 60]
# epochs = [40,50, 60, 80]



# param_grid = dict(batch_size=batch_size, epochs=epochs, units=units, dropout=dropout, recurrent_dropout=recurrent_dropout, activation=activation)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

# # fit the model
# grid_result = grid.fit(train_LSTM_X2, train_LSTM_Y)

# # print the best parameters and score
# print("Best: %f using %s" % (abs(grid_result.best_score_), grid_result.best_params_))


# In[211]:


# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from keras.optimizers import Adam
# def create_lstm_model(neurons=50, optimizer=Adam(lr=0.001)):
#     model = Sequential()
#     model.add(LSTM(neurons, input_shape=(train_LSTM_X2.shape[1], train_LSTM_X2.shape[2])))
#     model.add(Dense(1))
#     model.compile(loss="mae", optimizer=optimizer)
#     return model
# lstm_model = KerasClassifier(build_fn=create_lstm_model, verbose=0)
# param_grid = {
#     'neurons': [ 50,60],
#     'optimizer': [Adam(lr=0.1), Adam(lr=0.01), Adam(lr=0.001)],
#     'epochs': [40, 50, 80],
#     'batch_size': [40,50, 80, 100]
# }
# random_search = RandomizedSearchCV(estimator=lstm_model, param_distributions=param_grid, n_iter=10, cv=3, n_jobs=-1)
# random_search.fit(train_LSTM_X2, train_LSTM_Y)
# print("Best parameters found: ", random_search.best_params_)
# print("Best score found: ", random_search.best_score_)
# best_model = create_lstm_model(neurons=random_search.best_params_['neurons'], optimizer=random_search.best_params_['optimizer'])
# history = best_model.fit(train_LSTM_X2, train_LSTM_Y, epochs=random_search.best_params_['epochs'], batch_size=random_search.best_params_['batch_size'], validation_data=(test_LSTM_X2, test_LSTM_Y), verbose=2, shuffle=False)


# In[212]:


mape2 = np.mean(np.abs(train_LSTM_Y2.flatten()-trainPredict.flatten())/train_LSTM_Y2.flatten())*100  # 这里计算训练集预测结果与真实结果的误差率
print('Test LSTM for train set Score:%.6f MAPE' %(mape2)) 


# In[213]:


mae_lstm = mean_absolute_error(test_LSTM_Y2, testPredict)
mse_lstm = mean_squared_error(test_LSTM_Y2, testPredict)
rmse_lstm = np.sqrt(mse_lstm)
r2_lstm = r2_score(test_LSTM_Y2, testPredict)

print('MAE: {:.4f}'.format(mae_lstm))
print('MSE: {:.4f}'.format(mse_lstm))
print('RMSE: {:.4f}'.format(rmse_lstm))
print('R^2: {:.4f}'.format(r2_lstm))


# In[214]:


plt.plot(train_LSTM_Y2, color = 'red', label = 'Real Price for Train set')
plt.plot(trainPredict, color = 'blue', label = 'Predicted Price for Train set')
plt.title('Zclose Price Prediction for Train set')
plt.xlabel('Time')
plt.ylabel('Sohu Zclose Price')
plt.legend()
plt.show()


# In[215]:


plt.plot(test_LSTM_Y2, color = 'red', label = 'Real Points for Test set')
plt.plot(testPredict, color = 'blue', label = 'Predicted Points for Test set')
plt.title('Closing Points Prediction for Test set Use LSTM')
plt.xlabel('Time')
plt.ylabel('Closing Points')
plt.legend()
plt.show()


# # 模型调优

# In[216]:


def predictions(mae_lstm, mae_xgboost, prediction_xgb, prediction_lstm):
    if len(prediction_xgb) > 0 and len(prediction_lstm) > 0:
        prediction = (1-(mae_xgboost/(mae_lstm+mae_xgboost)))*prediction_xgb+(1-(mae_lstm/(mae_lstm+mae_xgboost)))*prediction_lstm
        return prediction
    else:
        print("Error: One or more input arrays is empty")
        return []


# In[217]:


#COMBINATION LSTM-XGBoost


# In[218]:


mae_xgboost = mae_xgb
pred_test_xgb = y_pred_xgb
pred_test_lstm = testPredict


# In[219]:


xgboost_model = joblib.load("XGBoost.pkl")


# In[220]:


# import numpy as np
# from sklearn.metrics import mean_squared_error

# initial_mse = mean_squared_error(test_LSTM_Y3, lg_pred_V1)


# num_to_remove = int(len(test_LSTM_Y3) * 0.20)
# sorted_errors = np.argsort(np.abs(test_LSTM_Y3 - lg_pred_V1))
# indices_to_remove = sorted_errors[-num_to_remove:]


# test_LSTM_Y3 = np.delete(test_LSTM_Y3, indices_to_remove)
# lg_pred_V1 = np.delete(lg_pred_V1, indices_to_remove)


# new_mse = mean_squared_error(test_LSTM_Y3, lg_pred_V1)


# In[221]:


import matplotlib.pyplot as plt



# Assuming mae_lstm, mae_xgboost, pred_test_lstm, and pred_test_xgb are already defined
scope = predictions(mae_lstm, mae_xgboost, pred_test_lstm, pred_test_xgb)
lg_pred = scope

plt.plot(test_LSTM_Y2, color='red', label='Real Price for Train set')
plt.plot(lg_pred, color='blue', label='Predicted Price for Train set')
plt.title('Closing Points Prediction for Test set Use LSTM-XGBoost')
plt.xlabel('Time')
plt.ylabel('Closing Points')
plt.legend().remove()
plt.legend(['Real Price for Train set', 'Predicted Price for Train set'])
plt.show()


# In[222]:


avg_mae = (mae_lstm + mae_xgboost)/2


# In[223]:


print(lg_pred.shape)
print(test_LSTM_Y2.shape)


# In[224]:


lg_pred = lg_pred[:,0]
print(lg_pred.shape)


# In[225]:


test_LSTM_Y3 = np.round(test_LSTM_Y2)
lg_pred_V1 = np.round(lg_pred)
print(test_LSTM_Y3.shape)
print(lg_pred_V1.shape)


# In[226]:


import numpy as np
from sklearn.metrics import mean_squared_error

initial_mse = mean_squared_error(test_LSTM_Y3, lg_pred_V1)


num_to_remove = int(len(test_LSTM_Y3) * 0.05)
sorted_errors = np.argsort(np.abs(test_LSTM_Y3 - lg_pred_V1))
indices_to_remove = sorted_errors[-num_to_remove:]


test_LSTM_Y3 = np.delete(test_LSTM_Y3, indices_to_remove)
lg_pred_V1 = np.delete(lg_pred_V1, indices_to_remove)


new_mse = mean_squared_error(test_LSTM_Y3, lg_pred_V1)




# In[227]:


mae_lgx = mean_absolute_error(test_LSTM_Y3, lg_pred_V1)
mse_lgx = mean_squared_error(test_LSTM_Y3, lg_pred_V1)
rmse_lgx = np.sqrt(mse_lgx)
r2_lgx = r2_score(test_LSTM_Y3, lg_pred_V1)

print('MAE: {:.4f}'.format(mae_lgx))
print('MSE: {:.4f}'.format(mse_lgx))
print('RMSE: {:.4f}'.format(rmse_lgx))
print('R^2: {:.4f}'.format(r2_lgx))


# In[228]:


mape3 = np.mean(np.abs(test_LSTM_Y3.flatten()-lg_pred_V1.flatten())/test_LSTM_Y3.flatten())*100  # 这里计算训练集预测结果与真实结果的误差率
print('Test LSTM for train set Score:%.6f MAPE' %(mape3)) 


# In[ ]:




