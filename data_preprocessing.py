import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def data_cleansing(entire_data):
    entire_data['YYYYMMDD']=pd.to_datetime(entire_data['YYYYMMDD'], format='%Y%m%d')
    entire_data['day_of_week'] = entire_data['YYYYMMDD'].dt.dayofweek
    entire_data['day_dummy'] = [0 if (i>=0) and (i<=4) else 1 for i in entire_data['day_of_week'].values]

    entire_data['work_time0'] = [1 if (x>=0) and (x<90000) else 0 for x in entire_data['HHMMSS']]
    entire_data['work_time1'] = [1 if (x>=90000) and (x<180000) else 0 for x in entire_data['HHMMSS']]
    entire_data['work_time2'] = [1 if (x>=180000) and (x<235900) else 0 for x in entire_data['HHMMSS']]

    entire_data['CPU_sma'] = entire_data['CPU'].rolling(12).mean()
    entire_data['CPU_ema'] = entire_data['CPU'].ewm(span=3).mean()

    entire_data['TEMP_sma'] = entire_data['TEMP1'].rolling(12).mean()
    entire_data['TEMP_ema'] = entire_data['TEMP1'].ewm(span=3).mean()

    entire_data = entire_data.dropna()
    entire_data = entire_data.loc[:,['day_dummy','work_time0','work_time1','work_time2','CPU_ema','CPU_sma','CPU', 'TEMP_ema', 'TEMP_sma', 'TEMP1']]
    entire_data.reset_index(drop=True)

    return entire_data

def normalized_data(processed_data):
    all_features = ['day_dummy','work_time0','work_time1','work_time2','CPU_ema','CPU_sma','CPU', 'TEMP_ema', 'TEMP_sma', 'TEMP1']
    norm_features = ['CPU_ema','CPU_sma','CPU', 'TEMP_ema', 'TEMP_sma', 'TEMP1']
    processed_data = processed_data.loc[:,all_features]
    normalized_data=(processed_data.loc[:,norm_features]-processed_data.loc[:,norm_features].mean())/processed_data.loc[:,norm_features].std()

    post_norm_data = pd.merge(processed_data, normalized_data, right_index=True, left_index=True)
    post_norm_data = post_norm_data.loc[:,['day_dummy','work_time0','work_time1','work_time2','CPU_ema_y','CPU_sma_y','CPU_y', 'TEMP_ema_y', 'TEMP_sma_y', 'TEMP1_y']]
    post_norm_data.columns = ['day_dummy','work_time0','work_time1','work_time2','CPU_ema','CPU_sma','CPU', 'TEMP_ema', 'TEMP_sma', 'TEMP1']
    return post_norm_data


def splitted_sequential_datasets(data, seq_len, shuffling):
  
    x = data.loc[:,['day_dummy','work_time0','work_time1','work_time2','CPU_ema','CPU_sma','CPU', 'TEMP_ema', 'TEMP_sma', 'TEMP1']].values
    y_cpu = data.loc[:,['CPU_sma','CPU']].values
    y_temp = data.loc[:,['TEMP_sma', 'TEMP1']].values
    x_nstep = data.loc[:,['day_dummy','work_time0','work_time1','work_time2','CPU_ema','CPU_sma','CPU', 'TEMP_ema', 'TEMP_sma', 'TEMP1']].values
    
    dataX = []
    dataY_cpu = []
    dataY_temp = []
    nstep_dataX = []
    
    for i in range(0, len(y_cpu)-30):
        _x = x[30-seq_len+i : 30+i]
        _x_nstep = x_nstep[30-seq_len+i : 30+i]
        cpu_y = y_cpu[30+i]
        temp_y = y_temp[30+i]
        
        dataX.append(_x)
        dataY_cpu.append(cpu_y)
        dataY_temp.append(temp_y)
        nstep_dataX.append(_x_nstep)
        
    dataX = np.array(dataX)
    data_cpu_Y = np.array([i for i in dataY_cpu])
    data_temp_Y = np.array([i for i in dataY_temp])
    
#     print(data_cpu_Y)
    np.random.seed(401)
    shuffle_index = np.random.choice(len(dataX), len(dataX), replace=False)

    global train_index 
    train_index = shuffle_index[0:int(len(dataX)*0.6)]
    global valid_index
    valid_index = shuffle_index[int(len(dataX)*0.6):int(len(dataX)*0.6)+int(len(dataX)*0.2)]
    global test_index
    test_index = shuffle_index[int(len(dataX)*0.6)+int(len(dataX)*0.2):]
    
#     return train_index, valid_index, test_index
    
    if shuffling == False :
        #split t rain, valid, test
        x_train, x_valid, y_train, y_valid = train_test_split(dataX, data_cpu_Y, test_size=0.4, shuffle=False)
        x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5, shuffle=False)

        x_train = np.array(x_train)
        x_valid = np.array(x_valid)
        x_test = np.array(x_test)

        y_cpu_train = np.array([i[0] for i in y_train])
        y_cpu_valid = np.array([i[0] for i in y_valid])
        y_cpu_test = np.array([i[0] for i in y_test])
        
        y_temp_train = np.array([i[1] for i in y_train])
        y_temp_valid = np.array([i[1] for i in y_valid])
        y_temp_test = np.array([i[1] for i in y_test])
        
    else :
        x_train = dataX[train_index]
        x_valid = dataX[valid_index]
        x_test = dataX[test_index]
        
        y_cpu_train = data_cpu_Y[train_index]
        y_cpu_valid = data_cpu_Y[valid_index]
        y_cpu_test = data_cpu_Y[test_index]
        
        y_temp_train = data_temp_Y[train_index]
        y_temp_valid = data_temp_Y[valid_index]
        y_temp_test = data_temp_Y[test_index]
        
    return [x_train, x_valid, x_test, y_cpu_train, y_cpu_valid, y_cpu_test, y_temp_train, y_temp_valid, y_temp_test, dataX, nstep_dataX, data_cpu_Y, data_temp_Y, train_index, valid_index, test_index]

  