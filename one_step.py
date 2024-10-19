import os
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, precision_score, recall_score

def one_step_forecasting(args, forecasting_model, x_test, y_cpu_test, y_temp_test, cpu_thres, temp_thres, cpu_mean_thres, temp_mean_thres):
    model_path = '/content/gdrive/My Drive/ATC/model/augmented_data_models/'
    excel_directory = '/content/gdrive/My Drive/ATC/results/ma_results/'

    forecasting_model.load_weights(f'./saved_models/device_{args.data_device}/{args.forecast_data}/{args.model_path}/aug_{args.forecast_data}_best_model_quantile_{args.tilt_loss_q}.hdf5')
    # TEMP_model.load_weights(model_path+'aug_temp_best_model_compile_w0.5_La0.3.hdf5')
    pred_y = forecasting_model.predict(x_test)
    # pred_temp_y_auc = TEMP_model.
    if args.forecast_data == 'temp':
        pred_y_temp_sma = np.array([pred_y[i][0] for i in range(len(pred_y))])
        true_y_temp_sma = np.array([y_temp_test[i][0] for i in range(len(y_temp_test))])

        pred_y_temp = np.array([pred_y[i][1] for i in range(len(pred_y))])
        true_y_temp = np.array([y_temp_test[i][1] for i in range(len(y_temp_test))])

        anomaly_y_true = [1 if i>=temp_thres else 0 for i in true_y_temp]
        anomaly_y_pred = [1 if i>=temp_thres else 0 for i in pred_y_temp]

        anomaly_y_true_mean = [1 if i>=temp_mean_thres else 0 for i in true_y_temp_sma]
        anomaly_y_pred_mean = [1 if i>=temp_mean_thres else 0 for i in pred_y_temp_sma]

    else:
        pred_y_temp_sma = np.array([pred_y[i][0] for i in range(len(pred_y))])
        true_y_temp_sma = np.array([y_cpu_test[i][0] for i in range(len(y_cpu_test))])

        pred_y_temp = np.array([pred_y[i][1] for i in range(len(pred_y))])
        true_y_temp = np.array([y_cpu_test[i][1] for i in range(len(y_cpu_test))])

        anomaly_y_true = [1 if i>=cpu_thres else 0 for i in true_y_temp]
        anomaly_y_pred = [1 if i>=cpu_thres else 0 for i in pred_y_temp]

        anomaly_y_true_mean = [1 if i>=cpu_mean_thres else 0 for i in true_y_temp_sma]
        anomaly_y_pred_mean = [1 if i>=cpu_mean_thres else 0 for i in pred_y_temp_sma]


    fvu = 1-r2_score(true_y_temp, pred_y_temp)
    precision = precision_score(anomaly_y_true, anomaly_y_pred)
    accuracy = accuracy_score(anomaly_y_true, anomaly_y_pred)
    recall = recall_score(anomaly_y_true, anomaly_y_pred)
    auc = roc_auc_score(anomaly_y_true, pred_y_temp)

    fvu_mean = 1-r2_score(true_y_temp_sma, pred_y_temp_sma)
    precision_mean = precision_score(anomaly_y_true_mean, anomaly_y_pred_mean)
    accuracy_mean = accuracy_score(anomaly_y_true_mean, anomaly_y_pred_mean)
    recall_mean = recall_score(anomaly_y_true_mean, anomaly_y_pred_mean)
    auc_mean = roc_auc_score(anomaly_y_true_mean, pred_y_temp_sma)

    print('CPU FVU:' ,fvu)
    print('CPU Acc:' ,accuracy)
    print('CPU Precision:' ,precision)
    print('CPU Recall:' ,recall)
    print('CPU AUC:' ,auc)
    print('='*35) 
    print('CPU SMA FVU:' ,fvu_mean)
    print('CPU SMA Acc:' ,accuracy_mean)
    print('CPU SMA Precision:' ,precision_mean)
    print('CPU SMA Recall:' ,recall_mean)
    print('CPU SMA AUC:' ,auc_mean)