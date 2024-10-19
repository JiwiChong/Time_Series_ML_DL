import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

def result2file(args, predict, real, threshold, n_step) :
  
    plt.plot(real,'red',label="real")
    plt.plot(predict,'blue',label="prediction")
    plt.xlabel('Time Period')
    plt.ylabel('Rate')
    plt.legend()
    #   plt.savefig(image_directory + '{0}_{2}_test_{1}_quantile_0.6_0.6.jpg'.format(mng_no, n_step+1, threshold))
    plt.savefig(f'./plots/device_{args.data_device}/quantile_0.5_0.6/{args.data_device}_{1}_test_{0}_quantile_0.5_0.6.jpg'.format(n_step+1, threshold))
    plt.show()

    real_value = np.asarray(real).reshape(-1,1)
    predict_value = np.asarray(predict).reshape(-1,1)
    r2 = r2_score(real_value, predict_value, multioutput='variance_weighted')
    for i in range(len(real)) :
        if(real[i] >= threshold) : real[i] = 1
        elif(real[i] < threshold) : real[i] = 0
    for i in range(len(predict)) :
        if(predict[i] >= threshold) : predict[i] = 1
        elif(predict[i] < threshold) : predict[i] = 0

    cf_matrix = confusion_matrix(real, predict)
    accuracy = accuracy_score(real, predict)
    precision = precision_score(real, predict)
    recall = recall_score(real, predict)
    return 1-r2 , cf_matrix, accuracy, precision, recall