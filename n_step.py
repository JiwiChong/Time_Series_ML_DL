import os
import numpy as np
import pandas as pd
from eval_utils import result2file
from models import forecasting_model

def n_step_forecasting(args, test_index, dataX, data_cpu_Y, data_temp_Y, nstep_dataX, cpu_threshold, temp_threshold):
    tot_index_test = test_index

    # n_step_x_test : n_step 용 테스트 셋
    n_step_x_test = np.array([dataX[i] for i in tot_index_test if i < len(dataX)-args.n_step_size])

    # 1_step_y_test_cpu : 1_step 용 cpu 정답지
    n_step_y_test_cpu = np.array([data_cpu_Y[i] for i in tot_index_test if i < len(data_cpu_Y)-args.n_step_size])

    # 1_step_y_test_temp : 1_step 용 temp 정답지
    n_step_y_test_temp = np.array([data_temp_Y[i] for i in tot_index_test if i < len(data_temp_Y)-args.n_step_size])

    # real_n_step : 테스트 셋의 12스텝에 대한 정답 데이터 셋
    # real_n_step = np.array([dataX[i+12] for i in tot_index_test if i < len(dataX)-12])
    real_n_step = np.array([nstep_dataX[i+args.n_step_size] for i in tot_index_test if i < len(dataX)-args.n_step_size])

    new_input = n_step_x_test
    seq_len, feature_len = n_step_x_test[0].shape

    tot_cpu_fvu = []
    tot_cpu_accuracy = []
    tot_cpu_precision = []
    tot_cpu_recall = []

    tot_temp_fvu = []
    tot_temp_accuracy = []
    tot_temp_precision = []
    tot_temp_recall = []

    cpu_model_directory = f'./{args.model_dir}/device_{args.data_device}/cpu/aug_cpu_best_model_quantile_{args.tilt_loss_q}_run_{args.run_num}.hdf5'
    temp_model_directory = f'./{args.model_dir}/device_{args.data_device}/temperature/aug_temp_best_model_quantile_{args.tilt_loss_q}_run_{args.run_num}.hdf5'

    for i in range(args.n_step_size):
    
        cpu_predict = forecasting_model(new_input, cpu_model_directory)
        temp_predict = forecasting_model(new_input, temp_model_directory)
        #   cpu_predict = model_load_predict(new_input, save_path + '{0}_{1}_mean_cpu_predict_best_model.hdf5'.format(mng_no, trial))
        #   temp_predict = model_load_predict(new_input, save_path + '{0}_{1}_mean_temp_predict_best_model.hdf5'.format(mng_no, trial)) 

        for j in range(len(real_n_step)) :
            new_feature = np.concatenate([real_n_step[j][0][:-2], cpu_predict[j], temp_predict[j]], axis=0).reshape(-1,feature_len)
            new_input[j] = np.concatenate([new_input[j][1:], new_feature], axis=0)

        n_step_y_cpu= [data[i][-2] for data in real_n_step]
        n_step_y_temp= [data[i][-1] for data in real_n_step]
            
        cpu_predict = cpu_predict.reshape(-1)
        temp_predict = temp_predict.reshape(-1)
        
        n_step_result_cpu = pd.DataFrame({'real_values' : n_step_y_cpu,
                                            'predict_value' : cpu_predict})

        n_step_result_temp = pd.DataFrame({'real_values' : n_step_y_temp,
                                            'predict_value' : temp_predict})  
        
        n_step_result_cpu.to_csv(f'./saved_data/n_step/device_{args.data_device}/{args.forecast_data}/run_{args.run_num}/{0}_step_integ_cpu_result.csv'.format(i))
        n_step_result_temp.to_csv(f'./saved_data/n_step/device_{args.data_device}/{args.forecast_data}/run_{args.run_num}/{0}_step_integ_temp_result.csv'.format(i))

        cpu_fvu, cpu_matrix, cpu_accuracy, cpu_precision, cpu_recall = result2file(cpu_predict, n_step_y_cpu, cpu_threshold, i)
        temp_fvu, temp_matrix, temp_accuracy, temp_precision, temp_recall = result2file(temp_predict, n_step_y_temp, temp_threshold, i)
        
        tot_cpu_fvu.append(cpu_fvu)
        tot_cpu_accuracy.append(cpu_accuracy)
        tot_cpu_precision.append(cpu_precision)
        tot_cpu_recall.append(cpu_recall)
        
        tot_temp_fvu.append(temp_fvu)
        tot_temp_accuracy.append(temp_accuracy)
        tot_temp_precision.append(temp_precision)
        tot_temp_recall.append(temp_recall)
        
        with open(f'./saved_data/n_step/device_{args.data_device}/run_{args.run_num}/{0}_step_{1}_integ_cpu_result.txt'.format(i, args.data_device), 'w') as f:
            f.write('test_matrix : {}, \n test_acc : {}, \n test_prec : {}, \n test_reca : {}, \n test_fvu : {}'.format(cpu_matrix, cpu_accuracy, cpu_precision, cpu_recall, cpu_fvu))
        with open(f'./saved_data/n_step/device_{args.data_device}/run_{args.run_num}/{0}_step_{1}_integ_temp_result.txt'.format(i, args.data_device), 'w') as f:
            f.write('test_matrix : {}, \n test_acc : {}, \n test_prec : {}, \n test_reca : {}, \n test_fvu : {}'.format(temp_matrix, temp_accuracy, temp_precision, temp_recall, temp_fvu))
            
    tot_result_cpu = pd.DataFrame({'fvu' : tot_cpu_fvu,
                                'accuracy' : tot_cpu_accuracy,
                                'precision' : tot_cpu_precision,
                                'recall' : tot_cpu_recall})

    tot_result_temp = pd.DataFrame({'fvu' : tot_temp_fvu,
                                'accuracy' : tot_temp_accuracy,
                                'precision' : tot_temp_precision,
                                'recall' : tot_temp_recall})

    tot_result_cpu.to_csv(f'./saved_data/n_step/device_{args.data_device}/run_{args.run_num}/{args.data_device}_tot_q_loss_cpu_result.csv')
    tot_result_temp.to_csv(f'./saved_data/n_step/device_{args.data_device}/run_{args.run_num}/{args.data_device}_tot_q_loss_temp_result.csv')