import argparse
import pickle
import numpy as np
from models import forecasting_model
from one_step import one_step_forecasting
from n_step import n_step_forecasting
from sklearn.metrics import r2_score, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score,roc_curve, auc, roc_auc_score 
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecasting of device data')
    parser.add_argument('--data_dir', type=str, help='Main directory of input dataset')
    parser.add_argument('--one_step', type=str, help='One step forecasting')
    parser.add_argument('--n_step', type=str, help='N-step forecasting')
    parser.add_argument('--n_step_size', type=str, help='N-step forecasting size')
    parser.add_argument('--scale_data', type=str, help='Whether to scale the data features or not')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--data_device', type=int, help='Data of selected device')
    parser.add_argument('--seq_len', type=int, help='Length of data sequence')
    parser.add_argument('--shuffling', type=int, default=True, help='Whether to shuffle data or not')
    parser.add_argument('--forecast_data', type=int, help='Variable to forecast')
    parser.add_argument('--loss_function', type=str, help='Loss function to use')
    parser.add_argument('--tilt_loss_q', type=float, help='Q value in tilt loss function')
    parser.add_argument('--custom_loss_prop', type=float, help='Proportion value in custom loss function')
    parser.add_argument('--learning_rate', type=float, default=0.0001,help='learning rate')
    parser.add_argument('--momentum', type=float, help='Momentum in learning rate')
    parser.add_argument('--validate', type=bool, default=False, help='whether to validate the model or not')
    parser.add_argument('--model_name', type=str, help='algorithm')
    parser.add_argument('--alpha', type=float, help='Reduction factor in Loss function')
    parser.add_argument('--optim_w_decay', type=float, help='optimizer weight decay value')
    parser.add_argument('--lr_decay', type=float, help='LR decay value')
    parser.add_argument('--num_epochs_decay', type=int, help='Number of epoch after which learning rate decays')
    parser.add_argument('--run_num', type=str, help='run number')
    args = parser.parse_args()

    if args.scale_data:
        cpu_thres, temp_thres, cpu_mean_thres, temp_mean_thres = 3.863, 2.654, 3.704, 2.385
    else:
        cpu_thres, temp_thres, cpu_mean_thres, temp_mean_thres = 73, 59, 66.5, 58.5

    with open(f'./dataset/device_{args.data_device}/data_w_moving_average/28_data.pickle', 'rb') as f :
        loaded_ma_dataset = pickle.load(f)

    x_train, x_valid, x_test, y_cpu_train, y_cpu_valid, y_cpu_test, y_temp_train, \
    y_temp_valid, y_temp_test, dataX, nstep_dataX, data_cpu_Y, data_temp_Y, train_index,\
    valid_index, test_index = loaded_ma_dataset
    
    if args.model_name == 'lstm':
        model = forecasting_model(args, x_train)

    if args.one_step:
        one_step_forecasting(args, model, x_test, cpu_thres, temp_thres, cpu_mean_thres, temp_mean_thres)
    elif args.n_step:
        n_step_forecasting(args, test_index, dataX, data_cpu_Y, data_temp_Y, nstep_dataX, cpu_thres, temp_thres)

        