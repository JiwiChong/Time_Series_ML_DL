import argparse
import os
import pickle
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from data_preprocessing import data_cleansing, normalized_data
from loss_functions import tilted_loss, custom_loss
from models import forecasting_model
from keras.callbacks import ModelCheckpoint


def forecast(args, model, x_train, y_train, x_valid, y_valid):
    if args.loss_function == 'tilt':
        checkpoint_dir = f'./{args.model_dir}/device_{args.data_device}/aug_{args.forecast_data}_best_model_quantile_{args.tilt_loss_q}_run_{args.run_num}.hdf5'
    elif args.loss_function == 'compile':
        checkpoint_dir = f'./{args.model_dir}/device_{args.data_device}/aug_{args.forecast_data}_best_model_compile_{args.custom_loss_prop}_run_{args.run_num}.hdf5'
    checkpoint = [ModelCheckpoint(filepath = f'./{args.model_dir}/device_{args.data_device}/aug_{args.forecast_data}_best_model_quantile_{args.tilt_loss_q}_run_{args.run_num}.hdf5', save_best_only = True, monitor = 'val_loss', verbose = 0)] 
    cpu_hist = model.fit(x_train, y_train, epochs=args.epochs, batch_size = args.batch_size, callbacks= checkpoint, validation_data=(x_valid, y_valid))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecasting of device data')
    parser.add_argument('--data_dir', type=str, help='Main directory of input dataset')
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

    if args.data_device == 28:
        df_2018 = pd.read_csv(os.path.join(args.data_dir, f'device_{args.data_device}/{args.data_device}_2018.csv'))
        df_2019 = pd.read_csv(os.path.join(args.data_dir, f'device_{args.data_device}/{args.data_device}_2019.csv'))
    elif args.data_device == 218:
        df_2018 = pd.read_csv(os.path.join(args.data_dir, f'device_{args.data_device}/{args.data_device}_2018.csv'))
        df_2019 = pd.read_csv(os.path.join(args.data_dir, f'device_{args.data_device}/{args.data_device}_2019.csv'))
    
    entire_data = pd.concat([df_2018, df_2019])
    entire_data = data_cleansing(entire_data)
    post_norm_data = normalized_data(entire_data)

    data = post_norm_data if args.scale_data else entire_data

    x_train, x_valid, x_test, y_cpu_train, y_cpu_valid, y_cpu_test, y_temp_train, \
    y_temp_valid, y_temp_test, dataX, nstep_dataX, data_cpu_Y, data_temp_Y, train_index,\
    valid_index, test_index = (data)

    data_list = [x_train, x_valid, x_test, y_cpu_train, y_cpu_valid, y_cpu_test, 
                 y_temp_train, y_temp_valid, y_temp_test, dataX, nstep_dataX, data_cpu_Y,
                   data_temp_Y, train_index, valid_index, test_index]
    
    with open(f'./dataset/device_{args.data_device}/data_w_moving_average/28_data.pickle', 'wb') as f:
        pickle.dump(data_list, f)

    if args.forecast_data == 'cpu':
        y_train, y_valid = y_cpu_train, y_cpu_valid
    else:
        y_train, y_valid = y_temp_train, y_temp_valid

    if args.model_name == 'lstm':
        model = forecasting_model(args, x_train)

    forecast(args, model, x_train, y_train, x_valid, y_valid)

# python main.py --data_dir .dataset/ --seq_len 12