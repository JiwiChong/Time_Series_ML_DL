from keras.layers import Input, Dense, CuDNNLSTM, GaussianNoise
from keras.optimizers import Adam
from keras.models import Model, Sequential
from loss_functions import tilted_loss, custom_loss

def forecasting_model(args, x_train):
    inp = Input(shape=(x_train.shape[1], x_train.shape[2]), name='input')
    noise = GaussianNoise(0.1)(inp)

    layer1 = CuDNNLSTM(256, return_sequences=False)(noise)
    layer2 = Dense(128, activation='relu')(layer1)
    outp = Dense(2)(layer2)

    model = Model(inputs=inp, outputs=outp)
    adam_optimizer = Adam(args.learning_rate)

    if args.loss_function == 'tilt':
        loss_func = lambda t, p: tilted_loss(0.4,t, p)
    elif args.loss_function == 'custom':
        loss_func = lambda t, p: tilted_loss(t, p)


    model.compile(loss=loss_func,
                  optimizer=adam_optimizer)

    return model