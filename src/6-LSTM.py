import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from keras import Sequential, layers, optimizers, callbacks

from skopt import gp_minimize
from skopt.space import Integer, Categorical

from plots import (plot_timeseries_seq, plot_train_test_data_seq, plot_train_val_cv_data_seq, 
                   plot_loss_seq, plot_test_data_seq, plot_prediction_seq)

from auxiliar import (setup_logging, log_model_summary,
                      seq_save_model_hps, seq_load_model_hps,
                      seq_save_model, seq_load_model,
                      seq_save_history, seq_load_history,
                      find_best_predictions, find_worst_predictions,
                      evaluate_model)

data_path = './data/'
save_path = 'Models/LSTM/'

def process_data(data_path, file) -> pd.DataFrame:
    df = pd.read_csv(data_path + file + '.csv', index_col='Date', parse_dates=True)

    df.insert(1, 'Week_Day_Sin', np.sin(2 * np.pi * df['Week_Day'] / 7))
    df.insert(3, 'Hour_Sin', np.sin(2 * np.pi * df['Hour'] / 24))
    df.insert(4, 'Week_Day_Cos', np.cos(2 * np.pi * df['Week_Day'] / 7))
    df.insert(5, 'Hour_Cos', np.cos(2 * np.pi * df['Hour'] / 24))

    df = df.drop(columns=['Year', 'Month', 'Day', 'Week_Num', 'Week_Day', 'Hour'])

    return df

def split_dataset(x_data, y_data, idx, start_test_data: datetime) -> dict[str, tuple[np.ndarray]]:
    X_train = x_data[idx < start_test_data]
    y_train = y_data[idx < start_test_data]
    idx_train = idx[idx < start_test_data]

    X_test = x_data[idx >= start_test_data]
    y_test = y_data[idx >= start_test_data]
    idx_test = idx[idx >= start_test_data]

    data = {
        'train': (X_train, y_train, idx_train),
        'test': (X_test, y_test, idx_test)
    }

    return data

def create_sequences(features, targets, indices, n_steps_in, n_steps_out=1, n_sliding_steps=1, window_type='sliding'):
    """
    Edited from 
    https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    
    Args:
    * features: univariate or multivariate input sequences
    * targets: univariate or multivariate output sequences
    * n_steps_in: length of input sequence for sliding window. 
    * n_steps_out: length of output sequence
    * n_sliding_steps: window step size
    * window_type: 'sliding' or 'expanding'
    """
    X, y, seq_indices = list(), list(), list()

    for i in range(0, len(features), n_sliding_steps):

        # Find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        # Check if we are beyond the sequence length
        if out_end_ix > len(features):
            break

        # Gather input and output parts of the pattern
        if window_type == 'sliding':  # Sliding window
            seq_x, seq_y = features[i:end_ix, :], targets[end_ix:out_end_ix, :]
        else:  # Expanding window
            seq_x, seq_y = features[0:end_ix, :], targets[end_ix:out_end_ix, :]

        X.append(seq_x)
        y.append(seq_y)
        
        # Add the index corresponding to the first target in the sequence
        seq_indices.append(indices[end_ix])

    return np.array(X), np.array(y), np.array(seq_indices)

def create_lstm_model(n_layers, units, lr, seq_len, n_features) -> Sequential:
    model = Sequential()
    model.add(layers.Input(shape=(seq_len, n_features)))

    for i in range(n_layers):
        model.add(layers.LSTM(units=units, return_sequences=(i != n_layers - 1)))

    model.add(layers.Dense(units=1))

    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mean_squared_error')

    return model

def tune_model_cv(X_train, y_train, max_trials=30):
    ''' Tune LSTM model hyperparameters using Bayesian Optimization and Time Series Cross-Validation. '''
    trial_num = 1
    
    def objective(hps_list):
        nonlocal trial_num, max_trials
        logging.info('Trial %s/%s:', trial_num, max_trials)

        n_layers = int (hps_list[0])
        units = int (hps_list[1])
        learning_rate = hps_list[2]
        batch_size = int (hps_list[3])

        logging.info('Hyperparameters used:')
        logging.info('num_layers: %s', n_layers)
        logging.info('units: %s', units)
        logging.info('learning_rate: %s', learning_rate)
        logging.info('batch_size: %s', batch_size)
        
        model = create_lstm_model(n_layers=n_layers, 
                                  units=units, 
                                  lr=learning_rate, 
                                  seq_len=X_train.shape[1], 
                                  n_features=X_train.shape[2])
        
        tscv = TimeSeriesSplit(n_splits=5)
        val_losses = []
        skip_trial = False

        early_stop = callbacks.EarlyStopping(monitor='loss', patience=3, mode='min', restore_best_weights=False)

        for train_idx, val_idx in tscv.split(X_train):
            X_train_cv, y_train_cv = X_train[train_idx], y_train[train_idx]
            X_val_cv, y_val_cv = X_train[val_idx], y_train[val_idx]

            # logging.info('X_train_cv.shape: %s', X_train_cv.shape)
            # logging.info('y_train_cv.shape: %s', y_train_cv.shape)
            # logging.info('X_val_cv.shape: %s', X_val_cv.shape)
            # logging.info('y_val_cv.shape: %s', y_val_cv.shape)

            model.fit(X_train_cv, y_train_cv, 
                      batch_size=batch_size, 
                      epochs=20, 
                      callbacks=[early_stop], 
                    #   validation_data=(X_val_cv, y_val_cv),
                    #   shuffle=False,
                      verbose=0)

            val_loss = model.evaluate(X_val_cv, y_val_cv, batch_size=batch_size, verbose=0)
            if np.isnan(val_loss):
                logging.info('Fold %s val_loss is NaN, setting to inf.', len(val_losses) + 1)
                val_loss = 1e9
                skip_trial = True
            
            logging.info('Fold %s val_loss: %s', len(val_losses) + 1, val_loss)
            val_losses.append(val_loss)
            
            if skip_trial:
                break
        
        mean_val_losses = np.mean(val_losses)
        logging.info('Mean val_loss of trial %s: %s', trial_num, mean_val_losses)

        trial_num += 1

        return mean_val_losses

    search_space = [
        Integer(1, 2, name='num_layers'), 
        # Categorical([32, 48, 64, 96, 128, 192, 256], name='units'),
        Categorical([32, 48, 64, 96, 128], name='units'),
        Categorical([0.0005, 0.001, 0.004, 0.008, 0.01], name='learning_rate'),
        # Categorical([8, 16, 32, 48, 64, 96, 128, 192, 256], name='batch_size')
        Categorical([16, 32, 48, 64, 96, 128, 192, 256], name='batch_size')
    ]

    # Perform Bayesian Optimization with gp_minimize
    res = gp_minimize(
        func=objective, 
        dimensions=search_space, 
        n_calls=max_trials,  
        verbose=True
    )

    best_hps = {
        'num_layers': int (res.x[0]),
        'units': int (res.x[1]),
        'learning_rate': float (res.x[2]),
        'batch_size': int (res.x[3])
    }

    return best_hps

def fit_model(model: Sequential, X_train, y_train, epochs, batch_size):
    early_stop = callbacks.EarlyStopping(monitor='loss', patience=5, mode='min', restore_best_weights=True)
    
    history = model.fit(X_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        callbacks=[early_stop], 
                        # shuffle=False,
                        verbose=1)
    
    return model, history

if __name__ == '__main__':
    tuning = False
    training = False
    
    N_STEPS_IN = 24
    start_test_data = datetime.strptime("01/09/2023", "%d/%m/%Y")

    spaces = ['Alameda', 'Torre_Norte', 'LSDC1']
    selected_space = spaces[2]

    setup_logging(save_path + 'logs/', selected_space)
    logging.info('Processing data for space: %s', selected_space)
    
    processed_data = process_data(data_path, 'Data_' + selected_space)
    # processed_data = processed_data.loc['2023-01-01':'2024-06-01']

    logging.info('Processed data shape: %s', processed_data.shape)
    logging.info('Processed data columns: %s', processed_data.columns)
    # print(processed_data.head())
    # print(processed_data.corr())
    
    X_processed, y_processed = processed_data.drop(columns=['Wh'], axis=1).values, processed_data.loc[:, 'Wh'].values
    y_processed = y_processed.reshape(-1, 1)
    
    processed_idx = processed_data.index
    
    # plot_timeseries_seq(save_path, selected_space, N_STEPS_IN, processed_idx, y_processed)

    logging.info('X_processed.shape: %s', X_processed.shape)
    logging.info('y_processed.shape: %s', y_processed.shape)

    logging.info('Creating sequences...')

    X_sequences, y_sequences, indices_sequences = create_sequences(features=X_processed, 
                                                                  targets=y_processed, 
                                                                  indices=processed_idx, 
                                                                  n_steps_in=N_STEPS_IN)
    
    split_data = split_dataset(X_sequences, y_sequences, indices_sequences, start_test_data)

    X_train, y_train, idx_train = split_data['train']
    X_test, y_test, idx_test = split_data['test']

    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    
    # plot_train_test_data_seq(save_path, selected_space, N_STEPS_IN, y_train, idx_train, y_test, idx_test)
    # plot_train_val_cv_data_seq(save_path, selected_space, N_STEPS_IN, y_train, idx_train)

    logging.info('Train time interval: from %s, until %s', idx_train[0], idx_train[-1])
    logging.info('Test time interval: from %s, until %s', idx_test[0], idx_test[-1])

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # scaler_x = StandardScaler()
    # scaler_y = StandardScaler()

    logging.info('Scaler X: %s', scaler_x)
    logging.info('Scaler Y: %s', scaler_y)

    input_scaler = scaler_x.fit(X_train.reshape(-1, X_train.shape[2]))
    output_scaler = scaler_y.fit(y_train)

    X_train = input_scaler.transform(X_train.reshape(-1, X_train.shape[2])).reshape(-1, N_STEPS_IN, X_train.shape[2])
    y_train = output_scaler.transform(y_train)

    X_test = input_scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(-1, N_STEPS_IN, X_test.shape[2])
    y_test = output_scaler.transform(y_test)

    X_train, y_train = X_train.astype('float32'), y_train.astype('float32')
    X_test, y_test = X_test.astype('float32'), y_test.astype('float32')

    logging.info('X_train.shape: %s', X_train.shape)
    logging.info('y_train.shape: %s', y_train.shape)
    logging.info('X_test.shape: %s', X_test.shape)
    logging.info('y_test.shape: %s', y_test.shape)

    logging.info('X_train dtype: %s', X_train.dtype)
    logging.info('y_train dtype: %s', y_train.dtype)
    logging.info('X_test dtype: %s', X_test.dtype)
    logging.info('y_test dtype: %s', y_test.dtype)

    if tuning:
        logging.info('Tuning model...')

        best_hps = tune_model_cv(X_train, y_train)

        logging.info('Tuning done...')
        logging.info('Best hyperparameters: %s', best_hps)

        seq_save_model_hps(save_path, selected_space, N_STEPS_IN, best_hps)
        logging.info('Model hyperparameters saved...')
        
    else:
        logging.info('Tuning skipped, loading tuned hyperparameters...')
        
        best_hps = seq_load_model_hps(save_path, selected_space, N_STEPS_IN)

        logging.info('Best hyperparameters: %s', best_hps)

    tuned_model = create_lstm_model(n_layers=best_hps.get('num_layers'),
                                    units=best_hps.get('units'),
                                    lr=best_hps.get('learning_rate'),
                                    seq_len=X_train.shape[1],
                                    n_features=X_train.shape[2])

    log_model_summary(tuned_model)
    
    if training:
        logging.info('Training model...')

        trained_model, history = fit_model(tuned_model, X_train, y_train,  
                            epochs=200, batch_size=best_hps.get('batch_size'))
        
        
        plot_loss_seq(save_path, selected_space, N_STEPS_IN, history)

        seq_save_model(save_path, selected_space, N_STEPS_IN, trained_model)
        seq_save_history(save_path, selected_space, N_STEPS_IN, history)
        logging.info('Model and training history saved...')
    
    elif tuning and not training:
        logging.info('Training skipped, hyperparameters tuned...')
        logging.info('Exiting...')
        exit()

    else:
        logging.info('Training skipped, loading trained model and history...')
        trained_model = seq_load_model(save_path, selected_space, N_STEPS_IN)
        history = seq_load_history(save_path, selected_space, N_STEPS_IN)
        # plot_loss_seq(save_path, selected_space, N_STEPS_IN, history)
    
    logging.info('Testing model...')
    y_pred = trained_model.predict(X_test)

    y_pred = output_scaler.inverse_transform(y_pred)
    y_test = output_scaler.inverse_transform(y_test)
    
    logging.info('y_pred.shape: %s', y_pred.shape)
    logging.info('y_test.shape: %s', y_test.shape)

    # plot_test_data_seq(save_path, selected_space, N_STEPS_IN, y_test, idx_test)
    plot_prediction_seq(save_path, selected_space, N_STEPS_IN, y_pred, y_test, idx_test)

    y_pred_ww, y_true_ww, y_ww_idx, ww_mape = find_worst_predictions(y_test, y_pred, idx_test, 'week')
    plot_prediction_seq(save_path, selected_space + '_ww', N_STEPS_IN, y_pred_ww, y_true_ww, y_ww_idx)
    # y_pred_wm, y_true_wm, y_wm_idx, wm_mape = find_worst_predictions(y_test, y_pred, idx_test, 'month')
    # plot_prediction_seq(save_path, selected_space + '_wm', N_STEPS_IN, y_pred_wm, y_true_wm, y_wm_idx)

    # y_pred_bw, y_true_bw, y_bw_idx, bw_mape = find_best_predictions(y_test, y_pred, idx_test, 'week')
    # plot_prediction_seq(save_path, selected_space + '_bw', N_STEPS_IN, y_pred_bw, y_true_bw, y_bw_idx)
    # y_pred_bm, y_true_bm, y_bm_idx, bm_mape = find_best_predictions(y_test, y_pred, idx_test, 'month')
    # plot_prediction_seq(save_path, selected_space + '_bm', N_STEPS_IN, y_pred_bm, y_true_bm, y_bm_idx)

    metrics = evaluate_model(y_test, y_pred)

    logging.info('Metrics:')
    logging.info('MSE: %s', metrics['mse'])
    logging.info('RMSE: %s', metrics['rmse'])
    logging.info('NRMSE: %s', metrics['nrmse'])
    logging.info('MAE: %s', metrics['mae'])
    logging.info('MAPE: %s', metrics['mape'])

    logging.info('Done!')