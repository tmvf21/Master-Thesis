import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from skopt import gp_minimize
from skopt.space import Real

from plots import (plot_timeseries, plot_train_test_data, plot_train_val_cv_data, 
                   plot_test_data, plot_prediction)

from auxiliar import (setup_logging, 
                      save_model_hps, load_model_hps, 
                      svr_save_model, svr_load_model,
                      find_best_predictions, find_worst_predictions,
                      evaluate_model)

data_path = './data/'
save_path = 'Models/SVR/'

def process_data(data_path, file) -> pd.DataFrame:
    df = pd.read_csv(data_path + file + '.csv', index_col='Date', parse_dates=True)

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

def create_svr_model(C, epsilon, gamma, kernel='rbf') -> SVR:

    model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)

    return model

def tune_model_cv(X_train, y_train, max_trials=30):
    trial_num = 1

    def objective(hps_list):
        nonlocal trial_num
        logging.info('Trial %s/%s:', trial_num, max_trials)

        C, epsilon, gamma = hps_list

        logging.info('Hyperparameters used:')
        logging.info('C: %s', C)
        logging.info('epsilon: %s', epsilon)
        logging.info('gamma: %s', gamma)

        model = create_svr_model(C, epsilon, gamma)

        tscv = TimeSeriesSplit(n_splits=5)
        val_losses = []

        for train_idx, val_idx in tscv.split(X_train):
            X_train_cv, y_train_cv = X_train[train_idx], y_train[train_idx]
            X_val_cv, y_val_cv = X_train[val_idx], y_train[val_idx]

            model.fit(X_train_cv, y_train_cv.ravel())

            y_pred_cv = model.predict(X_val_cv)
            val_loss = np.power(root_mean_squared_error(y_val_cv, y_pred_cv), 2)
            logging.info('Fold %s val_loss: %s', len(val_losses) + 1, val_loss)
            val_losses.append(val_loss)
            
        mean_val_losses = np.mean(val_losses)
        logging.info('Mean val_loss of trial %s: %s', trial_num, mean_val_losses)
        
        trial_num += 1

        return mean_val_losses
    
    search_space = [
        Real(1e-3, 100.0, prior='log-uniform', name='C'),
        Real(1e-4, 0.5, prior='log-uniform', name='epsilon'),
        Real(1e-5, 1.0, prior='log-uniform', name='gamma')
    ]
    
    res = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=max_trials,
        random_state=42,
        verbose=True
    )

    best_hps = {
        'C': res.x[0],
        'epsilon': res.x[1],
        'gamma': res.x[2]
    }

    return best_hps

def fit_model(model: SVR, X_train, y_train):
    model.fit(X_train, y_train.ravel())
    
    return model

if __name__ == '__main__':
    tuning = False
    training = False

    spaces = ['Alameda', 'Torre_Norte', 'LSDC1']
    selected_space = spaces[1]

    start_test_data = datetime.strptime("01/09/2023", "%d/%m/%Y")

    setup_logging(save_path + 'logs/', selected_space)
    logging.info('Processing data for space: %s', selected_space)
    
    processed_data = process_data(data_path, 'Data_' + selected_space)
    # processed_data = processed_data.loc['2023-03-01':'2023-03-07']
    
    logging.info('Processed data shape: %s', processed_data.shape)
    logging.info('Processed data columns: %s', processed_data.columns)
    # print(processed_data.head())
    # print(processed_data.corr())
    
    X_processed, y_processed = processed_data.drop(columns=['Wh'], axis=1).values, processed_data.loc[:, 'Wh'].values
    y_processed = y_processed.reshape(-1, 1)
    
    processed_idx = processed_data.index
    
    # plot_timeseries(save_path, selected_space, processed_idx, y_processed)

    logging.info('X_processed.shape: %s', X_processed.shape)
    logging.info('y_processed.shape: %s', y_processed.shape)
    
    split_data = split_dataset(X_processed, y_processed, processed_idx, start_test_data)
    
    X_train, y_train, idx_train = split_data['train']
    X_test, y_test, idx_test = split_data['test']

    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    # plot_train_test_data(save_path, selected_space, y_train, idx_train, y_test, idx_test)
    # plot_train_val_cv_data(save_path, selected_space, y_train, idx_train)

    logging.info('Train time interval: from %s, until %s', idx_train[0], idx_train[-1])
    logging.info('Test time interval: from %s, until %s', idx_test[0], idx_test[-1])

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # scaler_x = StandardScaler()
    # scaler_y = StandardScaler()

    logging.info('Scaler X: %s', scaler_x)
    logging.info('Scaler Y: %s', scaler_y)

    input_scaler = scaler_x.fit(X_train)
    output_scaler = scaler_y.fit(y_train)

    X_train = input_scaler.transform(X_train)
    y_train = output_scaler.transform(y_train)

    X_test = input_scaler.transform(X_test)
    y_test = output_scaler.transform(y_test)

    X_train, y_train = X_train.astype(np.float32), y_train.astype(np.float32)
    X_test, y_test = X_test.astype(np.float32), y_test.astype(np.float32)

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

        save_model_hps(save_path, selected_space, best_hps)
        logging.info('Model hyperparameters saved...')
    
    else:
        logging.info('Tuning skipped, loading tuned hyperparameters...')
        
        best_hps = load_model_hps(save_path, selected_space)

        logging.info('Best hyperparameters: %s', best_hps)

    tuned_model = create_svr_model(C=best_hps['C'], epsilon=best_hps['epsilon'], gamma=best_hps['gamma'])
        
    if training:
        logging.info('Training model...')
        trained_model = fit_model(tuned_model, X_train, y_train)

        svr_save_model(save_path, selected_space, trained_model)
        logging.info('Model saved...')

    elif tuning and not training:
        logging.info('Training skipped, hyperparameters tuned...')
        logging.info('Exiting...')
        exit()
        
    else:
        logging.info('Training skipped, loading trained model and history...')
        trained_model = svr_load_model(save_path, selected_space)
    
    logging.info('Testing model...')
    y_pred = trained_model.predict(X_test)

    y_pred = output_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test = output_scaler.inverse_transform(y_test)
    
    logging.info('y_pred.shape: %s', y_pred.shape)
    logging.info('y_test.shape: %s', y_test.shape)

    # plot_test_data(save_path, selected_space, y_test, idx_test)
    plot_prediction(save_path, selected_space, y_pred, y_test, idx_test)

    y_pred_ww, y_true_ww, y_ww_idx, ww_mape = find_worst_predictions(y_test, y_pred, idx_test, 'week')
    plot_prediction(save_path, selected_space + '_ww', y_pred_ww, y_true_ww, y_ww_idx)
    # y_pred_wm, y_true_wm, y_wm_idx, wm_mape = find_worst_predictions(y_test, y_pred, idx_test, 'month')
    # plot_prediction(save_path, selected_space + '_wm', y_pred_wm, y_true_wm, y_wm_idx)

    # y_pred_bw, y_true_bw, y_bw_idx, bw_mape = find_best_predictions(y_test, y_pred, idx_test, 'week')
    # plot_prediction(save_path, selected_space + '_bw', y_pred_bw, y_true_bw, y_bw_idx)
    # y_pred_bm, y_true_bm, y_bm_idx, bm_mape = find_best_predictions(y_test, y_pred, idx_test, 'month')
    # plot_prediction(save_path, selected_space + '_bm', y_pred_bm, y_true_bm, y_bm_idx)

    metrics = evaluate_model(y_test, y_pred)

    logging.info('Metrics:')
    logging.info('MSE: %s', metrics['mse'])
    logging.info('RMSE: %s', metrics['rmse'])
    logging.info('NRMSE: %s', metrics['nrmse'])
    logging.info('MAE: %s', metrics['mae'])
    logging.info('MAPE: %s', metrics['mape'])

    logging.info('Done!')