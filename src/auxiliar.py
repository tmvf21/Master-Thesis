import os
import logging
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from keras import Sequential
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

def setup_logging(path, selected_space):
    if not os.path.exists(path):
        os.makedirs(path)
    
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = path + 'session_' + selected_space + '_' + time + '.log'
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s: %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
                            logging.StreamHandler()
                        ])

def log_model_summary(model: Sequential):
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    for line in model_summary:
        logging.info(line)

def save_model_hps(path, selected_space, best_hps: dict):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'best_hps_' + selected_space + '.json', 'w') as f:
        json.dump(best_hps, f)

def load_model_hps(path, selected_space) -> dict:
    file_path = path + 'best_hps_' + selected_space + '.json'
    if not os.path.exists(file_path):
        print(file_path + 'not found...')
        exit()
    
    with open(file_path, 'r') as f:
        best_hps = json.load(f)

    return best_hps

def ann_save_history(path, selected_space, history):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'train_history_' + selected_space + '.pkl', 'wb') as f:
        pickle.dump(history, f)

def ann_load_history(path, selected_space):
    file_path = path + 'train_history_' + selected_space + '.pkl'
    if not os.path.exists(file_path):
        print(file_path + 'not found...')
        exit()
    
    with open(file_path, 'rb') as f:
        history = pickle.load(f)
    
    return history

def ann_save_model(path, selected_space, model: Sequential):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'model_' + selected_space + '.pkl', 'wb') as f:
        pickle.dump(model, f)

def ann_load_model(path, selected_space) -> Sequential:
    file_path = path + 'model_' + selected_space + '.pkl'
    if not os.path.exists(file_path):
        print(file_path + 'not found...')
        exit()
    
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def svr_save_model(path, selected_space, model: SVR):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'model_' + selected_space + '.pkl', 'wb') as f:
        pickle.dump(model, f)

def svr_load_model(path, selected_space) -> SVR:
    file_path = path + 'model_' + selected_space + '.pkl'
    if not os.path.exists(file_path):
        print(file_path + 'not found...')
        exit()
    
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def lr_save_model(path, selected_space, model: LinearRegression):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'model_' + selected_space + '.pkl', 'wb') as f:
        pickle.dump(model, f)

def lr_load_model(path, selected_space) -> LinearRegression:
    file_path = path + 'model_' + selected_space + '.pkl'
    if not os.path.exists(file_path):
        print(file_path + 'not found...')
        exit()
    
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def seq_save_model_hps(path, selected_space, n_steps_in, best_hps: dict):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'best_hps_' + selected_space + '_' + str (n_steps_in) + '.json', 'w') as f:
        json.dump(best_hps, f)

def seq_save_history(path, selected_space, n_steps_in, history):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'train_history_' + selected_space + '_' + str (n_steps_in) + '.pkl', 'wb') as f:
        pickle.dump(history, f)

def seq_save_model(path, selected_space, n_steps_in, model: Sequential):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'model_' + selected_space + '_' + str (n_steps_in) + '.pkl', 'wb') as f:
        pickle.dump(model, f)

def seq_load_model_hps(path, selected_space, n_steps_in) -> dict:
    file_path = path + 'best_hps_' + selected_space + '_' + str (n_steps_in) + '.json'
    if not os.path.exists(file_path):
        print(file_path + 'not found...')
        exit()
    
    with open(file_path, 'r') as f:
        best_hps = json.load(f)

    return best_hps

def seq_load_history(path, selected_space, n_steps_in):
    file_path = path + 'train_history_' + selected_space + '_' + str (n_steps_in) + '.pkl'
    if not os.path.exists(file_path):
        print(file_path + 'not found...')
        exit()
    
    with open(file_path, 'rb') as f:
        history = pickle.load(f)
    
    return history

def seq_load_model(path, selected_space, n_steps_in) -> Sequential:
    file_path = path + 'model_' + selected_space + '_' + str (n_steps_in) + '.pkl'
    if not os.path.exists(file_path):
        print(file_path + 'not found...')
        exit()
    
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def find_worst_predictions(y_true, y_pred, idx: datetime, interval: str):
    idx = pd.to_datetime(idx)
    worst_mae = 0
    
    y_pred_w, y_true_w, y_w_idx = None, None, None
    
    if interval.lower() == 'week':
        weeks = np.unique(idx.isocalendar().week)
        for week in weeks:
            idx_week = idx[idx.isocalendar().week == week]
            y_true_week = y_true[idx.isocalendar().week == week]
            y_pred_week = y_pred[idx.isocalendar().week == week]

            mae = mean_absolute_error(y_true_week, y_pred_week)

            if mae > worst_mae:
                worst_mae = mae
                y_pred_w = y_pred_week
                y_true_w = y_true_week
                y_w_idx = idx_week

    elif interval.lower() == 'month':
        months = np.unique(idx.to_period('M'))
        for month in months:
            idx_month = idx[idx.to_period('M') == month]
            y_true_month = y_true[idx.to_period('M') == month]
            y_pred_month = y_pred[idx.to_period('M') == month]

            mae = mean_absolute_error(y_true_month, y_pred_month)

            if mae > worst_mae:
                worst_mae = mae
                y_pred_w = y_pred_month
                y_true_w = y_true_month
                y_w_idx = idx_month

    logging.info('Worst %s MAE: %s', interval, worst_mae)
    # print('Worst %s var:' % interval, np.var(y_pred_w))

    return y_pred_w, y_true_w, y_w_idx, worst_mae

def find_best_predictions(y_true, y_pred, idx: datetime, interval: str):
    idx = pd.to_datetime(idx)
    best_mae = 1e9

    y_pred_b, y_true_b, y_b_idx = None, None, None

    if interval.lower() == 'week':
        weeks = np.unique(idx.isocalendar().week)
        for week in weeks:
            idx_week = idx[idx.isocalendar().week == week]
            y_true_week = y_true[idx.isocalendar().week == week]
            y_pred_week = y_pred[idx.isocalendar().week == week]

            # if np.var(y_pred_week) < 1e6:
            #     continue

            mae = mean_absolute_error(y_true_week, y_pred_week)

            if mae < best_mae:
                best_mae = mae
                y_pred_b = y_pred_week
                y_true_b = y_true_week
                y_b_idx = idx_week
    
    elif interval.lower() == 'month':
        months = np.unique(idx.to_period('M'))
        for month in months:
            idx_month = idx[idx.to_period('M') == month]
            y_true_month = y_true[idx.to_period('M') == month]
            y_pred_month = y_pred[idx.to_period('M') == month]

            # if np.var(y_pred_month) < 1e9:
            #     continue

            mae = mean_absolute_error(y_true_month, y_pred_month)

            if mae < best_mae:
                best_mae = mae
                y_pred_b = y_pred_month
                y_true_b = y_true_month
                y_b_idx = idx_month
    
    print('Best %s MAE:' % interval, best_mae)
    # print('Best %s var:' % interval, np.var(y_pred_b))
    
    return y_pred_b, y_true_b, y_b_idx, best_mae

def evaluate_model(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    mse = np.power(rmse, 2)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'nrmse': nrmse,
        'mae': mae,
        'mape': mape
    }

    return metrics
