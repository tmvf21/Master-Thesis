import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

font = {'size': 14}
plt.rc('font', **font)

def plot_timeseries_seq(path, selected_space, n_steps_in, x_axis, y_axis, start_date=None, end_date=None):
    if not os.path.exists(path + 'Img/' + selected_space + '/'):
        os.makedirs(path + 'Img/' + selected_space + '/')
    
    save_path = path + 'Img/' + selected_space + '/dataset_' + selected_space + '_' + str (n_steps_in) + '.png'
    
    if start_date is None:
        start_date = x_axis[0]
    if end_date is None:
        end_date = x_axis[-1]

    # convert to kWh
    y_axis = y_axis * 1e-3

    max_value = np.max(y_axis)
    min_value = np.min(y_axis)
    ylim_upper = int(max_value * 1.2)
    ylim_lower = 0 if min_value > 0 else (min_value * 1.1)

    plt.figure(figsize = (10, 6))
    plt.plot(x_axis, y_axis, linewidth=0.8)
    plt.xlabel('Date')
    plt.ylabel('Electricity Consumption [kWh]')
    plt.xlim(start_date, end_date)
    plt.ylim(ylim_lower, ylim_upper)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=400)
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=400)
    plt.close()

def plot_test_data_seq(path, selected_space, n_steps_in, test_data, test_idx):
    if not os.path.exists(path + 'Img/' + selected_space + '/'):
        os.makedirs(path + 'Img/' + selected_space + '/')

    save_path = path + 'Img/' + selected_space + '/test_data_' + selected_space + '_' + str (n_steps_in) + '.png'
    
    # convert to kWh
    test_data = test_data * 1e-3

    max_value =  np.max(test_data)
    min_value = np.min(test_data)
    ylim_upper = int(max_value * 1.2)
    ylim_lower = 0 if min_value > 0 else (min_value * 1.1)
    
    plt.figure(figsize = (10, 6))
    plt.plot(test_idx, test_data, linewidth=0.8, color='tab:orange')
    plt.xlabel('Date')
    plt.ylabel('Electricity Consumption [kWh]')
    plt.xlim(test_idx[0], test_idx[-1])
    plt.ylim(ylim_lower, ylim_upper)
    plt.grid(which='both', linestyle='--', linewidth=0.5)   
    plt.tight_layout() 
    plt.savefig(save_path, format='png', dpi=400)
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=400)
    plt.close()

def plot_train_test_data_seq(path, selected_space, n_steps_in, train_data, train_idx, test_data, test_idx):
    if not os.path.exists(path + 'Img/' + selected_space + '/'):
        os.makedirs(path + 'Img/' + selected_space + '/')

    # convert to kWh
    train_data, test_data = train_data * 1e-3, test_data * 1e-3
    
    max_value = max(np.max(train_data), np.max(test_data))
    min_value = min(np.min(train_data), np.min(test_data))
    ylim_upper = int(max_value * 1.2)
    ylim_lower = 0 if min_value > 0 else (min_value * 1.1)
    
    plt.figure(figsize = (10, 6))
    plt.plot(train_idx, train_data, label='Train set', linewidth=0.8, color='tab:blue')
    plt.plot(test_idx, test_data, label='Test set', linewidth=0.8, color='tab:orange')
    plt.xlabel('Date')
    plt.ylabel('Electricity Consumption [kWh]')
    plt.xlim(train_idx[0], test_idx[-1])
    plt.ylim(ylim_lower, ylim_upper)
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)

    save_path = path + 'Img/' + selected_space + '/data_split_' + selected_space + '_' + str (n_steps_in) + '.png'
    
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=400)
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=400)
    plt.close()

def plot_train_val_cv_data_seq(path, selected_space, n_steps_in, train_data, train_idx):
    if not os.path.exists(path + 'Img/' + selected_space + '/'):
        os.makedirs(path + 'Img/' + selected_space + '/')
    
    save_path = path + 'Img/' + selected_space + '/train_val_cv_splits_' + selected_space + '_' + str (n_steps_in) + '.png'
    
    tscv = TimeSeriesSplit(n_splits=5)

    # convert to kWh
    train_data = train_data * 1e-3
    
    max_value = np.max(train_data)
    min_value = np.min(train_data)
    ylim_upper = int(max_value * 1.2)
    ylim_lower = 0 if min_value > 0 else (min_value * 1.1)
    
    n_folds = tscv.get_n_splits()
    fig, axes = plt.subplots(n_folds, 1, figsize=(12, 1.2 * n_folds), sharex=True)

    for fold, (train_indices, val_indices) in enumerate(tscv.split(train_data)):
        train_data_cv, val_data_cv = train_data[train_indices], train_data[val_indices]
        train_idx_cv, val_idx_cv = train_idx[train_indices], train_idx[val_indices]

        axes[fold].plot(train_idx_cv, train_data_cv, label='Train', linewidth=0.6, color='tab:blue')
        axes[fold].plot(val_idx_cv, val_data_cv, label='Validation', linewidth=0.6, color='tab:green')
        axes[fold].set_title(f'Fold {fold + 1}')
        axes[fold].set_xlim(train_idx[0], train_idx[-1])
        # axes[fold].set_ylim(ylim_lower, ylim_upper)
        if fold == 0:
            axes[fold].legend()

    axes[-1].set_xlabel('Date')
    axes[2].set_ylabel('Electricity Consumption [kWh]')

    plt.tight_layout()    
    plt.savefig(save_path, format='png', dpi=400)
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=400)
    plt.close()

def plot_loss_seq(path, selected_space, n_steps_in, history):
    if not os.path.exists(path + 'Img/' + selected_space + '/'):
        os.makedirs(path + 'Img/' + selected_space + '/')
    
    save_path = path + 'Img/' + selected_space + '/loss_' + selected_space + '_' + str (n_steps_in) + '.png'
    
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'], label='Train loss')
    # plt.plot(history.history['val_loss'], label='Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=400)
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=400)
    plt.close()

def plot_prediction_seq(path, selected_space, n_steps_in, y_pred, y_test, index): 
    if selected_space[-3:] in ['_bw', '_bm', '_ww', '_wm']:
        selected_space_aux = selected_space[:-3]
    else:    
        selected_space_aux = selected_space
    
    if not os.path.exists(path + 'Img/' + selected_space_aux + '/'):
        os.makedirs(path + 'Img/' + selected_space_aux + '/')
    
    save_path = path + 'Img/' + selected_space_aux + '/pred_' + selected_space + '_' + str (n_steps_in) + '.png'
    
    # convert to kWh
    y_pred, y_test = y_pred * 1e-3, y_test * 1e-3

    if selected_space_aux == 'Alameda':
        max_value = 220
        ylim_upper = int(max_value)
    elif selected_space_aux == 'Torre_Norte':
        max_value = 40
        ylim_upper = int(max_value)
    elif selected_space_aux == 'LSDC1':
        max_value = 14
        ylim_upper = int(max_value)
    else:
        max_value = max(np.max(y_test), np.max(y_pred))
        ylim_upper = int(max_value * 1.2)
    
    min_value = min(np.min(y_test), np.min(y_pred))
    ylim_lower = 0 if min_value > 0 else (min_value * 1.1)

    plt.figure(figsize=(10, 6))
    plt.plot(index, np.array(y_test), label='True', marker='o', markersize=0.4, linestyle='-', linewidth=0.8)
    plt.plot(index, np.array(y_pred), label='Prediction', marker='o', markersize=0.4, linestyle='-', linewidth=0.8)
    plt.legend()
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Electricity Consumption [kWh]', fontsize=16)
    plt.xlim(index[0], index[-1])
    plt.ylim(ylim_lower, ylim_upper)
    plt.xticks(rotation=15, fontsize=13)
    plt.grid(which='both', linestyle='--', linewidth=0.5)    
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=400)
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=400)
    plt.close()

def plot_timeseries (path, selected_space, x_axis, y_axis, start_date=None, end_date=None):
    if not os.path.exists(path + 'Img/' + selected_space + '/'):
        os.makedirs(path + 'Img/' + selected_space + '/')
    
    save_path = path + 'Img/' + selected_space + '/dataset_' + selected_space + '.png'

    if start_date is None:
        start_date = x_axis[0]
    if end_date is None:
        end_date = x_axis[-1]

    # convert to kWh
    y_axis = y_axis * 1e-3

    max_value = np.max(y_axis)
    min_value = np.min(y_axis)
    ylim_upper = int(max_value * 1.2)
    ylim_lower = 0 if min_value > 0 else (min_value * 1.1)

    plt.figure(figsize = (12, 6))
    plt.plot(x_axis, y_axis, linewidth=0.8)
    plt.xlabel('Date')
    plt.ylabel('Electricity Consumption [kWh]')
    plt.xlim(start_date, end_date)
    plt.ylim(ylim_lower, ylim_upper)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=400)
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=400)
    plt.close()

def plot_test_data(path, selected_space, test_data, test_idx):
    if not os.path.exists(path + 'Img/' + selected_space + '/'):
        os.makedirs(path + 'Img/' + selected_space + '/')
    
    save_path = path + 'Img/' + selected_space + '/test_data_' + selected_space + '.png'

    # convert to kWh
    test_data = test_data * 1e-3

    max_value =  np.max(test_data)
    min_value = np.min(test_data)
    ylim_upper = int(max_value * 1.2)
    ylim_lower = 0 if min_value > 0 else (min_value * 1.1)
    
    plt.figure(figsize = (12, 6))
    plt.plot(test_idx, test_data, linewidth=0.8, color='tab:orange')
    plt.xlabel('Date')
    plt.ylabel('Electricity Consumption [kWh]')
    plt.xlim(test_idx[0], test_idx[-1])
    plt.ylim(ylim_lower, ylim_upper)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout() 
    plt.savefig(save_path, format='png', dpi=400)
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=400)
    plt.close()

def plot_train_test_data(path, selected_space, train_data, train_idx, test_data, test_idx):
    if not os.path.exists(path + 'Img/' + selected_space + '/'):
        os.makedirs(path + 'Img/' + selected_space + '/')

    save_path = path + 'Img/' + selected_space + '/data_split_' + selected_space + '.png'
    
    # convert to kWh
    train_data, test_data = train_data * 1e-3, test_data * 1e-3
    
    max_value = max(np.max(train_data), np.max(test_data))
    min_value = min(np.min(train_data), np.min(test_data))
    ylim_upper = int(max_value * 1.2)
    ylim_lower = 0 if min_value > 0 else (min_value * 1.1)
    
    plt.figure(figsize = (12, 6))
    plt.plot(train_idx, train_data, label='Train set', linewidth=0.8, color='tab:blue')
    plt.plot(test_idx, test_data, label='Test set', linewidth=0.8, color='tab:orange')
    plt.xlabel('Date')
    plt.ylabel('Electricity Consumption [kWh]')
    plt.xlim(train_idx[0], test_idx[-1])
    plt.ylim(ylim_lower, ylim_upper)
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=400)
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=400)
    plt.close()

def plot_train_val_cv_data(path, selected_space, train_data, train_idx):
    if not os.path.exists(path + 'Img/' + selected_space + '/'):
        os.makedirs(path + 'Img/' + selected_space + '/')
    
    save_path = path + 'Img/' + selected_space + '/train_val_cv_splits_' + selected_space + '.png'

    tscv = TimeSeriesSplit(n_splits=5)

    # convert to kWh
    train_data = train_data * 1e-3
    
    max_value = np.max(train_data)
    min_value = np.min(train_data)
    ylim_upper = int(max_value * 1.2)
    ylim_lower = 0 if min_value > 0 else (min_value * 1.1)
    
    n_folds = tscv.get_n_splits(train_data)
    fig, axes = plt.subplots(n_folds, 1, figsize=(12, 1.6 * n_folds), sharex=True)

    for fold, (train_indices, val_indices) in enumerate(tscv.split(train_data)):
        train_data_cv, val_data_cv = train_data[train_indices], train_data[val_indices]
        train_idx_cv, val_idx_cv = train_idx[train_indices], train_idx[val_indices]

        axes[fold].plot(train_idx_cv, train_data_cv, label='Train', linewidth=0.6, color='tab:blue')
        axes[fold].plot(val_idx_cv, val_data_cv, label='Validation', linewidth=0.6, color='tab:green')
        axes[fold].set_title(f'Fold {fold + 1}')
        axes[fold].set_xlim(train_idx[0], train_idx[-1])
        # axes[fold].set_ylim(ylim_lower, ylim_upper)
        if fold == 0:
            axes[fold].legend()

    axes[-1].set_xlabel('Date')
    axes[2].set_ylabel('Electricity Consumption [kWh]')

    plt.tight_layout()    
    plt.savefig(save_path, format='png', dpi=400)
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=400)
    plt.close()

def plot_loss(path, selected_space, history):
    if not os.path.exists(path + 'Img/' + selected_space + '/'):
        os.makedirs(path + 'Img/' + selected_space + '/')

    save_path = path + 'Img/' + selected_space + '/loss_' + selected_space + '.png'
    
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'], label='Train loss')
    # plt.plot(history.history['val_loss'], label='Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend()    
    plt.savefig(save_path, format='png', dpi=400)
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=400)
    plt.close()

def plot_prediction(path, selected_space, y_pred, y_test, index):
    if selected_space[-3:] in ['_bw', '_bm', '_ww', '_wm']:
        selected_space_aux = selected_space[:-3]
    else:    
        selected_space_aux = selected_space
    
    if not os.path.exists(path + 'Img/' + selected_space_aux + '/'):
        os.makedirs(path + 'Img/' + selected_space_aux + '/')
    
    save_path = path + 'Img/' + selected_space_aux + '/pred_' + selected_space + '.png'

    # convert to kWh
    y_pred, y_test = y_pred * 1e-3, y_test * 1e-3

    if selected_space_aux == 'Alameda':
        max_value = 220
        ylim_upper = int(max_value)
    elif selected_space_aux == 'Torre_Norte':
        max_value = 40
        ylim_upper = int(max_value)
    elif selected_space_aux == 'LSDC1':
        max_value = 14
        ylim_upper = int(max_value)
    else:
        max_value = max(np.max(y_test), np.max(y_pred))
        ylim_upper = int(max_value * 1.2)
    
    min_value = min(np.min(y_test), np.min(y_pred))
    ylim_lower = 0 if min_value > 0 else (min_value * 1.1)

    plt.figure(figsize=(10, 6))
    plt.plot(index, np.array(y_test), label='True', marker='o', markersize=0.5, linestyle='-', linewidth=1)
    plt.plot(index, np.array(y_pred), label='Predicted', marker='o', markersize=0.5, linestyle='-', linewidth=1)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Electricity Consumption [kWh]')
    plt.xlim(index[0], index[-1])
    plt.ylim(ylim_lower, ylim_upper)
    plt.xticks(rotation=15, fontsize=13)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=400)
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=400)
    plt.close()

def plot_time_series_split(path, selected_space, train_idx):
    if not os.path.exists(path + 'Img/' + selected_space + '/'):
        os.makedirs(path + 'Img/' + selected_space + '/')
    
    save_path = path + 'Img/' + selected_space + '/time_series_split_' + selected_space + '.png'

    tscv = TimeSeriesSplit(n_splits=5)

    plt.figure(figsize=(12, 6))
    for fold, (train_indices, val_indices) in enumerate(tscv.split(train_idx)):
        plt.barh(fold, len(train_indices), left=train_indices[0], color='tab:blue', label='Train' if fold == 0 else "")
        plt.barh(fold, len(val_indices), left=val_indices[0], color='tab:orange', label='Validation' if fold == 0 else "")

    plt.xlabel('Sample index')
    plt.ylabel('Fold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=400)
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=400)
    plt.close()
