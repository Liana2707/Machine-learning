import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb, lightgbm as lgb, catboost as cb
import optuna
from functools import partial
from time import time as tm
from tqdm import tqdm

from local_utils import start_optimization, train
from local_objectives import objective_subsample_time


def read_data(num_rows=None):
    data = pd.read_csv('data/application_train.csv')
    data.columns = ['_'.join([word.lower() for word in col_name.split(' ') if word != '-']) for col_name in data.columns]
    
    if num_rows is not None:
        data = data.sample(num_rows, random_state=911).reset_index(drop=True)
    return data


def split_data(data):
    np.random.seed(911)

    test_size = int(0.2 * data.shape[0])
    val_size = int(0.3 * (data.shape[0] - test_size))
    test_idx = np.random.choice(data.shape[0], size=test_size, replace=False)

    val_idx_candidates = np.setdiff1d(np.arange(data.shape[0]), test_idx)
    val_idx = np.random.choice(val_idx_candidates, size=val_size, replace=False)

    data_dict = dict()
    data_dict['tst'] = data.loc[test_idx].reset_index(drop=True)
    data_dict['val'] = data.loc[val_idx].reset_index(drop=True)

    not_train_idx = np.union1d(test_idx, val_idx)
    data_dict['tr'] = data.drop(index=not_train_idx)
    data_dict['tr'].reset_index(drop=True, inplace=True)

    for key, df in data_dict.items():
        print(key, 'shape:', df.shape)
        
    X_tr, y_tr = data_dict['tr'].iloc[:, 2:], data_dict['tr'].target
    X_val, y_val = data_dict['val'].iloc[:, 2:], data_dict['val'].target
        
    return X_tr, y_tr, X_val, y_val


def plot_lgb_gbdtpl_results(res_lgb_plain, X_tr, y_tr, X_val, y_val):
    param_info_lgb = {'num_threads': 16, 'objective': 'binary', 'eta': (1e-3, 1, 'float'), 'linear_lambda': (1e-3, 1, 'float'),
                     'linear_tree': True}
    study_lgb_gbdtpl = start_optimization('lgb', n_trials=40, X_tr=X_tr, y_tr=y_tr, param_info=param_info_lgb, X_val=X_val, y_val=y_val)
    
    del param_info_lgb['eta']
    del param_info_lgb['linear_lambda']
    lgb_tr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
    
    res_lgb_gbdtpl = train('lgb', study_lgb_gbdtpl, lgb_tr, lgb_val, n_trees=300, param_info=param_info_lgb)
    
    fig, ax = plt.subplot_mosaic([['gbdtpl']], sharey=True)
    ax['gbdtpl'].set_title('CPU, 16 ядер', fontsize=15)

    ax['gbdtpl'].plot(*res_lgb_plain, label=f'plain', lw=3)
    ax['gbdtpl'].plot(*res_lgb_gbdtpl, label=f'PL', lw=3)
    ax['gbdtpl'].axhline(max(res_lgb_plain[1]), ls='--', color='blue')
    ax['gbdtpl'].axhline(max(res_lgb_gbdtpl[1]), ls='--', color='orange')

    ax['gbdtpl'].set_ylabel('ROC AUC', fontsize=15)
    ax['gbdtpl'].set_xlabel('time, sec', fontsize=15)
    # ax['sklearn'].set_ylim(0.755, 0.763)
    ax['gbdtpl'].tick_params('both', labelsize=15)
    ax['gbdtpl'].legend(fontsize=15)
    fig.show()
    

def plot_xgb_lgb_results(X_tr, y_tr, X_val, y_val, n_trials=20):
    study_xgb = start_optimization('xgb', n_trials=20, X_tr=X_tr, y_tr=y_tr, param_info=None, X_val=X_val, y_val=y_val)
    study_lgb = start_optimization('lgb', n_trials=20, X_tr=X_tr, y_tr=y_tr, param_info=None, X_val=X_val, y_val=y_val)
    
    xgb_tr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    xgb_val = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

    lgb_tr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
    
    res_lgb = train('lgb', study_lgb, lgb_tr, lgb_val, n_trees=300, param_info=None)
    res_xgb = train('xgb', study_xgb, xgb_tr, xgb_val, n_trees=300, param_info=None)

    param_info_lgb = {'num_threads': 32}
    param_info_xgb = {'nthread': 32}
    res_lgb_1 = train('lgb', study_lgb, lgb_tr, lgb_val, n_trees=300, param_info=param_info_lgb)
    res_xgb_1 = train('xgb', study_xgb, xgb_tr, xgb_val, n_trees=300, param_info=param_info_xgb)
    res_lgb_sklearn = train('lgbSklearn', study_lgb, (X_tr, y_tr), (X_val, y_val), n_trees=300, param_info=None)
    
    param_info_lgb = {'device_type': 'gpu', 'num_threads': -1}
    param_info_xgb = {'tree_method': 'gpu_hist', 'nthread': -1}
    res_lgb_2 = train('lgb', study_lgb, lgb_tr, lgb_val, n_trees=300, param_info=param_info_lgb)
    res_xgb_2 = train('xgb', study_xgb, xgb_tr, xgb_val, n_trees=300, param_info=param_info_xgb)
    
    
    
    
    # PLOTTING

    fig, ax = plt.subplot_mosaic([['cpu_iter', 'cpu_time'], ['gpu_xgb', 'gpu_lgb']], sharey=True)

    ax['cpu_iter'].plot(res_lgb[1], label=f'lgbm', lw=3)
    ax['cpu_iter'].plot(res_xgb[1], label=f'xgboost', lw=3)
    ax['cpu_iter'].axhline(max(res_lgb[1]), ls='--', color='blue')
    ax['cpu_iter'].axhline(max(res_xgb[1]), ls='--', color='orange')

    ax['cpu_iter'].set_ylabel('ROC AUC', fontsize=15)
    ax['cpu_iter'].set_xlabel('iteration (tree_num)', fontsize=15)
    #ax['cpu_iter'].set_ylim(0.755, 0.763)
    ax['cpu_iter'].tick_params('both', labelsize=15)
    ax['cpu_iter'].legend(fontsize=15)

    ax['cpu_time'].set_title('CPU. nthread=1 vs nthread=32', fontsize=20)
    ax['cpu_time'].plot(*res_lgb, label=f'lgbm. 1 thread', lw=3)
    ax['cpu_time'].axhline(max(res_lgb[1]), ls='--', color='blue')
    ax['cpu_time'].plot(*res_xgb, label=f'xgboost. 1 thread', lw=3)
    ax['cpu_time'].axhline(max(res_xgb[1]), ls='--', color='orange')

    ax['cpu_time'].plot(*res_lgb_1, label=f'lgbm. 32 thread', lw=3, ls='--', color='blue')
    ax['cpu_time'].plot(*res_xgb_1, label=f'xgboost. 32 thread', lw=3, ls='--', color='orange')

    ax['cpu_time'].set_xlabel('time, sec', fontsize=15)
    # ax['cpu_time'].set_ylim(0.741, 0.763)
    ax['cpu_time'].tick_params('both', labelsize=15)
    ax['cpu_time'].legend(fontsize=15)


    ax['gpu_xgb'].set_title('XGBoost\nnthread=1 vs nthread=32 vs GPU', fontsize=18)
    ax['gpu_xgb'].plot(*res_xgb, label=f'xgboost. 1 thread', lw=3)
    ax['gpu_xgb'].axhline(max(res_xgb[1]), ls='--', color='orange')
    ax['gpu_xgb'].plot(*res_xgb_2, label=f'xgboost, gpu', lw=3)
    ax['gpu_xgb'].plot(*res_xgb_1, label=f'xgboost. 32 thread', lw=3)

    ax['gpu_lgb'].set_title('LightGBM\nnthread=1 vs nthread=32 vs GPU', fontsize=18)
    ax['gpu_lgb'].plot(*res_lgb, label=f'lgbm. 1 thread', lw=3)
    ax['gpu_lgb'].axhline(max(res_lgb[1]), ls='--', color='blue')
    ax['gpu_lgb'].plot(*res_lgb_2, label=f'lgbm, gpu', lw=3)
    ax['gpu_lgb'].plot(*res_lgb_1, label=f'lgbm. 32 thread', lw=3)


    ax['gpu_xgb'].set_ylabel('ROC AUC', fontsize=15)
    ax['gpu_xgb'].set_xlabel('time, sec', fontsize=15)
    #ax['gpu_xgb'].set_ylim(0.741, 0.763)
    ax['gpu_xgb'].tick_params('both', labelsize=15)
    ax['gpu_xgb'].legend(fontsize=15)

    ax['gpu_lgb'].set_xlabel('time, sec', fontsize=15)
    ax['gpu_lgb'].tick_params('both', labelsize=15)
    ax['gpu_lgb'].legend(fontsize=15)

    fig.set_size_inches(15, 12)
    fig.tight_layout(h_pad=5)
    
    return res_lgb_1, res_lgb_sklearn


def plot_lgb_sklearn_results(res_lgb_plain, res_lgb_sklearn):
    fig, ax = plt.subplot_mosaic([['sklearn']], sharey=True)
    ax['sklearn'].set_title('CPU, 32 ядра', fontsize=15)

    ax['sklearn'].plot(*res_lgb_plain, label=f'lgb.train', lw=3)
    ax['sklearn'].plot(*res_lgb_sklearn, label=f'lgb.LGBMClassifier', lw=3)
    ax['sklearn'].axhline(max(res_lgb_plain[1]), ls='--', color='blue')
    ax['sklearn'].axhline(max(res_lgb_sklearn[1]), ls='--', color='orange')

    ax['sklearn'].set_ylabel('ROC AUC', fontsize=15)
    ax['sklearn'].set_xlabel('time, sec', fontsize=15)
    # ax['sklearn'].set_ylim(0.755, 0.763)
    ax['sklearn'].tick_params('both', labelsize=15)
    ax['sklearn'].legend(fontsize=15)
    fig.show()
    
    
def start_lgb_random_search(param_info, X_tr, y_tr, X_val, y_val, n_trials=10):
    study = start_optimization(
        'lgb_rs', n_trials=n_trials,
        param_info=param_info,
        X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val, no_suggest=True
    )
    return study


def start_lgb_subsample_search(params, X_dummy, y_dummy):
    space = {
        'N': [1000, 10_000, 100_000, 1_000_000, 3_000_000],
        'd': [20, 50, 100, 200, 500]
    }
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.GridSampler(space))
    func = partial(objective_subsample_time, params=params, X_dummy=X_dummy, y_dummy=y_dummy)
    study.optimize(func)
    
    return study


def start_cb_random_search(param_info, X_tr, y_tr, X_val, y_val, n_trials=10, cat_features=None, timestamp=None):
    study = start_optimization(
        'cb_rs', n_trials=n_trials,
        param_info=param_info, cat_features=cat_features, timestamp=timestamp,
        X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val, no_suggest=True
    )
    return study


def plot_cb_ordered(X_tr, y_tr, X_val, y_val, n_trials=20, cat_features=None):
    study_cb = start_optimization('cb', n_trials, X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val, cat_features=cat_features)
    study_cb_ordered = start_optimization('cb', n_trials, param_info={'boosting_type': 'Ordered'},
                                          X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val, cat_features=cat_features)
    
    cb_tr = cb.Pool(X_tr, label=y_tr, cat_features=cat_features)
    cb_val = cb.Pool(X_val, label=y_val, cat_features=cat_features)
    
    res_cb = train('cb', study_cb, cb_tr, cb_val)
    res_cb_ordered = train('cb', study_cb_ordered, cb_tr, cb_val, param_info={'boosting_type': 'Ordered'})
    res_cb_allthreads = train('cb', study_cb, cb_tr, cb_val, param_info={'thread_count': -1})
    res_cb_gpu = train('cb', study_cb, cb_tr, cb_val,
                       param_info={'task_type': 'GPU', 'sampling_frequency': None, 'rsm': None})
    res_cb_gpu_ordered = train('cb', study_cb, cb_tr, cb_val,
                       param_info={'task_type': 'GPU', 'boosting_type': 'Ordered'})
    
    # PLOTTING
    fig, ax = plt.subplot_mosaic([['cpu_iter', 'cpu_iter']], sharey=True)

    ax['cpu_iter'].plot(res_cb[1], label=f'Plain. fit time: {res_cb[0]:.1f}', lw=3)
    ax['cpu_iter'].plot(res_cb_ordered[1], label=f'Ordered. fit time: {res_cb_ordered[0]:.1f}', lw=3)
    ax['cpu_iter'].plot(res_cb_allthreads[1], label=f'Plain. all threads. fit time: {res_cb_allthreads[0]:.1f}', lw=3)
    ax['cpu_iter'].plot(res_cb_gpu[1], label=f'GPU Plain. fit time: {res_cb_gpu[0]:.1f}', lw=3)
    ax['cpu_iter'].plot(res_cb_gpu_ordered[1], label=f'GPU Ordered. fit time: {res_cb_gpu_ordered[0]:.1f}', lw=3)
    ax['cpu_iter'].axhline(max(res_cb[1]), ls='--', color='blue')
    ax['cpu_iter'].axhline(max(res_cb_ordered[1]), ls='--', color='orange')

    ax['cpu_iter'].set_ylabel('ROC AUC', fontsize=15)
    ax['cpu_iter'].set_xlabel('iteration (tree_num)', fontsize=15)
#     ax['cpu_iter'].set_ylim(0.755, 0.763)
    ax['cpu_iter'].tick_params('both', labelsize=15)
    ax['cpu_iter'].legend(fontsize=15)
    
    fig.set_size_inches(16, 7)
    return study_cb


def plot_cb_ctr(study_cb, X_tr, y_tr, X_val, y_val, cat_features=None):
    cb_tr = cb.Pool(X_tr, label=y_tr, cat_features=cat_features)
    cb_val = cb.Pool(X_val, label=y_val, cat_features=cat_features)
    
    res_cb_ctr_borders = train('cb', study_cb, cb_tr, cb_val,
                        param_info={'thread_count': -1, 'simple_ctr': 'Borders', 'max_ctr_complexity': 1})
    res_cb_ctr_buckets = train('cb', study_cb, cb_tr, cb_val,
                        param_info={'thread_count': -1, 'simple_ctr': 'Buckets:CtrBorderCount=31', 'max_ctr_complexity': 1})
    res_cb_ctr_binarized = train('cb', study_cb, cb_tr, cb_val,
                        param_info={'thread_count': -1, 'simple_ctr': 'BinarizedTargetMeanValue:Prior=10', 'max_ctr_complexity': 1})
    res_cb_ctr_counter = train('cb', study_cb, cb_tr, cb_val,
                        param_info={'thread_count': -1, 'simple_ctr': 'Counter', 'max_ctr_complexity': 1})
    
    
    # PLOTTING
    
    fig, ax = plt.subplot_mosaic([['ctr']])

    ax['ctr'].set_title('CtrType', fontsize=20)
    ax['ctr'].plot(res_cb_ctr_borders[1], label=f'CtrType=Borders. fit time: {res_cb_ctr_borders[0]:.1f}', lw=3)
    ax['ctr'].plot(res_cb_ctr_buckets[1], label=f'CtrType=Buckets. fit time: {res_cb_ctr_buckets[0]:.1f}', lw=3)
    ax['ctr'].plot(res_cb_ctr_binarized[1],
                   label=f'CtrType=BinarizedTargetMeanValue. fit time: {res_cb_ctr_binarized[0]:.1f}', lw=3)
    ax['ctr'].plot(res_cb_ctr_counter[1], label=f'CtrType=Counter. fit time: {res_cb_ctr_counter[0]:.1f}', lw=3)

    ax['ctr'].set_ylabel('ROC AUC', fontsize=15)
    ax['ctr'].set_xlabel('iteration (tree_num)', fontsize=15)
#     ax['ctr'].set_ylim(0.75, 0.762)
    ax['ctr'].tick_params('both', labelsize=15)
    ax['ctr'].legend(fontsize=15)

    fig.set_size_inches(10, 7)
    
    
def plot_lgb_vs_catboost(X_dummy, y_dummy):
    sizes = [50_000, 200_000, 500_000, 1_000_000, 3_000_000, 10_000_000]
    time_cb_plain = []
    time_cb_ordered = []
    time_cb_gpu = []
    time_lgbm_plain = []
    time_lgbm_gpu = []

    n_trees = 10

    for size in tqdm(sizes):
        lgb_params = {
            'nthread': -1,
            'verbose': -1,
            'max_depth': 4,
    #         'max_leaves': 32,
            'force_col_wise': True,
            'eta': 1.,
            'metric': 'None',
            'objective': 'binary'
        }

        lgb_data = lgb.Dataset(X_dummy[:size], label=y_dummy[:size], free_raw_data=False)
        lgb_data.construct()

        stm = tm()
        model = lgb.train(lgb_params, lgb_data, num_boost_round=n_trees)
        time_lgbm_plain.append(tm() - stm)

        lgb_params.update({'device_type': 'gpu'})
        stm = tm()
        model = lgb.train(lgb_params, lgb_data, num_boost_round=n_trees)
        time_lgbm_gpu.append(tm() - stm)

        cb_params = {
            'iterations': n_trees,
            'thread_count': -1,
            'verbose': False,
            'depth': 4,
            'eta': 1.,
            'loss_function': 'Logloss:hints=skip_train~true',
            'leaf_estimation_iterations': 1,
        }

        cb_data = cb.Pool(X_dummy[:size], label=y_dummy[:size])
        cb_data.quantize()

        model = cb.CatBoost(cb_params)
        stm = tm()
        model.fit(cb_data)
        time_cb_plain.append(tm() - stm)

        cb_params.update({'boosting_type': 'Ordered'})
        model = cb.CatBoost(cb_params)
        stm = tm()
        model.fit(cb_data)
        time_cb_ordered.append(tm() - stm)

        cb_params.update({'boosting_type': 'Plain', 'task_type': 'GPU'})
        model = cb.CatBoost(cb_params)
        stm = tm()
        model.fit(cb_data)
        time_cb_gpu.append(tm() - stm)
        
    # PLOTTING
    
    plt.title('lgbm vs catboost, 120 features', fontsize=15)
    plt.plot(sizes, time_lgbm_plain, label='lgbm_plain', marker='o')
    plt.plot(sizes, time_lgbm_gpu, label='lgbm_gpu', marker='o')
    plt.plot(sizes, time_cb_plain, label='cb_plain', marker='o')
    plt.plot(sizes, time_cb_ordered, label='cb_ordered', marker='o')
    plt.plot(sizes, time_cb_gpu, label='cb_gpu', marker='o')
    plt.legend(fontsize=15)
    plt.gca().tick_params('both', labelsize=15)
    plt.ylabel('train time, sec', fontsize=15)
    plt.xlabel('data size', fontsize=15)
    plt.xscale('log')
    # plt.yscale('log')
    plt.gcf().set_size_inches(12, 8)

        