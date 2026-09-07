import optuna
from sklearn.metrics import roc_auc_score
from functools import partial
from lections.imbalance.local_utils1 import get_optimal_thr_fbeta
import lightgbm as lgb
from lections.imbalance.local_utils1 import *


def train_model_log_info(model_dict, lgb_tr, lgb_val, X_tr, X_val, name, params=None, beta=1.):
    eval_result = dict()
    init_params = {
        'objective': 'binary', # logloss
        'eta': 0.1,
        'verbose': -1,
        'nthread': 16,
        'metric': 'auc',
        'seed': 911
    }
    if params is not None:
        init_params.update(params)
    
    model = lgb.train(
        init_params, lgb_tr, num_boost_round=200, valid_sets=[lgb_val], valid_names=['val'],
        callbacks=[lgb.record_evaluation(eval_result), lgb.log_evaluation(0)],
        feval=[
            partial(custom_fbeta_score, beta=beta),
            partial(custom_fbeta_optimal_thr, beta=beta),
        ]
    )
    y_raw = model.predict(X_val, raw_score=True)
    y_raw_tr = model.predict(X_tr, raw_score=True)

    model_dict[name]['y_raw'] = y_raw
    model_dict[name]['y_raw_tr'] = y_raw_tr
    model_dict[name]['eval_result'] = eval_result
    model_dict[name]['beta'] = beta
    model_dict[name]['model'] = model


def objective(trial, X_tr, y_tr, X_val, y_val):
    balance_const = y_tr.eq(0).sum() / y_tr.eq(1).sum()
    params = {
        'objective': 'binary', # logloss
        'eta': 0.1,
        'verbose': -1,
        'nthread': 16,
        'metric': 'auc',
        'scale_pos_weight': trial.suggest_float('scale_pos_weight / is_unbalance', 0.001, 1000, log=True) * balance_const
    }
    lgb_tr = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, y_val, free_raw_data=False)
    
    model = lgb.train(
        params, lgb_tr, num_boost_round=200, valid_sets=[lgb_val], valid_names=['val'],
        callbacks=[lgb.early_stopping(10, verbose=False)]
    )
    y_raw = model.predict(X_val, raw_score=True)
    
    auc = roc_auc_score(y_val, y_raw)
    _, fbeta = get_optimal_thr_fbeta(y_val, y_raw, beta=2)
    return auc, fbeta



def study_scale_pos_weight(X_tr, y_tr, X_val, y_val, n_trials=120, n_jobs=4):
    sampler = optuna.samplers.TPESampler(n_startup_trials=100)
    study = optuna.create_study(sampler=sampler, directions=['maximize', 'maximize'])

    func = partial(objective, X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(func, n_trials=n_trials, n_jobs=n_jobs)
    
    return study