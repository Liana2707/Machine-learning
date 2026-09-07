import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib as mpl

from sklearn.metrics import precision_recall_curve, fbeta_score
import lightgbm as lgb
from functools import partial
import shap


'''
model_dict = {
    'model1': {
        'y_raw': y_raw, # val
        'eval_result': ...,
        'y_raw_tr': ..., # train
        'model': lgb.Booster
    }
}
'''
SINGLE_PLOT_FIGSIZE = (10, 5)
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.title_fontsize'] = 20
mpl.rcParams['legend.markerscale'] = 3
    

def get_optimal_thr_fbeta(y_true, y_raw, beta=1):
    prec, rec, thrs = precision_recall_curve(y_true, y_raw)
    fbeta = (1 + beta**2) * prec * rec / (rec + prec * beta**2)
    best_idx = np.nanargmax(fbeta)
    return thrs[best_idx], fbeta[best_idx]


def custom_fbeta_score(preds, eval_data, beta=1.):
    '''
    preds: raw scores if custom_objective else probas
    eval_data: lgb.Dataset
    
    returns: func_name, score, is_higher_better
    '''
    y_eval = eval_data.get_label()
    _, score = get_optimal_thr_fbeta(y_eval, preds)
    return 'fbeta', score, True


def custom_fbeta_optimal_thr(preds, eval_data, beta=1.):
    '''
    preds: raw scores if custom_objective else probas
    eval_data: lgb.Dataset
    
    returns: func_name, score, is_higher_better
    '''
    y_eval = eval_data.get_label()
    thr, _ = get_optimal_thr_fbeta(y_eval, preds)
    return 'optimal_thr', thr, True
    

def plot_scores_single(y_true, y_raw, bins=33, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=SINGLE_PLOT_FIGSIZE)
        
    sns.histplot(x=y_raw, hue=y_true, ax=ax, bins=bins)
    
    
def plot_scores_by_group(X_tr, y_tr, y_raw_tr, X_val, y_val, y_raw_val, group_col):
    unique_vals = np.unique(np.r_[X_tr[group_col], X_val[group_col]])
    for value in unique_vals:
        fig, ax = plt.subplot_mosaic([['tr', 'val']], figsize=(10, 3))
        
        cond = (X_tr[group_col] == value) if not pd.isna(value) else X_tr[group_col].isnull()
        plot_scores_single(y_tr[cond], y_raw_tr[cond], ax=ax['tr'])
        ax['tr'].set_yscale('log')
        ax['tr'].set_title(f'train | {group_col}={value}', fontsize=15)

        cond = (X_val[group_col] == value) if not pd.isna(value) else X_val[group_col].isnull()
        plot_scores_single(y_val[cond], y_raw_val[cond], ax=ax['val'])
        ax['val'].set_yscale('log')
        ax['val'].set_title(f'val | {group_col}={value}', fontsize=15)
        
        plt.show()
        
    
def plot_scores_pair(y_val, model_dict, key_x, key_y, ax=None):
    '''
    model_dict - словарь вида model_name: y_raw
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=SINGLE_PLOT_FIGSIZE)
        
    y_raw_x = model_dict[key_x]['y_raw']
    y_raw_y = model_dict[key_y]['y_raw']
    
    sns.scatterplot(x=y_raw_x, y=y_raw_y, hue=y_val, s=8, ax=ax)
    
    
def plot_pr(model_dict, y_val, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        
    for model_name in model_dict:
        y_raw = model_dict[model_name]['y_raw']
        prec, rec, _ = precision_recall_curve(y_val, y_raw)
        
        ax.plot(rec, prec, label=model_name, lw=3)
    
    ax.set_title('Precision-Recall curve', fontsize=25)
    ax.set_xlabel('recall', fontsize=20)
    ax.set_ylabel('precision', fontsize=20)
    ax.tick_params('both', labelsize=20)
        
        
def plot_iter_metric(model_dict, y_val, metric='auc', ax=None, beta=1.):
    if ax is None:
        fig, ax = plt.subplots(figsize=SINGLE_PLOT_FIGSIZE)
        
    for model_name in model_dict:
        eval_result = model_dict[model_name]['eval_result']['val'][metric][10:]
        label = model_name
        if metric == 'fbeta':
            beta = model_dict[model_name]['beta']
            label += f' | beta={beta}'
        ax.plot(eval_result, label=label, lw=2)
        
    ax.set_title(metric, fontsize=25)
    ax.tick_params('both', labelsize=20)
    if metric == 'auc':
        ax.legend(fontsize=15, bbox_to_anchor=[-0.3, 1])


def plot_info(model_dict, y_val, scores_keys=None, beta=1., plot_scores=True):
    mosaic = [['.', '.', 'auc', 'fbeta', 'thr']]
    if plot_scores:
        mosaic.append(model_dict.keys())
    fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize=(20, 4 + 5 * plot_scores))
    
    plot_iter_metric(model_dict, y_val, metric='auc', ax=ax['auc'])
    plot_iter_metric(model_dict, y_val, metric='fbeta', ax=ax['fbeta'])
    plot_iter_metric(model_dict, y_val, metric='optimal_thr', ax=ax['thr'])

    if plot_scores:
        for model_name in model_dict:
            y_raw_val = model_dict[model_name]['y_raw']
            sns.histplot(x=y_raw_val, hue=y_val, bins=33, ax=ax[model_name], legend=False)
            ax[model_name].set_title(model_name)
        
    #fig.tight_layout()
    plt.show()
    
    
def get_shap(model, X, target, features, max_display=30, return_values=False, max_sample=10000, order='max'):
    exp = shap.Explainer(model)

    dummy_val = X[np.r_[[target], features]].copy()
    for col in dummy_val.columns[dummy_val.dtypes == 'category']:
        print(col)
        dummy_val[col] = dummy_val[col].cat.codes
    shap_values = exp(dummy_val.groupby(target).apply(lambda x: x.sample(min(x.shape[0], max_sample)))[features])

    order = shap_values.abs.mean(0) if order == 'mean' else shap_values.abs.max(0)
    shap.plots.beeswarm(shap_values, max_display=max_display, order=order)

    if return_values:
        return shap_values