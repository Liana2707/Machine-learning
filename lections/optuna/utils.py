import matplotlib.pyplot as plt
import numpy as np

def plot_weights(model=None, features=None, weights=None, bias=None, top_k=20):
    '''
    рисует значения весов линейной модели при признаках
    
    top_k: рисовать первые top_k весов по модулю
    '''
    # подготовка необходимого
    num_features_to_plot = min(top_k, len(features))
    if weights is None:
        weights = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_ # классификация vs регрессия
    sorted_idx = np.argsort(-np.abs(weights))
    if bias is None:
        bias = model.intercept_[0] if not isinstance(model.intercept_, float) else model.intercept_ # классификация vs регрессия
    
    fig, ax = plt.subplots(figsize=(8, num_features_to_plot / 2))
    
    # сами бары
    container = ax.barh(y=features[sorted_idx][:top_k][::-1], width=weights[sorted_idx][:top_k][::-1])
    
    # приписать к ним значения весов
    ax.bar_label(container, weights[sorted_idx][:top_k][::-1].round(3), color='red', fontsize=15)
    
    # настройка ах'a
    ax.margins(0.2, 0.05)
    ax.set_title(f'bias: {bias:.1e}', fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_xlabel('weight', fontsize=15)
    
    plt.show()