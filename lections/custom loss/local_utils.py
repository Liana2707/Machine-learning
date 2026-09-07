import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit

def plot_scores(model, val, features, target_col, raw_scores=True, init_score=None):
    init_score_add = init_score if init_score is not None else 0
    y_raw = model.predict(val[features]) + init_score_add
    if not raw_scores:
        y_raw = expit(y_raw)
        
    sns.histplot(val, x=y_raw, hue=target_col, bins=33, legend=True)
    plt.yscale('log')
    plt.gcf().set_size_inches(6, 3)