from time import time
import xgboost as xgb


class XGBTimer(xgb.callback.TrainingCallback): # четкое API
    def __init__(self):
        self.res_list = []
        
    def before_training(self, model):
        self.start_time = time()
        return model
        
    def after_iteration(self, *args):
        self.res_list.append(time() - self.start_time)
        return False
    
    

def lgb_timer(times): # через одно место по аналогии с сурс кодом других коллбэков
    start_time = time()

    def _callback(env):
        times.append(time() - start_time)
        
    return _callback