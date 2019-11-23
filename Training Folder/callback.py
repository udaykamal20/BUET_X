# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:08:56 2019

@author: User
"""
from tqdm import tqdm


class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_trainer(self, model):
        self.trainer = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

class TQDM(Callback):

    def __init__(self):
        """
        TQDM Progress Bar callback

        This callback is automatically applied to 
        every SuperModule if verbose > 0
        """
        self.progbar = None
        super(TQDM, self).__init__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        if self.progbar is not None:
            self.progbar.close()

    def on_train_begin(self, logs):
        self.train_logs = logs


    def on_epoch_begin(self, epoch):
        try:
            self.progbar = tqdm(total=self.train_logs['num_batches'],
                                unit=' batches')
            self.progbar.set_description('Epoch %i/%i' % 
                            (epoch+1, self.train_logs['num_epoch']))
        except:
            pass

    def on_epoch_end(self, logs=None):
        log_data = {key: '%.04f' % logs[key] for key in logs.keys()}
        self.progbar.set_postfix(log_data)
        self.progbar.update()
        self.progbar.close()


    def on_batch_begin(self):
        self.progbar.update(1)

    def on_batch_end(self, logs=None):
        log_data = {key: '%.04f' % logs[key] for key in logs.keys()}
        self.progbar.set_postfix(log_data)
        

