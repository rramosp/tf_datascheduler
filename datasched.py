from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.framework import ops
from tensorflow.python.data.experimental.ops import random_ops
from tensorflow.python.ops import gen_stateless_random_ops
import numpy as np
import tensorflow as tf 
from tensorflow import data as td

from tensorflow.python.data.experimental.ops import interleave_ops 

import pandas as pd
import matplotlib.pyplot as plt



class ScheduleDataset(interleave_ops._DirectedInterleaveDataset):
    """
    combines several datasets according to weights. Same functionality as 
    [tf.data.experimental.sample_from_datasets](https://www.tensorflow.org/api_docs/python/tf/data/experimental/sample_from_datasets), 
    but weights are internally represented as tensors so that they are part of the computational graph
    and can be changed during training with `ScheduleDataset_Callback` as explained further below.
    """
    
    def __init__(self, datasets, weights=None, datasets_names=None):
        """
        datasets: a list to tensorflow datasets
        weights: a list of weights representing the proportions in which the datasets will be mixed. Must have the same length
                 as datasets. If None, defaults to all weights being equal.
        datasets_names: a list with descriptive names for the datasets for logging into TensorBoard. Must have the same length
                 as datasets. If None, they will be named D00, D01, etc.
        """
        if weights is None:
            weights = [1/len(datasets)]*len(datasets)
        
        if datasets_names is None:
            datasets_names = ["D%02d"%i for i in range(len(datasets))]

        assert len(datasets_names) == len(datasets), "you have %d datasets, but specified %d dataset names"%(len(datasets), len(datasets_names))
        assert np.alltrue([isinstance(i, td.Dataset) for i in datasets]), "all elements of 'datasets' must be instances of tf.data.Dataset"
        self.datasets = datasets
        self.datasets_names = datasets_names
        self.weights = None
        self.set_weights(weights)
        
        self.random_gen = dataset_ops.MapDataset(td.Dataset.from_tensor_slices([1]).repeat(-1), 
                                   lambda x: tf.random.categorical(tf.math.log(self.weights), 1)[0,0])
        interleave_ops._DirectedInterleaveDataset.__init__(self, self.random_gen, self.datasets)
        
    def set_weights(self, weights):
        """
        specifies new weights. Can be used directly, but mostly designed to be used by ScheduleDataset_Callback.
        """        
        assert type(weights) in (list, np.ndarray), "weights must be a list or an np array"        
        weights = np.r_[weights].reshape(-1)
        assert len(weights)==len(self.datasets), "must have the same number of weights as datasets"
        weights = weights.reshape(-1, len(self.datasets))
        assert np.allclose(np.sum(weights),1,atol=1e-3), "weights must add up to 1, got %s"%(str(weights))
        assert np.alltrue(weights>=0), "weights must be >= 0, got %s"%(str(weights))

        if self.weights is None:
            self.weights = tf.Variable(weights)
        else:
            self.weights.assign(weights)


class ScheduleDataset_Callback(tf.keras.callbacks.Callback):
    """
    a callback that, when used in .fit, dynamically changes the weights of a ScheduleDataset
    according to a schedule defined by the dataset_weights_fn function. See associated demo notebook
    for examples.
    """

    def __init__(self, dds, dataset_weights_fn, keras_tensorboard_callback=None):
        """
        dds: the ScheduleDataset 
        dataset_weights_fn: the function that, at each training step, must produce the weights with which
                            the dds ScheduleDataset will produce a mixture of data.
        keras_tensorboard_callback: used to log the dataschedule to TensorBoard during training.                            
        """
        assert type(dds)==ScheduleDataset, "must use a ScheduleDataset"        
        self.dds = dds
        if type(dataset_weights_fn)==str:
            self.dataset_weights_fn = self.__class__.get_fn(dataset_weights_fn)
        else:
            self.dataset_weights_fn = dataset_weights_fn
        
        self.weights_history = []
        self.keras_tensorboard_callback = keras_tensorboard_callback
        if self.keras_tensorboard_callback is not None:
            self.writer = tf.summary.create_file_writer(self.keras_tensorboard_callback.log_dir+"/dataschedule")

    @classmethod
    def get_available_funcs(cls):
        """
        returns the functions readily available with a string spec.
        """
        funcs = {
                    "linear": cls.linear_dataschedule_fn,
                    "log":    cls.log_dataschedule_fn,
                    "sin":    cls.sin_dataschedule_fn,
                    "sin2":   lambda x: cls.sin_dataschedule_fn(x, k=2),
                    "sin4":   lambda x: cls.sin_dataschedule_fn(x, k=4),
                    "sin10":   lambda x: cls.sin_dataschedule_fn(x, k=10),
                    "sqr":    cls.sqr_dataschedule_fn,
                    "sqrt":   cls.sqrt_dataschedule_fn
                }
        return funcs
            
    @classmethod
    def get_fn(cls, fname):
        """
        returns a data schedule function given a string spec.
        """
        funcs = cls.get_available_funcs()

        assert fname in funcs.keys(), "schedule '%s' function not known, allowed funcs are %s, or write your own"%(fname,str(list(funcs.keys())))
        return funcs[fname]

    """
    predefined data schedule functions. must return a list of weights or None (weights are not changed).
    see associated demo notebook for illustrations
    """
    @staticmethod
    def linear_dataschedule_fn(ds_callback):
        epochs = ds_callback.params["epochs"]
        current_epoch = ds_callback.current_epoch

        p1 = current_epoch / epochs 
        return [p1, 1-p1]        

    @staticmethod
    def log_dataschedule_fn(ds_callback):
        epochs = ds_callback.params["epochs"]
        current_epoch = ds_callback.current_epoch

        p1 = current_epoch / epochs 
        p1 = 1e-5 if p1<=1e-5 else p1
        p1 = np.log(p1)
        p1 = (p1-np.log(1e-5))/(-np.log(1e-5))
        return [p1, 1-p1]

    @staticmethod
    def sin_dataschedule_fn(ds_callback, k=1):
        epochs = ds_callback.params["epochs"]
        current_epoch = ds_callback.current_epoch

        p1 = current_epoch / epochs 
        p1 = (np.sin(k*np.pi*p1))**2
        return [p1, 1-p1]

    @staticmethod
    def sqr_dataschedule_fn(ds_callback):
        epochs = ds_callback.params["epochs"]
        current_epoch = ds_callback.current_epoch

        p1 = current_epoch / epochs 
        p1 = p1**2
        return [p1, 1-p1]

    @staticmethod
    def sqrt_dataschedule_fn(ds_callback):
        epochs = ds_callback.params["epochs"]
        current_epoch = ds_callback.current_epoch

        p1 = current_epoch / epochs 
        p1 = np.sqrt(p1)
        return [p1, 1-p1]
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch=epoch
        self.check_set_weights()

    def on_epoch_end(self, epoch, logs=None):
        """
        weights are set at the end of each epoch
        """
        self.current_epoch = epoch
        # log weights in tensorboard
        if self.keras_tensorboard_callback is not None: 
            with self.writer.as_default():
                for i,weight in enumerate(self.dds.weights.numpy().reshape(-1)):
                    steps  = self.params["steps"]
                    tf.summary.scalar('probability schedule for dataset %s'%self.dds.datasets_names[i], weight, step=epoch)    

    def check_set_weights(self):
        """
        calls the dataschedule function and sets weights accordingly
        """
        if self.dataset_weights_fn is not None:
            # obtain weights
            w = self.dataset_weights_fn(self)

            if w is not None:
                # set weights in schedule dataset
                self.dds.set_weights(w)
                self.weights_history.append({"epoch": self.current_epoch, "weights": w})

    @classmethod
    def plot_schedule(cls, n_epochs, steps_per_epoch, dataset_weights_fn):
        """
        simulates a training process on this callback and plots the weights generated all along.
        
        n_epochs: the number of epoch to simulate
        steps_per_epoch: the number of steps per epoch to simulate
        dataset_weights_fn: the data schedule function
        """
        d0 = tf.data.Dataset.from_tensor_slices([0]).repeat(-1)
        d1 = tf.data.Dataset.from_tensor_slices([1]).repeat(-1)

        di = ScheduleDataset([d0, d1])

        cb = cls(di, dataset_weights_fn)

        cb.params = {"epochs": n_epochs, "steps": steps_per_epoch}
        for epoch in range(n_epochs):
            cb.on_epoch_begin(epoch)
            cb.on_epoch_end(epoch)

        k = pd.DataFrame(cb.weights_history)
        w = np.r_[[i for i in k.weights.values]]
        plt.plot(w[:,0], label="dataset 0")
        plt.plot(w[:,1], label="dataset 1")
        plt.grid()
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("weight")
