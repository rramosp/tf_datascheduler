# TensorFlow dynamic data scheduler

combines several datasets according to weights, dynamically during training of a TF model. 

If it composed of two classes:

**ScheduleDataset**: It encapsulates the same functionality as tf.data.experimental.sample_from_datasets, but weights are internally represented as tensors so that they are part of the computational graph and can be changed during training with ScheduleDataset_Callback.

**ScheduleDataset_Callback**: A callback class that, when used in .fit, dynamically changes the weights of a ScheduleDataset
    according to a schedule defined by the dataset_weights_fn function. See associated demo notebook
    for examples.
    

See the [demo notebook](data_schedule_demo.ipynb) for further details.

