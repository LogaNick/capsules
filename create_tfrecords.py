# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def create_dataset(size=1024, dimensions=2, mean=0, scale=1, separate=False):
    """
    Creates data set consisting of two input vectors and an output vector
    representing the velocity (difference) between the input vectors
    
    Vectors generated are normally distributed with a mean at the origin
    by default
    
    separate set to false (default) flattens the example vectors. Otherwise,
    you'll get a list of lists representing the two vectors
    """
    
    examples = []
    labels = []
    
    for i in range(size):
        example_left = np.random.normal(size=dimensions, scale=scale, loc=mean)
        example_right = np.random.normal(size=dimensions, scale=scale, loc=mean)
        
        label = example_right - example_left
        
        if separate:
            examples.append([example_left, example_right])
        else:
            examples.append([*example_left, *example_right])
            
        labels.append(label)
        
        
    return examples, labels

def write_tfrecord(examples, labels, filename):
    """
    Writes a tfrecord to filename with the given examples, labels
    """
    with tf.python_io.TFRecordWriter(filename) as writer:
        for example, label in zip(examples, labels):
            # Create a feature dictionary
            feature = {
                    # See https://github.com/tensorflow/tensorflow/issues/9554#issuecomment-298761938
                    "example" : tf.train.Feature(float_list=tf.train.FloatList(value=[float(x) for x in example])),
                    "label" :  tf.train.Feature(float_list=tf.train.FloatList(value=[float(x) for x in label]))
                   }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            writer.write(tf_example.SerializeToString())