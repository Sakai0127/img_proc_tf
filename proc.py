import numpy as np
import tensorflow as tf

def dilation(inputs, filtersize, name='dilation'):
    assert filtersize % 2 == 1, 'filtersize must be odd.'
    with tf.name_scope(name):
        x = tf.nn.max_pool(inputs, [1, filtersize, filtersize, 1], [1, 1, 1, 1], 'SAME')
    return x

def erosion(inputs, filtersize, name='erosion'):
    assert filtersize % 2 == 1, 'filtersize must be odd.'
    with tf.name_scope(name):
        x = 1.0 - inputs
        x = tf.nn.max_pool(x, [1, filtersize, filtersize, 1], [1, 1, 1, 1], 'SAME')
        x = 1.0 - x
    return x

def gaussian_kernel(ksize=3, sigma=1.3):
    assert ksize % 2 == 1, 'kernel size must be odd.'
    def gausian2d(x, y, sigma):
        z = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        return z
    x = y = np.linspace(-sigma, sigma, num=ksize, dtype=np.float32)
    x, y = np.meshgrid(x, y)
    z = gausian2d(x, y, sigma)
    kernel = z / np.sum(z)
    return kernel

def Adaptivethreshold(inputs, filtersize=3, threshold_type='gaussian', sigma=1.3, c=2.0):
    with tf.name_scope('adaptive_threshold'):
        if threshold_type == 'gaussian':
            kernel = tf.constant(gaussian_kernel(ksize=filtersize, sigma=sigma).reshape(filtersize, filtersize, 1, 1), dtype=tf.float32, name='kernel')
        else:
            kernel = tf.ones([filtersize, filtersize, 1, 1]) / (filtersize**2)
        mean = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], 'SAME')
        threshold = mean - (c / 255.0)
        image = inputs - threshold
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.ceil(image)
    return image