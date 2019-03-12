import tensorflow as tf


def model(x):
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10

    with tf.name_scope('main_params'):
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    with tf.variable_scope('conv1') as scope:
    	# conv-1 (3x3, 32)
        conv = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        # conv-2 (3x3, 64)
        conv = tf.layers.conv2d(
            inputs=conv,
            filters=64,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        # max-pool-1 (2x2, strides=2)
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)

    with tf.variable_scope('conv2') as scope:
    	# conv-3 (3x3, 128)
        conv = tf.layers.conv2d(
            inputs=drop,
            filters=128,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        # max-pool-2 (2x2, stride=2)
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        # conv-4 (2x2, 128)
        conv = tf.layers.conv2d(
            inputs=pool,
            filters=128,
            kernel_size=[2, 2],
            padding='SAME',
            activation=tf.nn.relu
        )
        # max-pool-3 (2x2, stride=2)
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        # dropout for max-pool-3 ???
        drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)

    with tf.variable_scope('fully_connected') as scope:
        flat = tf.reshape(drop, [-1, 4 * 4 * 128])
        # fc (1500 units)
        fc = tf.layers.dense(inputs=flat, units=1500, activation=tf.nn.relu)
        # dropout for fc layer
        drop = tf.layers.dropout(fc, rate=0.5)
        # output layer
        before_softmax = tf.layers.dense(inputs=drop, units=_NUM_CLASSES, activation=None, name=scope.name)

    return before_softmax, global_step, learning_rate


def lr(epoch):
    learning_rate = 1e-3
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate
