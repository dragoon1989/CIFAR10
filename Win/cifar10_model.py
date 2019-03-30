import cifar10_input as mod_input
import tensorflow as tf
from tensorflow import keras

# cifar10 category number = 10
cifar10_categories = 10

# inference
def inference(images):
	''' input:	images --- batch of input images
		output:	logits_before_softmax --- batch of predictions before softmax '''
	#trun_nrm_init = keras.initializers.TruncatedNormal(mean=0, stddev=1)
	trun_nrm_init = tf.contrib.layers.variance_scaling_initializer()	# the he initializer
	zero_init = keras.initializers.Zeros()
	# reshape the input
	images = tf.reshape(images, shape=[-1, mod_input.cifar10_size, mod_input.cifar10_size, 3])
	with tf.variable_scope('conv1') as scope:
		conv_layer1 = keras.layers.Conv2D(filters=32,
								kernel_size=(3,3),
								strides=(1,1),
								padding='same',
								kernel_initializer=trun_nrm_init,
								bias_initializer=zero_init,
								activation=tf.nn.relu)
		conv = conv_layer1.apply(images)
		# summary conv layer weights
		#tf.summary.histogram(name='conv1_weights', values=conv_layer1.weights[0])
		#tf.summary.histogram(name='conv1-weights', values=conv_layer1.weights)
		conv_layer2 = keras.layers.Conv2D(filters=64,
								kernel_size=(3,3),
								strides=(1,1),
								padding='same',
								kernel_initializer=trun_nrm_init,
								bias_initializer=zero_init,
								activation=tf.nn.relu)
		conv = conv_layer2.apply(conv)
		pool = keras.layers.MaxPool2D(pool_size=(2,2),
							   strides=(2,2),
							   padding='same')(conv)
		#drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)
		drop = pool

	with tf.variable_scope('conv2') as scope:
		conv_layer3 = keras.layers.Conv2D(filters=128,
								kernel_size=(3,3),
								strides=(1,1),
								padding='same',
								kernel_initializer=trun_nrm_init,
								bias_initializer=zero_init,
								activation=tf.nn.relu)
		conv = conv_layer3.apply(drop)
		pool = keras.layers.MaxPool2D(pool_size=(2,2),
							   strides=(2,2),
							   padding='same')(conv)
		conv_layer4 = keras.layers.Conv2D(filters=128,
								kernel_size=(2,2),
								strides=(1,1),
								padding='same',
								kernel_initializer=trun_nrm_init,
								bias_initializer=zero_init,
								activation=tf.nn.relu)
		conv = conv_layer4.apply(pool)
		pool = keras.layers.MaxPool2D(pool_size=(2,2),
							   strides=(2,2),
							   padding='same')(conv)
		#drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)
		drop = pool

	with tf.variable_scope('fully_connected') as scope:
		flat = tf.reshape(drop, [-1, 4 * 4 * 128])
		# summary output of convolution layers
		#tf.summary.histogram(name='conv_output', values=flat)
		fc_layer1 = keras.layers.Dense(units=1500,
								  kernel_initializer=trun_nrm_init,
								  bias_initializer=zero_init,
								  activation=tf.nn.relu)
		fc = fc_layer1.apply(flat)
		# summary fc-1 weights
		#tf.summary.histogram(name='fc-1_weights', values=fc_layer1.weights[0])
		#fc = tf.layers.dense(inputs=flat, units=1500, activation=tf.nn.relu)
		#drop = tf.layers.dropout(fc, rate=0.5)
		drop = fc
		logits_before_softmax = keras.layers.Dense(units=cifar10_categories, activation=None)(drop)
		#logits_before_softmax = tf.layers.dense(inputs=drop, units=10, activation=None, name=scope.name)
	# over, this should be passed to a softmax layer to get correct prediction
	return logits_before_softmax

# compute the loss function
def loss_func(labels, logits_before_softmax):
	''' input:	labels --- batch of labels
		output:	logits_before_softmax --- predicitions before softmax '''
	# use sparse softmax cross entropy as loss function
	total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
																logits=logits_before_softmax)
	# compute batch loss
	batch_loss = tf.reduce_mean(input_tensor=total_loss)
	# compute total loss
	total_loss = tf.reduce_sum(total_loss)
	# over
	return batch_loss, total_loss
