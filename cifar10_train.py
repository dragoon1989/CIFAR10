import sys
import getopt

import tensorflow as tf
import numpy as np

from cifar10_input import BuildInputPipeline
from cifar10_model import inference
from cifar10_model import loss_func

train_batch_size = 50
test_batch_size = 50
train_set_size = 50000
test_set_size = 10000
train_data_path = ['./data/data_batch_%d.bin' % (i) for i in range(1,6)]
test_data_path = ['./data/test_batch.bin']
model_path = './tmp/'
best_model_ckpt = 'best_ckpt'
summary_path = './tensorboard/'
summary_name = 'summary-default'    # tensorboard default summary dir
num_epochs = 20
lr0 = 5e-3			# initial learning rate
lr_up_ratio = 0.1	# 10%
lr_dn_ratio = 0.5	# 50%

############################# build the global model #############################
with tf.name_scope('input_pipeline'):
	train_dataset = BuildInputPipeline(file_name_list=train_data_path,
										batch_size=train_batch_size,
										num_parallel_calls=4,
										num_epoch=1)
	test_dataset = BuildInputPipeline(file_name_list=test_data_path,
										batch_size=test_batch_size,
										num_parallel_calls=4,
										num_epoch=1)
	train_iterator = train_dataset.make_initializable_iterator()
	test_iterator = test_dataset.make_initializable_iterator()

	train_handle = train_iterator.string_handle()
	test_handle = test_iterator.string_handle()
	# build public data entrance
	handle = tf.placeholder(tf.string, shape=[])
	iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types)
	labels, images = iterator.get_next()

# set global step counter
global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
# inference
logits_before_softmax = inference(images)

with tf.name_scope('train_loss'):
	# compute loss function
	batch_loss, total_loss = loss_func(labels, logits_before_softmax)
	# summary the train loss
	tf.summary.scalar(name='train_loss', tensor=batch_loss)

with tf.name_scope('optimization'):
	# optimize the model
	# use a placeholder to control learning rate dynamically
	lr = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
	#grad_var_pairs = optimizer.compute_gradients(loss)
	#train_op = optimizer.apply_gradients(grad_var_pairs, global_step=global_step)
	#train_op = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(batch_loss, global_step)

	train_op = tf.train.AdamOptimizer(learning_rate=lr,
									   beta1=0.9,
									   beta2=0.999,
									   epsilon=1e-08).minimize(batch_loss, global_step=global_step)

with tf.name_scope('train_batch_accuracy'):
	# compute batch predictions (dtype cast to tf.int32)
	batch_predictions = tf.to_int32(tf.math.argmax(tf.nn.softmax(logits_before_softmax), axis=1))
	# compute batch accuracy
	correct_prediction = tf.math.equal(batch_predictions, labels)
	batch_accuracy = tf.math.reduce_mean(tf.to_float(correct_prediction))
	# summary the batch accuracy
	tf.summary.scalar(name='train_accuracy', tensor=batch_accuracy)

############################# define the validation #############################

# build the training and validation process
def train(cur_lr, sess, summary_writer, summary_op):
	# get iterator handles
	train_handle_val = sess.run(train_handle)
	# initialize iterator
	sess.run(train_iterator.initializer)
	# training loop
	current_batch = 0
	while True:
		try:
			_, batch_loss_val, batch_acc_val, global_step_val, train_summary_buff = \
				sess.run([train_op, batch_loss, batch_accuracy, global_step, summary_op],
						feed_dict={handle : train_handle_val, lr : cur_lr})
			current_batch += 1
			# print indication info
			if current_batch % 50 == 0:
				msg = '\tbatch number = %d, loss = %.2f, train accuracy = %.2f%%' % \
						(current_batch, batch_loss_val, batch_acc_val*100)
				print(msg)
				# write train summary
				summary_writer.add_summary(summary=train_summary_buff, global_step=global_step_val)
		except tf.errors.OutOfRangeError:
			break
	# over

def validate(sess, summary_writer):
	# get iterator handle
	test_handle_val = sess.run(test_handle)
	# initialize iterator
	sess.run(test_iterator.initializer)
	# validation loop
	correctness = 0
	loss_val = 0
	
	while True:
		try:
			# test on single batch
			labels_val, batch_predictions_val, total_loss_val, global_step_val = \
						sess.run([labels, batch_predictions, total_loss, global_step],
								 feed_dict={handle : test_handle_val})
			correctness += np.asscalar(np.sum(a=(batch_predictions_val==labels_val), dtype=np.float32))
			loss_val += np.asscalar(total_loss_val)
		except tf.errors.OutOfRangeError:
			break
	
	current_acc = correctness/test_set_size
	loss_val /= test_set_size
	# print and summary
	msg = 'test accuracy = %.2f%%' % (current_acc*100)
	test_acc_summary = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy',simple_value=current_acc)])
	test_loss_summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=loss_val)])
	# write summary
	summary_writer.add_summary(summary=test_acc_summary, global_step=global_step_val)
	summary_writer.add_summary(summary=test_loss_summary, global_step=global_step_val)
	# print message
	print(msg)
	# over
	return loss_val

# main entrance
if __name__ == "__main__":
	try:
		options, args = getopt.getopt(sys.argv[1:], '', ['logdir='])
	except getopt.GetoptError:
		print('invalid arguments!')
		sys.exit(-1)
	for option, value in options:
		if option == '--logdir':
			summary_name = value
	# train and test the model
	cur_lr = lr0
	best_loss = 1e10
	with tf.Session() as sess:
		# initialize variables
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		# initialize IO
		# build tf saver
		saver = tf.train.Saver()
		# build the tensorboard summary
		summary_writer = tf.summary.FileWriter(summary_path+summary_name)
		train_summary_op = tf.summary.merge_all()
		# train in epochs
		for cur_epoch in range(1, num_epoch+1):
			# print epoch title
			print('Current epoch No.%d, learning rate = %.2e' % (cur_epoch, cur_lr))
			# train
			train(cur_lr, sess, summary_writer, train_summary_op)
			# validate
			cur_loss = validate(sess, summary_writer)
			# update learning rate if necessary
			if cur_loss < best_loss:
				# save check point
				saver.save(sess=sess,save_path=model_path+best_model_ckpt)
				# update learning rate
				cur_lr *= 1+lr_up_ratio
				# print message
				print('model improved, save the ckpt.')
				# update best loss
				best_loss = cur_loss
			else:
				# we must restore old check point
				saver.restore(sess=sess, save_path=model_path+best_model_ckpt)
				# update learning rate
				cur_lr *= 1-lr_dn_ratio
				# print message
				print('model not improved, restore the ckpt.')
