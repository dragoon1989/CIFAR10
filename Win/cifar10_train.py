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
summary_path = './tensorboard/'
summary_name = 'summary-default'    # tensorboard default summary dir
num_epochs = 20

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
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    #grad_var_pairs = optimizer.compute_gradients(loss)
    #train_op = optimizer.apply_gradients(grad_var_pairs, global_step=global_step)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(batch_loss, global_step)
'''
    train_op = tf.train.AdamOptimizer(learning_rate=1e-3,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-08).minimize(batch_loss, global_step=global_step)
'''
with tf.name_scope('train_batch_accuracy'):
    # compute batch predictions (dtype cast to tf.int32)
    batch_predictions = tf.to_int32(tf.argmax(tf.nn.softmax(logits_before_softmax), axis=1))
    # compute batch accuracy
    correct_prediction = tf.equal(batch_predictions, labels)
    batch_accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
    # summary the batch accuracy
    tf.summary.scalar(name='train_accuracy', tensor=batch_accuracy)

############################# define the validation #############################

# build the training process
def train(epochs):
    ''' input:  epochs --- number of epochs '''
    # initialize counters
    accuracy = 0
    current_acc = 0
    current_epoch = 0
    # build tf saver
    saver = tf.train.Saver()
    # build the tensorboard summary
    summary_writer = tf.summary.FileWriter(summary_path+summary_name)
    merged_train_summary = tf.summary.merge_all()
    # create tf session
    with tf.Session() as sess:
        # save the graph
        summary_writer.add_graph(graph=sess.graph)
        # initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # get iterator handles
        train_handle_val, test_handle_val = sess.run([train_handle, test_handle])
        # training loop
        for current_epoch in range(1,epochs+1):
            print('Current epoch num = %d'%(current_epoch))
            # re-initialize dataset iterators
            sess.run([train_iterator.initializer])
            # training part
            current_batch = 0
            while True:
                try:
                    # train on single batch
                    _, batch_accuracy_val, global_step_val, train_summary_buff = \
                                sess.run([train_op, batch_accuracy, global_step, merged_train_summary],
                                         feed_dict={handle : train_handle_val})
                    current_batch += 1
                    # print indication info
                    if current_batch % 50 == 0:
                        msg = '\tbatch number = %d, train accuracy = %.2f%%' % (current_batch, batch_accuracy_val*100)
                        # write train summary
                        summary_writer.add_summary(summary=train_summary_buff, global_step=global_step_val)
                        # validation part
                        sess.run([test_iterator.initializer])
                        correctness = 0
                        loss_val = 0
                        while True:
                            try:
                                # test on single batch
                                labels_val, batch_predictions_val, total_loss_val = \
                                            sess.run([labels, batch_predictions, total_loss],
                                                     feed_dict={handle : test_handle_val})
                                correctness += np.asscalar(np.sum(a=(batch_predictions_val==labels_val), dtype=np.float32))
                                loss_val += np.asscalar(total_loss_val)
                            except tf.errors.OutOfRangeError:
                                break
                        current_acc = correctness/test_set_size
                        loss_val /= test_set_size
                        msg += ', test accuracy = %.2f%%' % (current_acc*100)
                        test_acc_summary = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy',simple_value=current_acc)])
                        test_loss_summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=loss_val)])
                        # write summary
                        summary_writer.add_summary(summary=test_acc_summary, global_step=global_step_val)
                        summary_writer.add_summary(summary=test_loss_summary, global_step=global_step_val)
                        # print message
                        print(msg)
                except tf.errors.OutOfRangeError:
                    break

            # save current best model parameters
            print('Epoch %d accuracy : %.2f%%'%(current_epoch, current_acc*100))
            # whether to save current weights or not
            if accuracy < current_acc:
                # save
                print('model is improved, save the result')
                saver.save(sess=sess, save_path=model_path, global_step=global_step_val)
                accuracy = current_acc

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
    train(num_epochs)
