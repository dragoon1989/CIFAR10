import numpy as np
import tensorflow as tf
from time import time
import math


from cifar10_input import BuildInputPipeline
from model import model, lr

# PARAMS
_TRAINING_DATA_PATH = ['batch_%d.bin'%(i) for i in range(5)]
_TEST_DATA_PATH = ['test.bin']
_BATCH_SIZE = 128
_TRAINING_DATA_SIZE = 50000
_TEST_DATA_SIZE = 10000
_EPOCH = 60
_SAVE_PATH = "./tensorboard/cifar-10-v1.0.0/"

tf.set_random_seed(21)

train_data = BuildInputPipeline(_TRAINING_DATA_PATH, _BATCH_SIZE, 
						num_parallel_calls=4, num_epoch=1)
test_data = BuildInputPipeline(_TEST_DATA_PATH, _BATCH_SIZE,
						num_parallel_calls=1, num_epoch=1)
						
train_iterator = train_data.make_initializable_iterator()
test_iterator = test_data.make_initializable_iterator()

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types)
labels, images = iterator.get_next()

before_softmax, global_step, learning_rate = model(x=images)

global_accuracy = 0
epoch_start = 0

# LOSS AND OPTIMIZER
# labels must be of dtype=tf.int32, logits must be of dtype=tf.float32
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, 
													    logits=before_softmax))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(loss, global_step=global_step)

# PREDICTION AND ACCURACY CALCULATION
logits_after_softmax = tf.nn.softmax(before_softmax)
predictions = tf.to_int32(tf.argmax(logits_after_softmax, axis=1))	# from tf.float32 cast to tf.int32
correct_prediction = tf.to_float(tf.equal(predictions, labels)	)		# cast back to tf.float32
accuracy = tf.reduce_mean(correct_prediction)

# create default session to be used
sess = tf.Session()

# SAVER
merged = tf.summary.merge_all()
saver = tf.train.Saver()
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)


try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())


def train(epoch):
    global epoch_start
    epoch_start = time()
    num_batch = int(_TRAINING_DATA_SIZE / _BATCH_SIZE))	# num of batches in one epoch
    i_global = 0
    # re-initialize the training iterator
    sess.run(train_iterator.initializer)

    for s in range(num_batch):
        start_time = time()
        # get the runtime string handle for training iterator
        train_handle_val = sess.run(train_iterator.string_handle())
        i_global, _, batch_loss, batch_acc = sess.run([global_step, optimizer, loss, accuracy],
            								   feed_dict={learning_rate: lr(epoch),
            								   		     handle: train_handle_val})
        duration = time() - start_time

        if s % 10 == 0:
            percentage = int(round((s/num_batch)*100))

            bar_len = 29
            filled_len = int((bar_len*int(percentage))/100)
            bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)

            msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
            print(msg.format(i_global, bar, percentage, batch_acc, batch_loss, _BATCH_SIZE / duration))

    test_and_save(i_global, epoch)


def test_and_save(_global_step, epoch):
    global global_accuracy
    global epoch_start
    
    # re-initialize test iterator
    sess.run(test_iterator.initializer)
	# get the runtime string handle for test iterator
    test_handle_val = sess.run(test_iterator.string_handle())
    
    predicted_class = np.zeros(shape=_TEST_DATA_SIZE, dtype=np.int)
    label_class = np.zeros(shape=_TEST_DATA_SIZE, dtype=np.int)
    i = 0
    # read test dataset in batches, and concatenate labels together
    while i < _TEST_DATA_SIZE:
        j = min(i + _BATCH_SIZE, _TEST_DATA_SIZE)
        # returned labels and predictions are all of dtype=tf.int32
        [label_class[i:j], predicted_class[i:j]] = sess.run(labels, predictions,
            					      				feed_dict={learning_rate: lr(epoch)})
        i = j
	# these are all numpy array operations
    correct = (labels == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()

    hours, rem = divmod(time() - epoch_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{}) - time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format((epoch+1), acc, correct_numbers, len(test_x), int(hours), int(minutes), seconds))

    if global_accuracy != 0 and global_accuracy < acc:

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
        ])
        train_writer.add_summary(summary, _global_step)

        saver.save(sess, save_path=_SAVE_PATH, global_step=_global_step)

        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format(acc, global_accuracy))
        global_accuracy = acc

    elif global_accuracy == 0:
        global_accuracy = acc

    print("###########################################################################################################")


def main():
    train_start = time()

    for i in range(_EPOCH):
        print("\nEpoch: {}/{}\n".format((i+1), _EPOCH))
        train(i)

    hours, rem = divmod(time() - train_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = "Best accuracy pre session: {:.2f}, time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format(global_accuracy, int(hours), int(minutes), seconds))


if __name__ == "__main__":
    main()


sess.close()
