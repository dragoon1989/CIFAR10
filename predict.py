import numpy as np
import tensorflow as tf

from cifar10_input import BuildInputPipeline
from model import model


test_x, test_y = get_data_set("test")
x, y, output, y_pred_cls, global_step, learning_rate = model()


_BATCH_SIZE = 128
_TEST_DATA_SIZE = 10000
_CLASS_SIZE = 10
_SAVE_PATH = "./tensorboard/cifar-10-v1.0.0/"

test_data = BuildInputPipeline('test', _BATCH_SIZE,
						num_parallel_calls=1, num_epoch=1)
test_iterator = test_data.make_initializable_iterator()

labels, images = test_iterator.get_next()

before_softmax, _, _ = model(images)
logits_after_softmax = tf.nn.softmax(before_softmax)
predictions = tf.to_int32(tf.argmax(logits_after_softmax, axis=1))	# from tf.float32 cast to tf.int32

saver = tf.train.Saver()
sess = tf.Session()


try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    

def main():
    i = 0
    # re-initialize test iterator
    sess.run(test_iterator.initializer)
    
    predicted_class = np.zeros(shape=_TEST_DATA_SIZE, dtype=np.int)
    label_class = np.zeros(shape=_TEST_DATA_SIZE, dtype=np.int)
    
    while i < _TEST_DATA_SIZE:
        j = min(i + _BATCH_SIZE, _TEST_DATA_SIZE)
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        [label_class[i:j], predicted_class[i:j]] = sess.run([labels, predictions])
        i = j

    correct = (label_class == predicted_class)
    acc = correct.mean() * 100
    correct_numbers = correct.sum()
    print()
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))


if __name__ == "__main__":
    main()


sess.close()
