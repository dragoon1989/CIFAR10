import tensorflow as tf


cifar10_size = 32

# parse single example from cifar10 dataset
def __parse_single_example(example):
	''' input : example --- single example (1 + 1024x3 bytes)
	   output: label --- int32 scalar tensor 
	   	       image --- float32 3D tensor (format = HWC)'''
	raw_bytes = tf.decode_raw(bytes=example, out_type=tf.uint8)
	# convert label to tf.int32
	label = tf.to_int32(raw_bytes[0])
	# reshape to scalar tensor
	label = tf.reshape(tensor=label, shape=[])
	# convert image to tf.float32
	image = tf.to_float32(raw_bytes[1:(1+cifar10_size*cifar10_size*3)])
	# reshape image to CHW format
	image = tf.reshape(tensor=image, shape=[cifar10_size, cifar10_size, 3])
	# permute the image to HWC format (use tf.transpose)
	image = tf.transpose(a=image, perm=[1, 2, 0]
	# over
	return label, image

# build a dataset that consists of all examples from a given data file
def __read_single_file(file_name):
	''' input:	file_name --- single data file name (scalar string tensor)
	   output:	dataset --- a dataset that consists of (label, image) elements'''
	# build a fixed length dataset
	dataset = tf.data.FixedLengthRecordDataset(file_name)
	# parse all examples and form a new dataset
	dataset = dataset.map(map_func=__parse_single_example
	# over
	return dataset

# build input pipeline using datasets
def BuildInputPipeline(file_name_list, 
				batch_size, 
				num_parallel_calls=1, 
				num_epoch=1):
	''' input:	file_name_list --- 1D string tensor
			batch_size --- size of batch
			num_parallel_calls
			num_epoch --- number of epochs
	   output:	dataset --- a dataset consisting of batches of (label,image) data from input files'''
	# build a file name dataset
	file_names_dataset = tf.data.Dataset.from_tensor_slices(file_name_list)
	# build a dataset that read all files named by file_names_dataset
	# the dataset consists of (label, image) pairs read from all files indicated in input list
	dataset = file_names_dataset.interleave(map_func=__read_single_file,
								     cycle_length=4,
								     block_length=16,
								     num_parallel_calls=num_parallel_calls)
	# set the epoch
	dataset = dataset.repeat(count=num_epoch)
	# shuffle the dataset
	dataset = dataset.shuffle(buffer_size=10*batch_size)
	# set the batch size
	dataset = dataset.batch(batch_size=batch_size)
	# use prefetch to allow asynchronous input
	# i think prefetch one batch is enouth
	dataset = dataset.prefetch(buffer_size=1)
	# over
	return dataset