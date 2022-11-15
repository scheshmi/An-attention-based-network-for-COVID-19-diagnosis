
from tensorflow import keras
import tensorflow as tf
import os
from shutil import copyfile
import random


# directories to create....
to_create = {
    'root': './dataset',
    'train_dir': './dataset/training',
    'test_dir': './dataset/testing',
    'covid_train_dir': './dataset/training/covid',
    'non-covid_train_dir': './dataset/training/non-covid',
    'covid_test_dir': './dataset/testing/covid',
    'non-covid_test_dir': './dataset/testing/non-covid'
}


def make_directory() -> None:
    '''
    make directories if they don't exist 
    '''
    for dir in to_create.values():
        if not os.path.isdir(dir):
            os.mkdir(dir)


def split_data(SOURCE: str, Train_path: str, Test_path: str, split_size: float) -> None:
    '''
    Shuffling dataset then splitting dataset into train and test set

    param SOURCE: path to whole dataset
    param Train_path: path to train set
    param Test_path: path to test path
    param split_size: fraction of train set
    '''
    all_files = []
    for image in os.listdir(SOURCE):
        image_path = os.path.join(SOURCE, image)
        if os.path.getsize(image_path):
            all_files.append(image)

    total_files = len(all_files)
    split_point = int(total_files * split_size)
    shuffled = random.sample(all_files, total_files)
    train = shuffled[:split_point]
    test = shuffled[split_point:]

    for image in train:
        copyfile(os.path.join(SOURCE, image), os.path.join(Train_path, image))

    for image in test:
        copyfile(os.path.join(SOURCE, image), os.path.join(Test_path, image))

# make_directory()
# split_data('./data/COVID',to_create.get('covid_train_dir'), to_create.get('covid_test_dir'), 0.8)
# split_data('./data/non-COVID',to_create.get('non-covid_train_dir'), to_create.get('non-covid_test_dir'), 0.8)


def parse_image(filename: str) -> tuple:
    '''
    read image and its label then decode image and convert it to appropriate size

    param filename: file path
    :return image an corrensponding label
    '''
    parts = tf.strings.split(filename, os.sep)
    label = parts[-2]
    # one-hot label
    label = (label == ['covid', 'non-covid'])
    # read image then decoding and resizing

    image = tf.io.read_file(filename)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    return image, label


AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16


def training_dataset(training_dir: str) -> tf.data.Dataset:
    '''
    preparing training dataset, it contains covid and non-covid samples which are paired together

    param  training_dir: training dataset path

    return paired dataset
    '''
    ct_data = tf.data.Dataset.list_files(
        training_dir + '/covid/*', shuffle=True)
    mutex_data = tf.data.Dataset.list_files(
        training_dir + '/non-covid/*', shuffle=True)
    ct_data = ct_data.map(parse_image, num_parallel_calls=AUTOTUNE).shuffle(
        500).batch(BATCH_SIZE, drop_remainder=True)
    mutex_data = mutex_data.map(parse_image, num_parallel_calls=AUTOTUNE).shuffle(
        500).batch(BATCH_SIZE, drop_remainder=True)

    return tf.data.Dataset.zip((ct_data, mutex_data)).prefetch(AUTOTUNE)


def testing_dataset(testing_dir: str) -> tf.data.Dataset:
    '''
    preparing test dataset, unlike training dataset we use both of covid and non-covid samples together here
    param tesring_dir : path to testdataset

    :return test dataset
    '''
    test_data = tf.data.Dataset.list_files(testing_dir + '/*/*', shuffle=True)
    test_data = test_data.map(
        parse_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    test_data = test_data.prefetch(AUTOTUNE)

    return test_data
