import os
import keras
import tensorflow as tf
import shutil

from keras.models import Sequential
from keras.layers import Dense
from utils import input_fn

paths = ['data/iris/iris_train.csv']

SepalLengthCm = tf.feature_column.numeric_column(key='SepalLengthCm',dtype=tf.dtypes.float32)
SepalWidthCm = tf.feature_column.numeric_column(key='SepalWidthCm',dtype=tf.dtypes.float32)
PetalLengthCm = tf.feature_column.numeric_column(key='PetalLengthCm',dtype=tf.dtypes.float32)
PetalWidthCm = tf.feature_column.numeric_column(key='PetalWidthCm',dtype=tf.dtypes.float32)

feature_columns = [SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]

model_dir = 'estimator'

estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
     n_classes=3,
     model_dir=model_dir
)

estimator.train(input_fn=input_fn(paths), steps=1000)

columns = [('SepalLengthCm', tf.float64),
           ('SepalWidthCm',tf.float64),
           ('PetalLengthCm',tf.float64),
           ('PetalWidthCm',tf.float64)]

feature_placeholders = {
 name: tf.placeholder(dtype, [1], name=name + "_placeholder")
 for name, dtype in columns
}
export_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
    feature_placeholders)
path = estimator.export_savedmodel(model_dir, export_input_fn)

shutil.move(path, "deployment")