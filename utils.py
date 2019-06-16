import tensorflow as tf

def input_fn(paths):
    """ model input function """

    names = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
    record_defaults = [[5.1], [3.5], [1.4], [0.3], [0]]

    def _parse_csv(rows_string_tensor):
        columns = tf.decode_csv(rows_string_tensor, record_defaults)
        features = dict(zip(names, columns[:-1]))
        labels = columns[-1]
        return features, labels

    def _input_fn():
        dataset = tf.data.TextLineDataset(paths)
        dataset = dataset.map(_parse_csv)
        dataset = dataset.batch(10)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    return _input_fn
