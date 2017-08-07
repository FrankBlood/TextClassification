from keras.utils import np_utils

y = [[1], [0], [2], [3], [4], [2], [4], [3], [4], [4]]

y = np_utils.to_categorical(y, num_classes=5)

print y