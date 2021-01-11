from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

ds_conf = {"num_classes": 10,
           "img_height": 32,
           "img_width": 32,
           "img_channels": 3,
           "epochs": 25}

