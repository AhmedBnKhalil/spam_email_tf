import time

from tensorflow.keras.callbacks import Callback


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_start_time)

    def on_train_end(self, logs={}):
        self.total_time = sum(self.times)
        print(f"Total training time: {self.total_time:.2f} seconds")
