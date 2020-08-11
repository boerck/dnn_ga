#import Genetic_Algorithm_part as gap
import dnn
from grayscale_colorization import binary_util, grayscale_colorization

binary_util.save("data_batch_train", grayscale_colorization.grayscale(True))
binary_util.save("data_batch_test", grayscale_colorization.grayscale(False))
#binary save

train_data = binary_util.load("data_batch_train", 1)
test_data = binary_util.load("data_batch_test", 1)
# batch size = 1로 설정함.

model = dnn.Net(4, [32*32, 256, 64, 16, 10], train_data, test_data, 100, hin=32*32, hout=10)


if __name__ == "__main__":
    model.run()
