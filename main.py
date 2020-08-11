import Genetic_Algorithm_part as gap
import dnn
import binary_util

data = binary_util.load("data_batch_train", 1)
model = dnn.Net(4, [32*32, 256, 64, 16, 10], data, data, 100, hin=32*32, hout=10)
model.run()
gap.descen_create()

if __name__ == "__main__":
    model
