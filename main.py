import Genetic_Algorithm_part as gap
from grayscale_colorization import binary_util, grayscale_colorization

binary_util.save("data_batch_train", grayscale_colorization.grayscale(True))
binary_util.save("data_batch_test", grayscale_colorization.grayscale(False))
#binary save


if __name__ == "__main__":
    genetic = gap.genetic(200)
    model = genetic.start()
    k = 0
    epoch = 15
    for i in range(epoch):
        k += 1
        su_model = genetic.competition(model)
        model = genetic.create(su_model)

        if k == epoch:
            print(su_model)

