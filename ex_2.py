import sys
import numpy as np

LABEL_NUM = 3
FEATURES_NUM = 12
K = 7
ETA = 0.001
EPOCH1 = 9
EPOCH2 = 1


def main():
    # convert files to arrays
    array_x, num_of_example = convert_train_to_array(sys.argv[1])
    array_y, num_of_y = convert_train_to_array(sys.argv[2])
    test_x, num_of_test = convert_train_to_array(sys.argv[3])

    # delete the third feature
    array_x = np.delete(array_x, 2, 1)
    test_x = np.delete(test_x, 2, 1)
    array_x, test_x = normalize(array_x, test_x)

    # add base
    array_x = add_base(array_x)
    test_x = add_base(test_x)

    # print prediction of all the algorithms
    print_prediction(array_x, array_y, test_x)


def knn(train_x, train_y, x):
    dist_array = []
    # calculate the distance from the point to all the example points
    for example_point, label in zip(train_x, train_y):
        distance = np.linalg.norm(example_point - x)
        dist_array.append([distance, label[0]])
    # sort and take only the k closest points
    dist_array.sort(key=sort_by)
    k_neighbors = dist_array[0:K]
    # find the tag that appears the most
    count_labels = [0, 0, 0]
    for elem in k_neighbors:
        count_labels[int(elem[1])] += 1
    x_label = np.argmax(count_labels, axis=0)
    return x_label


def knn_prediction(test_x, train_x, train_y):
    knn_yhat = np.zeros(len(test_x))
    # for each vector
    for i in range(len(test_x)):
        knn_yhat[i] = knn(train_x, train_y, test_x[i])
    return knn_yhat


def perceptron(train_x, train_y):
    w = np.zeros((LABEL_NUM, FEATURES_NUM))
    for i in range(EPOCH1):
        list_zip = list(zip(train_x, train_y))
        np.random.shuffle(list_zip)
        for x, y in list_zip:
            y_hat = np.argmax(np.dot(w, x))
            if y_hat != y:
                w[int(y[0])] += (ETA * x)
                w[y_hat] -= (ETA * x)
    return w


def perceptron_prediction(test_x, train_x, train_y):
    perceptron_yhat = np.zeros(len(test_x))
    w_perceptron = perceptron(train_x, train_y)
    for i in range(len(test_x)):
        perceptron_yhat[i] = np.argmax(np.dot(w_perceptron, test_x[i]))
    return perceptron_yhat


def pa(train_x, train_y):
    w = np.zeros((LABEL_NUM, FEATURES_NUM))
    for i in range(EPOCH2):
        list_zip = list(zip(train_x, train_y))
        np.random.shuffle(list_zip)
        for x, y in list_zip:
            mult = np.dot(w, x)
            mult[int(y[0])] = np.NINF
            y_hat = np.argmax(mult)
            if int(y[0]) != y_hat:
                # calculate the loss function
                loss = max(0, 1 - (np.dot(w[int(y[0]), :], x)) + (np.dot(w[y_hat, :], x)))
                x_x = (np.linalg.norm(x) ** 2) * 2
                if x_x != 0:
                    tau = loss / x_x
                else:
                    tau = 10
                w[int(y[0]), :] += (tau * x)
                w[y_hat, :] -= (tau * x)
    return w


def pa_prediction(test_x, train_x, train_y):
    pa_yhat = np.zeros(len(test_x))
    w_pa = pa(train_x, train_y)
    for i in range(len(test_x)):
        pa_yhat[i] = np.argmax(np.dot(w_pa, test_x[i]))
    return pa_yhat


def add_base(array):
    array_with_base = np.zeros([len(array), FEATURES_NUM])
    for i in range(len(array)):
        for j in range(FEATURES_NUM - 1):
            array_with_base[i][j] = array[i][j]
        array_with_base[i][FEATURES_NUM - 1] = 1
    return array_with_base


def normalize(train_x, test_x):
    train_normalized = np.zeros([len(train_x), FEATURES_NUM - 1])
    test_normalized = np.zeros([len(test_x), FEATURES_NUM - 1])
    for i in range(FEATURES_NUM - 1):
        mean = float(np.mean(train_x[:, i]))
        stand_dev = float(np.std(train_x[:, i]))
        if mean and stand_dev and stand_dev != 0:
            test_normalized[:, i] = (test_x[:, i] - mean) / stand_dev
            train_normalized[:, i] = (train_x[:, i] - mean) / stand_dev
    return train_normalized, test_normalized


def convert_train_to_array(filename):
    data = []
    count = 0
    with open(filename) as file:
        for line in file.readlines():
            # convert to array while 'R'=0 and 'W'=1
            data.append(line.replace('\n', '').replace('W', "1").replace('R', "0").split(','))
            count += 1
        x = np.array(data)
        y = x.astype(np.float)
    # return the array and the number of line in the file
    return y, count


def sort_by(e):
    return e[0]


def print_prediction(train_x, train_y, test_x):
    knn_yhat = knn_prediction(test_x, train_x, train_y)
    perceptron_yhat = perceptron_prediction(test_x, train_x, train_y)
    pa_yhat = pa_prediction(test_x, train_x, train_y)
    for i in range(len(test_x)):
        print(f"knn: {int(knn_yhat[i])}, perceptron: {int(perceptron_yhat[i])}, pa: {int(pa_yhat[i])}")


if __name__ == '__main__':
    main()
