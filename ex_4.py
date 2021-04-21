import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.legend_handler import HandlerLine2D
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
import copy


INPUT_SIZE = 784
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001


class ModelA(nn.Module):

    def __init__(self, size, p=0.0, batch_normalization=False):
        super(ModelA, self).__init__()
        self.batch_normalization = batch_normalization
        self.size = size
        self.fc0 = nn.Linear(size, 100)
        self.bn1 = nn.BatchNorm1d(num_features=100)
        self.fc1 = nn.Linear(100, 50)
        self.bn2 = nn.BatchNorm1d(num_features=50)
        self.fc2 = nn.Linear(50, 10)
        self.drop_layer = nn.Dropout(p=p)

    def forward(self, x):
        x = x.view(-1, self.size)
        if not self.batch_normalization:
            x = F.relu(self.fc0(x))
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.bn1(self.fc0(x)))
            x = F.relu(self.bn2(self.fc1(x)))

        x = self.fc2(x)
        self.drop_layer(x)
        return F.log_softmax(x, dim=1)


class ModelB(nn.Module):

    def __init__(self, size):
        super(ModelB, self).__init__()
        self.size = size
        self.fc0 = nn.Linear(size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def test(self, test_set):
        self.eval()
        with open("test_y", "w+") as fd:
            for data in test_set:
                output = self.forward(data)
                fd.write(f'{output.data.max(1, keepdim=True)[1].item()}\n')


class ModelE(nn.Module):

    def __init__(self, size):
        super(ModelE, self).__init__()
        self.size = size
        self.fc0 = nn.Linear(size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

    def test(self, test_set):
        with open("test_y", "w+") as fd:
            for data in test_set:
                output = self.forward(data)
                fd.write(f'{output.data.max(1, keepdim=True)[1].item()}\n')


class ModelF(nn.Module):

    def __init__(self, size):
        super(ModelF, self).__init__()
        self.size = size
        self.fc0 = nn.Linear(size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


def loss_graph(train_loss_avg, validation_loss_avg, model_name):
    line1, = plt.plot(list(train_loss_avg.keys()), list(train_loss_avg.values()), "red",
                      label='Train average Loss')
    line2, = plt.plot(list(validation_loss_avg.keys()), list(validation_loss_avg.values()), "blue",
                      label='Validation average Loss')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4), line2: HandlerLine2D(numpoints=4)})
    plt.title(model_name + " - Loss")
    plt.show()


def acc_graph(train_acc_avg, validation_acc_avg, model_name):
    line1, = plt.plot(list(train_acc_avg.keys()), list(train_acc_avg.values()), "red",
                      label='Train average Accuracy')
    line2, = plt.plot(list(validation_acc_avg.keys()), list(validation_acc_avg.values()), "blue",
                      label='Validation average Accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4), line2: HandlerLine2D(numpoints=4)})
    plt.title(model_name + " - Accuracy")
    plt.show()


def train(model, train_set, optimizer):
    model.train()
    train_loss = 0
    train_accuracy = 0
    for batch_idx, (data, labels) in enumerate(train_set):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels.long())
        train_loss += loss
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        train_accuracy += pred.eq(labels.data.view_as(pred)).cpu().sum()
    train_loss_avg = train_loss / len(train_set)
    train_accuracy_avg = train_accuracy / len(train_set.dataset) * 100
    return train_loss_avg, train_accuracy_avg


def validation(model, validation_set):
    model.eval()
    validation_loss = 0
    validation_accuracy = 0
    with torch.no_grad():
        for data, target in validation_set:
            output = model(data)
            validation_loss += F.nll_loss(output, target.long()).item()
            pred = output.data.max(1, keepdim=True)[1]
            validation_accuracy += (pred.eq(target.view_as(pred)).cpu().sum()).int()
    # print(float(validation_accuracy / len(validation_set.dataset) * 100))
    validation_loss_avg = validation_loss / len(validation_set.dataset)
    validation_accuracy_avg = float((validation_accuracy / len(validation_set.dataset)) * 100)
    return validation_loss_avg, validation_accuracy_avg


def run(model, train_set, optimizer, validation_set):
    for epoch in range(EPOCHS):
        train(model, train_set, optimizer)
        validation(model, validation_set)


class MyData(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.X = images
        self.y = labels
        self.transforms = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i, :]
        data = np.asarray(data).astype(np.uint8).reshape(28, 28)
        if self.transforms:
            data = self.transforms(data)
        if self.y is not None:
            return data, self.y[i]
        else:
            return data


def convert_to_tensor(train_x, train_y, validation_x, validation_y, test_x):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2858,), (0.3529,))])
    train_loader = torch.utils.data.DataLoader(
        MyData(train_x, train_y, transform),
        batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(
        MyData(validation_x, validation_y, transform),
        batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        MyData(test_x, None, transform))
    return train_loader, validation_loader, test_loader


def split_to_train_and_validation(train_x, train_y):
    size_of_validation = int(len(train_x) * 0.2)
    train_x = train_x[size_of_validation:]
    train_y = train_y[size_of_validation:]
    validation_x = train_x[:size_of_validation]
    validation_y = train_y[:size_of_validation]
    return train_x, train_y, validation_x, validation_y


def main():
    try:
        train_x_file, train_y_file, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    except IndexError:
        print("Files loading problem:")
        return
    # load files
    train_x = np.loadtxt(train_x_file)
    train_y = np.loadtxt(train_y_file, dtype=int)
    test_x = np.loadtxt(test_x)
    # split to 80% train and 20% validation
    train_x, train_y, validation_x, validation_y = split_to_train_and_validation(train_x, train_y)

    train_loader, validation_loader, test_loader = convert_to_tensor(train_x, train_y,
                                                                     validation_x, validation_y, test_x)
    # run the best model
    model_b = ModelB(INPUT_SIZE)
    optimizer_b = optim.Adam(model_b.parameters(), lr=LEARNING_RATE)
    run(model_b, train_loader, optimizer_b, validation_loader)
    model_b.test(test_loader)


if __name__ == "__main__":
    main()
