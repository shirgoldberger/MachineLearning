import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Linear, CrossEntropyLoss, Dropout
from gcommand_dataset import GCommandLoader
import torch.nn.functional as F

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001


class ConvNet(Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = Sequential(
            Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = Sequential(
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = Sequential(
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = Sequential(
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = Sequential(
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = Dropout(p=0.1)
        self.fc1 = Linear(7680, 1000)
        self.fc2 = Linear(1000, 512)
        self.fc3 = Linear(512, 30)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return F.log_softmax(out, dim=1)


def train(model, optimizer, train_loader, criterion, device, validation_loader):
    loss_list = []
    acc_list = []
    model.train()
    for epoch in range(EPOCHS):
        print("epoch " + str(epoch))
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

        #test(model, validation_loader, device)


def test(model, test_loader, device):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))

    # Save the model and plot
    torch.save(model.state_dict(), './conv_net_model.ckpt')


def prediction(test_loader, model, device, classes):
    model.eval()
    f = open("test_y", "w")
    i = 0
    predicted_list = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            predicted = output.data.max(1, keepdim=True)[1].item()
            data_ = int(test_loader.dataset.spects[i][0].split("/")[4].split('.')[0])
            predicted_list.append((data_,predicted))
            # print to file
            i += 1
    predicted_list = sorted(predicted_list)
    for e in predicted_list:
        line = str(e[0]) + ".wav, " + classes[e[1]] + '\n'
        f.write(line)
    f.close()


def convert_to_tensor():
    dataset_train = GCommandLoader('train')
    dataset_valid = GCommandLoader('valid')

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(
        dataset_valid, shuffle=False,
        pin_memory=True)
    return train_loader, validation_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, validation_loader = convert_to_tensor()

    # for input, label in test_loader:
    #     print(f"input shape : {input.shape}, label shape : {label}")
    dataset_test = GCommandLoader('test')
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        pin_memory=True)
    model = ConvNet().to(device)

    # Loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train(model, optimizer, train_loader, criterion, device, validation_loader)
    classes = train_loader.dataset.classes
    prediction(test_loader, model, device, classes)


if __name__ == '__main__':
    main()