import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from font_dataset import FontDataset
import time
from random import randrange


# ================================================================== #
#                        0. Define Hyper-parameters
# ================================================================== #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 50
in_channel = 3

batch_size = 100
val_batch_size = 100
shuffle = True
learning_rate = 1e-3
num_epochs = 100

# ================================================================== #
#                        1. Load Data
# ================================================================== #
train_data = FontDataset('./datasets/npy_train')
test_data = FontDataset('./datasets/npy_test')

# ================================================================== #
#                        2. Define Dataloader
# ================================================================== #
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=val_batch_size)

print(next(iter(train_loader)))
print("Train: {} batches with {} iteration".format(batch_size, len(train_loader)))

# ================================================================== #
#                        3. Define Model
# ================================================================== #
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channel, 16, 5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, 5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2))
        self.fc1 = nn.Linear(8*8*32, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

model = ConvNet(num_classes).to(device)


# ================================================================== #
#                        4. Set Loss & Optimizer
# ================================================================== #
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# ================================================================== #
#                        5. Train / Test
# ================================================================== #
if __name__ == '__main__':
    start_time = time.time()

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Assign Tensors to Configured Device
            images = images.to(device)
            labels = labels.to(device)

            # Forward Propagation
            outputs = model(images)

            # Get Loss, Compute Gradient, Update Parameters
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print Loss for Tracking Training
            if (i+1) % 10 == 0 or (i+1) == len(train_loader):
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                
#            if (i+1) % 100 == 0 or (i+1) == len(train_loader):
#                random_test_idx = randrange(len(test_loader))
#                for i, (test_image, test_label) in enumerate(test_loader):
#                    if i == random_test_idx:
#                        _, test_predicted = torch.max(model(test_image.to(device)).data, 1)
#                        print('Testing data: [Predicted: {} \n Real: {}]'.format(test_predicted, test_label))
#                    else:
#                        print("con-", i)
#                        continue

            if (i+1) % 100 == 0 or (i+1) == len(train_loader):
                test_image, test_label = next(iter(test_loader))
                _, test_predicted = torch.max(model(test_image.to(device)).data, 1)
                count = 0
                for j in range(len(test_label)):
                    if test_label.to(device)[j] == test_predicted[j]:
                        count += 1
                print('Testing data: [Predicted: {} \n Real: {} \n Acc: {}(Correct: {}, Wrong: {})]'.format(test_predicted, test_label, count*100/len(test_label), count, len(test_label)-count))

        if epoch+1 == num_epochs:
            torch.save(model.state_dict(), 'model.pth')

        print("Execution time: {} seconds".format(time.time() - start_time))

    # Test after Training is done
    model.eval() # Set model to Evaluation Mode (Batchnorm uses moving mean/var instead of mini-batch mean/var)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the {} test images: {}(correct: {}, wrong: {}) %'.format(len(test_loader)*batch_size, 100 * correct / total, correct, total-correct))
#
