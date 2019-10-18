import torch
import torchvision
import torchvision.transforms as transforms
from depth_classification_dataset import depth_classification_dataset
from depth_segmentation_dataset import depth_segmentation_dataset
from nyu_depth_segmentation_dataset import nyu_depth_segmentation_dataset
from resnet import ResNet18, ResNet50
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random



RES_OUT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources_out')

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 15)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class Net(nn.Module):
    def __init__(self,device, num_class=15, mode='classification', channels=64):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(16, momentum=0.99)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 36, 5,padding=2)
        self.batch_norm2 = nn.BatchNorm2d(36, momentum=0.99)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(36, 64, 5,padding=2)
        self.batch_norm3 = nn.BatchNorm2d(64, momentum=0.99)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, 3,padding=1)
        self.batch_norm4 = nn.BatchNorm2d(128, momentum=0.99)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128,256,2)
        self.batch_norm5 = nn.BatchNorm2d(256, momentum=0.99)
        self.conv6 = nn.Conv2d(256, 15, 1)

        # self.conv5 = nn.Conv2d(64, 64, 5,padding=2)
        # self.batch_norm5 = nn.BatchNorm2d(64)
        # self.conv6 = nn.Conv2d(64, num_class, 1)
        self.dense = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, 15)
        # self.conv7 = nn.Conv2d(channels,num_class,2, padding = 1)
        # self.conv8 = nn.Conv2d(channels,num_class,4, padding = 3)
        # self.conv9 = nn.Conv2d(channels, num_class, 8, padding=7)
        # self.conv_transpose = nn.ConvTranspose2d(num_class, num_class, kernel_size=32,stride=32)
        # self.pool_8x8 = nn.MaxPool2d((8, 8))
        # self.pool_4x4 = nn.MaxPool2d((4, 4))
        # self.pool_2x2 = nn.MaxPool2d((2, 2))
        self.device = device
        self.num_class = num_class
        self.mode = mode

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool4(x)
        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = self.conv6(x)
        x = x.view(-1, self.num_class)
        return x

    # def forward(self, x):
    #     x = F.relu(self.batch_norm1(self.conv1(x)))
    #     # 32x32
    #     x = self.pool(x)
    #     # 16x16
    #     x = F.relu(self.batch_norm2(self.conv2(x)))
    #     x = self.pool(x)
    #     # 8x8
    #     x = F.relu(self.batch_norm3(self.conv3(x)))
    #     x2 = self.pool_8x8(self.conv9(x))
    #     x = self.pool(x)
    #     # 4x4
    #     x = F.relu(self.batch_norm4(self.conv4(x)))
    #     x3 = self.pool_4x4(self.conv8(x))
    #     x = self.pool(x)
    #     # 2x2
    #     x = F.relu(self.batch_norm5(self.conv5(x)))
    #     x4 = self.pool_2x2(self.conv7(x))
    #     x = self.pool(x)
    #     # 1x1
    #     # if self.training:
    #     #x = F.relu(self.conv6(x))
    #     # else:
    #     if self.mode == 'segmentation':
    #         x = F.relu(self.conv6(x)) + x2 + x3 + x4
    #         x = self.conv_transpose(x)
    #     elif self.mode == 'classification':
    #         x = F.relu(self.conv6(x))
    #         x = x.view(-1, self.num_class)
    #     return x

def load_data(mode, image_size, batch_size,dataset=None):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if mode == 'segmentation':
        if image_size == 'small':
            train_dir = '/home/yotamg/data/jpg_images/patches'
            test_dir = '/home/yotamg/data/jpg_images/test/patches'
            label_dir = '/home/yotamg/data/depth_pngs/patches'
        else:
            train_dir = '/home/yotamg/data/jpg_images/'
            test_dir = '/home/yotamg/data/jpg_images/test/'
            label_dir = '/home/yotamg/data/depth_pngs/'
    else:
        train_dir = '/home/yotamg/data/rgb/train'
        test_dir = '/home/yotamg/data/rgb/val'
        label_dir = None

    if dataset == 'nyu':
        train_dir = '/home/yotamg/imaging/output_D25_Dn2150/rgb'
        test_dir = '/home/yotamg/imaging/output_D25_Dn2150/rgb/test'
        label_dir = '/home/yotamg/imaging/output_D25_Dn2150/GT_png'



    if dataset == 'nyu':
        trainset = nyu_depth_segmentation_dataset(root='./data', train=True, transform=transform, train_dir=train_dir,
                                              label_dir=label_dir, load_pickle=True)
        testset = nyu_depth_segmentation_dataset(root='./data', train=False, transform=transform, test_dir=test_dir,
                                             label_dir=label_dir, load_pickle=True)
    else:
        if mode == 'segmentation':
            trainset = depth_segmentation_dataset(root='./data', train=True, transform=transform, train_dir=train_dir,
                                                  label_dir=label_dir, load_pickle=True)
            testset = depth_segmentation_dataset(root='./data', train=False, transform=transform, test_dir=test_dir,
                                                 label_dir=label_dir, load_pickle=True)
        else:
            trainset = depth_classification_dataset(root='./data', train=True, transform=transform, train_dir=train_dir,
                                                    load_pickle=True)
            testset = depth_classification_dataset(root='./data', train=False, transform=transform, test_dir=test_dir,
                                                   load_pickle=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=True, num_workers=2)

    return trainloader, testloader


class DfdConfig(object):
    def __init__(self, batch_size=64, mode='segmentation', image_size='small', resume_path=os.path.join(RES_OUT,'best_model.pt'), start_epoch=0, end_epoch=50, model_state_name='best_model.pt', num_classes=15, lr=0.01,get_reduced_dataset=False,dataset=None):
        self.batch_size = batch_size
        self.mode = mode
        self.image_size = image_size
        self.resume_path = resume_path
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.model_state_name = model_state_name
        self.num_classes = num_classes
        self.lr = lr
        self.get_reduced_dataset = get_reduced_dataset
        self.dataset = dataset


def custom_loss(X,y):
    loss = nn.CrossEntropyLoss()

class Dfd(object):
    def __init__(self, net=None, config=DfdConfig(), device=None):
        self.config = config
        self.trainloader, self.testloader = load_data(config.mode,config.image_size, config.batch_size, dataset=config.dataset)
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = net if net else ResNet18(mode=config.mode, image_size=config.image_size, num_classes=self.config.num_classes)
        self.net = (self.net).to(self.device)
        if config.mode == 'segmentation':
            self.criterion = nn.CrossEntropyLoss(weight=self.get_class_weights())
        else:
            self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=self.config.lr, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr)
        # self.optimizer = optim.Adagrad(self.net.parameters(), lr=self.config.lr, weight_decay=0.0002)
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14','15')
        self.colors = (
        (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0,0,128))
        self.loss_history = list()
        self.acc_history = list()
        self.acc_plus_minus_history = list()
        self.steps_history = list()
        self.conf_matrix = np.zeros((config.num_classes, config.num_classes), dtype=np.uint32)
        if self.config.get_reduced_dataset:
            inds = random.sample(range(1, len(self.trainloader.dataset.train_data)), 10000)
            self.trainloader.dataset.train_data = self.trainloader.dataset.train_data[inds, :, :, :]
            self.trainloader.dataset.train_labels = list(np.array(self.trainloader.dataset.train_labels)[inds])

    def transform_image(self,x):
        x = ((x + 1) / 2)* 255
        # x = x.int()
        return x

    # functions to show an image
    def prepare_for_net(self, np_arr):
        np_arr = np.expand_dims(np_arr, 0)
        np_arr = np.transpose(np_arr, (0, 3, 1, 2))
        np_arr = torch.from_numpy(np_arr)
        np_arr = np_arr.type(torch.FloatTensor)
        np_arr = np_arr.to(self.device)
        return np_arr

    def imshow(self, img, transpose=True):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        if transpose:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
        else:
            plt.imshow(npimg)

    def plot_metrics(self):
        plt.subplot(131)
        plt.title('loss')
        plt.plot(self.loss_history, color='g')
        plt.subplot(132)
        plt.title('acc')
        plt.plot(self.acc_history, color='b')
        plt.subplot(133)
        plt.title('acc ±1')
        plt.plot(self.acc_plus_minus_history, color='r')
        # plt.xlabel('Steps')
        # plt.ylabel('loss (g), acc (b), acc+-1 (r)')
        plt.show()


    def show_random_training_images(self):
        # get some random training images
        dataiter = iter(self.trainloader)
        images, labels = dataiter.next()

        # show images
        self.imshow(torchvision.utils.make_grid(images))
        # print labels
        if not self.config.mode == 'segmentation':
            print(' '.join('%5s' % self.classes[labels[j]] for j in range(self.config.batch_size)))


    def resume(self):
        if self.config.resume_path:
            if os.path.isfile(self.config.resume_path):
                print("=> loading checkpoint '{}'".format(self.config.resume_path))
                checkpoint = torch.load(self.config.resume_path)
                self.start_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.loss_history = checkpoint['loss_history']
                self.acc_history= checkpoint['acc_history']
                self.acc_plus_minus_history = checkpoint['acc_plus_minus_history']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.config.resume_path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.config.resume_path))


    def show_imgs_and_seg(self, image_file=None, label_file=None, patch_size=32):
        self.net = self.net.eval()
        with torch.no_grad():
            if not image_file:
                for data in self.testloader:
                    images, labels = data
                    outputs = self.net(images.to(self.device))
                    predicted = torch.argmax(outputs.data, dim=1)
                    for i in range(4):
                        plt.subplot(2, 4, i)
                        self.imshow(images[i])
                        plt.subplot(2, 4, i+4)
                        plt.imshow(predicted[i])
                        plt.show()
            else:
                img = plt.imread(image_file)
                img_x = img.shape[0]
                img_y = img.shape[1]
                end_x = img_x // patch_size
                end_y = img_y // patch_size
                img = torch.from_numpy(img)
                img_patches_predictions = torch.zeros((end_x, end_y,patch_size,patch_size))
                img_predictions = np.zeros((img_x, img_y), dtype=np.uint8)
                img_predictions_image = np.zeros((img_predictions.shape[0],img_predictions.shape[1], 3), dtype=np.uint8)
                big_image = Image.open(image_file)
                big_image = np.array(big_image)
                big_image = self.prepare_for_net(big_image)
                big_image_pred = self.net(big_image)
                big_image_predicted = torch.argmax(big_image_pred, dim=1)
                big_image_predicted = big_image_predicted.cpu().numpy()
                big_image_predicted = np.squeeze(big_image_predicted, axis=0)
                for i in range(end_x):
                    for j in range(end_y):
                        patch = img[i * patch_size:(i + 1) * patch_size:, j * patch_size:(j + 1) * patch_size:, :]
                        patch = self.prepare_for_net(patch)
                        outputs = self.net(patch)
                        predicted = torch.argmax(outputs.data, dim=1)
                        img_patches_predictions[i, j, : , :] = predicted
                for i in range(end_x):
                    for j in range(end_y):
                        img_predictions[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = (img_patches_predictions[i,j]).numpy()
                labels = plt.imread(label_file)
                labels_img = np.zeros((labels.shape[0], labels.shape[1],3), dtype=np.uint8)
                labels = (((labels - np.min(labels)) / (np.max(labels) - np.min(labels))) * 14).astype(np.uint8)
                for i in range(1,16):
                    img_predictions_image[img_predictions == i,:] = self.colors[i]
                    labels_img[labels == i,:] = self.colors[i]
                plt.subplot(4,1,1)
                self.imshow(img, transpose=False)
                plt.subplot(4,1,2)
                plt.imshow(labels_img)
                plt.subplot(4,1,3)
                plt.imshow(img_predictions_image)
                plt.subplot(4,1,4)
                plt.imshow(big_image_predicted)
                plt.show()


    def evaluate_model(self, partial=False):
        self.net = self.net.eval()
        correct = 0
        total = 0
        correct_plus_minus = 0
        self.conf_matrix = np.zeros_like(self.conf_matrix)
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                # images = self.transform_image(images)
                outputs = self.net(images)
                if self.config.mode == 'segmentation':
                    total += labels.flatten().size()[0]
                    predicted = torch.argmax(outputs.data, dim=1)
                    labels = labels.long()
                    correct += (predicted == labels).sum().item()
                    labels_plus = labels + 1
                    labels_minus = labels - 1
                    correct_exact = (predicted == labels)
                    correct_plus  = (predicted == labels_plus)
                    correct_minus = (predicted == labels_minus)
                    correct_plus_minus += (correct_exact | correct_plus | correct_minus).sum().item()
                    pred_np = predicted.cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    for i in range (32):
                        for j in range (32):
                           self.conf_matrix[labels_np[0][i][j]][pred_np[0][i][j]] += 1
                    if partial and total >= 100000: #About 1000 images
                        break
                else:
                    total += labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    correct_exact = (predicted == labels)
                    correct_plus = (predicted == labels + 1).item()
                    correct_minus = (predicted == labels - 1).item()
                    correct_plus_minus += (correct_exact | correct_plus | correct_minus).sum().item()
                    self.conf_matrix[labels.item()][predicted.item()] += 1
                    # self.conf_matrix = self.conf_matrix / np.sum(self.conf_matrix)
                    if partial and total >= 10000:
                        break
        acc = 100 * correct / total
        acc_plus_minus = 100*correct_plus_minus / total
        print('Num of values compared: ', total)
        print('Accuracy on val images: %d %%' % (acc))
        print('Accuracy on label or -/+ 1 of the label: %d %%' % (acc_plus_minus))
        return acc, acc_plus_minus


    def train(self):
        acc = 1
        best_acc = 0
        # self.evaluate_model(partial=True)
        for epoch in range(self.config.start_epoch, self.config.end_epoch):  # loop over the dataset multiple times
            self.net = self.net.train()
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # inputs, labels = inputs.contiguous(), labels.contiguous()
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                # inputs = self.transform_image(inputs)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels.long())
                _, predicted = torch.max(outputs,1)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % (8000//self.config.batch_size) == (8000//self.config.batch_size)-1:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / (8000//self.config.batch_size)))
                    self.loss_history.append(running_loss / (8000//self.config.batch_size))
                    self.steps_history.append(i+1)
                    running_loss = 0.0

            prev_acc = acc
            acc, acc_plus_minus = self.evaluate_model(partial=True)
            self.acc_history.append(acc)
            self.acc_plus_minus_history.append(acc_plus_minus)
            if acc > best_acc:
                best_acc = acc
                print ("Saving model with best accuracy so far...")
                state = {
                    'epoch': epoch,
                    'state_dict': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'loss_history': self.loss_history,
                    'acc_history': self.acc_history,
                    'acc_plus_minus_history': self.acc_plus_minus_history
                }
                if not os.path.isdir(RES_OUT):
                    os.mkdir(RES_OUT)
                torch.save(state, os.path.join(RES_OUT, self.config.model_state_name))
            if (acc < prev_acc):
                current_lr = self.optimizer.defaults['lr']
                if current_lr > 1e-6:
                    new_lr = current_lr * 0.98
                    print ("Reducing learning rate from " + str(current_lr) +" to ", str(new_lr))
                    self.optimizer = optim.Adam(self.net.parameters(), lr=new_lr)
                #     #self.optimizer = optim.SGD(self.net.parameters(), lr=current_lr/10, momentum=0.9)
        print('Finished Training')

    def acc_per_class(self):
        class_correct = list(0. for i in range(self.config.num_classes))
        class_total = list(0. for i in range(self.config.num_classes))
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(self.config.num_classes):
            print('Accuracy of %5s : %2d %%' % (
                self.classes[i], 100 * class_correct[i] / class_total[i]))


    def get_class_weights(self):
        class_items = np.bincount(np.concatenate(np.concatenate(self.trainloader.dataset.train_labels)), minlength=16)
        class_weights = 1 / class_items
        norm_class_weights = class_weights * (1 / min(class_weights))
        norm_class_weights[0] = 0
        norm_class_weights[15] = 0
        return torch.from_numpy(norm_class_weights).float().to(self.device)

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    torch.cuda.empty_cache()
    # torch.backends.cudnn.enabled = False
    np.set_printoptions(linewidth=320)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = DfdConfig(image_size='small', batch_size=32, mode='segmentation', lr=0.001, get_reduced_dataset=False, num_classes=15,dataset='ours')
    net = Net(device=device, num_class=config.num_classes,mode=config.mode, channels=64)
    # net = ResNet18(mode='segmentation', num_classes=16,image_size='small')
    dfd = Dfd(config=config, net=net, device=device)
    # dfd.resume()
    dfd.train()
    dfd.plot_metrics()
    dfd.show_imgs_and_seg(image_file='/home/yotamg/data/jpg_images/City_0212_rot1.jpg', label_file='/home/yotamg/data/depth_pngs/City_0212_rot1.png', patch_size=32)