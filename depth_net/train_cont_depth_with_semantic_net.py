import torch
import torchvision
import torchvision.transforms as transforms
from depth_classification_dataset import depth_classification_dataset
from depth_segmentation_dataset import depth_segmentation_dataset
from nyu_depth_segmentation_dataset import nyu_depth_segmentation_dataset
from cont_depth_segmentation_dataset import cont_depth_segmentation_dataset
from resnet import ResNet18, ResNet50
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import pickle
# from cont_seg_net import Net
from cont_seg_with_semantic_net import Net





RES_OUT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources_out')

def load_data(mode, image_size, batch_size,dataset=None,target_mode='discrete'):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if mode == 'segmentation':
        if target_mode == 'cont':
            train_dir = '/home/yotamg/data/raw_rgb_images/'
            test_dir = '/home/yotamg/data/raw_rgb_images/test/'
            label_dir = '/home/yotamg/data/depth_maps_cont/'
        elif image_size == 'small':
            train_dir = '/home/yotamg/data/jpg_images/patches'
            test_dir = '/home/yotamg/data/jpg_images/test/patches'
            label_dir = '/home/yotamg/data/depth_pngs/patches'
        else:
            train_dir = '/home/yotamg/data/tmp_raw/train'
            test_dir = '/home/yotamg/data/tmp_raw/test/'
            label_dir = '/home/yotamg/data/depth_maps/'
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
        if mode == 'segmentation' and target_mode == 'discrete':
            trainset = depth_segmentation_dataset(root='./data', train=True, transform=transform, train_dir=train_dir,
                                                  label_dir=label_dir, load_pickle=False, add_noise=False)
            testset = depth_segmentation_dataset(root='./data', train=False, transform=transform, test_dir=test_dir,
                                                 label_dir=label_dir, load_pickle=False, add_noise=False)
        elif mode == 'segmentation' and target_mode == 'cont':

            trainset = cont_depth_segmentation_dataset(root='./data', train=True, transform=transform, train_dir=train_dir,
                                                  label_dir=label_dir, load_pickle=True, add_noise=False)
            testset = cont_depth_segmentation_dataset(root='./data', train=False, transform=transform, test_dir=test_dir,
                                                 label_dir=label_dir, load_pickle=True, add_noise=False)
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
    def __init__(self, batch_size=64, mode='segmentation', target_mode='discrete', image_size='small', resume_path=os.path.join(RES_OUT,'best_model.pt'), start_epoch=0, end_epoch=50, model_state_name='best_model.pt', num_classes=15, lr=0.01,get_reduced_dataset=False,dataset=None, skip_layer=True):
        self.batch_size = batch_size
        self.mode = mode
        self.target_mode = target_mode
        self.image_size = image_size
        self.resume_path = resume_path
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.model_state_name = model_state_name
        self.num_classes = num_classes
        self.lr = lr
        self.get_reduced_dataset = get_reduced_dataset
        self.dataset = dataset
        self.skip_layer = skip_layer

class Dfd(object):
    def __init__(self, net=None, config=DfdConfig(), device=None):
        self.config = config
        self.trainloader, self.testloader = load_data(config.mode,config.image_size, config.batch_size, dataset=config.dataset, target_mode=config.target_mode)
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = net if net else ResNet18(mode=config.mode, image_size=config.image_size, num_classes=self.config.num_classes)
        self.net = (self.net).to(self.device)
        if config.mode == 'segmentation':
            if config.target_mode == 'discrete':
                # self.criterion = nn.CrossEntropyLoss(weight=self.get_class_weights())
                self.criterion = nn.CrossEntropyLoss(ignore_index=0, weight=self.get_class_weights())
            elif config.target_mode == 'cont':
                self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            #self.criterion = nn.MSELoss(reduction='elementwise_mean')
            #self.criterion = nn.CrossEntropyLoss(reduction='elementwise_mean')
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
        return x

    # functions to show an image
    def prepare_for_net(self, np_arr):
        np_arr = np.expand_dims(np_arr, 0)
        np_arr = np.transpose(np_arr, (0, 3, 1, 2))
        np_arr = torch.from_numpy(np_arr)
        np_arr = np_arr.type(torch.FloatTensor)
        np_arr = np_arr.to(self.device)
        np_arr = ((np_arr / 255) * 2) - 1
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
        plt.show()

    def resume(self, resume_path=None, ordinal=False):
        if not resume_path:
            resume_path = self.config.resume_path
        if resume_path:
            if os.path.isfile(resume_path):
                print("=> loading checkpoint '{}'".format(resume_path))
                checkpoint = torch.load(resume_path)
                self.start_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'], strict=False)
                self.net.state_dict().update(checkpoint['state_dict'])
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.loss_history = checkpoint['loss_history']
                self.acc_history= checkpoint['acc_history']
                self.acc_plus_minus_history = checkpoint['acc_plus_minus_history']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume_path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(resume_path))


    def show_imgs_and_seg(self, image_file=None, label_file=None, patch_size=32, load_pickle=True):
        self.net = self.net.eval()
        self.reg_conv = nn.Conv2d(16,1,1)
        self.reg_conv.weight[0,:,0,0] = torch.arange(16)
        self.reg_conv.to(self.device)
        with torch.no_grad():
            if not image_file:
                for i, data in enumerate(self.testloader):
                    images, labels = data
                    outputs = self.net(images.to(self.device))
                    if config.target_mode == 'discrete':
                        predicted,_ = torch.argmax(outputs.data, dim=1)
                    elif config.target_mode == 'cont':
                        predicted = outputs
                    #     predicted = torch.round(outputs)
                    # predicted_reg = self.reg_conv(outputs.data)
                    # predicted_reg = predicted_reg[0,0,:,:].cpu().numpy()
                    if i == 4:
                        break
                    plt.subplot(4, 4, i*4+1)
                    self.imshow(images[0])
                    plt.subplot(4, 4, i*4+2)
                    plt.imshow(labels[0],cmap='jet')
                    plt.subplot(4, 4, i*4+3)
                    plt.imshow(predicted,cmap='jet')
                    # plt.subplot(4, 4, i*4+4)
                    # plt.imshow(predicted_reg,cmap='jet')
                plt.show()
            else:
                if load_pickle:
                    with open(image_file, 'rb') as f:
                        big_image = pickle.load(f)
                        orig_image = Image.fromarray(big_image)
                else:
                    big_image = Image.open(image_file)
                    orig_image = big_image
                big_image = np.array(big_image)
                big_image = self.prepare_for_net(big_image)
                start_time = time.time()
                big_image_pred = self.net(big_image)
                print ("Inference time: ", time.time() - start_time)
                # predicted_reg = self.reg_conv(big_image_pred)
                # predicted_reg = predicted_reg[0, 0, :, :].cpu().numpy()
                big_image_predicted = big_image_pred
                if config.target_mode != 'cont':
                    big_image_predicted = torch.argmax(big_image_pred, dim=1)
                    big_image_predicted = big_image_predicted.cpu().numpy()
                    big_image_predicted = np.squeeze(big_image_predicted, axis=0)
                plt.subplot(2, 1, 1)
                plt.title("Big image predictions discrete")
                plt.imshow(big_image_pred_discrete, cmap='jet')
                # plt.show()
                plt.subplot(2,1,2)
                plt.title("Big image predictions")
                plt.imshow(big_image_predicted, cmap='jet')
                plt.show()

    def evaluate_model(self, partial=False, net=None):
        if not net:
            net = self.net;
        net = net.eval()
        correct = 0
        total = 0
        loss = 0.0
        correct_plus_minus = 0
        self.conf_matrix = np.zeros_like(self.conf_matrix)
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                # images = self.transform_image(images)
                outputs = net(images)
                if self.config.mode == 'segmentation':
                    total += labels.flatten().size()[0]
                    if self.config.target_mode == 'discrete':
                        predicted = torch.argmax(outputs.data, dim=1)
                    else: #Cont
                        predicted = torch.round(outputs.data)
                        predicted = predicted.long()
                        predicted = torch.unsqueeze(predicted,dim=0)
                    labels = labels.long()
                    loss += self.criterion(outputs, labels)
                    correct += (predicted == labels).sum().item()
                    labels_plus = labels + 1
                    labels_minus = labels - 1
                    correct_exact = (predicted == labels)
                    correct_plus  = (predicted == labels_plus)
                    correct_minus = (predicted == labels_minus)
                    correct_plus_minus += (correct_exact | correct_plus | correct_minus).sum().item()
                    pred_np = predicted.cpu().numpy()
                    pred_np = np.minimum(np.maximum(pred_np, 0),15)
                    labels_np = labels.cpu().numpy()
                    for i in range (pred_np.shape[1]):
                        for j in range (pred_np.shape[2]):
                            self.conf_matrix[labels_np[0][i][j]][pred_np[0][i][j]] += 1
                    if partial and total >= max(100000, pred_np.shape[1] * pred_np.shape[2] * 10): #About 1000 small images
                        break
                elif self.config.mode == 'segmentation' and self.config.target_mode == 'cont':
                    total += labels.flatten().size()[0]
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
        print('Validation loss: %f ' % (loss / 80))
        return acc, acc_plus_minus


    def train(self):
        acc = 1
        best_acc = 0
        self.evaluate_model(partial=True)
        for epoch in range(self.config.start_epoch, self.config.end_epoch):  # loop over the dataset multiple times
            self.net = self.net.train()
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                # inputs = self.transform_image(inputs)

                outputs = self.net(inputs)
                if self.config.mode == 'segmentation' and self.config.target_mode == 'cont':
                    labels = labels.float()
                    labels = labels.squeeze()
                else:
                    labels = labels.long()
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % (80//self.config.batch_size) == (80//self.config.batch_size)-1:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / (80//self.config.batch_size)))
                    self.loss_history.append(running_loss / (80//self.config.batch_size))
                    self.steps_history.append(i+1)
                    running_loss = 0.0
            if epoch % 5 == 4:
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
                    print (self.conf_matrix)
                if (acc < prev_acc):
                    current_lr = self.optimizer.defaults['lr']
                    if current_lr > 1e-6:
                        new_lr = current_lr * 0.98
                        print ("Reducing learning rate from " + str(current_lr) +" to ", str(new_lr))
                        self.optimizer = optim.Adam(self.net.parameters(), lr=new_lr)
                    #     #self.optimizer = optim.SGD(self.net.parameters(), lr=current_lr/10, momentum=0.9)
        print('Finished Training')


    def get_class_weights(self):
        class_items = np.bincount(np.concatenate(np.concatenate(self.trainloader.dataset.train_labels)).astype(np.uint8), minlength=16)
        class_weights = 1 / (class_items + 1)
        norm_class_weights = class_weights * (1 / min(class_weights))
        norm_class_weights[0] = 0
        # norm_class_weights[15] = 0
        return torch.from_numpy(norm_class_weights).float().to(self.device)

    def show_imgs_cont_and_discrete(self, image, labels):
        plt.subplot(3,1,1)
        plt.imshow(labels[0,:,:])
        plt.subplot(3,1,2)
        plt.imshow(net(image).detach())

if __name__ == '__main__':
    torch.cuda.empty_cache()
    np.set_printoptions(linewidth=320)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = DfdConfig(image_size='big', batch_size=4, mode='segmentation', lr=0.01,target_mode='discrete', get_reduced_dataset=False, num_classes=16,dataset='ours', end_epoch=300)
    net = Net(device=device, num_class=config.num_classes,mode=config.mode, channels=64, config=config)
    dfd = Dfd(config=config, net=net, device=device)
    # dfd.resume(resume_path='/home/yotamg/PycharmProjects/dfd/resources_out/discrete_segmentation_80t1_97pm1.pt')
    # dfd.resume(resume_path='/home/yotamg/PycharmProjects/dfd/resources_out/best_bu.pt')
    # net = ResNet18(mode='segmentation', num_classes=16,image_size='small')
    # config = DfdConfig(image_size='big', batch_size=16, mode='segmentation', target_mode='cont',lr=0.0001, get_reduced_dataset=False, num_classes=16,dataset='ours', end_epoch=300)



    # with open('/home/yotamg/data/raw_rgb_images/City_0062_rot2.raw', 'rb') as f:
    #     img = pickle.load(f)
    #
    # feed_dict = dict()
    # img = dfd.prepare_for_net(img)
    # # img = torch.Tensor(img)
    # feed_dict['img_data'] = img
    # segmentation_module.to(device)


    # dfd.resume(resume_path=os.path.join(RES_OUT,'cont_segmentation_96pm1.pt'))
    # for param in dfd.net.segmentation_module.parameters():
    #     param.requires_grad = False
    dfd.train()
    dfd.plot_metrics()
