import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

N_PARAMS = {'projective': 5,
            'affine': 6,
            'translation': 2,
            'rotation': 1,
            'center_rotation': 1,
            'scale': 2,
            'shear': 2,
            'rotation_scale': 3,
            'translation_scale': 4,
            'rotation_translation': 3,
            'rotation_translation_scale': 5}

def projective_stn(x, theta):
    shape = x.size()
    orig_img = x.clone() #TODO - Remove
    tx = torch.linspace(-1,1,shape[2]).unsqueeze(0).repeat(shape[2],1)
    ty = torch.linspace(-1,1,shape[3]).unsqueeze(1).repeat(1,shape[3])
    theta1 = Variable(torch.zeros([x.size(0), 3, 3], dtype=torch.float32, device=x.get_device()), requires_grad=True)
    theta1 = theta1 + 0
    theta1[:,2,2] = 1.0
    angle = theta[:, 0]
    # angle = torch.Tensor([0])
    theta1[:, 0, 0] = torch.cos(angle)
    theta1[:, 0, 1] = -torch.sin(angle)
    theta1[:, 1, 0] = torch.sin(angle)
    theta1[:, 1, 1] = torch.cos(angle)
    theta1[:, 2, 0] = theta[:, 1] #x_perspective
    # theta1[:, 2, 0] = 0  # x_perspective
    theta1[:, 2, 1] = theta[:, 2] #y_perspective
    theta1[:, 0, 2] = theta[:, 3] #x_translation
    # theta1[:, 0, 2] = 0 #x_translation
    theta1[:, 1, 2] = theta[:, 4] #y_translation
    # theta1[:, 1, 2] = 0 #y_translation
    grid = Variable(torch.zeros((1,shape[2], shape[3], 3), dtype=torch.float32, device=x.get_device()), requires_grad=False)
    grid[0,:,:,0] = tx
    grid[0,:,:,1] = ty
    grid[0,:,:,2] = torch.ones(shape[2], shape[3])
    # theta1 = theta1.reshape((3,3))
    grid = torch.mm(grid.reshape(-1,3), theta1[0].t()).reshape(1, shape[2], shape[3], 3)
    grid[0, :, :, 0] = grid[0, :, :, 0].clone() / grid[0, :, :, 2].clone()
    grid[0, :, :, 1] = grid[0, :, :, 1].clone() / grid[0, :, :, 2].clone()
    # grid = grid
    #This is due to cudnn bug
    try:
        x = F.grid_sample(x, grid[:,:,:,:2])
    except:
        x = F.grid_sample(x, grid[:,:,:,:2])
    # import matplotlib.pyplot as plt
    # print(theta1)
    # plt.subplot(121)
    # plt.imshow(orig_img[0].permute(1,2,0).detach().cpu())
    # plt.subplot(122)
    # plt.imshow(x[0].permute(1,2,0).detach().cpu())
    # plt.show()
    return x, theta1


# Spatial transformer network forward function
def stn(x, theta, mode='affine'):
    if mode == 'affine':
        theta1 = theta.view(-1, 2, 3)
    else:
        theta1 = Variable(torch.zeros([x.size(0), 2, 3], dtype=torch.float32, device=x.get_device()),
                          requires_grad=True)
        theta1 = theta1 + 0
        theta1[:, 0, 0] = 1.0
        theta1[:, 1, 1] = 1.0
        if mode == 'translation':
            theta1[:, 0, 2] = theta[:, 0]
            theta1[:, 1, 2] = theta[:, 1]
        elif mode == 'rotation':
            angle = theta[:, 0]
            theta1[:, 0, 0] = torch.cos(angle)
            theta1[:, 0, 1] = -torch.sin(angle)
            theta1[:, 1, 0] = torch.sin(angle)
            theta1[:, 1, 1] = torch.cos(angle)
        elif mode == 'center_rotation':
            angle = theta[:, 0]
            theta1[:, 0, 0] = torch.cos(angle)
            theta1[:, 0, 1] = -torch.sin(angle)
            theta1[:, 1, 0] = torch.sin(angle)
            theta1[:, 1, 1] = torch.cos(angle)
            theta1[:, 0, 2] = -0.5 * torch.cos(angle) + 0.5 * torch.sin(angle) + 0.5
            theta1[:, 1, 2] = -0.5 * torch.sin(angle) - 0.5 * torch.cos(angle) + 0.5
        elif mode == 'scale':
            theta1[:, 0, 0] = theta[:, 0]
            theta1[:, 1, 1] = theta[:, 1]
        elif mode == 'shear':
            theta1[:, 0, 1] = theta[:, 0]
            theta1[:, 1, 0] = theta[:, 1]
        elif mode == 'rotation_scale':
            angle = theta[:, 0]
            theta1[:, 0, 0] = torch.cos(angle) * theta[:, 1]
            theta1[:, 0, 1] = -torch.sin(angle)
            theta1[:, 1, 0] = torch.sin(angle)
            theta1[:, 1, 1] = torch.cos(angle) * theta[:, 2]
        elif mode == 'translation_scale':
            theta1[:, 0, 2] = theta[:, 0]
            theta1[:, 1, 2] = theta[:, 1]
            theta1[:, 0, 0] = theta[:, 2]
            theta1[:, 1, 1] = theta[:, 3]
        elif mode == 'rotation_translation':
            angle = theta[:, 0]
            theta1[:, 0, 0] = torch.cos(angle)
            theta1[:, 0, 1] = -torch.sin(angle)
            theta1[:, 1, 0] = torch.sin(angle)
            theta1[:, 1, 1] = torch.cos(angle)
            theta1[:, 0, 2] = theta[:, 1]
            theta1[:, 0, 2] = theta[:, 1]
            theta1[:, 1, 2] = theta[:, 2]
        elif mode == 'rotation_translation_scale':
            angle = theta[:, 0]
            theta1[:, 0, 0] = torch.cos(angle) * theta[:, 3]
            theta1[:, 0, 1] = -torch.sin(angle)
            theta1[:, 1, 0] = torch.sin(angle)
            theta1[:, 1, 1] = torch.cos(angle) * theta[:, 4]
            theta1[:, 0, 2] = theta[:, 1]
            theta1[:, 1, 2] = theta[:, 2]
    grid = F.affine_grid(theta1, x.size())
    x = F.grid_sample(x, grid)
    return x, theta1


class ConfigNet(nn.Module):
    def __init__(self, stereo_model, stn_mode='projective'):
        super(ConfigNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.stn_mode = stn_mode
        self.stn_n_params = N_PARAMS[stn_mode]
        self.stereo_model = stereo_model

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(153760, 36),
            nn.Linear(36, 32),
            nn.ReLU(True),
            nn.Linear(32, self.stn_n_params)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[3].weight.data.fill_(0)
        self.fc_loc[3].weight.data.zero_()
        if self.stn_mode == 'projective':
            self.fc_loc[3].bias.data.copy_(torch.tensor([0, 0.1, 0.1, 0, 0], dtype=torch.float))
        elif self.stn_mode == 'affine':
            self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif self.stn_mode in ['translation', 'shear']:
            self.fc_loc[3].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        elif self.stn_mode == 'scale':
            self.fc_loc[3].bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation':
            self.fc_loc[3].bias.data.copy_(torch.tensor([0], dtype=torch.float))
        elif self.stn_mode == 'center_rotation':
            self.fc_loc[3].bias.data.copy_(torch.tensor([0], dtype=torch.float))
        elif self.stn_mode == 'rotation_scale':
            self.fc_loc[3].bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
        elif self.stn_mode == 'translation_scale':
            self.fc_loc[3].bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation':
            self.fc_loc[3].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
        elif self.stn_mode == 'rotation_translation_scale':
            self.fc_loc[3].bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))

    def stn(self, x):
        x,theta1 = projective_stn(x, self.theta(x))
        return x, theta1

    def theta(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 153760)
        theta = self.fc_loc(xs)
        # theta = torch.unsqueeze(theta,0)
        return theta

    def forward(self, left_img, right_img):
        # transform the input
        right_img_transformed, theta = self.stn(right_img)
        # stereo_out = self.stereo_model(left_img, right_img_transformed)
        return theta, right_img_transformed

