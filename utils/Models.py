import torch
from torch import nn
import torchvision.models as models


class Residual(nn.Module):
    def __init__(self, num_channels, stride=1):
        super(Residual, self).__init__()
        self.num_channels = num_channels
        
        self.net = nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=stride),
            nn.LazyBatchNorm2d(),
            nn.PReLU(),
            
            
            nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=stride),
            nn.LazyBatchNorm2d()
        )
        
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=stride)
        
    def forward(self, x):
        
        out = self.net(x)
        
        if self.num_channels != x.shape[1]:
            x = self.conv3(x)
        else:
            self.conv3 = None
            
        out += x 
        
        return out
    
    
class SR_Generator(nn.Module):
    def __init__(self, body_arch):
        super(SR_Generator, self).__init__()

        # The first convolutional layer (c1) with a 9x9 kernel, 64 filters, and PReLU activation
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )

        # The main body of the generator containing residual blocks (body_arch defines the structure)
        self.body = nn.Sequential()
        for i, b in enumerate(body_arch):
            self.body.add_module(f'b{i+2}', self.res_block(*b))
        

        # The second convolutional layer (c2) with a 3x3 kernel, 64 filters, and Batch Normalization
        self.c2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
        )

        # The third convolutional block (c3) containing two upsampling layers and the final output layer
        self.c3 = nn.Sequential(
            # First upsampling layer: 3x3 kernel, 256 filters, PixelShuffle, and PReLU activation
            nn.Conv2d(64, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.PixelShuffle(2),
            nn.PReLU(),

            # Second upsampling layer: 3x3 kernel, 256 filters, PixelShuffle, and PReLU activation
            nn.Conv2d(64, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.PixelShuffle(2),
            nn.PReLU(),

            # Final output layer: 9x9 kernel, 3 filters (RGB channels), and no activation
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
        )

    def forward(self, x):
        # Pass input through the first convolutional layer (c1)
        out = self.c1(x)
        
        # Pass the output through the residual blocks (body)
        x = self.body(out)

        # Pass the output through the second convolutional layer (c2)
        x = self.c2(x)

        # Add the skip connection from the output of c1
        x += out

        # Pass the output through the third convolutional block (c3) to generate the final image
        x = self.c3(x)

        return x

    # Function to create a residual block with a specified number of residuals and channels
    def res_block(self, num_residuals, num_channels):
        blk = []
        for i in range(num_residuals):
            blk.append(Residual(num_channels))
        return nn.Sequential(*blk)

    
class PretrainGeneratorLoss(nn.Module):
    def __init__(self):
        super(PretrainGeneratorLoss, self).__init__()
        self.pre_train = True
        self.mse_loss = nn.MSELoss()

    def forward(self, sr, hr):
        # Calculate the MSE loss between the super-resolved image and the high-resolution image
        content_loss = self.mse_loss(sr, hr)
        return content_loss
    
class GeneratorLoss(nn.Module):
    def __init__(self, content_weight=1e-3, adversarial_weight=1):
        super(GeneratorLoss, self).__init__()
        self.pre_train = False
        self.content_weight = content_weight
        self.adversarial_weight = adversarial_weight
        
        self.vgg19 = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1')
        self.vgg19 = nn.Sequential(*list(self.vgg19.features.children())[:35])
        self.vgg19 = self.vgg19.cuda() if torch.cuda.is_available() else self.vgg19
        self.vgg19.eval()
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, sr, hr, discriminator_pred):
        # Calculate the content loss
        hr_features = self.vgg19(hr)
        sr_features = self.vgg19(sr)
        content_loss = self.mse_loss(sr_features, hr_features.detach())

        # Calculate the adversarial loss
        real_label = torch.ones(discriminator_pred.size())
        real_label = real_label.cuda() if torch.cuda.is_available() else real_label
        adversarial_loss = self.bce_loss(discriminator_pred, real_label)

        # Total generator loss
        total_loss = self.content_weight * content_loss + self.adversarial_weight * adversarial_loss
        return total_loss
    
    
class SR_Discriminator(nn.Module):
    def __init__(self, body_arch):
        super(SR_Discriminator, self).__init__()

        self.body = nn.Sequential()
        for i, b in enumerate(body_arch):
            self.body.add_module(f'b{i+2}', self.dis_block(*b, i))

        # Add the fully connected part before the final nn.Linear layer
        self.fc = nn.Sequential(
            nn.LazyLinear(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        # Pass the output through the discriminator blocks (body)
        x = self.body(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Pass the output through the fully connected part
        x = self.fc(x)

        return x

    def dis_block(self, kernel_size, num_channels, stride, use_bn=True):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=kernel_size, stride=stride),
            nn.LazyBatchNorm2d() if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2),
        )
    
    
class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, real_pred, fake_pred):
        real_label = torch.ones(real_pred.size())
        fake_label = torch.zeros(fake_pred.size())

        if torch.cuda.is_available():
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()

        # Loss for real high-resolution images
        real_loss = self.bce_loss(real_pred, real_label)

        # Loss for generated high-resolution images
        fake_loss = self.bce_loss(fake_pred, fake_label)

        # Total discriminator loss
        total_loss = (real_loss + fake_loss) * 0.5
        return total_loss