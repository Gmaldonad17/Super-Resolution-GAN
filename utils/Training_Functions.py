import os
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import pyplot as plt


class SRGAN_Dataset(Dataset):
    
    # Init has to find all paths to image and labels
    def __init__(self, root_dir, low_res=(32,32), transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.low_res = low_res
        
        # Classifications of the objects
        self.definitions = os.listdir(self.root_dir)
        
        data = []
        # Go through each defination and add each image path to the csv
        for label in self.definitions:
            label_path = self.root_dir + label + "/"
            
            for image in os.listdir(label_path):
                
                # Check if file is of an image type
                if image.split('.')[-1] not in ["png", "jpg", "jpeg"]:
                    continue
                
                # If it is, then append the path to the image as well as the label given by the folder
                data.append({
                    'path': label_path + image,
                    'label': self.definitions.index(label)
                })
        
        # Save to a dataframe
        self.csv = pd.DataFrame(data)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        
        # Extract the path and label of a datapoint
        img_path = self.csv.iloc[idx]['path']
        label = self.csv.iloc[idx]['label']

        # Open the image and ensure 3 channels
        img_high = Image.open(img_path)
        img_high = img_high.convert('RGB')

        if self.transform:
            img_high = self.transform(img_high)
            
        img_low = transforms.Resize(size=self.low_res)(img_high)

        return (img_low, img_high, label)
    
    
def visualize_images(lr_image, hr_image, fake_image, label, epoch, pre_train, show=True):
    
    lr_image = np.transpose(lr_image.cpu().numpy(), (1, 2, 0))
    hr_image = np.transpose(hr_image.cpu().numpy(), (1, 2, 0))
    fake_image = np.transpose(fake_image.cpu().numpy(), (1, 2, 0))
    
    p = "p" if pre_train else ""
    cv2.imwrite(f"progress/fk/{p}{epoch}.png", fake_image*255)
    
    
    if not os.path.exists(f"progress/hr/0.png"):
        cv2.imwrite(f"progress/hr/0.png", hr_image*255)
        cv2.imwrite(f"progress/lr/0.png", lr_image*255)
        

    if show:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs = axs.ravel()
        
        axs[0].imshow(lr_image, cmap='gray')
        axs[0].set_title('Low Resolution')

        axs[1].imshow(hr_image, cmap='gray')
        axs[1].set_title('High Resolution')

        axs[2].imshow(fake_image, cmap='gray')
        axs[2].set_title('Generated Image')

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(definitions[label])
        plt.show()


def Train_SRGAN(Generator, Discriminator, Generator_Loss, Discriminator_Loss, TrainLoader, TestLoader, lr, EPOCHs, e_start=0, device="cpu"):
    
    Generator.train()
    Discriminator.train()
    
    Gen_optimizer = torch.optim.Adam(Generator.parameters(), lr=lr)
    Dis_optimizer = torch.optim.Adam(Discriminator.parameters(), lr=lr*0.1)
    
    Metrics = {"Generator Loss": [0], "Discriminator Loss": [0]}
    
    for epoch in range(EPOCHs):
        
        genLossSum = 0
        disLossSum = 0

        # Iterates through dataloader
        for i, (low_res, high_res, labels) in enumerate(tqdm(TrainLoader)):
            
            # Moves inputs and outputs to GPU and makes the labels one-hot vectors
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            
            # Model makes prediction which is passed into a loss function
            fake_img = Generator(low_res)
            
            if not Generator_Loss.pre_train:
                fake_result = Discriminator(fake_img)
                real_result = Discriminator(high_res)

                # Update Discriminator
                Dis_loss = Discriminator_Loss(real_result, fake_result)
                Dis_optimizer.zero_grad()
                Dis_loss.backward(retain_graph=True)
                Dis_optimizer.step()

            # Update Generator
            gen_loss_updators = [fake_img, high_res]
            if not Generator_Loss.pre_train:
                gen_loss_updators.append(Dis_loss.detach())
            
            Gen_loss = Generator_Loss(*gen_loss_updators)
            Gen_optimizer.zero_grad()
            Gen_loss.backward()
            Gen_optimizer.step()
            
            # Running Loss and accuracy
            genLossSum += Gen_loss.item()
            
            if not Generator_Loss.pre_train:
                disLossSum += Dis_loss.item()

            
        # Update metrics based on running loss and accuracy
        Metrics["Generator Loss"].append(genLossSum / len(TrainLoader))
        if not Generator_Loss.pre_train:
            Metrics["Discriminator Loss"].append(disLossSum / len(TrainLoader))
        
        print(f"Epoch {epoch+1+e_start} - Generator Loss: {Metrics['Generator Loss'][-1]}, Discriminator Loss: {Metrics['Discriminator Loss'][-1]}")
        
        if (epoch + 1 + e_start) % 1 == 0:
            
            Generator.eval()
            for i, (low_res, high_res, labels) in enumerate(TestLoader):
                with torch.no_grad():
                    sample_lr = low_res[0].unsqueeze(0)
                    sample_hr = high_res[0].unsqueeze(0)
                    sample_fake = Generator(sample_lr.to(device))
                    
                    show = False
                    if (epoch + 1 + e_start) % 20 == 0:
                        show = True
                    
                    visualize_images(sample_lr[0], sample_hr[0], sample_fake[0], labels[0].item(), epoch+e_start, Generator_Loss.pre_train, show)
                break
                
            Generator.train()
            
        if (epoch + 1 + e_start) % 10 == 0:
            
            torch.save(Generator.state_dict(), f'generator_model_{epoch}.pth')
            torch.save(Discriminator.state_dict(), f'discriminator_{epoch}_model.pth')
