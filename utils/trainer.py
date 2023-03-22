import cv2
import os
import pandas as pd

import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from torchmetrics.classification import BinaryAccuracy
from torchmetrics import ConfusionMatrix

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score



# There are issues with using a subset, therefore subsets must be converted back into a dataset
def convert_subset_to_imagefolder(subset, root, transform=None):
    dataset = datasets.ImageFolder(root, transform)
    dataset.samples = [subset.dataset.samples[i] for i in subset.indices]
    return dataset

# This code was copied from online since Pytorch does not have a built in Early stopping system
class EarlyStopping():
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""
    
    def __call__(self, model, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
            
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
            
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.status = f"Stopped on {self.counter}"
            
            if self.restore_best_weights:
                model.load_state_dict(self.best_model.state_dict())
            return True
        
        self.status = f"{self.counter}/{self.patience}"
        return False
    
    
    
    # Custom Dataset class
class CustomDataset(Dataset):
    
    # Init has to find all paths to image and labels
    def __init__(self, root_dir, generator=None, transform=None, normalize=True, device="cuda"):
        
        self.device = device
        self.root_dir = root_dir
        self.transform = transform
        self.generator = generator
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
        if normalize else nn.Identity()
        
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
        img = Image.open(img_path)
        img = img.convert('RGB')

        # if there is a transformation, apply it
        if self.transform:
            img = self.transform(img)
        
        # If generator to upscale image
        if self.generator:
            with torch.no_grad():
                img = self.generator(img.reshape(1,3,32,32).to(self.device))
                img = img.to("cpu")
                img = img[0]
                
        img = self.normalize(img)

        return (img, label)
    
    
    
    # Creates a large trainer class for training and saving metrics of our model
class ModelTrainer:
    
    def __init__(self, model, loss, optimizer, device="cuda"):
        
        # Sets model to GPU and basic loss function and optimizer used
        self.device = device
        self.model = model.to(self.device)
        self.Loss_Function = loss
        self.optimizer = optimizer
        
        # Place to store metrics of our model throughout training and testing
        self.Metrics = {"Training Loss":[], "Validation Loss":[], 
                        "Training Accuracy":[], "Validation Accuracy":[],
                        "Test Accuracy":0} 
        
        # Place to save confidence matrix 
        self.ConfMatrix = None
    
    
    # Defines the training loop for training our model
    def Training_Loop(self, Loader):
        
        # Sets model into training mode
        self.model.train()
        
        # Sets up metric grabing and an accuracy function
        BA = BinaryAccuracy()
        tLossSum = 0
        tAccuracy = 0
        
        # Iterates through dataloader
        for images, labels in tqdm(Loader):
            
            # Moves inputs and outputs to GPU and makes the labels one-hot vectors
            images = images.to(self.device)
            labels = torch.eye(2)[labels].to(self.device)
            
            # Model makes prediction which is passed into a loss function
            pred = self.model(images)
            loss_val = self.Loss_Function(pred, labels)
            
            # Backpropagation model etc, etc...
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()
            
            # Set the predictions and labels back into integers for accuracy calculation
            pred = torch.Tensor([torch.argmax(i).item() for i in pred])
            labels = torch.Tensor([torch.argmax(i).item() for i in labels])
            
            # Running Loss and accuracy
            tLossSum += loss_val.item()
            tAccuracy += BA(pred, labels)
        
        # Update metrics based on running loss and accuracy
        self.Metrics["Training Loss"].append(tLossSum / len(Loader))
        self.Metrics["Training Accuracy"].append(tAccuracy / len(Loader))
        
        
    # Defines a function for validating our model is generalizing
    def Validation_Loop(self, Loader):
        
        # Sets model into evaluation mode
        self.model.eval()
        
        # Sets up metric grabing and an accuracy function
        BA = BinaryAccuracy()
        tLossSum = 0
        tAccuracy = 0
        
        # Iterates through dataloader
        for images, labels in Loader:
            
            # Moves inputs and outputs to GPU and makes the labels one-hot vectors
            images = images.to(self.device)
            labels = torch.eye(2)[labels].to(self.device)
            
            # No Backpropagation, use no_grad to get simple prediction and loss
            with torch.no_grad():
                pred = self.model(images)
            loss_val = self.Loss_Function(pred, labels)
            
            # Set the predictions and labels back into integers for accuracy calculation
            pred = torch.Tensor([torch.argmax(i).item() for i in pred])
            labels = torch.Tensor([torch.argmax(i).item() for i in labels])
            
            # Running Loss and accuracy
            tLossSum += loss_val.item()
            tAccuracy += BA(pred, labels)
            
        # Update metrics based on running loss and accuracy
        self.Metrics["Validation Loss"].append(tLossSum / len(Loader))
        self.Metrics["Validation Accuracy"].append(tAccuracy / len(Loader))
        
    
    # Fits model to training while also validating model 
    def fit(self, trainingLoader, validationLoader, EPOCHS):
        
        # Initate Earlystopping class to keep track of best model
        ES = EarlyStopping()
        
        for i in range(EPOCHS):
            
            # Training and Validation loop
            self.Training_Loop(trainingLoader)
            self.Validation_Loop(validationLoader)
                
            # Print epoch metrics
            print("EPOCH:", i+1)
            print("Training Loss:", self.Metrics["Training Loss"][-1], " | Validation Loss:", self.Metrics["Validation Loss"][-1])
            print("Training Accuracy:", self.Metrics["Training Accuracy"][-1].item(), " | Validation Accuracy:", self.Metrics["Validation Accuracy"][-1].item())
            
            # Check if model is overfitting and break if it is
            if ES(self.model, self.Metrics["Validation Loss"][-1]):
                print("Stopping Model Early")
                break

    
    # Evaluate model on data unseen 
    def Test_Model(self, testLoader):
        
        # Sets model into evaluation mode
        self.model.eval()
        
        # Sets up confusion matrix and accuracy
        confusion = ConfusionMatrix(task="binary", num_classes=2)
        BA = BinaryAccuracy()
        
        # A data structure for storing all labels and predictions
        predMax = torch.empty(0)
        labelMax = torch.empty(0)
    
        # Iterates through dataloader
        for images, labels in testLoader:

            # Moves inputs and outputs to GPU and makes the labels one-hot vectors
            images = images.to(self.device)
            labels = torch.eye(2)[labels].to(self.device)

            # No Backpropagation, use no_grad to get simple prediction
            with torch.no_grad():
                pred = self.model(images)

            # Set the predictions and labels back into integers for accuracy calculation
            pred = torch.Tensor([torch.argmax(i).item() for i in pred])
            labels = torch.Tensor([torch.argmax(i).item() for i in labels])

            # Concatenate labels to store and use later
            predMax = torch.cat((predMax, pred))
            labelMax = torch.cat((labelMax, labels))
        
        # Create confusion matrix and determine accuarcy 
        self.ConfMatrix = confusion(predMax, labelMax)
        self.Metrics["Test Accuracy"] = BA(predMax, labelMax).item()
        
    
    # Show representations of model metrics
    def Graph_Metrics(self):
        
        # Create subplots of a certain size and spacing
        fig, (ax11, ax2) = plt.subplots(1, 2, figsize=(11,4))
        fig.subplots_adjust(wspace=0.3)
        
        # Plot loss of both training and validation on a seperate axis
        ax12 = ax11.twinx()
        ax11.plot(self.Metrics["Training Loss"], color='b')
        ax11.plot(self.Metrics["Validation Loss"], color='c')
        ax11.set_ylabel("Loss")
        ax11.legend(["Training Loss", "Validation Loss"], bbox_to_anchor=(0.40, -0.3), loc='lower right', borderaxespad=0.5)
        
        # Plot accuracy of both training and validation on a seperate axis
        ax12.plot(self.Metrics["Training Accuracy"], color='r')
        ax12.plot(self.Metrics["Validation Accuracy"], color='m')
        ax12.set_ylabel("Percentage")
        ax12.legend(["Training Accuracy", "Validation Accuracy"], bbox_to_anchor=(1.02, -0.3), loc='lower right', borderaxespad=0.5)

        ax11.set_title("Model Metrics Across Epochs")

        ax2.imshow(self.ConfMatrix, cmap='Blues')
        
        # Add total number of predictions for each box
        for i in range(self.ConfMatrix.shape[0]):
            for j in range(self.ConfMatrix.shape[1]):
                ax2.text(j, i, self.ConfMatrix[i, j].item(), ha='center', va='center', color='black')

        # Removes y labels for confusion matrix
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax2.set_xlabel('Predicted labels')
        ax2.set_ylabel('True labels')
        ax2.set_title("Model Confusion Matrix for Test")
       