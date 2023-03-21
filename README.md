# Super-Resolution-GAN-for-Medical-Imaging
The purpose of this project is to both train and understand the uses of SRGANs for data manipulation and adding detail. The SRGAN in this repository was created adhering to the original SRGAN paper [ADD LINK] using the same models and training protocol. The data used was the Retinal OCT dataset from kaggle which is for a binary classification of DME or DRUSEN conditions. SRGAN were used to increase the resolution from 32x32 to 128x128.

![Progression of Training, Clear distiction between pre and post training](figures/0-149.gif)

# Installation 

Clone the repsistory into {ROOT}

`conda env create -f environment.yml`
This will install all required prerequisite for running the main body of the program
Next install pytorch, torchvision with your correct version of cuda or cpu


# Data preparation
The directory tree should look like this:
```
{ROOT}
|-- datasets
    |-- Retinal_OCT
    |   |   |-- test
    |   |   |   | -- DME
    |   |   |   | -- DRUSEN
    |   |   |-- train
    |   |   |   | -- DME
    |   |   |   | -- DRUSEN
|-- models
    |-- BC_A.pth
    |-- BC_B.pth
    |-- generator_model.pth
    |-- discriminator_model.pth
|-- progress
    |-- fk
    |-- hr
    |-- lr
```

# Results
This section will show the results of training on the different models of the model

## Model Trained on full Res infomation 128x128 Model A
![Model_A](figures/BC_A_training.png)

## Model Trained on low Res infomation 32x32 Model A_32
![Model_A_32](figures/BC_A_32_training.png)

## Model Trained on generated infomation 128x128 Model B
![Model_B](figures/BC_B_training.png)

As we can see from the following trainings the model trained on the original infomation at the highest resolution of 128 preformed the best at 99.0% accuracy. The same model on original data scaled down to 32 preformed the worst of all the models at 85.8% (Model A_32) accuracy showing the additional infomation from the generator provides a better ability to generalize on training infomation. The model which worked with generated infomation preformed at almost 10% higher than the A_32 Model at an accuracy of 94.8%.
