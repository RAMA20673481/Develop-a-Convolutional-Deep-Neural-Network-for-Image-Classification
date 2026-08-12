# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Image classification is an important application of Deep Learning and Computer Vision. Manually identifying different types of clothing from images can be time-consuming and may lead to classification errors. Therefore, there is a need for an automated system that can accurately classify clothing images into their respective categories.

In this project, a Convolutional Neural Network (CNN) is developed using PyTorch to classify images from the Fashion-MNIST dataset. The dataset contains 28×28 grayscale images belonging to 10 different clothing categories such as T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

The CNN learns important visual features from the training images using convolutional and pooling layers and then classifies the images using fully connected layers. The trained model is evaluated using classification accuracy, confusion matrix, precision, recall, and F1-score. Finally, the model is used to predict the class of new unseen images.


The **Fashion-MNIST dataset** is used for training and testing the Convolutional Neural Network (CNN). It contains grayscale images of clothing and footwear belonging to 10 different categories.

| Property | Details |
|---|---|
| Dataset | Fashion-MNIST |
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Size | 28 × 28 pixels |
| Image Type | Grayscale |
| Number of Channels | 1 |
| Number of Classes | 10 |
| Task | Multi-class Image Classification |

### Classes

| Label | Category |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Neural Network Model

<img width="1024" height="692" alt="Screenshot 2026-08-12 173614" src="https://github.com/user-attachments/assets/4d3857f6-73db-4dab-b65a-dac16da45790" />


## DESIGN STEPS
### STEP 1: 
Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

### STEP 2: 
Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

### STEP 3: 
Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

### STEP 4: 
Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.

### STEP 5: 
Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

### STEP 6: 
Evaluate the trained model on test images and verify the classification accuracy for new unseen images.




## PROGRAM

### Name: G.Ramanujam

### Register Number: 212224240129

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
```

```python
transform=transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5,),(0.5))  
])
```

```python
train_set=torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,
transform=transform)
test_set=torchvision.datasets.FashionMNIST(root='./data',train=False,download=True,
transform=transform)
```

```python
im,lbl=train_set[0]
print(im.shape)
print(len(train_set))
print(len(test_set))
```

```python
trl=DataLoader(train_set,batch_size=32,shuffle=True)
tstl=DataLoader(test_set,batch_size=32,shuffle=False)
```

```python
class CNNclassifier1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.c2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.c3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.l1=nn.Linear(128*3*3,64)
        self.l2=nn.Linear(64,32)
        self.l3=nn.Linear(32,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.c1(x)))
        x=self.pool(torch.relu(self.c2(x)))
        x=self.pool(torch.relu(self.c3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.l1(x))
        x=torch.relu(self.l2(x))
        x=self.l3(x)
        return x
```

```python
from torchsummary import summary

model=CNNclassifier1()
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)
if torch.cuda.is_available():
    print(torch.cuda.is_available())
    device=torch.device('cuda')
    model.to(device)


summary(model,input_size=(1,28,28))
```

```python
epochs=3
running_loss=0.0

for i in range(epochs):
    model.train()
    for a,b in trl:
        op.zero_grad()
        pred=model(a)
        loss=criterion(pred,b)
        loss.backward()
        op.step()
        running_loss+=loss.item()
    print(f"Loss:{i}",running_loss/len(trl))
```

```python
t=0
c=0
act=[]
pre=[]
model.eval()
with torch.no_grad():
    for img,labels in tstl:
        output=model(img)
        _,predicted=torch.max(output,1)
        t=t+labels.size(0)
        c+=(predicted==labels).sum().item()
        pre.extend(predicted.cpu().numpy())
        act.extend(labels.cpu().numpy())
accuracy=c/t*100
print("Accuracy Score:",accuracy)
conf_matrix=confusion_matrix(act,pre)
class_report=classification_report(act,pre,target_names=test_set.classes)
print("Classification Report:",class_report)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues',xticklabels=test_set.classes,yticklabels=test_set.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

```python
with torch.no_grad():
    img1,label=test_set[0]
    output=model(img1.unsqueeze(0))
    _,pred=torch.max(output,1)
    classes=test_set.classes
    img1 = img1 * 0.5 + 0.5
    plt.imshow(img1.squeeze(),cmap="gray")
    plt.title("Predicted Image")
    plt.axis('off')
    plt.show()
    print(f"Actual:{classes[label]}\n")
    print(f"Predicted:{classes[pred.item()]}\n")
```
### OUTPUT

## Training Loss per Epoch

<img width="342" height="120" alt="Screenshot 2026-08-12 171251" src="https://github.com/user-attachments/assets/585fd2de-8608-4ad7-9f4a-7e9d5101ea34" />


## Confusion Matrix

<img width="748" height="635" alt="Screenshot 2026-08-12 171307" src="https://github.com/user-attachments/assets/5220d60c-4714-41bb-aa77-b1b0d6319aa2" />


## Classification Report

<img width="831" height="452" alt="Screenshot 2026-08-12 171303" src="https://github.com/user-attachments/assets/688c3216-9c91-4dee-a496-df60c4e3c965" />


### New Sample Data Prediction

<img width="600" height="672" alt="Screenshot 2026-08-12 171315" src="https://github.com/user-attachments/assets/d0e105e0-05d5-4758-b058-c18a413b05a1" />


<img width="482" height="510" alt="Screenshot 2026-08-12 171320" src="https://github.com/user-attachments/assets/f86611ec-a9be-4811-8714-f126b54727b8" />


## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
