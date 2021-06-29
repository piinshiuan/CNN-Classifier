import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets , models, transforms
import numpy as np
import matplotlib.image as mpimg
from torchsummary import summary
from sklearn.model_selection import KFold

import json

    
# Create CNN Model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0) #output_shape=(16,220,220) #(224-5+1)/1 #(weigh-kernel+1)/stride 無條件進位
        self.relu1 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,110,110) #(220/2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) #output_shape=(32,106,106)
        self.relu2 = nn.ReLU() # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,53,53)
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0) #output_shape=(16,51,51)
        self.relu3 = nn.ReLU() # activation
        # Max pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,25,25)
        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0) #output_shape=(8,23,23)
        self.relu4 = nn.ReLU() # activation
        # Max pool 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) #output_shape=(8,11,11)
        # Fully connected 1 ,#input_shape=(8*12*12)
        self.fc1 = nn.Linear(8 * 11 * 11, 512) 
        self.relu5 = nn.ReLU() # activation
        self.fc2 = nn.Linear(512, 2) 
        self.output = nn.Softmax(dim=1)
        
    
    def forward(self, x):
        out = self.cnn1(x) # Convolution 1
        out = self.relu1(out)
        out = self.maxpool1(out)# Max pool 1
        out = self.cnn2(out) # Convolution 2
        out = self.relu2(out) 
        out = self.maxpool2(out) # Max pool 2
        out = self.cnn3(out) # Convolution 3
        out = self.relu3(out)
        out = self.maxpool3(out) # Max pool 3
        out = self.cnn4(out) # Convolution 4
        out = self.relu4(out)
        out = self.maxpool4(out) # Max pool 4
        out = out.view(out.size(0), -1) # last CNN faltten con. Linear NN
        out = self.fc1(out) # Linear function (readout)
        out = self.fc2(out)
        out = self.output(out)

        return out

"""for showing the pic"""

classes = ['fake','real']
mean , std = torch.tensor([0.485, 0.456, 0.406]),torch.tensor([0.229, 0.224, 0.225])

def denormalize(image):
  image = transforms.Normalize(-mean/std,1/std)(image) #denormalize
  image = image.permute(1,2,0) #Changing from 3x224x224 to 224x224x3
  image = torch.clamp(image,0,1)
  return image

# helper function to un-normalize and display an image
def imshow(img):
    img = denormalize(img) 
    plt.imshow(img)
  
def draw(data):
  # number of subprocesses to use for data loading
  num_workers = 0
  # how many samples per batch to load
  batch_size = 32
  # learning rate
  LR = 0.01
  train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=num_workers,shuffle=False)
  images,labels=next(iter(train_loader))
  # obtain one batch of training images
  dataiter = iter(train_loader)
  images, labels = dataiter.next()
  # convert images to numpy for display

  # plot the images in the batch, along with the corresponding labels
  fig = plt.figure(figsize=(25, 5))
  # display 20 images
  for idx in np.arange(20):
    ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title("{} ".format( classes[labels[idx]]))
  plt.show()

"""for showing the pic"""

""" for k-fold """
def kfold(train_data,train_on_gpu):
  k_folds = 5
  kfold = KFold(n_splits=k_folds, shuffle=True)
  for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_data)):
    print(f'FOLD {fold}')
    print('=====================================')
    kfold = KFold(n_splits=k_folds, shuffle=True)
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      train_data, 
                      batch_size=32, sampler=train_subsampler)
    validloader = torch.utils.data.DataLoader(
                      train_data,
                      batch_size=32, sampler=valid_subsampler)
    #draw(trainloader)
    model = CNN_Model()
    if train_on_gpu:
      model.cuda()
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss() 
    n_epochs=50
    valid_loss_min = np.Inf
    y=[]
    x=[]
    for epoch in range(1, n_epochs+1):
      # keep track of training and validation loss
      train_loss = 0.0
      valid_loss = 0.0
      print('running epoch: {}'.format(epoch))
      print('--------------------------------')
      ###################
      # train the model #
      ###################
      model.train()
      for data, target in trainloader:
          # move tensors to GPU if CUDA is available
          if train_on_gpu:
              data, target = data.cuda(), target.cuda()
          # clear the gradients of all optimized variables
          optimizer.zero_grad()
          # forward pass: compute predicted outputs by passing inputs to the model
          output = model(data)
          # calculate the batch loss
          loss = criterion(output, target)
          # backward pass: compute gradient of the loss with respect to model parameters
          loss.backward()
          # perform a single optimization step (parameter update)
          optimizer.step()
          # update training loss
          train_loss += loss.item()*data.size(0)
      model.eval()
      correct = 0.
      total = 0.
      for data, target in validloader:
          #print(type(data))
          # move tensors to GPU if CUDA is available
          if train_on_gpu:
              data, target = data.cuda(), target.cuda()
          
          # forward pass: compute predicted outputs by passing inputs to the model
          output = model(data)
          # calculate the batch loss
          loss = criterion(output, target)
          # update average validation loss 
          valid_loss += loss.item()*data.size(0)
          pred = output.data.max(1, keepdim=True)[1]

          # compare predictions to true label
          correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
          total += data.size(0)
          
      # calculate average losses
      train_loss = train_loss/len(trainloader.dataset)
      valid_loss = valid_loss/len(validloader.dataset)
      
      # print training/validation statistics 
      print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
          train_loss, valid_loss))
      print('Test Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))  
      # save model if validation loss has decreased
      if valid_loss <= valid_loss_min:
          print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
          valid_loss_min,
          valid_loss))
          torch.save(model.state_dict(), 'model_CNN{}.pth'.format(fold))
          valid_loss_min = valid_loss
      y.append(valid_loss)
      x.append(epoch)
    plt.title('{}-fold loss!'.format(fold))
    plt.plot(x, y)
    plt.show()
  
""" k-fold """  
    

""" test """
def test(test_data, use_cuda):
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = len(test_data)
    # learning rate
    LR = 0.01
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)
    model0 = CNN_Model()
    model1 = CNN_Model()
    model2 = CNN_Model()
    model3 = CNN_Model()
    model4 = CNN_Model()
    if use_cuda:
      model0.cuda()
      model1.cuda()
      model2.cuda()
      model3.cuda()
      model4.cuda()
    model0.load_state_dict(torch.load('{}'.format('model_CNN0.pth')))
    model1.load_state_dict(torch.load('{}'.format('model_CNN1.pth')))
    model2.load_state_dict(torch.load('{}'.format('model_CNN2.pth')))
    model3.load_state_dict(torch.load('{}'.format('model_CNN3.pth')))
    model4.load_state_dict(torch.load('{}'.format('model_CNN4.pth')))
    criterion = torch.nn.CrossEntropyLoss() 
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    correct0 =0.
    correct1 =0.
    correct2 =0.
    correct3 =0.
    correct4 =0.
    total = 0.

    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    
    predictions=[]
    for batch_idx, (data, target) in enumerate(test_loader):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        with torch.no_grad():
          output0=model0(data)
          output1=model1(data)
          output2=model2(data)
          output3=model3(data)
          output4=model4(data)
          output=(output0+output1+output2+output3+output4)/5.0
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        pred0 = output0.data.max(1, keepdim=True)[1]
        pred1 = output1.data.max(1, keepdim=True)[1]
        pred2 = output2.data.max(1, keepdim=True)[1]
        pred3 = output3.data.max(1, keepdim=True)[1]
        pred4 = output4.data.max(1, keepdim=True)[1]

        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        correct0 += np.sum(np.squeeze(pred.eq(target.data.view_as(pred0))).cpu().numpy())
        correct1 += np.sum(np.squeeze(pred.eq(target.data.view_as(pred1))).cpu().numpy())
        correct2 += np.sum(np.squeeze(pred.eq(target.data.view_as(pred2))).cpu().numpy())
        correct3 += np.sum(np.squeeze(pred.eq(target.data.view_as(pred3))).cpu().numpy())
        correct4 += np.sum(np.squeeze(pred.eq(target.data.view_as(pred4))).cpu().numpy())

        total += data.size(0)
        predlist=pred.tolist()
        for i in range(len(predlist)):
          predictions.append(predlist[i][0])
            
    print('Test Loss: {:.6f}'.format(test_loss))
    print('----Model average -----')
    print('Test Accuracy : %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total)) 
    print('----Model 0 -----')
    print('Test Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct0 / total, correct0, total)) 
    print('----Model 1 -----')
    print('Test Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct1 / total, correct1, total)) 
    print('----Model 2 -----')
    print('Test Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct2 / total, correct2, total)) 
    print('----Model 3 -----')
    print('Test Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct3 / total, correct3, total)) 
    print('----Model 4 -----')
    print('Test Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct4 / total, correct4, total))   


def prediction(unknown_data, use_cuda):
    batch_size =100
    test_loader = torch.utils.data.DataLoader(unknown_data, batch_size=batch_size)
    model0 = CNN_Model()
    model1 = CNN_Model()
    model2 = CNN_Model()
    model3 = CNN_Model()
    model4 = CNN_Model()
    if use_cuda:
      model0.cuda()
      model1.cuda()
      model2.cuda()
      model3.cuda()
      model4.cuda()
      
    model0.load_state_dict(torch.load('{}'.format('model_CNN0.pth')))
    model1.load_state_dict(torch.load('{}'.format('model_CNN1.pth')))
    model2.load_state_dict(torch.load('{}'.format('model_CNN2.pth')))
    model3.load_state_dict(torch.load('{}'.format('model_CNN3.pth')))
    model4.load_state_dict(torch.load('{}'.format('model_CNN4.pth')))
    
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    predictions=[]
    correct=0
    total=0
    
    for(data, target) in test_loader:
        # move to GPU
        if use_cuda:
            data = data.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        with torch.no_grad():
          output0=model0(data)
          output1=model1(data)
          output2=model2(data)
          output3=model3(data)
          output4=model4(data)
          output=(output0+output1+output2+output3+output4)/5.0
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        predlist=pred.tolist()
        
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

        total += data.size(0)
        
        for i in range(len(predlist)):
          predictions.append(predlist[i][0])
    
      

path_train="train"
train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
path_test="test"
test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
path_unknown="icons"
unknown_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


"""train part"""

train_data=datasets.ImageFolder(path_train, transform=train_transforms)

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

kfold(train_data,train_on_gpu)



"""test part"""
"""check the performance of the Model"""
test_data=datasets.ImageFolder(path_test, transform=test_transforms)

use_cuda = torch.cuda.is_available()
if not use_cuda:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
test(test_data,use_cuda)


""" predict """
unknown_data=datasets.ImageFolder(path_unknown,transform=unknown_transforms)

use_cuda = torch.cuda.is_available()
if not use_cuda:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    
prediction(unknown_data,use_cuda)

import json
with open('0713412_4_result.json', newline='') as jsonfile:
    data = json.load(jsonfile)
    # 或者這樣
    # data = json.loads(jsonfile.read())
    print(len(data))
    