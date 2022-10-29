import os
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 
from matplotlib import pyplot as plt
import numpy as np  

#---------------------------------------------------------------------------------------------------------------------------------# 
#---------------------------------------------------------- MODEL TRAINER --------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------# 

class Model:
    def __init__(self):
        # instantiate model + optimizer + loss function + any other stuff you need
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # gpu or cpu

        self.model = noise2noise(3, 3).to(self.device)
        self.Optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.99))
        self.Criterion = torch.nn.MSELoss()  # loss function

        # learning parameters
        self.num_epochs = 200  # number of epochs to train
        self.pretrained = False  # if we want to use pretrained weights
        self.train_ratio = 0.9  # part of train data in a full dataset
        self.batch_size = 256
        self.plot = False 

    def load_pretrained_model(self):
        # This loads the parameters saved in bestmodel.pth into the model
        if os.path.isfile(f"./bestmodel.pth"):
            checkpoint = torch.load(f"./bestmodel.pth", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.Optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self, train_input, train_target):
        #:train_input: tensor of size (N, C, H, W) containing a noisy version of the images
        # .same images, which only differs from the input by their noise.
        #:train_target: tensor of size (N, C, H, W) containing another noisy version of the

        # load initial data to dataloader
        len_dataset = train_input.shape[0]
        train_len = int(self.train_ratio * len_dataset)
        train_dataset = NoiseDataset(train_input[:train_len], train_target[:train_len])
        val_dataset = NoiseDataset(train_input[train_len:], train_target[train_len:])
        TrainLoader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)  # create train loader
        ValLoader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)  # create val loader

        if self.pretrained: 
            self.load_pretrained_model()
  
        res_memory = {"train": [], "val": []}
        print("*" * 60 + " Start training " + "*" * 60)
        for epoch in tqdm(range(self.num_epochs)):  # run epoch
            # train
            train_loss, val_loss = [], []
            self.model.train()
            for idx, (noise, clean) in enumerate(TrainLoader):  # run 1 batch
                noise, clean = noise.to(self.device).float(), clean.to(self.device).float()  # convert to GPU
                with torch.set_grad_enabled(True):
                    self.Optimizer.zero_grad()
                    predicted = self.model(noise)  # get predictions
                    cur_loss = self.Criterion(predicted, clean)  # get loss
                    train_loss.append(cur_loss.item())
                    cur_loss.backward()  # backward loss for model fitting
                    self.Optimizer.step()  # update optimizer

            # validation
            self.model.eval()
            for idx, (noise, clean) in enumerate(ValLoader):  # run 1 batch
                noise, clean = noise.to(self.device).float(), clean.to(self.device).float()  # convert to GPU
                if idx == 0:
                  val_image = noise.cpu()
                  val_clean = clean.cpu()
                with torch.set_grad_enabled(False):
                    predicted = self.model(noise)  # get predictions
                    val_result = predicted.cpu()
                    cur_loss = self.Criterion(predicted, clean)  # get loss
                    val_loss.append(cur_loss.item())
            
            res_memory['train'].append(np.mean(train_loss))
            res_memory['val'].append(np.mean(val_loss))
            print(f"Epoch {epoch} / {self.num_epochs}: Train loss {res_memory['train'][-1]}, "
                  f"Validation loss: {res_memory['val'][-1]}")
        
        # plot losses 
        if self.plot: 
          plt.plot(np.arange(self.num_epochs), res_memory['train'], label="Train")
          plt.plot(np.arange(self.num_epochs), res_memory['val'], label="Val")
          plt.legend()
          plt.title("Losses")
          plt.show()

          sample_show = 6
          fig, ax = plt.subplots(ncols=sample_show, nrows=3, figsize=(8, 8), sharex=True, sharey=True, constrained_layout = True)
          ax[0, 0].set_ylabel("Original")
          ax[1, 0].set_ylabel("Clean")
          ax[2, 0].set_ylabel("Predicted")
          for idx in range(sample_show):
            ax[0, idx].set_axis_off()
            ax[1, idx].set_axis_off()
            ax[2, idx].set_axis_off()
            ax[0, idx].imshow(val_image[idx].permute(1, 2, 0).numpy().astype(int))
            ax[1, idx].imshow(val_clean[idx].permute(1, 2, 0).numpy().astype(int))
            ax[2, idx].imshow(val_result[idx].permute(1, 2, 0).numpy().astype(int))
          
          plt.show()
          print("PSNR for validation batch", psnr(val_result, val_clean))

        # save results
        print("*" * 60 + " Saving results " + "*" * 60)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.Optimizer.state_dict()
        }, 'bestmodel.pth')

    def predict(self, test_input):
            #:test_input: tensor of size (N1, C, H, W) that has to be denoised by the trained
            # or the loaded network.
            self.model.eval()
            if self.pretrained:
                self.load_pretrained_model()
            results = []
            for idx, noise in enumerate(test_input):  # run 1 batch
                noise = noise.to(self.device)  # convert to GPU
                with torch.set_grad_enabled(False):
                    predicted = self.model(noise)  # get predictions
                    results.append(predicted)
            return results

    def run(self):
        print("Device:", self.device)
        noisy, clean = torch.load('/content/drive/MyDrive/DeepLearning/train_data.pkl')
        self.train(noisy, clean)

#---------------------------------------------------------------------------------------------------------------------------------# 
#-------------------------------------------------------- NOISE2NOISE MODEL-------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------# 


class noise2noise(nn.Module):
    """
    Noise2Noise model
    """
    def __init__(self, n_dim=3, out_dim=3):
        super(noise2noise, self).__init__()
        kernel_size = (1, 1)
        pooling_size = (2, 2)
        channel_diff = 48 
        padding = 0
        dilation = 1
        p=0

        # input : 32x32

        # LAYERS
        self.enc_conv0 = nn.Sequential(nn.Conv2d(3, channel_diff, kernel_size=kernel_size, padding=padding, dilation=dilation), nn.LeakyReLU(), nn.Dropout(p=p))
        # (#, 48, 32, 32)
        self.enc_conv1 = nn.Sequential(nn.Conv2d(channel_diff,  channel_diff, kernel_size=kernel_size, padding=padding, dilation=dilation), nn.LeakyReLU(), nn.Dropout(p=p))
        # (#, 48, 32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=pooling_size)
        # (#, 48, 16, 16)
        self.enc_conv2 = nn.Sequential(nn.Conv2d(channel_diff, channel_diff, kernel_size=kernel_size, padding=padding, dilation=dilation), nn.LeakyReLU(), nn.Dropout(p=p))
        # (#, 48, 16, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=pooling_size)
        # (#, 48, 8, 8)
        self.enc_conv3 = nn.Sequential(nn.Conv2d(channel_diff,  channel_diff, kernel_size=kernel_size, padding=padding, dilation=dilation), nn.LeakyReLU(), nn.Dropout(p=p))
        # (#, 48, 8, 8)
        self.pool3 = nn.MaxPool2d(kernel_size=pooling_size)

        # CONCAT3
        self.dec_conv3a = nn.Sequential(nn.Conv2d(channel_diff * 2,  channel_diff * 2, kernel_size=kernel_size, padding=padding, dilation=dilation), nn.LeakyReLU(), nn.Dropout(p=p))
        self.dec_conv3b = nn.Sequential(nn.Conv2d(channel_diff * 2,  channel_diff * 2, kernel_size=kernel_size, padding=padding, dilation=dilation), nn.LeakyReLU(), nn.Dropout(p=p))
        self.upsample2 = nn.Upsample(scale_factor=2)
        # CONCAT2
        self.dec_conv2a = nn.Sequential(nn.Conv2d(channel_diff * 3,  channel_diff * 2, kernel_size=kernel_size, padding=padding, dilation=dilation), nn.LeakyReLU(), nn.Dropout(p=p))
        self.dec_conv2b = nn.Sequential(nn.Conv2d(channel_diff * 2,  channel_diff * 2, kernel_size=kernel_size, padding=padding, dilation=dilation), nn.LeakyReLU(), nn.Dropout(p=p))
        self.upsample1 = nn.Upsample(scale_factor=2)
        # CONCAT1
        self.dec_conv1a = nn.Sequential(nn.Conv2d(channel_diff * 2 + 3,  channel_diff, kernel_size=kernel_size, padding=padding, dilation=dilation), nn.LeakyReLU(), nn.Dropout(p=p))
        self.dec_conv1b = nn.Sequential(nn.Conv2d(channel_diff,  channel_diff // 2, kernel_size=kernel_size, padding=padding, dilation=dilation), nn.LeakyReLU(), nn.Dropout(p=p))
        self.dec_conv1c = nn.Sequential(nn.Conv2d(channel_diff // 2, 3, kernel_size=kernel_size, padding=padding, dilation=dilation), nn.LeakyReLU(), nn.Dropout(p=p))

    def forward(self, input):
        # encoder 
        x = self.enc_conv0(input)
        # (#, 48, 32, 32)
        x = self.enc_conv1(x)
        # (#, 48, 32, 32)
        out1 = self.pool1(x)

        # (#, 48, 16, 16)

        x = self.enc_conv2(out1)
        # (#, 48, 16, 16)
        out2 = self.pool2(x)
  
        # (#, 48, 8, 8)

        x = self.enc_conv3(out2)
        # (#, 48, 8, 8)
        x = torch.concat([x, out2], dim=1)
        # (#, 96, 8, 8)
        x = self.dec_conv3a(x)
        # (#, 96, 8, 8)
        x = self.dec_conv3b(x)
        # (#, 96, 8, 8)
        x = self.upsample2(x)
        # (#, 96, 16, 16)

        x = torch.concat([x, out1], dim=1)
        # (#, 144, 16, 16)
        x = self.dec_conv2a(x)
        # (#, 96, 16, 16)
        x = self.dec_conv2b(x)
        # (#, 96, 16, 16)
        x = self.upsample1(x)
        # (#, 96, 32, 32)

        x = torch.concat([x, input], dim=1)
        # (#, 99, 32, 32)
        x = self.dec_conv1a(x)
        # (#, 48, 32, 32)
        x = self.dec_conv1b(x)
        # (#, 24, 32, 32)
        x = self.dec_conv1c(x)
        # (#, 3, 32, 32)
        return x

#---------------------------------------------------------------------------------------------------------------------------------# 
#------------------------------------------------------- NOISE DATASET HANDLER ---------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------# 

class NoiseDataset(Dataset):
    """
    PyTorch Dataset which loads validation and training images
    """
    def __init__(self, train_input, train_target):
        """
        Initialize dataset parameters
        :param path: path to validation and train images
        :param data_type: {"train" or "val"} type of data
        """
        self.limitations = 100
        self.input, self.target = train_input, train_target

    def __len__(self):
        return len(self.input) // self.limitations 

    def __getitem__(self, item):
        return self.input[item], self.target[item]



def show_img_results(img_idx, model, url='/content/drive/MyDrive/DeepLearning/val_data.pkl'):
  noisy_val, clean_val = torch.load(url)
  
  # Forward pass of the noisy validation images
  denoised = model.forward(noisy_val.type(torch.FloatTensor))
  denoised = denoised.byte()

  # Create figure and fill it with noisy, denoised and clean images
  fig = plt.figure(figsize=(5, 7))
  for i in range(len(img_idx)):
    # Plot noisy image
    fig.add_subplot(len(img_idx), 3, 3*i+1) 
    plt.imshow(noisy_val[img_idx[i], :, :, :].permute(1, 2, 0))
    
    # Plot denoised image
    fig.add_subplot(len(img_idx), 3, 3*i+2)
    plt.imshow(denoised[img_idx[i], :, :, :].permute(1, 2, 0))

    # Plot clean image
    fig.add_subplot(len(img_idx), 3, 3*i+3)
    plt.imshow(clean_val[img_idx[i], :, :, :].permute(1, 2, 0))


