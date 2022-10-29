from torch import empty
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
import math # Juste to check for nan values
from torch.nn.functional import unfold, fold 
import torch # only to test convolution

#---------------------------------------------------------------------------------------------------------------------------------# 
#---------------------------------------------------------- MODEL TRAINER --------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------# 

class Model(object):
    def __init__(self):
        self.criterion = MSE()
        self.optimizer = SGD(lr=1e-2)
        
        self.num_epochs = 10  # number of epochs to train
        self.pretrained = False  # if we want to use pretrained weights
        self.train_ratio = 0.8  # part of train data in a full dataset
        self.batch_size = 256

        self.Model = Sequential(Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=1), ReLU(),
                            Conv2d(in_channels=16, out_channels=48, kernel_size=2, stride=1), ReLU(),
                            TransposeConv2d(in_channels=48, out_channels=16, kernel_size=2, stride=1), ReLU(),
                            TransposeConv2d(in_channels=16, out_channels=3, kernel_size=2, stride=1), Sigmoid(),
                            criterion=self.criterion, optimizer=self.optimizer)     
            
        if self.pretrained:
          self.load_pretrained_model()
          print("Pretrained model successfully loaded")            

    def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel.pth into the model
        with open("bestmodel.pth", "rb") as file:
            params = pickle.load(file)
            self.Model.set_params(params)


    def train(self, train_input, train_target): 
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images , which only differs from the input by their noise .

        # Compute dataset length
        len_dataset = train_input.shape[0]
        train_len = int(self.train_ratio * len_dataset)
        # Create Dataset
        train_dataset = NoiseDataset(train_input[:train_len], train_target[:train_len])
        val_dataset = NoiseDataset(train_input[train_len:], train_target[train_len:])
        # Create dataloader for validation and train datasets
        TrainLoader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)  # create train loader
        ValLoader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)  # create val loader

        acc_loss = {"train":[], "val":[]}
        loss_train_mean = 0

        for e in range(self.num_epochs):
            loss_psnr = 0
            loss_train = 0
            loss_train = []

            for idx, (input_batch, target_batch) in enumerate(TrainLoader):

                # EST CE QUE INPUT BATCH ET TARGET BATCH SONT BIEN NORMALISE ?

                output = self.Model.forward(input_batch)

                # Compute loss
                loss_train.append(self.Model.criterion.forward(target_batch, output).item())

                # Compute backward
                self.Model.backward()

                # Optimize
                self.Model.step()
            
            loss_train_mean = sum(loss_train)/len(loss_train)
            acc_loss["train"].append(loss_train_mean)
            loss_val = self.Model.criterion.forward(val_dataset.target, self.Model.forward(val_dataset.input))
            acc_loss["val"].append(loss_val.item())

            # Computation of the PSNR loss using validation dataset
            loss_psnr = psnr(self.Model.forward(val_dataset.input[:train_len]), val_dataset.target[:train_len])

            print("Epoch " + str(e+1) + "/" + str(self.num_epochs) +
                  " || MSE Loss (Train) : " + str(round(loss_train_mean, 3)) + " [-]" +
                  " || MSE Loss (Val) : " + str(round(loss_val.item(), 3)) + " [-]" +
                  " || PSNR (Val) : " + str(round(loss_psnr.item(), 3)) + " [dB]")

        # Plot the learning curve
        plt.figure(figsize=(5, 5))
        plt.plot(acc_loss["train"], label="train")
        plt.plot(acc_loss["val"], label="validation")
        plt.grid()
        plt.legend()
        plt.show()

        # Save the model
        with open("model.pth", "wb") as file:
            param = self.Model.param()
            pickle.dump(param, file)

    def predict(self, test_input): # -> torch.Tensor
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a byte tensor of the size (N1 , C, H, W)
        
        test_input = test_input.float()
        test_output = 255. * self.Model.forward(test_input/255.)
        return test_output.byte()

#---------------------------------------------------------------------------------------------------------------------------------# 
#----------------------------------------------------------- MODULE TEMPLATE -----------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------# 

# empty template
class Module(object):
    """
    Prototype of the modules
    """
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    
    def param(self):
        return [] # empty for parameterless modules

#---------------------------------------------------------------------------------------------------------------------------------# 
#----------------------------------------------------------- SEQUENTIAL ----------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------# 

class Sequential(Module):
    """
    Sequential module that contains all the module of the network
    """
    def __init__(self, *args, criterion, optimizer):
        # list of the network's modules
        self.modules = args
        # set minimization criterion and optimizer
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, input):
        """
        Call the forward function of all modules
        """
        x = input
        for idx, m in enumerate(self.modules):
            x = m.forward(x)
            
        return x
    
    def backward(self):
        """
        Call the backward funtion of all modules in reversed order
        """
        # compute initial derivative to backpass thanks to criterion
        x = self.criterion.backward()

        # propagates the gradient through the network
        for idx, m in enumerate(reversed(self.modules)):
            x = m.backward(x)

    def step(self):
        """
        Update parameters of all modules
        """
        for m in self.modules:
            m.step(self.optimizer)

    def param(self):
        """
        Return a list of all list of modules' parameters
        """
        params = []
        for m in self.modules:
            params.append(m.param())
        return params

    def set_params(self, params):
        for idx, m in enumerate(self.modules):
            m.set_param(params[idx])


#---------------------------------------------------------------------------------------------------------------------------------# 
#-------------------------------------------------------------- Conv2d -----------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------# 

class Conv2d(Module):
    """
    Convolution module similar to torch.nn.Conv2d
    uses only dilation = 1 and padding_mode = 'zeros'
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, is_bias=True): 
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int): 
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        self.is_bias = is_bias

        # parameters & other variables
        self.weight = self._init_weight()
        self.grad_weight = empty(self.weight.size()).zero_()
        if is_bias:
            self.bias = self._init_bias()
            self.grad_bias = empty(self.bias.size()).zero_()
        else:
            self.bias = None
            self.grad_bias = None
            

        self.in_im_size = None
        self.in_unfolded = None

    def _init_weight(self):
        """
        initialize weight with standard torch.nn.Conv2d initialization
        weight.size() = (out_channels, in_channels, kernel_h, kernel_w)
        """
        n = empty(1).zero_()
        n[0] = self.kernel_size[0] * self.kernel_size[1]
        return empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).uniform_(-1, 1).mul(n.sqrt())

    def _init_bias(self):
        """
        initialize bias with standard torch.nn.Conv2d initialization
        bias.size() = (out_channels)
        """
        n = empty(1).zero_()
        n[0] = self.kernel_size[0] * self.kernel_size[1]
        return empty(self.out_channels).uniform_(-1, 1).mul(n.sqrt())

    def _compute_output_size(self, input):
        """
        compute output image height and width from parameters :
        input, kernel height and width, stride, padding
        """
        out_h = (input.size(2) + 2 * self.padding[0] - self.kernel_size[0])/self.stride[0] + 1
        out_w = (input.size(3) + 2 * self.padding[1] - self.kernel_size[1])/self.stride[1] + 1

        # assess stride value
        if not(out_h.is_integer()) or not(out_w.is_integer()):
            raise NameError('inadequate stride value')

        return (int(out_h),int(out_w))

    def test_convolution(self, input): 
        """
        Tests the forward pass of the convolution with weight and bias from standard torch.nn.Conv2d initialization
        import torch to use it
        """
        if self.is_bias: bias = self.bias
        else: bias = None
        torch.testing.assert_allclose(self.forward(input), torch.nn.functional.conv2d(input, weight=self.weight,bias=bias,padding=self.padding,stride=self.stride))

    def forward(self, input):
        """
        Forward pass of the convolution
        returns the output of the convolution
        """
        # extract batch size and image size (height and width)
        self.batch_size = input.size(0) 
        self.in_im_size = (input.size(2), input.size(3))

        # unfold input
        self.in_unfolded = unfold(input, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride) # save for backward pass

        # perform linear convolution
        conv_out = self.weight.view(self.out_channels, -1).matmul(self.in_unfolded)

        # compute output size for folding back
        output_size = self._compute_output_size(input)

        # reshape output and add bias
        output =  fold(conv_out, output_size, (1,1)) # equivalent to view(...)
        if self.is_bias:
            output += self.bias.view(1, -1, 1, 1)
        return output


    def backward(self, gradwrtoutput):
        """
        Backward pass of the convolution
        returns the gradient of the loss wrt the input
        """
        # accumulate gradient for bias
        # accumulate : sum along 1st dimension
        self.grad_bias = gradwrtoutput.sum(dim=[0,2,3])

        # reshape grad_out for more lisibility
        grad_out = gradwrtoutput.view(self.batch_size, self.out_channels, -1)

        # accumulate gradient for weight :
        # grad_weight = grad_out*unfolded' 
        # accumulate : sum along 1st dimension
        grad_weight = grad_out.matmul(self.in_unfolded.transpose(1,2)) # grad_weight.size() = (batch_size, out_ch, in_ch * ker_rows * ker_col)
        grad_weight = grad_weight.sum(dim=0) # add the grad of each weight of the batch
        self.grad_weight = grad_weight.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]) # save in self.grad_weight in original form
        
        # compute gradient wrt input : 
        grad_in_unfolded = self.weight.view(self.out_channels, -1).t().matmul(grad_out)

        # fold with same parameters as unfold in forward pass to retrieve input like shape
        grad_in = fold(grad_in_unfolded, output_size=self.in_im_size, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        return grad_in

    def step(self, optimizer):
        """
        Updates the weights of the convolutional layer
        also reset accumulated gradients to 0 (for safety)
        """
        self.bias = optimizer(self.bias, self.grad_bias)   
        self.weight = optimizer(self.weight, self.grad_weight)
        self.grad_bias.zero_()
        self.grad_weight.zero_()    
    
    def param(self):
        return [(self.weight, self.grad_weight),(self.bias, self.grad_bias)]

    def set_param(self, param):
        self.weight = param[0][0]
        self.bias = param[1][0]

#---------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------- TRANSPOSECONV2D -----------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------#

class TransposeConv2d(Module):
    """
    Transpose Convolution module similar to torch.nn.TransposeConv2d
    Here we perform a so called fractionally-strided convolution
    Associated transposed convolution : kernel_size, stride and padding are the same parameters used for the initialization of the associated convolution
    
    Uses only dilation = 1 and padding_mode = 'zeros'
    Also, output_padding is not necessary because in the associated convolution, we uses a input and kernel sizes, padding and stride so that there is no ambiguity 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, is_bias=True):
        
        self.in_channels = in_channels
        # keep record of the stride to build the fractionally strided input in the forward pass
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        
        # calculate associated parameters (padding and stride) for the convolution
        assos_stride = 1
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assos_padding = (kernel_size[0] - 1 - padding[0], kernel_size[1] - 1 - padding[1])
        

        # initialize the convolution (will take fractionally strided input)
        self.fracst_conv = Conv2d(in_channels, out_channels, kernel_size, assos_stride, assos_padding, is_bias)

        conv_params = self.fracst_conv.param()

        # parameters
        # take the init value from the convolution module
        self._fracst_conv_params_to_self_params()

        # retain params for testing purpose :
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        self.kernel_size = kernel_size
        self.is_bias = is_bias

    def _fracst_conv_params_to_self_params(self):
        """
        Get params from self.fracst_conv and save them as self params
        flip the kernel and transpose channel dims (dim are that way in torch conv_transpose2d)
        """
        # weight.size() = (in_channels, out_channels, 'fliped_kernel' )
        conv_params = self.fracst_conv.param()
        self.weight = conv_params[0][0].transpose(0,1).flip(dims=[2,3])
        self.grad_weight = conv_params[0][1].transpose(0,1).flip(dims=[2,3])
        self.bias = conv_params[1][0]
        self.grad_bias = conv_params[1][1]

    def forward(self, input):
        """
        Forward pass of the transpose convolution module
        Build a fractionally strided input then call the forward pass the convolution
        """
        batch_size = input.size(0)
        in_im_size = (input.size(2), input.size(3))

        # Build a fractionally strided input
        self.fracst_input_size = ((in_im_size[0] - 1) * self.stride[0] + 1, (in_im_size[1] - 1) * self.stride[1] + 1)
        fracst_input = empty(batch_size, self.in_channels, self.fracst_input_size[0], self.fracst_input_size[1]).zero_() # full of zeros
        fracst_input[:, :, 0:self.fracst_input_size[0]:self.stride[0], 0:self.fracst_input_size[1]:self.stride[1]] = input
        
        # Call forward pass of the convolution with fracst_input
        output = self.fracst_conv.forward(fracst_input)
        return output

    def backward(self, gradwrtoutput):
        """
        Backward pass of the transpose convolution module
        Calls the backward pass of the convolution, retrieve the shape of the gradient wrt the input and updates the accumulated gradients
        """
        # Call backward pass of the convolution
        grad_fracst_in = self.fracst_conv.backward(gradwrtoutput)

        # Retrieve the shape of the original input for the gradient wrt the input
        grad_in = grad_fracst_in[:, :, 0:self.fracst_input_size[0]:self.stride[0], 0:self.fracst_input_size[1]:self.stride[1]]

        # Updates gradients
        self._fracst_conv_params_to_self_params()
        return grad_in
    
    def step(self, optimizer):
        """
        Updates the weights of convolution and of the module
        """
        self.fracst_conv.step(optimizer)
        # Updates the weights and grad with weights from the convolution 
        self._fracst_conv_params_to_self_params()  
    
    def param(self):
        return [(self.weight, self.grad_weight),(self.bias, self.grad_bias)]

    def set_param(self, param):
        self.weight = param[0][0]
        self.bias = param[1][0]
        
#---------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------- SGD OPTIMIZER -------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------#

class SGD(Module):
    """
    Stochastic gradient descent module (optimizer)
    """
    def __init__(self, lr=1e-3):
        self.learning_rate = lr

    def __call__(self, param, acc_grad):
        return param - self.learning_rate * acc_grad
    
    def param(self):
        return super().param()

    def set_param(self, param):
        pass
#---------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------- MSE CRITERION -------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------#

class MSE(Module):
    """
    Mean Square Error class
    computes MSE loss 
    """
    def __init__(self):
        self.y = None
        self.y0 = None
        self.nb = 0 # Number of elements 
    
    def forward(self, *input):
        """
        INPUT : y0 = ground truth images [C, H, W]
                y = Output of the NN [C, H, W]
        OUTPUT : MSE Loss
        """
        if len(input) == 2:
            self.y0 = input[0] # Ground truth
            self.y = input[1] # Output of the NN
            N, C, H, W = self.y.size()
            self.nb = N*C*H*W
        else:
            print("Invalid number of arguments in MSE Loss")
            return -1

        output = (self.y - self.y0).pow(2).mean()
        return output

    def backward(self):
        return 2*(self.y - self.y0)/self.nb
    
    def param(self):
        return super().param() # returns nothing, no parameters

    def set_param(self, param):
        pass

#---------------------------------------------------------------------------------------------------------------------------------# 
#----------------------------------------------------------- RELU ACTIVATION -----------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------# 

class ReLU(Module):
    """
    Relu module (activation)
    """
    def __init__(self):
        self.x_in = 0 # store for backward pass
    
    def forward(self, input):
        self.x_in = input.float() # save for backprop
        return self.x_in.clamp(0, None)

    def backward(self, gradwrtoutput):
        return gradwrtoutput * (self.x_in >= 0).float() # Convert bool into float

    def step(self, *args):
        return 0

    def param(self):
        return super().param() # returns nothing, no parameters

    def set_param(self, param):
        pass

#---------------------------------------------------------------------------------------------------------------------------------# 
#----------------------------------------------------------- SIGMOID ACTIVATION --------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------# 

class Sigmoid(Module):
    """
    Sigmoid module (activation)
    """
    def __init__(self):
        self.sig_out = 0

    def forward(self, input):        
        self.sig_out = 1 / (1 + input.mul(-1).exp()) # save for backprop
        return self.sig_out

    def backward(self, gradwrtoutput):
        grad_sig = self.sig_out.mul(1 - self.sig_out)
        grad_in = grad_sig.mul(gradwrtoutput)
        return grad_in

    def step(self, *args):
        return 0

    def param(self):
        return super().param() # returns nothing, no parameters

    def set_param(self, param):
        pass
