import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def create_dataset():
    import numpy as np
    import sklearn.datasets

    # Generate a dataset and plot it
    np.random.seed(0)
    num_samples = 500

    X, y = sklearn.datasets.make_moons(num_samples, noise=0.20)

    # define train, validation, and test sets
    X_tr = X[:100].astype('float32')
    X_val = X[100:400].astype('float32')
    X_te = X[400:].astype('float32')

    # and labels
    y_tr = y[:100].astype('int32')
    y_val = y[100:400].astype('int32')
    y_te = y[400:].astype('int32')

    #plt.scatter(X_tr[:,0], X_tr[:,1], s=40, c=y_tr, cmap=plt.cm.Spectral)
    return X_tr, y_tr, X_val, y_val

class Net(nn.Module):

    def __init__(self, actions, num_features, num_output, layer_limit): 
        super(Net, self).__init__()
      

        max_layers = 5
        if max_layers < layer_limit:
            print('Maximum layers that ChildNet accepts is '.format(max_layers))
            raise 

        from copy import deepcopy
        hidd_units_layers = deepcopy(actions)
        hidd_units_layers[-1] = num_output
        self.nb_layers = len(hidd_units_layers)

        while len(hidd_units_layers)<max_layers:
            hidd_units_layers.append(64)

        self.l_1 = nn.Linear(in_features=num_features, 
                          out_features=hidd_units_layers[0],
                          bias=True)
        
        self.l_2 = nn.Linear(in_features=hidd_units_layers[0], 
                          out_features=hidd_units_layers[1],
                          bias=True)

        self.l_3 = nn.Linear(in_features=hidd_units_layers[1], 
                          out_features=hidd_units_layers[2],
                          bias=True)

        self.l_4 = nn.Linear(in_features=hidd_units_layers[2], 
                          out_features=hidd_units_layers[3],
                          bias=True)

        self.l_5 = nn.Linear(in_features=hidd_units_layers[3], 
                          out_features=hidd_units_layers[4],
                          bias=True)

        self.layers = [self.l_1, self.l_2, self.l_3, self.l_4, self.l_5]
    
    def forward(self, x):
        for i,layer in enumerate(self.layers):
            if i < self.nb_layers-1:
                #print(layer)
                x = F.relu(layer(x))
            elif i < self.nb_layers:
                x = layer(x)   
                #print('here', layer)       
        #assert False
        return x 

def accuracy(ys, ts):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = torch.eq(ts.long(), torch.max(ys, 1)[1])
    # averaging the one-hot encoded vector
    return torch.mean(correct_prediction.float())
    
def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

class ChildNet():

    def __init__(self, layer_limit):
        self.criterion = nn.CrossEntropyLoss()

        X_tr, y_tr, X_val, y_val = create_dataset()
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_val = X_val
        self.y_val = y_val
        
        self.num_features = X_tr.shape[-1]
        self.num_output = 2
        self.layer_limit = layer_limit

    def compute_reward(self, hidd_units_layers, num_epochs):
        # store loss and accuracy for information
        train_losses = []
        net = Net(hidd_units_layers, self.num_features, self.num_output, self.layer_limit)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        max_val_acc = 0
        
        # get training input and expected output as torch Variables and make sure type is correct
        tr_input = Variable(torch.from_numpy(self.X_tr))
        tr_targets = Variable(torch.from_numpy(self.y_tr))

        # get validation input and expected output as torch Variables and make sure type is correct
        val_input = Variable(torch.from_numpy(self.X_val))
        val_targets = Variable(torch.from_numpy(self.y_val))

        # training loop
        for e in range(num_epochs):

            # zeroize accumulated gradients in parameters
            optimizer.zero_grad()
            # predict by running forward pass
            tr_output = net(tr_input)
            # compute cross entropy loss
            tr_loss = self.criterion(tr_output.float(), tr_targets.long())
            train_losses.append(tr_loss.data.numpy())
            # compute gradients given loss
            tr_loss.backward()
            #print(net.l_1.weight.grad)
            # update the parameters given the computed gradients
            optimizer.step()
            
            if num_epochs - e < 6:
                #AFTER TRAINING

                # predict with validation input
                val_output = net(val_input)
                # compute loss and accuracy
                val_loss = self.criterion(val_output.float(), val_targets.long())
                val_acc = accuracy(val_output, val_targets)
                val_acc = float(val_acc.numpy())
                
                if val_acc > max_val_acc:
                	max_val_acc = val_acc
                
                #reset weights
                net.apply(weight_reset)
            
        #return float(val_acc.numpy()), num_epochs, val_loss, train_losses, val_output, val_targets #-float(val_loss.detach().numpy()) 
        return max_val_acc**3 #-float(val_loss.detach().numpy()) 
