import torch
from torch.autograd import Variable

def indexes_to_actions(actions, batch_size, total_actions):
    ''' Transform the index of the predicted actions to their corresponding hyperparameter, 
    s.t. the ChildNet can be trained with this architecture. '''
    
    batch_hid_units = []
    batch_act_functions = []
    batch_index_eos = []
    
    for b in range(batch_size):
        batch_actions = actions[b,:]                
        hid_units = [total_actions[int(action)] for i,action in enumerate(batch_actions)]
        
        #cut when 'EOS' is reached
        try:
            index_eos = hid_units.index('EOS')
            hid_units = hid_units[:index_eos + 1]
        except ValueError:
            hid_units = hid_units[:-1] + ['EOS']
            index_eos = len(hid_units)
    
        batch_hid_units.append(hid_units)
        batch_index_eos.append(index_eos)
        
    return batch_hid_units, batch_index_eos
    
def fill_tensor(tensor_to_fill, size, ones=True):
    '''Fill a tensor with zeros or ones at the end. Useful because the policy generates architecture with 
    different number of layers. ''' 
    
    if len(tensor_to_fill.size()) >= 2: #dim >= 2
        tensor_to_fill = tensor_to_fill.view(-1)
    
    if ones:
        #fill with ones, useful if afterwards it will be applied log (log 1 = 0, log 0 = -inf)
        size_remaining = size - tensor_to_fill.size()[0]
        return torch.cat((tensor_to_fill, torch.ones(size_remaining)))
    else:        
        size_remaining = size - tensor_to_fill.size()[0]
        return torch.cat((tensor_to_fill, torch.zeros(size_remaining)))
    
    
def get_variable(inputs, cuda=False, **kwargs):
    '''Variable on GPU or CPU, depending on the availability of cuda. '''
    
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out