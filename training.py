import torch
from childNet import ChildNet
from utils import fill_tensor, indexes_to_actions
from torch.autograd import Variable

def training(policy, batch_size, total_actions, verbose = False, num_episodes = 500):
    ''' Optimization/training loop of the policy net. Returns the trained policy. '''
    
    # training settings
    decay = 0.9
    training = True
    
    # childNet
    cn = ChildNet(policy.layer_limit)
    nb_epochs = 100
    
    # train policy network
    training_rewards, val_rewards, losses = [], [], []
    baseline = torch.zeros(15, dtype=torch.float)
    
    print('start training')
    for i in range(num_episodes):
        if i%100 == 0: print('Epoch {}'.format(i))
        rollout, batch_r, batch_a_probs = [], [], []
        #forward pass
        with torch.no_grad():
            prob, actions = policy(training)
        batch_hid_units, batch_index_eos = indexes_to_actions(actions, batch_size, total_actions)
        
        #compute individually the rewards
        for j in range(batch_size):
            # policy gradient update 
            if verbose:
                print(batch_hid_units[j])
            r = cn.compute_reward(batch_hid_units[j], nb_epochs)**3
            if batch_hid_units[j]==['EOS']:
                r -= -1
            a_probs = prob[j, :batch_index_eos[j] + 1]

            batch_r += [r]
            batch_a_probs += [a_probs.view(1, -1)] 

        #rearrange the action probabilities
        a_probs = []
        for b in range(batch_size):
            a_probs.append(fill_tensor(batch_a_probs[b], policy.n_outputs, ones=True))
        a_probs = torch.stack(a_probs,0)

        #convert to pytorch tensors --> use get_variable from utils if training in GPU
        batch_a_probs = Variable(a_probs, requires_grad=True)
        batch_r = Variable(torch.tensor(batch_r), requires_grad=True)
        
        # classic traininng steps
        loss = policy.loss(batch_a_probs, batch_r, torch.mean(baseline))
        policy.optimizer.zero_grad()  
        loss.backward()
        policy.optimizer.step()

        # actualize baseline
        baseline = torch.cat((baseline[1:]*decay, torch.tensor([torch.mean(batch_r)*(1-decay)], dtype=torch.float)))
        
        # bookkeeping
        training_rewards.append(torch.mean(batch_r).detach().numpy())
        losses.append(loss.item())
        
        # print training
        if verbose and (i+1) % val_freq == 0:
            print('{:4d}. mean training reward: {:6.2f}, mean loss: {:7.4f}'.format(i+1, np.mean(training_rewards[-val_freq:]), np.mean(losses[-val_freq:])))

    print('done training')  
 
    return policy