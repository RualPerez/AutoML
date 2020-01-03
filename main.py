from policy import PolicyNet
from training import training 
import warnings
warnings.filterwarnings("ignore")
import argparse
import torch 

if __name__ == "__main__":
        
    # input parameters
    parser = argparse.ArgumentParser(description='Documentation in the following link: https://github.com/RualPerez/AutoML', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--batch', help='Batch size of the policy (int)', nargs='?', const=1, type=int, default=15)
    parser.add_argument('--max_layer', help='Maximum nb layers of the childNet (int)', nargs='?', const=1, type=int, default=6)
    parser.add_argument('--possible_hidden_units', default=[1,2,4,8,16,32], nargs='*',
                        type=int, help='Possible hidden units of the childnet (list of int)')
    parser.add_argument('--possible_act_functions', default=['Sigmoid', 'Tanh', 'ReLU', 'LeakyReLU'], nargs='*', 
                        type=int, help='Possible activation funcs of the childnet (list of str)')
    parser.add_argument('--verbose', help='Verbose while training the controller/policy (bool)', nargs='?', const=1, 
                        type=bool, default=False)
    parser.add_argument('--num_episodes', help='Nb of episodes the policy net is trained (int)', nargs='?', const=1, 
                        type=int, default=500)
    args = parser.parse_args()
    
    # parameter settings
    args.possible_hidden_units += ['EOS']
    total_actions = args.possible_hidden_units + args.possible_act_functions
    n_outputs = len(args.possible_hidden_units) + len(args.possible_act_functions) #of the PolicyNet
    
    # setup policy network
    policy = PolicyNet(args.batch, n_outputs, args.max_layer)
    
    # train
    policy = training(policy, args.batch, total_actions, args.verbose, args.num_episodes)
    
    # save model
    torch.save(policy.state_dict(), 'policy.pt')