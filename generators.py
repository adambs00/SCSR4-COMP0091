import pandas as pd
from risk import Risk


def insurer_generator(n_insurers, premium_formulae, insurer_cash_values):
    """
    Generates a group of insurer agent parameters.
    
    Arguments:
        - n_insurers, the number (integer) of insurer agents
        - premium_formulae, a list of functions that return a premium (scalar) for insuring a risk
        - insurer_cash, a list of the insurer agents' initial cash values (scalars)
    """
    insurer_parameters = []
    for n in range(n_insurers):        
        insurer = {"premium_formula": premium_formulae[n],
                   "cash": insurer_cash_values[n]}
        insurer_parameters.append(insurer)

    return insurer_parameters


def dqn_insurer_generator(n_dqn_insurers, dqn_insurer_algorithms, dqn_insurer_weights,
                          state_space, action_space, dqn_insurer_cash_values, 
                          hyperparameters, scaling_constants):
    """
    Generates a group of DQN insurer agent parameters.
    
    Arguments:
        - n_dqn_insurers, the number (integer) of DQN insurer agents
        - dqn_insurer_algorithms, a list of the DQN insurer agents' reinforcement learning algorithms (strings)
        - dqn_insurer_weights, a list of weights (strings) to initialise the DQN insurer agents
        - state_space, the dimensions (integer) of the observational state space
        - action_space, a list of actions (scalars)
        - dqn_insurer_cash_values, a list of DQN insurer agents' initial cash values (scalars)
        - hyperparameters, a dictionary of hyperparameters (scalars)
        - scaling_constants, a dictionary of scaling constants (scalars)
    """
    dqn_insurer_parameters = []
    for n in range(n_dqn_insurers):
        dqn_insurer = {"algorithm": dqn_insurer_algorithms[n],
                       "weights": dqn_insurer_weights[n],
                       "state_space": state_space, 
                       "action_space": action_space, 
                       "cash": dqn_insurer_cash_values[n],
                       "hyperparameters": hyperparameters,
                       "scaling_constants": scaling_constants}
        dqn_insurer_parameters.append(dqn_insurer)
        
    return dqn_insurer_parameters

