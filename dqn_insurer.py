import os
import abcEconomics as abce
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
from contract import Contract

class DQNInsurer(abce.Agent):
    def init(self, algorithm, weights, state_space, action_space, cash, hyperparameters, scaling_constants):
        """
        DQN insurer agents learn a policy network to insure risks.
        
        Arguments:
            - algorithm, the DQN insurer's reinforcement learning algorithm (string)
            - weights, the initial weights (string) for the policy network
            - state_space, the dimensions (integer) of the observational state space
            - action_space, a list of the DQN insurer agent's actions (scalars)
            - cash, the DQN insurer's initial cash value (scalar)
            - hyperparameters, a dictionary of hyperparameters (scalars) containing the number of hidden
              dense layers in the model ("n_hidden_layers"), a list of the number of nodes in each layer of the
              model ("n_hidden_nodes"), the exploration probability ("epsilon"), the minimum exploration 
              probability ("min_epsilon"), the rate of exploration decay ("decay"), the minibatch size 
              ("batch_size"), the learning rate ("learning_rate"), the discount factor ("gamma"), and the 
              prioritised experience replay parameter ("alpha")
            - scaling_constants, a dictionary of scaling constants (scalars) for the monthly market capacity 
              ("market_capacity_scale"), the monthly number of risks ("n_risks_scale"), the capacity of a 
              risk ("risk_capacity_scale"), the reward ("reward_scale", "reward_shift"), and the length of a
              risk ("risk_length_scale")
        """
        self.type = "dqn_insurer"
        self.algorithm = algorithm
        self.initial_weights = weights
        
        self.create("cash", cash)
        self.bankrupt = False
        self.requests = []
        self.contracts = []
        self.claims = {}
        
        self.state_space = state_space
        self.action_space = action_space
        self.memory = {}
        self.priority = {}
        self.prioritise_queue = []
        
        self.market_capacity_scale = scaling_constants.get("market_capacity_scale")
        self.n_risks_scale = scaling_constants.get("n_risks_scale")
        self.risk_capacity_scale = scaling_constants.get("risk_capacity_scale")
        self.reward_scale = scaling_constants.get("reward_scale")
        self.reward_shift = scaling_constants.get("reward_shift")
        self.risk_length_scale = scaling_constants.get("risk_length_scale")                      
        
        self.n_hidden_layers = hyperparameters.get("n_hidden_layers")
        self.n_hidden_nodes = hyperparameters.get("n_hidden_nodes")
            
        self.epsilon = hyperparameters.get("epsilon")
        self.min_epsilon = hyperparameters.get("min_epsilon")
        self.decay = hyperparameters.get("decay")
        self.batch_size = hyperparameters.get("batch_size")
        self.learning_rate = hyperparameters.get("learning_rate")
        self.gamma = hyperparameters.get("gamma")
        self.alpha = hyperparameters.get("alpha")

        self.state = None
        self.policy = self.model()
        self.target = self.model()
        self.action = None
        
        self.measure_time = []
        self.measure_action = []
        self.measure_state = []
        self.measure_reward = []
        self.measure_capacity = []
        
    def model(self):
        """
        Compile the DQN insurer's policy network for predicting state-action values.
        """        
        model = Sequential()
        for layer in range(self.n_hidden_layers):
            if layer == 0:
                model.add(Dense(self.n_hidden_nodes[layer], input_dim=self.state_space, activation="relu"))
            else:
                model.add(Dense(self.n_hidden_nodes[layer], activation="relu"))
            
        model.add(Dense(len(self.action_space), activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model
    
    def observe(self, features):
        """
        Observe the state of the environment on the current time-step.
        
        Arguments,
            - features, a dictionary of whole market features (scalars) on the current time-step containing the 
                        total number of risks incepted ("n_risks") and the total market capacity 
                        ("market_capacity")
        """        
        n_risks = len(features.get("risk_ids"))
        n_risks *= self.n_risks_scale
        
        market_capacity = features.get("market_capacity")
        if market_capacity != 0:
            market_capacity *= self.market_capacity_scale
                
        self.requests = self.get_messages("request_quote")
        for request in self.requests:
            risk = request["risk"]
            risk_capacity = risk.capacity
            if risk_capacity != 0:
                risk_capacity *= self.risk_capacity_scale
            
            risk_length = risk.length
            risk_length *= self.risk_length_scale
            
            observation = [n_risks] + [market_capacity] + [risk_capacity] + [risk_length]
            observation = np.reshape(np.array(observation), [1, self.state_space])
            self.state = observation
    
    def quote(self):
        """
        DQN insurer agents quote all customer agents seeking coverage a premium, in the form of a contract, to 
        insure their risks. 
        """
        for request in self.requests:
            self.measure_time.append(self.time)
            if np.random.rand() <= self.epsilon and self.algorithm != "PASSIVE":
                self.action = np.random.randint(len(self.action_space))
                self.measure_action.append(self.action)
            else:
                q_values = self.policy.predict(self.state)
                self.action = np.argmax(q_values)
                self.measure_action.append(self.action)
                
            def premium_formula(risk):
                capacity = risk.capacity
                premium = np.abs(np.round(capacity * self.action_space[self.action])).item()
                return premium

            risk = request["risk"]
            premium = premium_formula(risk)
            insurer = (self.type, self.id)
            customer = request["customer"]

            contract = Contract(risk, premium, insurer, customer)
            self.send(("customer", customer), "quote", {"contract": contract})

            risk_id = risk.risk_id
            self.memory.update({str(risk_id): {"state": self.state, "action": self.action, "reward": 0}})

            if self.memory.get(str(risk_id - 1)):
                update = self.memory.get(str(risk_id - 1))
                update["next_state"] = self.state
                self.memory.update({str(risk_id - 1): update})
            
                if self.algorithm == "DQNPER" or self.algorithm == "DDQNPER":
                    self.prioritise_queue.append(str(risk_id - 1))
                                
    def underwrite(self):
        """
        DQN insurer agents receive premiums and underwrite risks for all subscribed contracts.
        """
        subscriptions = self.get_messages("subscription")
        for subscription in subscriptions:
            contract = subscription["contract"]
            self.contracts.append(contract)
            
            premium = contract.premium
            self.create("cash", premium)
            reward = premium
                        
            risk = contract.risk
            claims = risk.claims
            for claim in claims:
                claim_date = pd.to_datetime(claim[0])
                value = int(self.claims.get(str(claim_date)) or 0) + claim[1]
                self.claims.update({str(claim_date): value})
                reward -= claim[1]
            
            reward += self.reward_shift
            reward *= self.reward_scale

            self.measure_reward.append(reward)
            self.measure_capacity.append(risk.capacity)
            
            risk_id = risk.risk_id
            update = self.memory.get(str(risk_id))
            update["reward"] = reward
            self.memory.update({str(risk_id): update})
                            
        rejections = self.get_messages("rejection")
        for rejection in rejections:
            contract = rejection["contract"]
            premium = contract.premium
            reward = premium
            
            risk = contract.risk
            claims = risk.claims
            for claim in claims:
                reward -= claim[1]
            
            reward -= self.reward_shift
            reward *= -self.reward_scale
            
            self.measure_reward.append(reward)
            self.measure_capacity.append(0)
            
            risk_id = risk.risk_id
            update = self.memory.get(str(risk_id))
            update["reward"] = reward
            self.memory.update({str(risk_id): update})
            
        if self.algorithm == "DQNPER" or self.algorithm == "DDQNPER":
            for risk_id in self.prioritise_queue:
                self.prioritise(risk_id)
                
            self.prioritise_queue = []
                    
    def payout(self):
        """
        DQN insurer agents make payouts for claims and terminate contracts.
        """
        cash = self["cash"]
        value = int(self.claims.get(str(self.time)) or 0)
        if value < cash:
            self.destroy("cash", value)
        else:
            self.destroy("cash")
        
        contracts = list(self.contracts)
        for contract in contracts:
            contract.time_step() 
            if contract.terminate:
                self.contracts.remove(contract)
        
    def replay(self):
        """
        Sample a minibatch of the history and perform a gradient descent step on the target network.
        """
        if self.algorithm != "PASSIVE":
            if len(self.memory) > self.batch_size:
                if self.algorithm == "DQNPER" or self.algorithm == "DDQNPER":
                    minibatch = self.priority_minibatch()
                else:
                    minibatch = np.random.choice(list(self.memory.keys()), size=self.batch_size)
                
                for risk_id in minibatch:
                    sample = self.memory[risk_id]
                    state = sample["state"]
                    action = sample["action"]
                    reward = sample["reward"]
                    try:
                        next_state = sample["next_state"]
                    except:
                        continue

                    if self.algorithm == "DQN" or self.algorithm == "DQNPER":
                        next_reward = self.target.predict(next_state)[0]
                        target_reward = reward + self.gamma * np.max(next_reward) 

                    if self.algorithm == "DDQN" or self.algorithm == "DDQNPER":
                        next_reward = self.policy.predict(next_state)[0]
                        target_reward = reward + self.gamma * self.target.predict(next_state)[0][np.argmax(next_reward)]
                    
                    target = self.policy.predict(state)
                    target[0][action] = target_reward
                    
                    if self.algorithm == "DQN" or self.algorithm == "DDQN":
                        self.policy.fit(state, target, epochs=1, verbose=0)

                    if self.algorithm == "DQNPER" or self.algorithm == "DDQNPER":
                        importance = np.array([self.memory[risk_id].get("importance") ** (1 - self.epsilon)])
                        self.policy.fit(state, target, sample_weight=importance, epochs=1, verbose=0)

                    self.measure_state.append(state)
                
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
    
    def prioritise(self, risk_id):
        """
        Prioritise an experience sample in the replay buffer.
        
        Arguments:
            - risk_id, the risk ID (integer)
        """
        sample = self.memory[risk_id]
        state = sample["state"]
        action = sample["action"]
        reward = sample["reward"]
        next_state = sample["next_state"]
                
        next_reward = self.policy.predict(next_state)[0]
        target_reward = reward + self.gamma * np.max(next_reward)
        current_reward = self.policy.predict(state)[0][action]
        priority = (np.abs(target_reward - current_reward) + np.exp(-10)) ** self.alpha
        self.priority.update({str(risk_id): priority})
        
    def priority_minibatch(self):
        """
        Return a minibatch, with importance values, sampled from the priority distribution.
        """
        sum_priority = np.sum(list(self.priority.values()))
        distribution = list(self.priority.values()) / sum_priority
        minibatch = np.random.choice(list(self.priority.keys()), size=self.batch_size, p=distribution)
        
        for risk_id in minibatch:
            importance = sum_priority / (self.priority.get(str(risk_id)) * len(self.priority))
            update = self.memory.get(str(risk_id))
            update["importance"] = importance
            self.memory.update({str(risk_id): update})
                                         
        return minibatch
    
    def initialise(self, path):
        """
        Load the initial policy network weights and save the model architecture.
        
        Arguments:
            - path, the path (string) to load the initial weights and save the model
        """
        if self.initial_weights:
            self.policy.load_weights(path + "model/weights_" + self.initial_weights + ".hdf5")
            self.target.load_weights(path + "model/weights_" + self.initial_weights + ".hdf5")
        
        self.policy.save(path + "model/model")
        tf.keras.utils.plot_model(self.policy, to_file=path + "plot/model" + str(self.id) + ".png", 
                                  show_shapes=True, show_layer_names=True)
                    
    def load(self, path_a, path_b, epsilon):
        """
        Load the policy network weights and exploration constant.
        
        Arguments:
            - path_a, the path (string) to load the initial weights from
            - path_b, the path (string) to load the most recent weights from
            - epsilon, the exploration constant (scalar)
        """
        if self.algorithm == "PASSIVE":
            self.policy.load_weights(path_a + self.initial_weights + ".hdf5")
        else:
            self.policy.load_weights(path_a + path_b + str(self.id) + ".hdf5")
            self.target.load_weights(path_a + path_b + str(self.id) + ".hdf5")
        
        self.epsilon = epsilon

    def save(self, path):
        """
        Save the current policy network weights and return the exploration constant.
        
        Arguments:
            - path, the path (string) to save the weights under
        """
        if self.algorithm != "PASSIVE":
            self.policy.save_weights(path + str(self.id) + ".hdf5")
        
        return self.epsilon

    def measure(self, path):
        """
        Save the DQN insurer agent's actions and states over the course of the simulation
        
        Arguments:
            - path, the path (string) to save the files under
        """
        action_df = pd.DataFrame({"time": self.measure_time, 
                                  self.type + str(self.id): self.measure_action,
                                  "reward": self.measure_reward,
                                  "capacity": self.measure_capacity})
        action_df.to_csv(path + "_" + self.type + str(self.id) + "_actions.csv")

        states = np.array(self.measure_state).reshape(len(self.measure_state), self.state_space)
        np.save(path + "_" + self.type + str(self.id) + "_states.npy", states)
