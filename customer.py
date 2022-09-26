import abcEconomics as abce
import pandas as pd

class Customer(abce.Agent):
    def init(self, simulation_parameters, risk):
        """
        Customer agents hold a risk they seek to insure.
        
        Arguments:
            - simulation_parameters, a dictionary containing the number of insurers ("n_insurers") (integer),
                                     the number of DQN insurers ("n_dqn_insurers") (integer), and the total
                                     simulation time ("time") (datetime range)
            - risk, a risk object (Risk)
        """
        self.risk = risk
        self.entry = risk.entry
        self.risk_id = risk.risk_id
        self.insurers = range(simulation_parameters["n_insurers"])
        self.dqn_insurers = range(simulation_parameters["n_dqn_insurers"])
        self.contract = False
        
    def seek(self, risk_id):
        """
        Customer agents enter the market and seek coverage for their risk from all insurer agents.
        
        Arguments:
            - risk_id, the risk ID (integer)
        """
        if self.risk_id == risk_id and not self.contract:
            for insurer_id in self.insurers:
                self.send(("insurer", insurer_id), "request_quote", 
                          {"risk": self.risk, "customer": self.id})
                
            for dqn_insurer_id in self.dqn_insurers:
                self.send(("dqn_insurer", dqn_insurer_id), "request_quote", 
                          {"risk": self.risk, "customer": self.id})
                
    def subscribe(self):
        """
        Customer agents subscribe to the contract with the lowest premium quote.
        """
        quotes = self.get_messages("quote")
        
        if quotes:
            premiums = {quote["contract"].premium: quote["contract"] for quote in quotes}
            chosen_contract = premiums[min(premiums)]
            chosen_insurer_type = chosen_contract.insurer[0]
            chosen_insurer_id = chosen_contract.insurer[1]
        
            for insurer_id in self.insurers:
                if chosen_insurer_type == "insurer" and chosen_insurer_id == insurer_id:
                    self.send(("insurer", insurer_id), "subscription", {"contract": chosen_contract})
                else:
                    self.send(("insurer", insurer_id), "rejection", {"contract": chosen_contract})

            for dqn_insurer_id in self.dqn_insurers:
                if chosen_insurer_type == "dqn_insurer" and chosen_insurer_id == dqn_insurer_id:
                    self.send(("dqn_insurer", dqn_insurer_id), "subscription", {"contract": chosen_contract})
                else:
                    self.send(("dqn_insurer", dqn_insurer_id), "rejection", {"contract": chosen_contract})

            self.contract = True
