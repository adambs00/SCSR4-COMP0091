import abcEconomics as abce
import numpy as np
import pandas as pd
from risk import Risk
from contract import Contract


class Insurer(abce.Agent):
    def init(self, premium_formula, cash):
        """
        Insurer agents insure risks.
        
        Arguments:
            - premium_formula, a function of a risk object (Risk) that returns a premium (scalar)
            - cash, the insurer's initial cash value (scalar)
        """
        self.type = "insurer"
        
        self.premium_formula = premium_formula

        self.create("cash", cash)
        self.bankrupt = False
        self.contracts = []
        self.claims = {}
                
        self.measure_time = []
        self.measure_action = []
        self.measure_capacity = []

    def quote(self):
        """
        Insurer agents quote all customer agents seeking coverage a premium, in the form of a contract, to 
        insure their risks. 
        """
        requests = self.get_messages("request_quote")
        for request in requests:
            risk = request["risk"]
            premium = self.premium_formula(risk)
            insurer = (self.type, self.id)
            customer = request["customer"]

            contract = Contract(risk, premium, insurer, customer)
            self.send(("customer", customer), "quote", {"contract": contract})
            
            action = premium / risk.capacity
            self.measure_time.append(self.time)
            self.measure_action.append(action)

    def underwrite(self):
        """
        Insurer agents receive premiums and underwrite risks for all subscribed contracts.
        """        
        subscriptions = self.get_messages("subscription")
        for subscription in subscriptions:
            contract = subscription["contract"]
            self.contracts.append(contract)

            premium = contract.premium
            self.create("cash", premium)
            
            risk = contract.risk            
            claims = risk.claims
            for claim in claims:
                claim_date = pd.to_datetime(claim[0])
                value = int(self.claims.get(str(claim_date)) or 0) + claim[1]
                self.claims.update({str(claim_date): value})
            
            self.measure_capacity.append(risk.capacity)
        
        rejections = self.get_messages("rejection")
        for rejection in rejections:
            self.measure_capacity.append(0)

    def payout(self):
        """
        Insurer agents make payouts for claims and terminate contracts.
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

    def measure(self, path):
        """
        Save the insurer agent's actions over the course of the simulation.
        
        Arguments:
            - path, the path (string) to save the file under
        """
        measure_df = pd.DataFrame({"time": self.measure_time,
                                   self.type + str(self.id): self.measure_action,
                                   "capacity": self.measure_capacity})
        measure_df.to_csv(path + "_" + self.type + str(self.id) + "_actions.csv")
