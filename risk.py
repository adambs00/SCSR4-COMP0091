import numpy as np
import pandas as pd


class Risk:
    def __init__(self, entry, risk_id, capacity, length, claims):
        """
        Risks are held by customer agents and insured by insurer agents.
        
        Arguments: 
            - entry, the inception date (datetime)
            - risk_id, the risk ID (integer) indicating the order of the simulation
            - capacity, the total exposure value (scalar)
            - length, the contract length (integer)
            - claims, a list of claim tuples, which consist of a claim's date (datetime) and its value (scalar)
        """
        self.entry = entry
        self.risk_id = risk_id
        self.capacity = capacity
        self.length = length
        self.claims = claims
