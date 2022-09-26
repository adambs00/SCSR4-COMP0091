class Contract:

    def __init__(self, risk, premium, insurer, customer):
        """
        Contracts are held between a customer agent and an insurer agent to insure a risk.
        
        Arguments:
            - risk, a risk object (Risk)
            - premium, a premium value (scalar)
            - insurer, a tuple consisting of the insurer type (string) and the insurer agent ID (integer)
            - customer, the customer agent ID (integer)
        """
        self.risk = risk
        self.length = risk.length
        self.premium = premium
        self.insurer = insurer
        self.customer = customer
        self.terminate = False

    def time_step(self):
        """
        Check whether or not the contract has expired.
        """
        self.length -= 1
        if self.length == 0 or self.risk.capacity == 0:
            self.terminate = True
