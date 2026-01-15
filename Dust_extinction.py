class DustExtinction:

    """This class calculates the dust extinction magnitude """

    def __init__(self, filter, colour_excess):

        """Initialise with R_V and colour_excess E(B-V), band
        wavelengths, and filter of choice."""

        self.R_V = 3.1
        self.lambda_B = 0.43 #micrometres
        self.lambda_V = 0.55 
        self.colour_excess = colour_excess
        self.filter = filter
        if self.filter != "B" and self.filter != "V":
            raise ValueError(f"The band filter must be either B or V")

    def compute_extinction_mag(self):

        """Compute dust extinction magnitude in B & V filters"""

        #V-band extinction
        A_V = self.R_V * self.colour_excess

        if self.filter == "V":
            return A_V
        
        #B-band extinction
        x = 1 / self.lambda_B

        y = x - 1.82
  
        a = 1 + 0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) + \
        0.72085*(y**4) +  0.01979*(y**5) - 0.77530*(y**6) + \
        0.32999*(y**7)

        b =  1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) - \
        5.38434*(y**4) - 0.62251*(y**5) + 5.30260*(y**6) - \
        2.09002*(y**7)

        A_B = A_V * (a + b / self.R_V)

        return A_B

    