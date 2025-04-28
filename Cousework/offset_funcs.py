import numpy as np
import math
import matplotlib.pyplot as plt
import random
import sys
import math
import random
import os
import time as chrono
import csv
from datetime import datetime


def schedule_offsetfn(t):
    # Gradient is drift (growth) with small gaussian fluctuations
    # Wavelength controls the frequency of market cycles
    # Amplitude is volatility, wild swings etc. Dependent on time - markets become riskier over time (speculation, development and external factors increase price fluctuation)
    # pi2 = math.pi * 2  
    # c = math.pi * 3000  
    # wavelength = t / c #+ random.lognormvariate(0, math.sqrt(t) * 0.1)
    # gradient = 100 * t / (c / pi2) + random.gauss(0, math.log1p(t)) # scales with time 
    # amplitude = 10 * t / (c / pi2) #* random.uniform(0.8, 1.2)
    # offset = gradient + amplitude * math.sin(wavelength * t) + random.gauss(0, 10 + 0.05 * t)
    
    return int((t % 75)/2)#int(round(offset, 0))


def brownian_offset(t, X0):
    r = 0.02  # 2% annual drift
    sigma = 0.1  # 20% annual volatility
    dt = t / (365 * 24 * 60)  # Convert time to years
    W = random.normalvariate(0, math.sqrt(dt))  # Proper Wiener process
    X = X0 * np.exp(sigma * W + (r - 0.5 * sigma**2) * dt)

    # Add occasional market shocks
    # if random.random() < 0.01:
    #     X *= random.uniform(0.95, 1.05)  

    return X #int(round(X,0))




class GeometricBrownianMotion:
    def __init__(self, S0, r, sigma, T=1.0, N=252):
        """
        Derived from this article: https://medium.com/the-quant-journey/a-gentle-introduction-to-geometric-brownian-motion-in-finance-68c37ba6f828

        Name: "Log normal asset return model"

        Generates a path for an initial asset price S0, using the Geometric Brownian Stochastic Differential Equation Solution
        Can I use this as an offset function?

        Build in a variance in r and sigma to have heteroskedasticity, instead of the same drift and volatility rates over and over.

        Try:
        - plotting a kernal density estimation (KDE) distribution for the final path values of each trial. 
        - Introducing more realistic asset price dynamics by letting r and sigma be dependent on time.
        """
        self.S0 = S0 #initial asset price
        self.r = r /(365*60*24) # drift
        self.sigma = sigma /(365*60*24) # volatility / diffusion
        self.T = T #total time period
        self.N = N #number of time steps
        self.dt = T / N #

    def generate_paths(self):
        W = np.random.standard_normal(size=self.N)
        W = np.cumsum(W) * np.sqrt(self.dt)
        time_steps = np.linspace(0, self.T, self.N)  # Create time steps
        X = (self.r - 0.5 * self.sigma ** 2) * time_steps
        X += self.sigma * W
        S = self.S0 * np.exp(X)
        return S, time_steps # Return both the path and time steps
    

def schedule_offsetfn_read_file(filename, col_t, col_p, scale_factor=75):
    """
    Read in a CSV data-file for the supply/demand schedule time-varying price-offset value
    :param filename: the CSV file to read
    :param col_t: column in the CSV that has the time data
    :param col_p: column in the CSV that has the price data
    :param scale_factor: multiplier on prices
    :return: on offset value event-list: one item for each change in offset value
            -- each item is percentage time elapsed, followed by the new offset value at that time
    """
    
    vrbs = False
    
    # does two passes through the file
    # assumes data file is all for one date, sorted in time order, in correct format, etc. etc.
    rwd_csv = csv.reader(open(filename, 'r'))
    
    # first pass: get time & price events, find out how long session is, get min & max price
    minprice = None
    maxprice = None
    firsttimeobj = None
    timesincestart = 0
    priceevents = []
    
    first_row_is_header = True
    this_is_first_row = True
    this_is_first_data_row = True
    first_date = None
    
    for line in rwd_csv:
        
        if vrbs:
            print(line)
        
        if this_is_first_row and first_row_is_header:
            this_is_first_row = False
            this_is_first_data_row = True
            continue
            
        row_date = line[col_t][:10]
        
        if this_is_first_data_row:
            first_date = row_date
            this_is_first_data_row = False
            
        if row_date != first_date:
            continue
            
        time = line[col_t][11:19]
        if firsttimeobj is None:
            firsttimeobj = datetime.strptime(time, '%H:%M:%S')
            
        timeobj = datetime.strptime(time, '%H:%M:%S')
        
        price_str = line[col_p]
        # delete any commas so 1,000,000 becomes 1000000
        price_str_no_commas = price_str.replace(',', '')
        price = float(price_str_no_commas)
        
        if minprice is None or price < minprice:
            minprice = price
        if maxprice is None or price > maxprice:
            maxprice = price
        timesincestart = (timeobj - firsttimeobj).total_seconds()
        priceevents.append([timesincestart, price])
        
        if vrbs:
            print(row_date, time, timesincestart, price)
        
    # second pass: normalise times to fractions of entire time-series duration
    #              & normalise price range
    pricerange = maxprice - minprice
    endtime = float(timesincestart)
    offsetfn_eventlist = []
    for event in priceevents:
        # normalise price
        normld_price = (event[1] - minprice) / pricerange
        # clip
        normld_price = min(normld_price, 1.0)
        normld_price = max(0.0, normld_price)
        # scale & convert to integer cents
        price = int(round(normld_price * scale_factor))
        normld_event = [event[0] / endtime, price]
        if vrbs:
            print(normld_event)
        offsetfn_eventlist.append(normld_event)
    
    return offsetfn_eventlist


def schedule_offsetfn_from_eventlist(time, params):
    """
    Returns a price offset-value for the current time, by reading from an offset event-list.
    :param time: the current time
    :param params: a list of parameter values...
        params[1] is the final time (the end-time) of the current session.
        params[2] is the offset event-list: one item for each change in offset value
                    -- each item is percentage time elapsed, followed by the new offset value at that time
    :return: integer price offset value
    """

    final_time = float(params[0])
    offset_events = params[1]
    # this is quite inefficient: on every call it walks the event-list
    percent_elapsed = time/final_time
    offset = None
    for event in offset_events:
        offset = event[1]
        if percent_elapsed < event[0]:
            break
    return offset


def schedule_offsetfn_increasing_sinusoid(t, params):
    """
    Returns sinusoidal time-dependent price-offset, steadily increasing in frequency & amplitude
    :param t: time
    :param params: set of parameters for the offsetfn: this is empty-set for this offsetfn but nonempty in others
    :return: the time-dependent price offset at time t
    """
    if params is None:  # this test of params is here only to prevent PyCharm from warning about unused parameters
        pass
    scale = -7500
    multiplier = 7500000    # determines rate of increase of frequency and amplitude
    offset = ((scale * t) / multiplier) * (1 + math.sin((t*t)/(multiplier * math.pi)))
    return int(round(offset, 0))



# if __name__ == '__main__':


#     # Example usage:
#     S0 = 100      # Initial price
#     mu = 0.1      # Drift
#     sigma = 0.2   # Volatility
#     T = 1.0       # Time horizon (1 year)
#     N = 252      # Number of time steps (e.g., trading days in a year)

#     gbm = GeometricBrownianMotion(S0, mu, sigma, T, N)

#     plt.figure()

#     num_paths = 5  # Generate 5 different paths
#     for i in range(num_paths):
#         S, time_steps = gbm.generate_paths()
#         plt.plot(time_steps, S)

#     plt.xlabel("Time")
#     plt.ylabel("Asset Price")
#     plt.title("Geometric Brownian Motion Simulation")
#     plt.grid(True)
#     plt.show()



#     # # Example usage:
#     # S0 = 100
#     # mu = 0.1  # Drift
#     # sigma = 0.005  # Volatility

#     # gbm2 = GeometricBrownianMotion(S0, mu, sigma)

#     # # Generate prices at some random times:
#     # num_samples = 100  # Number of random times to sample
#     # T = 1.0  #Example Time Horizon. Could be any value.
#     # random_times = np.sort(np.random.uniform(0, T, num_samples)) #Random times, sorted.

#     # prices = []
#     # for t in random_times:
#     #     price = gbm2.generate_price_at_t(t)
#     #     prices.append(price)

#     # plt.figure()

#     # plt.plot(random_times, prices)  # Scatter plot
#     # plt.xlabel("Time (t)")
#     # plt.ylabel("Asset Price")
#     # plt.title("Geometric Brownian Motion - Prices at Random Times")
#     # plt.grid(True)
#     # plt.show()
