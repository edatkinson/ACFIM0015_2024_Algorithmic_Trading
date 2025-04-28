


'''
File containing all of the tools (functions & classes) I have made to make the process of retrieving data easy. And also because I am bored af, and think this could be a good use of my time.
Look into using parallel processing tools like dask or polars or h5py for high-speed data manipulation.

New tools:
- Review Lectures from this week on statistical analysis
- Get the data from the market simulations and make tools which do many different stats methods.
- Make visualisation tools.


Understanding the anonymised LOB:
Layout:
    Time => ["Bid:", #nBids ,(bid, Quantity)*nBids] <-> ["Ask:", #nAsks, (ask, Quantity)*nAsks]

    The anonymized LOB is a list structure, with the bids and asks each sorted into
    ascending order (so the best bid is the last item on the bid LOB, but the best ask is the
    first item on the ask LOB). Each item in the list is a pair of items: a price, and the number
    of shares bid or offered at that price. 

    With new orders entering and existing orders fulfulled, the gap between best bid and best ask evolves over time depending on liquidity (volume of asks/bids)

Blotters:
[TID, numtrades]
Individual Trades
[TID, "Trade", Time, BuyerTID/SellerTID, BuyerTID/SellerTID, Q] (2 ways to make a trade: Buyer matches ask or seller matches bid)


# Currently tape_dump only writes a list of transactions (ignores cancellations)

Traders:
Trader_spec = (<trader type>, <number of this type of trader>, optionally: <params for this type of trader>)

If a supply or demand schedule mode is "random" and more than one range is supplied in ranges[],
then each time a price is generated one of the ranges is chosen equiprobably and
the price is then generated uniform-randomly from that range

if len(range)==2, interpreted as min and max values on the schedule, specifying linear supply/demand curve
if len(range)==3, first two vals are min & max, third value should be a function that generates a dynamic price offset
                  -- the offset value applies equally to the min & max, so gradient of linear sup/dem curve doesn't vary
if len(range)==4, the third value is function that gives dynamic offset for schedule min,
                  and fourth is a function giving dynamic offset for schedule max, so gradient of sup/dem linear curve can vary


Made to be run in the root directory (the same directory as where the script is using the functions and classes)
'''
# from BSEv1_9 import market_session
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import shutil
import logging
from collections import Counter


import seaborn as sns
import multiprocessing as mp
from BSEv1_9 import market_session
"""
<---------------------------- Market Tools -------------------------------->
"""



class RunMarket:
    def __init__(self, trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose):
        self.trial_id = trial_id
        self.start_time = start_time
        self.end_time = end_time
        self.traders_spec = traders_spec
        self.order_sched = order_sched
        self.dump_flags = dump_flags
        self.verbose = verbose
        
    
    def run_session(self, sesh=0):
        """Run a single market session"""
        # trial = self.trial_id + '_'
        # trial = trial + str(sesh) #need to replace the final digit instead of add.
        trial = f'{self.trial_id}_{sesh}'
        args = (trial,self.start_time, self.end_time, self.traders_spec, self.order_sched, self.dump_flags, self.verbose)
        print(f"Starting session: {trial}")  # or use a logging framework
        result = market_session(*args)
        print(f"End session: {trial}")  # or use a logging framework
        return result
    
    # def multiple_sessions(self, num_sessions=1):
    #     """Run multiple market sessions"""
    #     for sesh in range(num_sessions):
    #         self.run_session(sesh=sesh)

    #     return None
    
    def multiple_sessions(self, num_sessions=1):
        """Run multiple market sessions in parallel using multiprocessing"""

        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Pair the scenario instance and session index for each run

            pool.map(self.run_session, [sesh for sesh in range(num_sessions)])

        return None


class SetUpScenario:
    """
    - Sets up a market scenario 
    - Idea is to run this for many permutations of market dynamics for different trading algorithms.
    - Manage the data ready for statistical analysis.

    - Need to make more accessible for testing

    obj.run_scenario(): Runs market session for this scenario.
        - Args: num_sessions 
    
    obj.folderiser(): Organises scenario data into separate files, to work for iteratice runs of different scenarios.

    obj.organise_data(): Will call the getData class to organise the big data outputted by the current scenario market session.
    - Updates: pointing to the correct folders for data analysis, then making new folders to store analysed data, ready for statistical analysis

    """

    def __init__(self, 
                 days,
                 start_time, 
                 end_time,
                 chart_range,
                 step_mode, #fixed
                 order_interval,
                 time_mode, # drip-poission,fixed,jitter
                 seller_spec, 
                 buyer_spec,
                 proptrader_spec,
                 scenario_id, 
                 dump_flags,
                 verbose=False):
        
        #Validation Checks (incomplete for now)

        # Check times
        if not isinstance(start_time, (int,float)) or not isinstance(end_time, (int,float)) or start_time >= end_time:
            raise ValueError(f"Invalid start/end times: start_time ({start_time}) must be an integer and less than end_time ({end_time})")
        
        if isinstance(chart_range, tuple):
            chart_range = [chart_range]

 
        #Define params
        self.days = days
        self.start_time = start_time
        self.end_time = end_time * days
        self.step_mode = step_mode
        self.order_interval = order_interval
        self.time_mode = time_mode
        self.seller_spec = seller_spec
        self.buyer_spec = buyer_spec
        self.proptrader_spec = proptrader_spec
        self.scenario_id = scenario_id
        self.dump_flags = dump_flags
        self.verbose = verbose

        self.chart_range = chart_range # {'supply': (min,min,func), 'demand':(max,max,func)}
        
        self.supply_schedule = [{'from': self.start_time, 'to': self.end_time, 'ranges': [self.chart_range['supply']], 'stepmode': self.step_mode}] 
        self.demand_schedule = [{'from': self.start_time, 'to': self.end_time, 'ranges': [self.chart_range['demand']], 'stepmode': self.step_mode}] 
        
        self.order_schedule = {'sup': self.supply_schedule, 'dem': self.demand_schedule,
            'interval': self.order_interval, 'timemode': self.time_mode}

        self.trader_spec = {'sellers': self.seller_spec, 'buyers': self.buyer_spec, 'proptraders': self.proptrader_spec}

        self.args = (self.scenario_id, self.start_time, self.end_time, self.trader_spec, self.order_schedule, self.dump_flags, self.verbose)

    def dynamics(self):
        """
        Here we will implement certain market dynamics by:
        - Splitting up the start and end time into random chunks centered around a mean to facilitate shock ranges for the demand and supply schedule.
        """
        return None
    

    def run_scenario(self, num_sessions):
        """
        Runs market simulations for num_sessions, organises data 
        """
        market_session = RunMarket(*self.args)
        # market_session.multiple_sessions(num_sessions=num_sessions)
        market_session.run_session(num_sessions)
        return self.organise_data()
    

    def organise_data(self):
        """
        
        """
        manage_files(self.dump_flags) # Puts the outputted data into 5 folders for 5 info types.
        # self.folderiser() # Makes folders for the current scenario, and moves files from scenario into these folders.
        """
        The getData class can be called after all the data is ready.
        """
        # grab_data = getData(self.dump_flags, self.scenario_id)
        # t,x = grab_data.tape(flag='tape') #or grab_data.tapes()
        # data = grab_data.avg_bal(flag='avgbals') #or grab_data.avg_bal()
        #or
        # grab_data.analyse_flags(save_files=True)
        # t,p = grab_data.tape(flag='tape')


        return None
    

    
    def __str__(self):

        return f'Chart: {self.scenario_id}'

"""
<---------------------------- Data Tools -------------------------------->

"""


def locate_flag_dir(cwd, flag: str, scenario_id: str):
    """
    Locates the file path of the flag in args
    """
    flag_dir = f'{cwd}/{flag}/{scenario_id}'
    return flag_dir


def save_file(cwd, data: pd.DataFrame, flag:str, scenario_id: str):
    """
    Saves the formatted data for a specified flag

    Add:
    Checks that data is a DataFrame, else write to file.
    """
    flag_dir = locate_flag_dir(cwd, flag, scenario_id)
    analysis_dir = f'{flag_dir}/analysis'
    os.makedirs(analysis_dir, exist_ok=True)
    data.to_csv(f'{analysis_dir}/{flag}_formatted.csv', index=False)


class getData:
    def __init__(self, dump_flags: dict, scenario_id: str):
        """
        Class which extracts important information from the respective directories of the type of data you want.
        """
        self.cwd = os.getcwd()
        self.dump_flags = dump_flags
        self.scenario_id = scenario_id

    def analyse_flags(self, save_files=True):
        flags = check_flags(self.dump_flags)
        for flag in flags:
            if flag == 'tape':
                #Change to save to a file.
                t, p = self.tape(flag)
            elif flag == 'avgbals':
                data = self.avg_bal(flag)
                if save_files == True:  
                    save_file(self.cwd, data, flag, self.scenario_id)
            else:
                print('Flag format not available right now: ', flag) # for now

        return None
                

    def tape(self, flag='tape'):
        """Navigate to the tapes directory, cycle through the files and produce x (time) and y (price)"""
        tape_dir = locate_flag_dir(self.cwd, flag, self.scenario_id)
        tapes = os.listdir(tape_dir) # csv files
        
        prices_list = []
        times_list = []

        for tape in tapes:
            tape_data = pd.read_csv(f'{tape_dir}/{tape}', sep=',', names=['trade', 'time', 'price'])
            prices_list.append(tape_data['price'])
            times_list.append(tape_data['time'])

        prices = np.array(pd.concat(prices_list, ignore_index=True))
        times = np.array(pd.concat(times_list, ignore_index=True))


        return times, prices
    
    def avg_bal(self, flag='avgbals'):
        """
        Navigate to the avg_bal directory, cycle throught the files, analyse data and produce important metrics.

        """

        bals_dir = locate_flag_dir(self.cwd, flag, self.scenario_id)
        bals = os.listdir(bals_dir)
        
        formatted_data = []
        for bal in bals:
            df = pd.read_csv(f'{bals_dir}/{bal}', sep=',', header=None, low_memory=False)
            for _, row in df.iterrows():
                chart, time, bid, ask = row[:4]
                traders_data = row[4:-1]  # Extract trader-specific values
                # Process trader data in groups of 4 (TraderType, TotalProfit, NumTraders, AvgProfit). len trader data=17
                for i in range(0, len(traders_data), 4):
                    trader_type = traders_data.values[i]
                    total_profit = traders_data.values[i+1]
                    num_traders = traders_data.values[i+2]
                    avg_profit = traders_data.values[i+3]
                    
                    formatted_data.append([chart, time, bid, ask, trader_type, total_profit, num_traders, avg_profit])

        big_data = pd.DataFrame(formatted_data, columns=["Chart", "Time","Bid","Ask","Trader", "TotalProfit", "NumTraders", "AvgProfit"])
            
        return big_data
    
    def find_pt1_pt2_indices(self, filepath):
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()

        # Split the line into values
        columns = first_line.split(',')

        # Strip whitespace
        columns = [col.strip() for col in columns]

        # Find the indices of PT1 and PT2
        pt1_idx = columns.index('PT1')
        pt2_idx = columns.index('PT2')

        return pt1_idx, pt2_idx

    def get_pt1_pt2_total_profits(self, flag='avgbals'):
        """
        From avgbals files, extract Chart, Time, Bid, Ask, and TotalProfit for PT1 and PT2.
        """
        bals_dir = locate_flag_dir(self.cwd, flag, self.scenario_id)
        bals = os.listdir(bals_dir)

        formatted_data = []

        for bal in bals:

            df = pd.read_csv(f'{bals_dir}/{bal}', sep=',', header=None, low_memory=False)
            # Find indices of PT1 and PT2
            pt1_idx, pt2_idx = self.find_pt1_pt2_indices(f'{bals_dir}/{bal}')
            
            # Extract desired columns: 0-3 (chart, time, bid, ask), pt1 total profit, pt2 total profit
            selected_cols = [0, 1, 2, 3, pt1_idx+1, pt2_idx+1]
            df_filtered = df.iloc[:, selected_cols].copy()
            df_filtered.columns = ["Chart", "Time", "Bid", "Ask", "PT1_TotalProfit", "PT2_TotalProfit"]
            formatted_data.append(df_filtered)

        # Concatenate all files
        result_df = pd.concat(formatted_data, ignore_index=True)

        return result_df

    
    def blotters(self):
        """Navigate to the blotters directory, cycle throught the files, analyse data and produce important metrics"""
        return None
    
    def LOBframes(self):
        """Navigate to the LOB_frames directory, cycle throught the files, analyse data and produce important metrics"""
        return None
    
    def strats(self):
        """Navigate to the strats directory, cycle throught the files, analyse data and produce important metrics"""
        return None


class AvgBalAnalyser:
    def __init__(self, avgbal_records):
        """
        Arg: Either a dictionary/dataframe from the main program, or a json file saved for analysis
        - Separates the traders into their own respective dataframes.
        -Attr:
         - stats
            - means, std over num sessions for a scenario
            - max profit of trading session, for every session in scenario
            TODO:
                - Trade Expectancy: Average profit or loss per trade by considering win rate and size of win versus losses. 
                - Win rate (blotters)
                - Profit per trade (blotters)
                - Sharpe Ratio 
                - Ratio of Profits to losses (profit factor)
         - plots
         - trader_dfs
        """
        self._computed_values = {}  # Dictionary to store computed attributes
        
        if isinstance(avgbal_records, str):
            if avgbal_records.split('.')[-1] == 'json':
                with open(avgbal_records, 'r') as f:
                    data = json.load(f)
                    return pd.DataFrame(data['test-1'])
            else:
                self.df = pd.read_csv(avgbal_records, sep=',', low_memory=False)
        else:
            self.df = avgbal_records


    def _compute_statistics(self):
        """
        - Lazy computation of statistical metrics (mean, std) for AvgProfit.
        
        - returns stats_dict:
            - dict holds mean and std over the x num_sessions for each scenario.
                - also max_profit of each session
        """
        # print(f"Computing statistics for {self.df['Chart'][0].split('_')[0]} ")
        # stats_dict = {}
        # chart_order = sorted(self.df['Chart'].unique())

        # for key in ["PT1", "PT2"]:
        #     session_group = self.df.groupby('Chart')[f'{key}_TotalProfit'] # Takes for bal each session repeated for scenarios
        #     max_values = session_group.max().reindex(chart_order).values
        #     last_values = session_group.last().reindex(chart_order).values
        #     sorted_values = sorted(session_group.last().values) 
        #     mean_val = np.mean(sorted_values)
        #     std_val = np.std(sorted_values)
        #     unique_bal = [] # each update to the overall balance for each session
        #     ppt = [] # profit per trade

        #     for chart, group in self.df.groupby('Chart', sort=False):
        #         vals = group[f'{key}_TotalProfit']
        #         stepped_vals = vals[vals != vals.shift()].tolist()  # drop repeated values
        #         unique_bal.append(stepped_vals)
        #         even_vals = np.array(stepped_vals[::2])
        #         profit_diffs = np.diff(even_vals)
        #         ppt.extend(profit_diffs)

        #     # zip these up with their corresponding session ids, purposefully to maintian consistency when processing each file for analysis
        #     n_trades = list(zip(chart_order, [len(bal) for bal in unique_bal])) # returns a series of the number of trades, being the number of unique balances
        #     ppt = list(zip(chart_order, ppt))
        #     max_prof = list(zip(chart_order, max_values))
        #     last_bal = list(zip(chart_order, last_values))

        #     stats_dict[key] = {"mean": mean_val, "std": std_val,'n_trades':n_trades, 'ppt':ppt , 'unique_bal': unique_bal, "max_prof": max_prof, "last_bal": last_bal}


        # return stats_dict
        stats_dict = {}

        for key in ["PT1", "PT2"]:
            charts = self.df['Chart'].unique()  # preserves original order

            # Group by 'Chart' without sorting
            session_group = self.df.groupby('Chart', sort=False)[f'{key}_TotalProfit']

            max_values = session_group.max().reindex(charts).values
            last_values = session_group.last().reindex(charts).values
            sorted_values = sorted(last_values)
            mean_val = np.mean(sorted_values)
            std_val = np.std(sorted_values)

            unique_bal = []  # unique balance updates per session
            ppt = []         # profit per trade

            for chart in charts:
                group = self.df[self.df['Chart'] == chart]
                vals = group[f'{key}_TotalProfit']
                stepped_vals = vals[vals != vals.shift()].tolist()  # drop repeats
                unique_bal.append(stepped_vals)

                even_vals = np.array(stepped_vals[::2])
                profit_diffs = np.diff(even_vals)
                ppt.append(profit_diffs)  # store per-chart profits

            # Align everything by chart
            n_trades = list(zip(charts, [len(bal) for bal in unique_bal]))
            # ppt_flat = list(zip(charts, [val for sublist in ppt for val in sublist]))
            ppt = list(zip(charts, [list(l) for l in ppt]))
            max_prof = list(zip(charts, max_values))
            last_bal = list(zip(charts, last_values))
            
            # Store results
            stats_dict[key] = {
                'n_trades': n_trades,
                'ppt': ppt, # so this is a list of the ppt per session
                'max_prof': max_prof,
                'last_bal': last_bal,
                'mean': mean_val,
                'std': std_val
            }
        return stats_dict


    def _generate_plots(self):
        """Lazy computation of plots for visualizing AvgProfit distributions."""
        print("Generating plots...")
        trader_dfs = self.trader_dfs  # Ensure traders are processed
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 10))  
        axes = axes.flatten()

        for ax, (key, df) in zip(axes, trader_dfs.items()):
            
            session_group = df.groupby('Chart')['AvgProfit'].max()
            sorted_values = sorted(session_group.values)

            mean_val = np.mean(sorted_values)
            std_val = np.std(sorted_values)

            sns.histplot(sorted_values, kde=True, stat="density", bins=30, color='gray', alpha=0.4, ax=ax)
            x_range = np.linspace(min(sorted_values), max(sorted_values), 100)
            ax.plot(x_range, stats.norm.pdf(x_range, mean_val, std_val), 'r--', label=f"Normal Fit\nmean={mean_val:.2f}\nstd={std_val:.2f}")
            ax.set_title(f"Trader: {key}", fontsize=12)  # Set title for each subplot
            ax.set_ylabel("Density")
            ax.legend(loc='upper right')
            

        plt.tight_layout()
        plt.show()

        return fig

    def __getattr__(self, attr):
        """Lazy computation of expensive attributes."""
        calculations = {
            "stats": self._compute_statistics,
            "plot": self._generate_plots,
        }

        if attr in calculations:
            self._computed_values[attr] = calculations[attr]()  # Compute and store result
            return self._computed_values[attr]

        raise AttributeError(f"'AvgBalAnalyser' object has no attribute '{attr}'")

# Example Usagea

# avg_bals_file = "/Users/edatkinson/University/4thYear/AlgoTrading/BSE_demos/avgbals/bse_d001_i10_0001_avg_balance.csv"

# data_obj = getData({'dump_blotters': True, 'dump_lobs': False, 'dump_strats': True,
#                           'dump_avgbals': True, 'dump_tape': True}, avg_bals_file.split('/')[-1])

# data_obj.avg_bal()

# analyser = AvgBalAnalyser("/Users/edatkinson/University/4thYear/AlgoTrading/BSE_demos/avgbals/bse_d001_i10_0001_avg_balance.csv")  # Or pass a dictionary
# analyser.trader_dfs 
# analyser.statistics 
# analyser.plots


def check_flags(dump_flags):
    """
    Check which flags are specified
    """
    dirs = []
    #Where dumpflags value == True, make those dirs.
    for key, value in dump_flags.items():
        if value == True:
            dirs.append(key.split('_')[1])
        else:
            continue
    return dirs


def manage_files(dump_flags: dict):
    """
    At this point ALL market sessions have been run. So now we organise each file into their respective directories.
    This makes cycling through the data easier and more logical.

    """
    cwd = os.getcwd()
    
    dirs = check_flags(dump_flags)

    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

    for file in os.listdir(cwd):
        try:
            if file.split('_')[-1] == 'balance.csv':
                os.rename(f'{cwd}/{file}', f'{cwd}/avgbals/{file}')
            elif file.split('_')[-1] == 'blotters.csv':
                os.rename(f'{cwd}/{file}', f'{cwd}/blotters/{file}')
            elif file.split('_')[-1] == 'frames.csv':
                os.rename(f'{cwd}/{file}', f'{cwd}/lobs/{file}')
            elif file.split('_')[-1] == 'tape.csv':
                os.rename(f'{cwd}/{file}', f'{cwd}/tape/{file}')
            elif file.split('_')[-1] == 'strats.csv':
                os.rename(f'{cwd}/{file}', f'{cwd}/strats/{file}')
        except Exception as e:
            print("File does not exist. Please revisit manage_files() in tools.py to fix", e)





"""
<--------------------------- Stat's Tools ---------------------------->
Need to Design Experiments.
Visualise and Analyse the results.

Statistical measures which are important:
- Profitability 
- Risk of loss

Plan;
How can you tell if one algo is better than another?
- You must set up a range of market dynamics which test how the trader performs in different scenarios.
- You must then simulate this many, many, many times.
- Obtain key performance metrics for both traders.
- Compare using statistical methods for each scenario.
- Could average over each scenario for a final statistic which determines the winning strategy.

Framework for this:
- For each simulation, information for traders is dumped in their respective folders.
- Simulation_x:
    - Scenario_1, scenario_2, scenario_3, .... , scenario_x. Each of the scenario's are a singular market simulation of said market dynamics.

- Repeat Simulation_x many, many times to get a huge amount of data for each possible market permutation of BSM.
- This will return information for each scenario, x amount of times.

For each scenario, analyse the profitability and risk of loss for Trading Algo A.

Repeat the whole process for trading Algo B.

Compare the results using statistical analyses to determine if there is a difference in these algorithm's metrics.

OR

Framework:
- Each simulation has a probability distribution of market dynamics, which are happen at a similar rate to real market dynamics.
- Run over many trials will mean all possible outcomes will have happened.



Ideas?

- What is the best data structure to use for this code?


"""
 





"""

How it will work:

X scenarios, each with N number of sessions, For Trader A
X scenarios, each with N number of sessions, for Trader B

2(X*N) Sessions in total.
How many sessions is enough?
How many scenarios are there?

TODO:
- Generate a good amount of permutations for the market dynamics with a good level of customisability.
- Statistically Analyse the in-built traders first, so develop a set of tools to do this, including visualisation.
- Make my own trading algorithm
- Analyse trading algorithm




Need to set up a function or class which generates permutations of market dynamics.
Before I do this I need to decide what the best way to do it is:
- Look at papers and BSE on all the different possibilities.
- Check that this approach is actually valid.
- Rate of new customer orders Order Interval

Furthermore, I need to decide how I am going to analyse the data which will influence the getData class too, so I will leave that for now

"""


"""
<---------------------- Market Permutation Analysis ------------------------>
"""



"""

Parameters for making permutations of the market dynamic:

- Number of traders (fixed, say 32) 1:1 sell and buy. Done
- Types of traders: ZIP, SNPR, AA, GVWY, ZIC, SHVR, mine (7) Done Pick 4 of these
- Order schedule: random, or regular in time. Balance between sell orders and buy orders, how the price of each order is generated.
- Supply Schedule: fixed, random, jittered
- Timemode: time-distribution of customer orders. [drip-fixed, drip-jitter or drip poisson]
- Offset Functions: "Drunkards Walk" refer to readme


---------------------------------------------------------------------------

After these are done, could I make a model-based trader which trains on the various dynamical simulated data to compete with the 
normal traders?
Then use Tim Masters book as a way to verify my model is working or not?


"""



def buyer_seller_ratios(equal_ratio: int, traders):
    """
    Calculates all possible permutations of buyer/seller ratios for N traders,
    ensuring an equal number of buyers and sellers.

    Args:
        equal_ratio: The ratio of buyers to sellers for each trader type.  
                     e.g., 2 means each trader type has twice as many buyers as sellers
        traders: A list of trader names (strings).

    Returns:
        A list of dictionaries. Each dictionary represents a valid configuration
        of buyer/seller ratios, where keys are trader names and values are the 
        number of buyers for that trader type.  Returns an empty list if no valid
        combinations are found.
    """

    n_trader_types = len(traders)
    n_traders = equal_ratio * n_trader_types  # Total number of buyers (and sellers)

    results = []

    # Generate all possible combinations of buyer counts for each trader type.
    # The sum of these counts must equal n_traders. We use a generator to avoid
    # storing all combinations in memory at once.
    for combination in _generate_combinations(n_traders, n_trader_types):
        # Check if the combination is valid (sums to n_traders)
        if sum(combination) == n_traders and all(count >= 1 for count in combination): #Check for at least 1 of each
            buyer_spec = {}
            for i, trader in enumerate(traders):
                buyer_spec[trader] = combination[i]
            results.append(buyer_spec)
    
    results_formatted = []
    for res in results:
        results_formatted.append(list(res.items()))

    return results_formatted


def _generate_combinations(total: int, num_elements: int):
    """
    Generates all possible combinations of non-negative integers that sum to a given total and have a specific number of elements.  Uses recursion.

    Args:
        total: The target sum.
        num_elements: The number of integers in each combination.

    Yields:
        A tuple representing a valid combination.
    """
    if num_elements == 0:
        if total == 0:
            yield ()  # Base case: empty tuple if total is also 0
        return

    if total < 0: # Pruning to avoid unnecessary recursion
        return

    for i in range(total + 1):  # Iterate through possible values for the current element
        for sub_combination in _generate_combinations(total - i, num_elements - 1):
            yield (i,) + sub_combination  # Add current value to the sub-combination





# traders = ['ZIP', 'ZIC', 'SHVR', 'GVWY', 'TraderA']
# equal_ratio = 5
# ratios = buyer_seller_ratios(equal_ratio, traders) #10626 combos in form [('ZIP', 13), ('ZIC', 1), ('SHVR', 1), ('GVWY', 1), ('TraderA', x)] If I ran 50 sessions for each combo, it would take 531,300 scenarios. Definitly not viable.





    
"""

- Order schedule: random, or regular in time. Balance between sell orders and buy orders, how the price of each order is generated.
- Supply Schedule: fixed, random, jittered
- Timemode: time-distribution of customer orders. [drip-fixed, drip-jitter or drip poisson]
- Offset Functions: "Drunkards Walk" refer to readme

"""



# def range_generator(lower_range, upper_range, n_ranges):
#     """
#     Function to generate chart_ranges which vary in complexity
#     n_ranges is the number of different ranges
#     Lower range is a window of min prices
#     Upper range is a window of max prices
#     Increment n_ranges amount in each window to get lower and upper values.
    
#     """

#     #Lower range = [10,20]
#     #Upper range = [100,140]
#     #Pick a random number n times between [minmin, maxmin] and[maxmin, maxmax]
#     #Then combine into a tuple of (min, min, func1, func2), (max,max, func1,func2)
#     #Return n (1x4) tuples


    

    
    
    


# range_generator([90,100],[105,115],3)



# dump_flags = {'dump_blotters': True, 'dump_lobs': True, 'dump_strats': True, 'dump_avgbals': True, 'dump_tape': True}
# sellers_spec = [('ZIP', 10), ('ZIC', 10), ('SHVR', 10), ('GVWY', 10)]
# buyers_spec = sellers_spec
# traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}
# order_interval = 10
# minutes = 10
# start_time = 0
# end_time = 20
# step_mode = 'fixed'
# min_price = 80
# max_price = 320
# time_mode = 'drip-poisson'
# scenario_id = 'test_3'
# verbose = False



    
        


