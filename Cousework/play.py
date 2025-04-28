import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import os
import json
import sys
from tools import SetUpScenario, getData, AvgBalAnalyser, check_flags
from offset_funcs import schedule_offsetfn_from_eventlist, schedule_offsetfn_increasing_sinusoid, schedule_offsetfn_read_file
import multiprocessing as mp
import time
import logging
import shutil

"""
To do: 
Cloud Computing is set up, I have run some tests in the ssh platform and it works all good with command line arguments for BTC file.
What I need to do:
    - Transform the process using multiprocessing library so I can parallelise the process.
    - Automate the process of running tests:
        - Download many BTC files.
        - Generate many different market dynamic scenarios in a separate file and save each parameter for them into a yaml file.
    - Main program:
        - Runs market sessions using yaml file inputs and saves them.
    - Testing:
        - Taps into all of the data which is produced.
        - Aim: For the different scenarios (each scenario has 1000 trials) we generate core statistics such as mean, variance (and others) of all performance metrics.
        - Then perform a range of non/not-non parametric hypothesis tests at different SigLevels. Condidence Intervals etc.

Takes about 4mins per session per core   


Understanding BSE:
Simulates the market for x number of days/years based off of BTC data. The BTC data is daily, 5min intervals, so shouldn't we simulate BSE daily if we are using daily data??


"""


# # Set up logging
# logging.basicConfig(
#     filename="simulation.log",  # Save logs to this file
#     filemode="a",  # Overwrite log file each run
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     level=logging.INFO  # INFO level logs progress, DEBUG for more details
# )

def run_scenario_parallel(scenario):
    scenario_id = scenario.scenario_id
    logging.info(f"Starting scenario {scenario_id}...")

    try:
        scenario.run_scenario(num_sessions=100)  # Run the simulation for number of sessions
        logging.info(f"Scenario {scenario_id} completed successfully.")
        return scenario_id
    except Exception as e:
        logging.error(f"Scenario {scenario_id} failed: {str(e)}")
        return None  # Return None to indicate failure

def organise_files_by_scenario(dump_flags):
    """
    Organizes files by their scenario ID into separate folders.
    """
    # list_of_directories = ['avgbals', 'blotters', 'lobs', 'strats', 'tape']
    list_of_directories = check_flags(dump_flags=dump_flags)
    cwd = os.getcwd()

    for dir in list_of_directories:
        path = os.path.join(cwd, dir)  # Full path to directory
        files = os.listdir(path)  # List all files in the directory

        for file in files:
            if "_" in file:  # Ensure filename contains an underscore
                scenario_id = file.split("_")[0]  # Extract scenario ID (first part before "_")
                scenario_folder = os.path.join(path, scenario_id)  # New folder path
                
                os.makedirs(scenario_folder, exist_ok=True)  # Create folder if not exists
                
                src = os.path.join(path, file)  # Source file
                dest = os.path.join(scenario_folder, file)  # Destination file
                
                shutil.move(src, dest)  # Move file

        print(f"Organized files in {dir}.")



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


"""


"""




if __name__ == "__main__":
    num_cores = mp.cpu_count()

    logging.info(f"Starting simulations with {num_cores} CPU cores.")
    

    if len(sys.argv) > 1:
        price_offset_filename = sys.argv[1] # File of data we are using
        name = sys.argv[2] # Name of the chart
    else:
        price_offset_filename = '/Users/edatkinson/University/4thYear/AlgoTrading/BSE_demos/MarketData/offset_BTC_USD_20250210.csv'
        name = 'test' #default name = test

    
    #ALL PARAMS REQUIRED
    
    dump_flags = {'dump_blotters': True, 'dump_lobs': False, 'dump_strats': True, 'dump_avgbals': True, 'dump_tape': True}
    sellers_spec = [('ZIP', 13), ('ZIC', 2), ('SHVR', 5), ('GVWY', 5)]
    buyers_spec = sellers_spec
    proptraders_spec = [('PT1', 1, {'bid_percent': 0.95, 'ask_delta': 7}), ('PT2', 1, {'n_past_trades': 25})]
    traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec, 'proptraders': proptraders_spec}

    # Different trading scenarios
    traders = ['ZIP', 'ZIC', 'GVWY']
    equal_ratio = 40 # so there is one instance where they are equal at 40, then the rest are permutations around this
    list_of_trader_specs = buyer_seller_ratios(equal_ratio, traders) #10626 combos in form [('ZIP', 13), ('ZIC', 1), ('SHVR', 1), ('GVWY', 1), ('TraderA', x)] If I ran 50 sessions for each combo, it would take 531,300 scenarios. Definitly not viable.
    print(list_of_trader_specs, len(list_of_trader_specs))

    
    order_interval = 10
    days = 1000 # 3 years=3*365, So the market is simulated for 3 years.
    start_time = 0
    hours = 24
    end_time = 60*60*hours #then multiplied by days
    step_mode = 'random'
    # chart_range = (min_price, max_price, schedule_offsetfn)
    offsetfn_events = None
    
    if price_offset_filename is not None:
        offsetfn_events = schedule_offsetfn_read_file(price_offset_filename, 0, 1)
    

    chart_range = {'supply': (75,110,(schedule_offsetfn_from_eventlist, [[end_time, offsetfn_events]])), 'demand':(120,300,(schedule_offsetfn_from_eventlist, [[end_time, offsetfn_events]]))}
    
    time_mode = 'drip-poisson'
    scenario_id = 'test-1'
    verbose = False

    scenario_id_list = []


    ### Loop through scenarios ###

    for i in range(1): # number of market scenarios, automates different scenarios (tbd)
        scenario_id = name +'-'+str(i)
        #need a function which defines the new args for the new test
        args = (days, 
            start_time, 
            end_time, 
            chart_range,
            step_mode,
            order_interval, 
            time_mode,
            sellers_spec, 
            buyers_spec,
            proptraders_spec,
            scenario_id, 
            dump_flags,
            verbose)
        
        scenario = SetUpScenario(*args)
        scenario_id_list.append(scenario)


    #Run each scenario with n sessions
    # run_scenario = True
    # if run_scenario == True:
    #     for scenario in scenario_id_list:
    #         scenario.run_scenario(num_sessions=2) # modified for tapes
    
    
    run_scenario = True
    if run_scenario:
        num_cores = min(mp.cpu_count(), len(scenario_id_list))  # Use available cores
        with mp.Pool(num_cores) as pool:
            results = pool.map(run_scenario_parallel, scenario_id_list)

        logging.info(f"All scenarios completed. Results: {results}")

    organise_files_by_scenario()



    ## Data Analysis ##
    # data_analysis = True
    # if data_analysis == True:
            
    #     tape_records = {} 
    #     avgbals_records = {} 

    #     for scenario in scenario_id_list:
    #         id = scenario.scenario_id
    #         flags = scenario.dump_flags
    #         data_obj = getData(flags, id)

    #         #Handle Tapes
    #         times, prices = data_obj.tape() #Concatenated session times and prices for a single scenario
    #         plt.scatter(times, prices, marker='x', color='black')
    #         plt.show()
    #         tape_records[id] = {'times': times.tolist(), 'prices': prices.tolist()}

            # #Handle Avg Bals
            # avgbal_data = data_obj.avg_bal() # DataFrame
            # analyser = AvgBalAnalyser(avgbal_data)
            # trader_df = analyser.trader_dfs
            # trader_stats = analyser.stats # for each scenario, change this to store statistics from each scenario, so i can plot them.
            # # print(trader_stats)
            # analyser.plot


# #     #     list_of_scenario_ids = [scenario.scenario_id for scenario in scenario_id_list]




        


