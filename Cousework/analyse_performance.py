import csv
import yaml
import numpy as np
import pandas as pd
from tools import AvgBalAnalyser, getData
import os
import sys
import argparse
from scipy import stats as st


def obtain_key_stats(yaml_file, directory):
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)

    dump_flags = cfg['dump_flags']

    avg_bal_dir = os.path.join(directory, 'avgbals')
    scenario_folder = os.listdir(avg_bal_dir)

    tape_records = {} 
    avgbals_records = {} 

    track_PT = {
        'PT1' : {'mean': 0, 'std': 0, 'n_trades': 0 ,'max_prof': 0, 'last_bal': 0}, #Mean & std over all sessions in that scenario
        'PT2' : {'mean': 0, 'std': 0, 'n_trades': 0,'max_prof': 0, 'last_bal': 0} 
    }

    track_by_scenario = {

    }

    os.chdir(directory) # change to the dir 

    # Instead, lets save mean, std maxprofits into csv files for analysis.
    for folder in scenario_folder:
        scenario_id = folder

        data = getData(dump_flags, scenario_id)

        # avgbal_data = data.avg_bal() # DataFrame
        avgbal_data = data.get_pt1_pt2_total_profits()
        analyser = AvgBalAnalyser(avgbal_data)
        trader_stats = analyser.stats

        track_PT['PT1']['mean'] = trader_stats['PT1']['mean']
        track_PT['PT1']['std'] = trader_stats['PT1']['std']
        track_PT['PT1']['max_prof'] = trader_stats['PT1']['max_prof'] # maximum profit recorded in all trials
        track_PT['PT1']['last_bal'] = trader_stats['PT1']['last_bal'] # Final balance recorded after market session in all trials
        track_PT['PT1']['n_trades'] = trader_stats['PT1']['n_trades']
        track_PT['PT1']['ppt'] = trader_stats['PT1']['ppt']
        
        track_PT['PT2']['mean'] = trader_stats['PT2']['mean']
        track_PT['PT2']['std'] = trader_stats['PT2']['std']
        track_PT['PT2']['max_prof'] = trader_stats['PT2']['max_prof']
        track_PT['PT2']['last_bal'] = trader_stats['PT2']['last_bal']
        track_PT['PT2']['n_trades'] = trader_stats['PT2']['n_trades']
        track_PT['PT2']['ppt'] = trader_stats['PT2']['ppt']
        # print(track_PT, '\n'*3)

        track_by_scenario[scenario_id] = {k: v.copy() for k, v in track_PT.items()} #track_PT # saves PT tracker for each scenario

    return track_PT, track_by_scenario

def check_ppt(proft):
    if isinstance(proft[0], list):
        print('true')

def preprocess_profit(profit_metric):
    """
    Preprocess the metrics.
    Struct: list[tuple(session_id, prof)] 
    I called it profit, but it is actually any metric out of max_prof, final_bal, profit_per_time, number of trades

    For ppt it is list[tuple[session, list(ppt)]] need to hanfle differently. 
    """
    profit = [i[1] for i in profit_metric]
    session_id = [i[0].split('_')[-1] for i in profit_metric]

    return profit, session_id


def get_median(profit):
    """
    Calculate the median for the scenario in questions final balances or whatever.
    Median:
        - sort the profits from lowest to highest
        - choose the middle observation
    """
    sorted_profit = sorted(profit)
    n = len(profit)
    mid_point = n // 2 # get the middle index rounded down for first instance
    median = sorted_profit[mid_point]

    return median

def median_abs_deviation(profit):
    """
    Similar to STD, but more robust to outliers. Could be handy
    """
    median_abs = st.median_abs_deviation(profit)
    return median_abs

def get_CI(profit, threshold = 0.95):
    """
    Calculate the threshold CI for for the scenario in questions final balances or whatever 
    """
    # get scipy to do it, more efficient
    confidence_interval = st.norm.interval(confidence=threshold, loc=np.mean(profit), scale=st.sem(profit))

    return confidence_interval

def get_mode(profit):
    """
    Calculate the mode for the scenario in questions final balances or whatever 
    Mode:
        - Scipy.stats.mode()
    """
    mode = st.mode(profit)
    
    return [mode[0],mode[1]] # mode, count

def get_IQR(profit):
    """
    Calculate the inter-quartile range for the scenario in questions final balances or whatever 
    """
    iqr = list(np.quantile(profit, [0.25, 0.75]))
    return iqr

def get_mean(profit):
    mean = np.mean(profit)
    return mean

def get_std(profit):
    std = np.std(profit)
    return std

def get_more_stats(profit, threshold=0.95):

    if isinstance(profit, tuple):
        profit = profit[0] # for metrics which are in a tuple, take the first value which are the balances
    
    if isinstance(profit, np.ndarray):
        profit = profit # for n_trades - not needed anymore but keeping just incase

    if len(profit) <2:
        return 0
    try:
        mode = get_mode(profit)
    except:
        mode = 0
    iqr = get_IQR(profit)
    median = get_median(profit)
    median_abs_dev = median_abs_deviation(profit)
    ci = get_CI(profit, threshold=threshold)
    mean = get_mean(profit)
    std = get_std(profit)

    return [mode, iqr, median, median_abs_dev, ci]


def write_scenario_stats(scenario_tracker, filename = 'comparative_stats'):
    """
    write mean's and std's + other stats from get_more_stats() to a csv file for 2 observation metics (maxprofit and final bal) each scenario
    """
    
    evaluating = ['max_prof', 'last_bal', 'n_trades', 'ppt'] # 

    for metric in evaluating:
        with open(f'{filename}_{metric}.csv', 'w') as f:
            print(f'Writing file for: {metric}')
            writer = csv.writer(f)
            writer.writerow(['scenario_id', 'mean2','std2', 'mean1','std1', metric+'_stats_2',metric+'_stats_1', metric+'_values_2', metric+'_values_1'])
            for scenario_id, trader_stats in scenario_tracker.items():
                pt2 = trader_stats['PT2']
                pt1 = trader_stats['PT1']
                pt1_metric = preprocess_profit(pt1[metric])
                pt2_metric = preprocess_profit(pt2[metric])
                if metric ==  'ppt':
                    pt1_metric_stats = 0
                    pt2_metric_stats = 0 # no need to do this now, as there are problems, will do after in statistical_tests.py
                else:
                    pt1_metric_stats = get_more_stats(pt1_metric, threshold=0.95) # [mode,count], iqr:[25,75], median, median_abs_deviation, ci: [lower,upper]
                    pt2_metric_stats = get_more_stats(pt2_metric, threshold=0.95)
                
                writer.writerow([scenario_id, pt2['mean'], pt2['std'], pt1['mean'], pt1['std'], pt2_metric_stats, pt1_metric_stats, pt2_metric, pt1_metric])


if __name__ == '__main__':
    """
    What this does:
        - Runs analysis on avgbals, writes files for each key metric: ['max_prof', 'last_bal', 'n_trades', 'ppt']
        - These files are used for statistical analysis.    
    Example usage:
    python3 analyse_performance.py --yaml_file yaml_files/GoldenStandard.yaml --dir wd21585

    """

    parser = argparse.ArgumentParser(description="Obtain key statistical information from scenarios")

    parser.add_argument("--yaml_file", required=True, help="Path to the yaml file used in scenario")
    parser.add_argument("--dir", required=True, help="Directory which holds avgbals")
    parser.add_argument("--f", required=False, help="Input a file name for the comparative statistics")

    args = parser.parse_args()

    if 'avgbals' not in os.listdir(args.dir):
        print("Please enter a directory which has 'avgbals' directory inside it")
        exit
    else:
        print("Args accepted. Please wait")

    track_PT, track_by_scenario = obtain_key_stats(args.yaml_file, args.dir)
    # print(track_by_scenario)
    write_scenario_stats(track_by_scenario, args.f)

    


