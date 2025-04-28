import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
import os
sys.path.append('/Users/edatkinson/University/4thYear/AlgoTrading/BSE_demos/') # so I can access indicators in modules
import pandas as pd
from modules import Indicators
import argparse

def generate_position(tape_df):
    tape_df['positions'] = 0
    tape_df.loc[tape_df['MACD'] > tape_df['EMA9'], 'positions'] = 1
    tape_df.loc[tape_df['MACD'] <= tape_df['EMA9'], 'positions'] = -1
    return tape_df


def evaluate_performance(data: pd.DataFrame) -> dict:
    data = data.copy()

    tape_df['positions'] = tape_df['positions'].shift(1)
    tape_df['positions'] = tape_df['positions'].fillna(0)
    
    data['Return'] = data['price'].pct_change().fillna(0)
    
    data['StrategyReturn'] = data['Return'] * data['positions']
    
    data['CumulativeStrategyReturn'] = (1 + data['StrategyReturn']).cumprod() - 1
    sharpe_ratio = (data['StrategyReturn'].mean() / data['StrategyReturn'].std()) 

    cumulative_pnl = (1 + data['price'].pct_change() * data['positions'].shift(1)).cumprod() -1  # assume initial capital=500

    cumulative = (1 + data['StrategyReturn']).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return data, {
        "Sharpe": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Cumulative Return": data['CumulativeStrategyReturn'].iloc[-1],
        # "Cumulative_pnl": cumulative_pnl
    }

def plot_diagrams(tape_df):
    fig, axes= plt.subplots(2,1, figsize=(14,8))

    axes[0].plot(tape_df.index, tape_df['price'], label = 'Market Price')
    axes[0].plot(tape_df.index, tape_df['EMA12'], label = 'Fast-EMA')
    axes[0].plot(tape_df.index, tape_df['EMA24'], label = 'Slow-EMA')
    # axes[0].plot(tape_df.index, tape_df['MA25'], label = 'MA')

    axes[1].plot(tape_df.index, tape_df['MACD'], label = 'MACD')
    axes[1].plot(tape_df.index, tape_df['EMA9'], label = 'Signal', color='red', linewidth=0.5)

    axes[0].legend()
    axes[1].legend()

    plt.show()



if __name__ == '__main__':
    """
    Example usage:
    python3 statistical_methods/technical_analysis.py --dir tape/test4-0 --period 10S/1m/2hr/1day --plot True/False/None
    """

    parser = argparse.ArgumentParser(description="Do Technical Analysis on tape data from scenarios")

    parser.add_argument("--dir", required=True, help="Directory where the tape scenario is located")
    parser.add_argument("--period", required=True, help="Time period you want to reorder the tape into, in minutes or seconds")
    parser.add_argument("--plot", required=False, help="Plot figures of the 1st tape")
    args = parser.parse_args()
    
    period = str(args.period)

    if period[-1].lower() == 'm':
        period = int(period[:-1])
        period = period * 60 # secs
        period = str(period) + 'S'
    elif period[-2:].lower() == 'hr':
        period = int(period[:-2])
        period = period * 60 * 60
        period = str(period) + 'S'
    elif period[-1].lower() == 'd':
        period = int(period[:-1])
        period = period * 60 * 60 * 24
        period = str(period) + 'S'

    directory = args.dir
    files = os.listdir(directory)
    list_of_tapes = []

    for tape in files:
        tape_df = pd.read_csv(os.path.join(directory,tape), sep=',', names = ['trade', 'time', 'price'])
        tape_df['time'] = pd.to_timedelta(tape_df['time'], unit='s') + pd.Timestamp('2025-02-10')
        tape_df = tape_df.set_index('time')
        tape_df_resampled = tape_df.resample(period).agg({'price': 'last'}) # 5 minutes
        resampled_indicators = Indicators(tape_df_resampled)
        resampled_indicators.add_ema_custom(mu=12, col='price')
        resampled_indicators.add_ema_custom(mu=24, col='price')

        resampled_indicators.data['MACD'] = resampled_indicators.data['EMA12'] - resampled_indicators.data['EMA24']
        resampled_indicators.add_ema_custom(mu=9, col='MACD')

        tape_df = resampled_indicators.data
        list_of_tapes.append(tape_df)

    # needs to be in work folder for HPC
    os.makedirs('work/statistical_methods', exist_ok = True)
    os.makedirs('work/statistical_methods/technical', exist_ok = True)
    os.chdir('work/statistical_methods/') # change to the stats dir

    with open(f'technical/{directory.split('/')[-1]}_period:{args.period}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'sharpe', 'max_drawdown', 'cumulative_return'])
        for index, tape in enumerate(list_of_tapes):
            tape_df = generate_position(tape)
            _, stats = evaluate_performance(tape)
            writer.writerow([files[index], stats['Sharpe'], stats['Max Drawdown'], stats['Cumulative Return']])
    
    print(f"Results written to: work/technical/{directory.split('/')[-1]}.csv")
    
    if args.plot:
        plot_diagrams(list_of_tapes[0])