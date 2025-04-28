# built in
import csv
import os
import sys
import argparse
import ast
import re
from pprint import pprint

# required
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st 
from scipy.stats import ttest_rel, wilcoxon, shapiro, pearsonr
import pandas as pd
import seaborn as sns


"""
For statistical tests, we want to do pairwise significance. This means test each stat of both traders, across all scenarios.
- So write a function for each observation you want to know about for all metrics. 
"""

# could use a class, but I cba
def retrieve_mode(data, pt1_info, pt2_info):
    modes1 = [data[pt1_info][i][0][0][0] for i in range(1, len(data[pt1_info]))]
    modes2 = [data[pt2_info][i][0][0][0] for i in range(1, len(data[pt2_info]))]
    return modes2, modes1


def retrieve_median(data, pt1_info, pt2_info):
    pt1_median = [data[pt1_info][i][2] for i in range(1, len(data[pt1_info]))]
    pt2_median = [data[pt2_info][i][2] for i in range(1, len(data[pt2_info]))]
    return pt2_median, pt1_median # int

def retrieve_iqr(data, pt1_info, pt2_info):

    pt1_iqr = [data[pt1_info][i][1] for i in range(1, len(data[pt1_info]))]
    pt2_iqr = [data[pt2_info][i][1] for i in range(1, len(data[pt2_info]))]
    return pt2_iqr, pt1_iqr # lists [lower, upper]

def retrieve_mad(data, pt1_info, pt2_info):
    pt1_mad = [data[pt1_info][i][3] for i in range(1, len(data[pt1_info]))]
    pt2_mad = [data[pt2_info][i][3] for i in range(1, len(data[pt2_info]))]
    return pt2_mad, pt1_mad # floats 

def retrieve_CI(data,pt1_info, pt2_info):
    pt1_ci = [data[pt1_info][i][-1] for i in range(1, len(data[pt1_info]))]
    pt2_ci = [data[pt2_info][i][-1] for i in range(1, len(data[pt2_info]))]

    return pt2_ci, pt1_ci # tuples (lower, upper)

def retrieve_mean(data,pt1_info, pt2_info):
    # if pt1_info != 'n_trades_values_1':
    #     pt1_mean = [np.mean((np.array(data[pt1_info][i][0])-500) / 500) for i in range(1, len(data[pt1_info]))]
    #     pt2_mean = [np.mean((np.array(data[pt2_info][i][0])-500) / 500) for i in range(1, len(data[pt2_info]))]
    # else:
    pt1_mean = [np.mean(data[pt1_info][i][0]) for i in range(1, len(data[pt1_info]))]
    pt2_mean = [np.mean(data[pt2_info][i][0]) for i in range(1, len(data[pt2_info]))]

    return pt2_mean, pt1_mean # lists the means of scenarios

def retrieve_std(data,pt1_info, pt2_info):
    """Currently normalised for profit"""
    pt1_std = [np.std((np.array(data[pt1_info][i][0])-500) / 500) for i in range(1, len(data[pt1_info]))]
    pt2_std = [np.std((np.array(data[pt2_info][i][0])-500) / 500) for i in range(1, len(data[pt2_info]))]

    return pt2_std, pt1_std # 




def clean_and_eval(x):
    if isinstance(x, str):
        # Strip NumPy array wrappers, e.g., "array([8675])" → "[8675]"
        x = re.sub(r'array\((\[.*?\])\)', r'\1', x)
        try:
            return ast.literal_eval(x)
        except Exception:
            return x  # or return np.nan if you prefer
    return x

def get_data(metric, phase):
    file = f'{phase}_{metric}.csv'
    data = pd.read_csv(file, names = ['scenario_id','mean2','std2','mean1','std1',f'{metric}_stats_2',f'{metric}_stats_1',f'{metric}_values_2',f'{metric}_values_1'])
    
    data[f'{metric}_stats_2'] = data[f'{metric}_stats_2'].apply(clean_and_eval)
    data[f'{metric}_stats_1'] = data[f'{metric}_stats_1'].apply(clean_and_eval)
    data[f'{metric}_values_2'] = data[f'{metric}_values_2'].apply(clean_and_eval)
    data[f'{metric}_values_1'] = data[f'{metric}_values_1'].apply(clean_and_eval)
    return data


def process_file(metric: str, phase: str, observable: str) -> tuple:
    """
    - takes in the metric you are using to retrieve means for
    - test phase for file and reading

    - Problem, mode is dtype np.array which is not a python literal and so we cannot use ast.

    """

    data = get_data(metric, phase)

    pt1_info = f'{metric}_stats_1'
    pt2_info = f'{metric}_stats_2'    

    pt1_values = f'{metric}_values_1'
    pt2_values = f'{metric}_values_2'

    if observable == 'mean':
        return retrieve_mean(data, pt1_values, pt2_values)
    elif observable == 'std':
        return retrieve_std(data, pt1_values, pt2_values)
    elif observable == 'mode':
        return retrieve_mode(data, pt1_info, pt2_info)
    elif observable == 'median':
        return retrieve_median(data, pt1_info, pt2_info)
    elif observable == 'IQR':
        return retrieve_iqr(data, pt1_info, pt2_info)
    elif observable == "MAD": # Mean Abs Deviation
        return retrieve_mad(data, pt1_info, pt2_info)
    elif observable == "CI":
        return retrieve_CI(data, pt1_info, pt2_info)
    else:
        print('Invalid observable')
        sys.exit('Nope, wrong')

def calc_sharpe(metric, phase):
    pt2_mean, pt1_mean = process_file(metric, phase, 'mean')
    pt2_std, pt1_std = process_file(metric, phase, 'std')

    pt2_mean = np.array(pt2_mean)
    pt1_mean = np.array(pt1_mean)

    pt2_std = np.array(pt2_std)
    pt1_std = np.array(pt1_std)

    pt2_sharpe = pt2_mean  / pt2_std
    pt1_sharpe = pt1_mean / pt1_std

    return pt2_sharpe, pt1_sharpe

def profit_per_trade(metric, phase):
    pt2_mean, pt1_mean = process_file(metric, phase, 'mean') # mean return
    pt2_std, pt1_std = process_file('n_trades', phase, 'mean') # mean trades

    pt2_mean = np.array(pt2_mean)
    pt1_mean = np.array(pt1_mean)

    pt2_n_trades = np.array(pt2_std)
    pt1_n_trades = np.array(pt1_std)

    pt2_ppt= pt2_mean  / pt2_n_trades
    pt1_ppt = pt1_mean / pt1_n_trades

    return pt2_ppt, pt1_ppt # mean profit per trade for all scenarios

def final_drawdown(phase):
    
    last_bal = get_data('last_bal',phase)
    pt1_last_bal = np.array([i[0] for i in last_bal['last_bal_values_1'][1:]])
    pt2_last_bal = np.array([i[0] for i in last_bal['last_bal_values_2'][1:]])

    max_prof = get_data('max_prof', phase)
    pt1_max_prof = np.array([i[0] for i in max_prof['max_prof_values_1'][1:]]) # shape is scenarios by num_sessions (6x500)
    pt2_max_prof = np.array([i[0] for i in max_prof['max_prof_values_2'][1:]])

    pt1_drawdown = np.mean(pt1_max_prof - pt1_last_bal, axis = 1) # compute means horizontally for rows
    pt2_drawdown = np.mean(pt2_max_prof - pt2_last_bal, axis = 1)

    return pt2_drawdown, pt1_drawdown


def test_each_scenario(data, pt1_col, pt2_col, metric_name, alpha=0.05, max_samples=-1):
    scenario_ids = data['scenario_id'].tolist()[1:]
    pt1_all = [data[pt1_col][i][0] for i in range(1, len(data[pt1_col]))]
    pt2_all = [data[pt2_col][i][0] for i in range(1, len(data[pt2_col]))]
    
    results = []

    for sid, pt1_vals, pt2_vals in zip(scenario_ids, pt1_all, pt2_all):
        try:
            pt1_vals = np.array(pt1_vals[:max_samples])
            pt2_vals = np.array(pt2_vals[:max_samples])

            diffs = pt2_vals - pt1_vals

            # Check normality of the paired differences
            W, p_norm = shapiro(diffs)
            is_normal = p_norm > alpha

            if is_normal:
                stat, p = ttest_rel(pt2_vals, pt1_vals)
                p = p / 2 if stat > 0 else 1 - p / 2
                test_used = 't-test'
            else:
                stat, p = wilcoxon(pt2_vals, pt1_vals, alternative='greater')
                test_used = 'wilcoxon'

            results.append((sid, test_used, stat, f'{p:2f}', p < alpha))
        
        except Exception as e:
            results.append((sid, 'ERROR', None, None, f'{str(e)}'))

    return results



def compare_ci_width(pt2_ci, pt1_ci):
    pt2_widths = [high - low for (low, high) in pt2_ci]
    pt1_widths = [high - low for (low, high) in pt1_ci]

    return run_statistical_tests("CI Width", pt2_widths, pt1_widths, direction='less')

def compare_iqr(pt2_iqr, pt1_iqr):
    pt2_ranges = [q3 - q1 for (q1, q3) in pt2_iqr]
    pt1_ranges = [q3 - q1 for (q1, q3) in pt1_iqr]

    return run_statistical_tests("IQR", pt2_ranges, pt1_ranges, direction='less')


def compute_cohens_d(pt2, pt1):
    pt2 = np.array(pt2)
    pt1 = np.array(pt1)
    diff = pt2 - pt1
    mean_diff = np.mean(diff)
    std_pooled = np.sqrt((np.std(pt2, ddof=1)**2 + np.std(pt1, ddof=1)**2) / 2)
    d = mean_diff / std_pooled
    return d

def run_statistical_tests(metric_name, pt2, pt1, direction='greater'):
    """
    direction: 'greater' for PT2 > PT1, 'less' for PT2 < PT1
    """
    assert direction in ['greater', 'less'], "Direction must be 'greater' or 'less'"

    print(f"=== ({'PT2 > PT1' if direction == 'greater' else 'PT2 < PT1'}) ===")

    pt2 = np.array(pt2)
    pt1 = np.array(pt1)

    # One-sided paired t-test
    t_stat, p_two_sided = ttest_rel(pt2, pt1)
    if direction == 'greater':
        p_one_sided_t = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    else:  # direction == 'less'
        p_one_sided_t = p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2

    print(f"One-sided t-test ({'PT2 > PT1' if direction == 'greater' else 'PT2 < PT1'}): "
          f"t = {t_stat:.4f}, p = {p_one_sided_t:.4f} {'**' if p_one_sided_t < 0.05 else ''}")

    # One-sided Wilcoxon test
    try:
        w_stat, p_wilcoxon = wilcoxon(pt2, pt1, alternative=direction)
        print(f"Wilcoxon test ({'PT2 > PT1' if direction == 'greater' else 'PT2 < PT1'}): "
              f"W = {w_stat:.4f}, p = {p_wilcoxon:.4f} {'**' if p_wilcoxon < 0.05 else ''}")
    except ValueError as e:
        print(f"Wilcoxon test skipped: {e}")

    # Cohen's d
    d = compute_cohens_d(pt2, pt1)
    print(f"Cohen's d: {d:.4f} {'(small)' if abs(d) < 0.5 else '(medium)' if abs(d) < 0.8 else '(large)'}\n")

    return [w_stat, p_wilcoxon] if p_wilcoxon is not None else [t_stat, p_one_sided_t]


def run_normality_check(pt2, pt1, metric_name):
    diff = np.array(pt2) - np.array(pt1) # the differences between them must be normally distributed

    # Shapiro-Wilk
    W, p = shapiro(diff)
    print(f"{metric_name}: Shapiro-Wilk W = {W:.4f}, p = {p:.4f}")
    if p > 0.05:
        print("YES-Assumption of normality appears reasonable.\n")
    else:
        print("NO-Data is likely not normally distributed.\n")



# Function to plot boxplots in subplots for each metric
def plot_metric_boxplots(metrics, phase='phase1'):
    n_metrics = len(metrics)
    cols = 2
    rows = (n_metrics + 1) // cols
    label_dict = {
        'ppt': 'Profit Per Trade',
        'last_bal':'Final Profit',
        'max_prof': 'Maximum Profit',
        'n_trades': 'Number of Trades'
    }
    if phase == 'phase1':
        scenario_dict = {
            'BTC-USD-20250404-A20-0': '04-04-2025',
            'BTC-USD-20250414-A20-0': '14-04-2025',
            'BTC-USD-20250211-1D-A20-0': '11-02-2025',
            'BTC-USD-20250210-1D-A20-0': '10-02-2025',
            'BTC-USD-20250405-1D-A20-0': '05-04-2025',
            'BTC-USD-20250406-1D-A20-0': '06-04-2025'
        }
    elif phase == 'phase2':
        scenario_dict = {
            'BTC-USD-20250211': '10-5-5-5',
            'BTC-USD-20250211-1D-51055-0': '5-10-5-5',
            'BTC-USD-20250211-1D-55105-0' : '5-5-10-5',
            'BTC-USD-20250211-1D-55510-0' : '5-5-5-10',
            'BTC-USD-20250211-1D-77207-0' : '7-7-21-7',
            'BTC-USD-20250211-1D-77720-0' : '7-7-7-21',
            'BTC-USD-20250211-1D-72077-0' : '7-21-7-7',
            'BTC-USD-20250211-1D-20777-0' : '21-7-7-7',
            'BTC-USD-20250211-1D-551010-0' : '5-5-10-10',
            'BTC-USD-20250211-1D-105105-0' : '10-5-10-5',
            'BTC-USD-20250211-1D-510510-0' : '5-10-5-10',
            'BTC-USD-20250211-1D-101055-0' : '10-10-5-5',
            'BTC-USD-20250211-1D-105510-0' : '10-5-5-10',

        }
    elif phase == 'phase3':
        scenario_dict = {
            'elastic-supply-fixed-0': 'ES-fixed',
            'elastic-demand-fixed-0': 'ED-fixed',
            'elastic-supply-jit-0': 'ES-jit',
            'elastic-demand-jit-0': 'ED-jit',
            'elastic-supply-rand-0': 'ES-rand',
            'elastic-demand-rand-0': 'ED-rand'
        }


    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 7))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        data = get_data(metric, phase)

        scenario_ids = data['scenario_id'].tolist()[1:]

        plot_data = []
        for scenario, pt1_vals, pt2_vals in zip(scenario_ids,
                                                data[f'{metric}_values_1'][1:],
                                                data[f'{metric}_values_2'][1:]):
            for val in pt1_vals[0]:
                plot_data.append({'Scenario': scenario_dict[scenario], 'Trader': 'PT1', label_dict[metric]: val})
            for val in pt2_vals[0]:
                plot_data.append({'Scenario': scenario_dict[scenario], 'Trader': 'PT2', label_dict[metric]: val})

        plot_df = pd.DataFrame(plot_data)
        if phase == 'phase2':
            scenario_order = ['10-5-5-5', '5-10-5-5', '5-5-10-5','5-5-5-10','21-7-7-7','7-21-7-7','7-7-21-7','7-7-7-21','5-5-10-10','10-5-10-5','5-10-5-10','10-10-5-5','10-5-5-10']
        elif phase == 'phase3':
            scenario_order = ['ES-fixed','ES-jit','ES-rand','ED-fixed','ED-jit','ED-rand']
        else:
            scenario_order = ['10-02-2025', '11-02-2025', '04-04-2025','05-04-2025', '06-04-2025', '14-04-2025']

        # Set categorical order for sorting
        plot_df['Scenario'] = pd.Categorical(plot_df['Scenario'], categories=scenario_order, ordered=True)      
        
        sns.boxplot(x='Scenario', y=label_dict[metric], hue='Trader', data=plot_df, ax=axes[idx], order = scenario_order)
        # axes[idx].set_title(f'{label_dict[metric].capitalize()} by Scenario')
        axes[idx].set_xlabel('Scenario', fontsize = 14)
        axes[idx].set_ylabel(label_dict[metric], fontsize= 14)
        axes[idx].legend(title='Trader', fontsize=12)
        axes[idx].tick_params(axis='x', rotation=45, labelsize=14)
        axes[idx].tick_params(axis='y', labelsize=14)

    # Remove any unused subplots
    for idx in range(len(metrics), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    plt.savefig(f'{phase}_box_plot.png', bbox_inches='tight', dpi=300)
    
    plt.show()


if __name__ == '__main__':
    """
    Set up statistical tests for all 3 Areas of investigation.
        - Each investigation outputs N files (metrics like final balance, max profit, profit-per-trade per session, number of trades per session)

    Architecture:
        - Process files outputted from analyse_performance.py 
        - Set up a series of tests which tell, to a significance wether PT2 is better than PT1 (Hopefully is)
        
        - Results:
            - View in Terminal
            - Write to a txt file for Appendix
    
    
    Tomorrow TODO:
        - Visualise the results - how?
        - Get Phase 1 results (Data Variance), phase 2 depends on phase 1, so make sure this gets done.
        - Phase 2 started. should be done saturday

    Friday TODO:
        - Hopefully, Phase 2 is done, if it is get results, then start Phase 3.

    Saturday TODO:
        - Phase 3 results should be in by now. 
        - Run results. Have a break for easter weekend.
    
    Monday - Tuesday TODO:
        - All results should be done by now
        - So write up the results section of the report.
    
    Wednesday TODO:
        - Do references and make the report smooth and whatever. 
    
    Aim to be done Thursday/Friday
            
    """

    # print(final_drawdown('phase1'))

    # #Plot box plots for the data files
    metrics = ['last_bal', 'n_trades']
    phase = 'phase2'
    """
    Look into each scenarios last balances.
    Look into each scenarios n_trades. half these as trades come in pairs

    Divide the final balances by each trade for each trader to get ppt for each session. Then calculate the SHARPE ratios etc.

    """

    #trades:
    # n_trades = 'n_trades'
    # trades = get_data(n_trades, phase)
    # pt1_trades = trades[f'{n_trades}_values_1']
    # pt2_trades = trades[f'{n_trades}_values_2']

    # final_bals = 'last_bal'
    # data = get_data(final_bals, phase)
    # pt1_bal = data[f'{final_bals}_values_1'] # remember this for each scenario
    # pt2_bal = data[f'{final_bals}_values_2']

    # # Calculate Pearson correlation for pt1
    # pt1_corr, pt1_pval = pearsonr(pt1_trades, pt1_bal)
    # print(f"PT1 Pearson correlation: {pt1_corr:.3f} (p-value: {pt1_pval:.3g})")

    # # Calculate Pearson correlation for pt2
    # pt2_corr, pt2_pval = pearsonr(pt2_trades, pt2_bal)
    # print(f"PT2 Pearson correlation: {pt2_corr:.3f} (p-value: {pt2_pval:.3g})")


    




        


    # plot_metric_boxplots(metrics, 'phase1')
    plot_metric_boxplots(metrics, 'phase2')
    # plot_metric_boxplots(metrics, 'phase3')
    
    # Sharpe ratios for the absolute profit, change profits to returns.
    # pt2_sharpe, pt1_sharpe = calc_sharpe('last_bal', 'phase1')
    # print("Phase 1 Sharpe Ratios:")
    # print(f"  PT2 Sharpe: {pt2_sharpe}")
    # print(f"  PT1 Sharpe: {pt1_sharpe}\n")

    # pt2_sharpe, pt1_sharpe = calc_sharpe('last_bal', 'phase2')
    # print("Phase 2 Sharpe Ratios:")
    # print(f"  PT2 Sharpe: {pt2_sharpe}")
    # print(f"  PT1 Sharpe: {pt1_sharpe}\n")
    # # pt2_sharpe, pt1_sharpe = calc_sharpe('last_bal', 'phase3')
    

    # # mean profit per trade for phase 1 scenarios
    # pt2_ppt, pt1_ppt = profit_per_trade('last_bal', 'phase1')
    # print("Phase 1 Profit Per Trade:")
    # print(f"  PT2 PPT: {pt2_ppt}")
    # print(f"  PT1 PPT: {pt1_ppt}\n")
    
    # run_normality_check(pt2_ppt, pt1_ppt, "Mean Profit")
    # run_statistical_tests('mean', pt2_ppt, pt1_ppt, direction ='greater')


    # pt2_ppt, pt1_ppt = profit_per_trade('last_bal', 'phase2')
    # print("Phase 2 Profit Per Trade:")
    # print(f"  PT2 PPT: {pt2_ppt}")
    # print(f"  PT1 PPT: {pt1_ppt}\n")
    # run_normality_check(pt2_ppt, pt1_ppt, "Mean Profit")
    # run_statistical_tests('mean', pt2_ppt, pt1_ppt, direction ='greater')



    # # pt2_ppt, pt1_ppt = profit_per_trade('last_bal', 'phase3')


    # observables = ['mean', 'std', 'mode', 'MAD', 'CI', 'median', 'IQR']
    # phases = ['phase1', 'phase2']
    # for phase in phases:
    #     print(f" #################### {phase} #################### \n ")

    #     for metric in metrics:
            
    #         print(f"**************** {metric.upper()} ****************\n")
    #         print("======= Mean =======\n")
    #         pt2, pt1 = process_file(metric, phase, 'mean')
    #         run_normality_check(pt2, pt1, "Mean Profit")
    #         run_statistical_tests(metric, pt2, pt1, direction ='greater')

    #         print("======= Standard Deviation =======\n")
    #         pt2, pt1 = process_file(metric,phase, 'std')
    #         run_normality_check(pt2, pt1, "STD Profit")
    #         run_statistical_tests(metric, pt2, pt1, direction='less')
        
    #         print("======= Mean Absolute Deviation =======\n")
    #         pt2, pt1 = process_file(metric, phase, 'MAD')
    #         run_normality_check(pt2, pt1, "MAD Profit")
    #         run_statistical_tests(metric, pt2, pt1, direction='less')
        
    #         print("======= Median =======\n")
    #         pt2, pt1 = process_file(metric, phase, 'median')
    #         run_normality_check(pt2, pt1, "Median")
    #         run_statistical_tests(metric, pt2, pt1, direction='greater')
        
    #         print("======= Mode =======\n")
    #         pt2, pt1 = process_file(metric, phase, 'mode')
    #         run_normality_check(pt2, pt1, "Mode")
    #         run_statistical_tests(metric, pt2, pt1, direction='greater')
        

    #         print("======= Confidence Interval Width =======\n")
    #         pt2_ci, pt1_ci = process_file(metric, phase, 'CI')
    #         compare_ci_width(pt2_ci, pt1_ci)

    #         print("======= Inter-Quartile Ranges =======\n")
    #         pt2_iqr, pt1_iqr = process_file(metric, phase, 'IQR')
    #         compare_iqr(pt2_iqr, pt1_iqr)

    #     print("########## Scenario-Specific ###########\n")
    #     print("************ Wilcoxon or T-test **************\n")
    #     for metric in metrics:
    #         print(f'======= {metric} =======\n')
    #         data = get_data(metric,phase)

    #         results = test_each_scenario(
    #             data, 
    #             pt1_col=f'{metric}_values_1', 
    #             pt2_col=f'{metric}_values_2',
    #             metric_name=metric
    #         )

    #         pprint(results)
    #         print('\n')

    # print("************* Paired T-test ***********\n")

    # for metric in metrics:
    #     print(f'======= {metric} =======\n')
    #     data = get_data(metric, 'phase1')

    #     results = test_each_scenario(
    #         data, 
    #         pt1_col=f'{metric}_values_1', 
    #         pt2_col=f'{metric}_values_2',
    #         metric_name=metric
    #     )

    #     pprint(results)
    #     print('\n')



