# AlgoTrading
Supporting code for Algorithmic Trading Coursework

Inside coursework the following pipline is used for HPC;
1) Trader is in BSEv1.9
2) All parameters for BSE are in `yaml_files` folder. 
3) `tools.py` is the main code which does all the analysis;
   1) it sets up the market scenario using SetUpScenario Class
   2) uses multiprocessing to run multiple sims at once, and has the option to change while debugging or developing.
   3) manages_files function manages files outputted by bse into directories using the scenario ID name for each dump file
   4) no hardcoded file paths as intended for use on HPC
   5) getData class gets the avgBals data for PT1 and PT2 efficiently, as we analyse up to 50,000,000 lines per scenario.
   6) AvgBalAnalyser class computes statistics on the data preprocessed from getData - uses lazy computation of class attributes for efficiency.
4) `run_session.py` is the main function which sets up a scenario using a yaml file as input, this feeds all the parameters in to the setupscenario class and runs single scenatio in BSE for those parameters. 
   1) Made to be repeatable so HPC can do many of these with different YAML files in array job
5) `run_array.sh` runs scenarios for the folders in yaml_folders, change the number of tasks depending on how many yaml files you have as this cycles through the yaml files and runs `run_session.py` for each one.
6) `get_stats.sh` uses `analyse_performance.py` to get all the statistics, using command line flags to control where you want the output to be written to, and also the name of the file. 
7) Inside the `statistical_methods` directory, all the csv files are proof of all simulations I have done as they contain the outputted data for each scenario. Then `statistical_tests.py` processess this information and runs all the stats etc for the results. These include box plots, statistical tests (Wilcoxon, t-test, shapiro-wilks etc).

The `final_report.pdf` achieved 75/100. Feedback: Profit-Per-Trade is not as important as the reasons why my trader can secure more trades than PT1. 
The maximum grade for the year is 76/100, only 7/52 people achieved >70. 

