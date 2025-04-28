import numpy as np
# from BSEv1_9 import *
import sys
import yaml
from offset_funcs import *
from tools import SetUpScenario
from play import organise_files_by_scenario
import logging
import time as tm
import os
"""
Sets up a scenario and runs sessions using run_scenario()
    - Multiprocessed.
    - Need to configure yaml files to represent different market dynamics
    - Then run: ./run_tests.sh and that will run this python script over all of the yaml files, and organise them into the respective folders.
    - I can run this in the cloud overnight.
    
Yaml scenarios are separated into folders and each vm is given a different folder of yaml's to complete at the same time. 

"""


if __name__ == "__main__":
    # Load YAML
    yaml_file = sys.argv[1]
    work_dir = sys.argv[2]
    os.makedirs(work_dir, exist_ok=True)
    
    log_file = os.path.join(work_dir, "simulation.log")

    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)

    # Handle offset file loading
    offsetfn_events = schedule_offsetfn_read_file(cfg['price_offset_filename'], 0, 1)
    supply_chart = (cfg['chart_range']['supply']['min_price'],
                    cfg['chart_range']['supply']['max_price'],
                    (schedule_offsetfn_from_eventlist, [[cfg['end_time'], offsetfn_events]]))

    demand_chart = (cfg['chart_range']['demand']['min_price'],
                    cfg['chart_range']['demand']['max_price'],
                    (schedule_offsetfn_from_eventlist, [[cfg['end_time'], offsetfn_events]]))

    chart_range = {
        'supply': supply_chart,
        'demand': demand_chart
    }

    # Generate scenario_id
    scenario_id = cfg['chart_name'] + "-" + str(cfg['scenario_index'])

    # Build args
    args = (
        cfg['days'],
        cfg['start_time'],
        cfg['end_time'],
        chart_range,
        cfg['step_mode'],
        cfg['order_interval'],
        cfg['time_mode'],
        cfg['sellers_spec'],
        cfg['buyers_spec'],
        cfg['proptraders_spec'],
        scenario_id,
        cfg['dump_flags'],
        cfg['verbose']
    )

    # change to working dir for HPC
    os.chdir(work_dir)

    start = tm.time()
    scenario = SetUpScenario(*args)
    scenario.run_scenario(num_sessions=cfg['num_sessions'])
    end = tm.time()

    organise_files_by_scenario(dump_flags = cfg['dump_flags'])
    logging.info(f"Finished simulation: {scenario_id}, time taken = {end-start}")

    


