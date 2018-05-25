import pandas as pd
import json
import single_column_cat_profitability as sccp
import create_pairplot as cp

config_file_path='src/config/config_list.json'
with open(config_file_path) as f:
    config  = json.loads(f.read())
if config['sccp']:
    for path in config['single_column_cat_profitability_config_files']:
        sccp.Single_Col_Cat_Prof(path)
if config['cpc']:
    for path in config['create_pairplot_config_files']:
        cp.Create_Pairplot(path)
