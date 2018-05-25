import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

class Create_Pairplot:
    def __init__(self, config_file_path='src/config/create_pairplot_config.json'):
        with open(config_file_path) as f:
            self.config  = json.loads(f.read())
        self.data  = self.read_in_data()
        sns.pairplot(self.data)
        plt.savefig(self.config['image_file_location'])

    def read_in_data(self):
        if self.config['multiple_files']:
            dataframe_list = []
            if self.config['limit_columns']:
                for path,cols in zip(self.config['multiple_file_paths'],self.config['multiple_columns_to_use']):
                    dataframe_list.append(pd.read_csv(path), usecols=cols)
            else:
                for path in self.config['multiple_file_paths']:
                    dataframe_list.append(pd.read_csv(path))
            raw_data = pd.concat(dataframe_list)
        elif self.config['limit_columns']:
            raw_data = pd.read_csv(self.config['data_file_path'], usecols=self.config['columns_to_use'])
        else:
            raw_data = pd.read_csv(self.config['data_file_path'])
        return raw_data
