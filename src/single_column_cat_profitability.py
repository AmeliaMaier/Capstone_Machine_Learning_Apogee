import pandas as pd
import json

class Single_Col_Cat_Prof:

    def __init__(self, config_file_path='src/config/single_column_cat_profitablity_config.json'):
        with open(config_file_path) as f:
            self.config  = json.loads(f.read())
        self.data_dummies, self.data  = self.read_in_data()
        if self.config['profit_per_interaction']: self.profit_per_interaction()
        if self.config['profit_per_user']:self.profit_per_user()
        if self.config['profit_per_cat']:self.profit_per_cat()


    def read_in_data(self):
        columns = [self.config['user_id_col_name'],self.config['conversion_yes_no_col_name'],
                self.config['categorical_col_name'], self.config['revenue_col_name'],self.config['cost_col_name']]
        raw_data = pd.read_csv(self.config['data_file_path'], usecols=columns)
        raw_data.rename(index=str, columns={self.config['user_id_col_name']:'user_id',
                                self.config['conversion_yes_no_col_name']:'conversion',
                                self.config['categorical_col_name']:'category',
                                self.config['revenue_col_name']:'revenue',
                                self.config['cost_col_name']:'cost' }, inplace=True)
        raw_data.category = self.config['category_name'] + '_' + raw_data.category.astype(str)
        raw_data['interaction'] = 1
        return pd.get_dummies(raw_data, columns=['category']), raw_data

    def profit_per_interaction(self):
        self.data['profit'] = self.data.revenue.astype(float) - self.data.cost.astype(float)
        self.data.to_csv(self.config['profit_per_interaction_file']+'.csv')
        self.data_dummies['profit'] = self.data_dummies.revenue - self.data_dummies.cost
        self.data_dummies.to_csv(self.config['profit_per_interaction_file']+'_dummies'+'.csv')

    def profit_per_user(self):
        if not self.config['profit_per_interaction']:
            self.data_dummies['profit'] = self.data_dummies.revenue.astype(float)  - self.data_dummies.cost.astype(float)
        temp = self.data_dummies.groupby('user_id').sum()
        temp = temp.reset_index().merge(self.data.groupby('user_id')['category'].nunique().reset_index() ,on='user_id')
        temp = temp.rename(index=str, columns={'interaction':'interaction_count', 'conversion':'conversion_count', 'category': 'unique_'+self.config['category_name']+'_count'})
        temp.to_csv(self.config['profit_per_user_file']+'.csv')

    def profit_per_cat(self):
        if not self.config['profit_per_interaction']:
            self.data['profit'] = self.data.revenue.astype(float)  - self.data.cost.astype(float) 
        temp = self.data.groupby('category').agg({'user_id': 'nunique',
                                                    'interaction': 'sum',
                                                    'conversion': 'sum',
                                                    'revenue': 'sum',
                                                    'cost': 'sum',
                                                    'profit': 'sum'})
        temp = temp.rename(index=str, columns={'interaction':'interaction_count', 'conversion':'conversion_count', 'user_id':'unique_users_count'})
        temp.to_csv(self.config['profit_per_cat_file']+'.csv')
