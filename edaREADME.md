
I have created two scripts that will

1) create profitability information for a column of categorical data (ie browser type, internet speed...ect). This script is call single_column_cat_profitability.py and it has a json configuration file called single_column_cat_profitabilty_config.json.
2) create a pairplot of the data provided. This script is called create_pairplot.py and it has a json configuration file called create_pairplot_config.json.

There is main control script called loop_through_multiple_analysis.py with configuration file called config_list.json. This is the only configuration file that can't be renamed without making changes in the python code. Please note that these processes are ram intensive. A 1 to 1.5 Gb file will take up about 3-4 Gb of ram while being processed. A future improvement would be moving as much of the processing as possible into Spark so you can process more data at a time.

I suggest always running the scripts through the main control script as it will be the easiest to call from your computer's terminal and you can set it up to run as many or as few scripts as you would like.

json files: json files are made up keys and values. The key names come first and must not be renamed. The values are what you change to control the scripts.

```
config_list.json
{
  "sccp": either true or false - controls if the files listed on the next line are run
  "single_column_cat_profitability_config_files":[ a list of config file names, separated by commas and in double quotation marks]
  "cpc":either true or false - controls if the files listed on the next line are run
  "create_pairplot_config_files":[a list of config file names, separated by commas and in double quotation marks]
}

create_pairplot.json
{
  "data_file_path": if the data is only coming from one file, put the file name here in double quotes
  "image_file_location": put the full path and file name for the image file you want created. It will overwrite if the file already exists
  "multiple_files": if you need data pulled from multiple files, put true, otherwise put false
  "multiple_file_paths":if you are only pulling data from one file, put null, otherwise put a list [with each file name wrapped in double quotes and separated by commas]
  "limit_columns":if you want to only look a a few of the columns in the file(s), put true, otherwise put false and all the columns will be read in
  "columns_to_use":if false above, put null. Otherwise, if multiple files are being read in, put null here and move to the next line. Otherwise, put a list [ with each column name wrapped in double quotes and separated by commas ]
  "multiple_columns_to_use":leave as null unless you are reading in multiple files and need to select specific columns from each. If you are using this, it will be a list of lists [[],[],[]] each list will match to a file listed above based on the order they are each provided. each inner list should contain the columns wanted wrapped in double quotes and separated by columns.
 }

single_column_cat_profitablity_config.json
{
    "data_file_path": put the file name here in double quotes
    "user_id_col_name": put the name of the column in the file that you want used as the user_id
    "conversion_yes_no_col_name":put the name of the column in the file that you want to use to denote conversion. should be binary (0,1) for no and yes
    "categorical_col_name":put the name of the column that has categorical values and you want analyzed
    "revenue_col_name":"put the name of the column you want used for revenue
    "cost_col_name":put the name of the column you want used for cost
    "category_name": put the name you want added to the beginning of each category, I suggest the name of the column
    "profit_per_cat":true if you want to get the profit per categorical value, false otherwise
    "profit_per_cat_file":null if above is false, otherwise the name of the file you want created. In this case, leave off the .csv
    "profit_per_user":true if you want the profit per user, false otherwise.
    "profit_per_user_file": null if above is false, otherwise the name of the file you want created. In this case, leave off the .csv
    "profit_per_interaction":true if you want the profit per interaction, false otherwise
    "profit_per_interaction_file":null if above is false, otherwise the name of the file you want created. In this case, leave off the .csv
}
```
