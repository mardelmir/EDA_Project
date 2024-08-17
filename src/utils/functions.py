import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def variable_type(df, discrete_threshold = 9, continuous_threshold = 15, sort_ascending = None, sugg_type = None, index = None):
    '''
    Calculates cardinality and suggest a variable type to each column of a dataframe. It also suggests variables to use as index.

    Args:
        df (DataFrame): dataframe to analyze
        discrete_threshold (int): minimum cardinality threshold to consider the variable as a numeric discrete type.
        continuous_threshold (int): minimum cardinality threshold to consider the variable as a numeric continuous type.
        sort_ascending (None | bool): sorts by % cardinality, useful if suggested index is not correct.
        sugg_type (string | None): filters dataframe by specified suggested types.
        index (None | bool): filters dataframe by possible index.
    
    Returns:
        DataFrame
    '''
    # Dataframe creation
    df_temp = pd.DataFrame([df.nunique(), df.nunique() / len(df) * 100, df.dtypes]).T
    df_temp = df_temp.rename(columns = {0: 'cardinality', 1: '%_cardinality', 2: 'type'})
    
    # Suggested type based on calculated cardinality
    df_temp['suggested_type'] = 'Categorical'
    df_temp.loc[df_temp['cardinality'] == 1, '%_cardinality'] = 0.00
    df_temp.loc[df_temp['cardinality'] == 2, 'suggested_type'] = 'Binary'
    df_temp.loc[df_temp['%_cardinality'] >= discrete_threshold, 'suggested_type'] ='Numeric (discrete)'
    df_temp.loc[df_temp['%_cardinality'] >= continuous_threshold, 'suggested_type'] = 'Numeric (continuous)'
    
    # Index suggestion
    df_temp['possible_index'] = False
    index_cond = (df_temp['%_cardinality'] == 100) & (df_temp.index.str.contains('id', case = False, regex = False))
    df_temp.loc[index_cond, 'possible_index'] = True
    
    # Returns dataframe sorted by % cardinality, useful if suggested index is not correct
    if type(sort_ascending) is bool:
        df_temp.sort_values(by = '%_cardinality', ascending = sort_ascending, inplace = True)
    
    # Returns dataframe that only includes specified suggested types
    if sugg_type:
        df_temp = df_temp.loc[df_temp['suggested_type'].str.contains(sugg_type, case = False)]
    
    # Returns dataframe with possible index. Can also be set to exclude possible index suggestions
    if type(index) is bool:
        df_temp = df_temp.loc[df_temp['possible_index'] == index]
    
    return df_temp


