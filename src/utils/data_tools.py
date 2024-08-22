import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats


def classify_by_cardinality(df, discrete_threshold = 9, continuous_threshold = 15, sort_ascending = 'origin', sugg_type = None, index_first = None):
    '''
    Classifies the columns of a DataFrame based on their cardinality and suggests a variable type for each column.
    It also identifies potential columns to use as an index.

    Args:
        df (DataFrame): The DataFrame to analyze.
        discrete_threshold (int): Minimum cardinality threshold to consider a variable as a Numerical discrete type. Defaults to 9.
        continuous_threshold (int): Minimum cardinality threshold to consider a variable as a Numerical continuous type. Defaults to 15.
        sort_ascending (None | bool): If specified, sorts the DataFrame by percentage cardinality. Useful if the suggested index is not correct.
        sugg_type (string | None): If specified, filters the DataFrame to include only columns with the suggested type.
        index (None | bool): If specified, filters the DataFrame to include or exclude possible index columns based on the boolean value.

    Returns:
        DataFrame: A DataFrame with the following columns:
            - 'Cardinality': Number of unique values in the column.
            - '% Cardinality': Percentage of unique values relative to the total number of rows.
            - 'Type': The data type of the column.
            - 'Suggested Type': Suggested type based on cardinality (e.g., 'Categorical', 'Binary', 'Numerical (discrete)', 'Numerical (continuous)').
            - 'Possible Index': Boolean flag indicating if the column could be used as an index.
    '''
    
    # Dataframe creation
    df_temp = pd.DataFrame([df.nunique(), df.nunique() / len(df) * 100, df.dtypes]).T
    df_temp = df_temp.rename(columns = {0: 'Cardinality', 1: '% Cardinality', 2: 'Type'})
    
    # Initial suggested type based on calculated cardinality
    df_temp['Suggested Type'] = 'Categorical'
    df_temp.loc[df_temp['Cardinality'] == 1, '% Cardinality'] = 0.00
    df_temp.loc[df_temp['Cardinality'] == 2, 'Suggested Type'] = 'Binary'
    df_temp.loc[df_temp['Cardinality'] >= discrete_threshold, 'Suggested Type'] ='Numerical (discrete)'
    df_temp.loc[df_temp['% Cardinality'] >= continuous_threshold, 'Suggested Type'] = 'Numerical (continuous)'
    
    # Adjust classification for datetime columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df_temp.at[col, 'Suggested Type'] = 'Date/Time'
    
    # Adjust classification for possible identifiers (alphabetic, Numerical or alphaNumerical), adds index suggestion
    df_temp['Possible Index'] = False
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].str.contains(r'\w').any() and df_temp.at[col, '% Cardinality'] == 100.0:
            df_temp.at[col, 'Suggested Type'] = 'Categorical (id)'
            df_temp.at[col, 'Possible Index'] = True  
        elif pd.api.types.is_integer_dtype(df[col]) and df_temp.at[col, '% Cardinality'] == 100.0:
            df_temp.at[col, 'Suggested Type'] = 'Numerical (id)'
            df_temp.at[col, 'Possible Index'] = True
    
    # Sort by % cardinality if specified
    if isinstance(sort_ascending, bool):
        df_temp.sort_values(by = '% Cardinality', ascending = sort_ascending, inplace = True)
    
    # Filter by suggested type if specified
    if sugg_type:
        df_temp = df_temp.loc[df_temp['Suggested Type'].str.contains(sugg_type, case = False)]
    
     # Filter by possible index if specified
    if isinstance(index_first, bool):
        df_temp = df_temp[df_temp['Possible Index'] == index_first]
    
    return df_temp

# Adds 1 space for strings written in PascalCase or camelCase
add_space = lambda text: re.sub(r'(?<!^)(?=[A-Z])', ' ', text)