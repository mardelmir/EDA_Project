import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency


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

def categorical_correlation(df, cat_col1, cat_cols2):
    '''
    Computes the chi-squared correlation between a primary categorical column and one or more secondary categorical columns. It identifies columns from `cat_cols2` that are significantly associated with `cat_col1` based on a p-value threshold of 0.05. The function also returns detailed information for each chi-squared test conducted.

    Parameters:
        df : pandas.DataFrame
            The DataFrame containing the categorical columns to be analyzed.
        
        cat_col1 : str
            The name of the primary categorical column in `df` for which correlations with other columns are assessed.
        
        cat_cols2 : str or list of str
            A column name or a list of column names in `df` to compare with `cat_col1`. The function will compute the chi-squared
            statistic for each column in this list against `cat_col1`.

    Returns:
        correlated_cols : dict
            A dictionary where the keys are the names of the columns from `cat_cols2` that have a significant association
            (p-value < 0.05) with `cat_col1`, and the values are their corresponding p-values.
        
        all_info : dict
            A dictionary containing detailed results for each chi-squared test conducted. The keys are the names of the columns
            from `cat_cols2`, and the values are dictionaries with the following keys:
                - 'chi2': The chi-squared statistic.
                - 'p': The p-value of the test.
                - 'dof': The degrees of freedom of the test.
                - 'expected': The expected frequencies table computed for the chi-squared test.
    
    Notes:
        - If `cat_cols2` is passed as a string, it will be converted into a list containing that string.
        - The function skips comparing `cat_col1` with itself to avoid meaningless self-correlation.
        - The chi-squared test is only valid for categorical data with sufficient sample size in each category.
    '''
    
    # Validate cat_cols2 type; if a string is passed, convert it into a list
    if isinstance(cat_cols2, str):
        cat_cols2 = [cat_cols2]
        
    # Initialize dictionaries to store results
    correlated_cols = {}
    all_info = {}
    
    # Loop through each column in cat_cols2 to compare with cat_col1
    for col in cat_cols2:
        # Contingency table and chi-squared test
        contingency_table = pd.crosstab(df[cat_col1], df[col], margins = False)
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        # Check if the p-value is significant and avoid self-correlation
        if p < 0.05 and col != cat_col1:
            correlated_cols[col] = p
            
        # Store detailed test results for the current column
        all_info[col] = {'chi2': chi2, 'p': p, 'dof': dof, 'expected': expected}
        
    # Return the dictionary of correlated columns and the detailed information
    return correlated_cols, all_info


# Adds 1 space for strings written in PascalCase or camelCase
add_space = lambda text: re.sub(r'(?<!^)(?=[A-Z])', ' ', text)