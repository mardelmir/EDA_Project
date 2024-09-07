import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr


def classify_by_cardinality(df, discrete_threshold = 9, continuous_threshold = 15, sort_ascending = 'origin', sugg_type = None, index_first = None):
    '''
    Classifies the columns of a DataFrame based on their cardinality and suggests a variable type for each column.
    It also identifies potential columns to use as an index.

    Parameters:
    -----------
        df (DataFrame)
            The DataFrame to analyze.
        discrete_threshold (int)
            Minimum cardinality threshold to consider a variable as a Numerical discrete type. Defaults to 9.
        continuous_threshold (int)
            Minimum cardinality threshold to consider a variable as a Numerical continuous type. Defaults to 15.
        sort_ascending (None | bool) 
            If specified, sorts the DataFrame by percentage cardinality. Useful if the suggested index is not correct.
        sugg_type (string | None)
            If specified, filters the DataFrame to include only columns with the suggested type.
        index (None | bool)
            If specified, filters the DataFrame to include or exclude possible index columns based on the boolean value.

    Returns:
    --------
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

def categorical_correlation_test(df, target, cat_cols, *, alpha=0.05, significant_only=False):
    '''
    Computes the chi-squared correlation between a primary categorical column and one or more secondary categorical columns. 
    Identifies columns from `cat_cols` that are significantly associated with `target` based on a p-value threshold of `alpha`. 
    Returns a DataFrame containing detailed results for each chi-squared test conducted.

    Parameters:
    -----------
        df : pandas.DataFrame
            The DataFrame containing the categorical columns to be analyzed.
        
        target : str
            The name of the primary categorical column in `df` for which correlations with other columns are assessed.
        
        cat_cols : str or list of str
            A column name or a list of column names in `df` to compare with `target`. The function will compute the chi-squared
            statistic for each column in this list against `target`.
        
        alpha : float, optional
            The significance level for determining whether a p-value indicates a significant association. Default is 0.05.
        
        significant_only : bool, optional
            If True, only columns with a p-value less than `alpha` will be included in the returned DataFrame. Default is False.

    Returns:
    --------
        pandas.DataFrame
            A DataFrame where rows correspond to columns from `cat_cols`, with the following columns:
                - 'chi2': The chi-squared statistic.
                - 'p_value': The p-value of the test.
                - 'dof': The degrees of freedom of the test.
                - 'expected': The expected frequencies table computed for the chi-squared test.
                - 'significant': Boolean indicating if the p-value is less than the alpha threshold.
    
    Notes:
    ------
        - If `cat_cols` is passed as a string, it will be converted into a list containing that string.
        - The function skips comparing `target` with itself to avoid meaningless self-correlation.
        - The chi-squared test is only valid for categorical data with sufficient sample size in each category.
    '''
    
    # Ensure cat_cols is a list
    if isinstance(cat_cols, str):
        cat_cols = [cat_cols]
        
    # Initialize dictionary to store results
    results = {}
    
    # Compute chi-squared test for each column in cat_cols (excluding the target column itself)
    for col in cat_cols:
        if col != target:
            # Create contingency table and perform chi-squared test
            contingency_table = pd.crosstab(df[target], df[col], margins=False)
            chi2, p, dof, expected = chi2_contingency(contingency_table)
                
            # Store detailed test results for the current column
            results[col] = {
                'chi2': chi2,
                'p_value': p,
                'dof': dof,
                'expected': expected.tolist()  # Convert NumPy array to list for DataFrame compatibility
            }
    
    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame(results).T.astype({'chi2': 'float64', 'p_value': 'float64', 'dof': 'int32'})
    results_df['significant'] = results_df['p_value'] < alpha
    results_df = results_df.sort_values('p_value', ascending=False)
    
    # Filter results to include only significant results if specified
    if significant_only:
        results_df = results_df[results_df['significant']]
        
    return results_df

def categorical_numerical_test(df, target, alpha = 0.05, significant_only = False):
    '''
    Performs the Mann-Whitney U test to determine if there are significant differences in the distributions of numerical columns between two groups defined by a binary target variable.

    Parameters:
    -----------
        df : pandas.DataFrame
            The DataFrame containing the data.
        target : str
            The name of the binary target variable (categorical) used to split the data into two groups.
        alpha : float, optional
            The significance level to determine if the p-value indicates a statistically significant difference,
            by default 0.05.
        significant_only : bool, optional
            If True, the function returns only the columns with statistically significant results (p-value < alpha), 
            by default False.

    Returns:
    --------
        results_df : pandas.DataFrame
            A DataFrame containing the U statistic, p-value, and a boolean flag indicating statistical significance for each numerical column tested. 
            If `significant_only` is True, only the columns with significant results are returned.
    '''
    
    # Identify all numerical columns in the DataFrame
    num_cols = df.select_dtypes(include = np.number).columns.tolist()

    # Initialize a dictionary to store the test results
    results = {}

    # Retrieve the unique values of the binary target variable
    target_values = df[target].value_counts().index.to_list()

    # Validate that the target variable has only two unique values
    if len(target_values) != 2:
        raise ValueError(f'The target variable "{target}" must have exactly two unique values.')

    # Split the DataFrame into two groups based on the binary target variable
    group_a = df[df[target] == target_values[0]]
    group_b = df[df[target] == target_values[1]]

    # Loop through each numerical column to compare its distribution between the two groups
    for col in num_cols:
        if col != target:
            # Perform the Mann-Whitney U test
            u_stat, p_value = mannwhitneyu(group_a[col], group_b[col])

            # Store the U statistic and p-value in the results dictionary
            results[col] = {'u_stat': u_stat, 'p_value': p_value}

    # Convert the results dictionary to a DataFrame for easier handling
    results_df = pd.DataFrame(results).T
    results_df['significant'] = results_df['p_value'] < alpha

    # If significant_only is True, filter the results to include only significant results
    if significant_only:
        results_df = results_df[results_df['significant'] == True]

    # Return the DataFrame with the test results
    return results_df

def numerical_correlation_spearman(df, target, *, show_heatmap = True, annot = False, threshold = 0, significant_only = False):
    '''
    Computes the Spearman correlation between a binary target variable and all other numerical variables in the DataFrame, and optionally visualizes the results using a heatmap.

    Parameters:
    -----------
        df : pandas.DataFrame
            The DataFrame containing the data.
        target : str
            The name of the binary target variable (categorical) used to compute correlations with numerical features.
        show_heatmap : bool, optional, default=True
            If True, displays a heatmap of the Spearman correlation coefficients.
        annot : bool, optional, default=False
            If True, annotates the heatmap with the correlation coefficients.
        significant_only : bool, optional, default=False
            If True, filters the output to include only significant correlations (p-value < 0.05).

    Returns:
    --------
        spearman_df : pandas.DataFrame
            A DataFrame containing the Spearman correlation coefficients, p-values, and significance indicator.

    Notes:
    ------
        - The function computes the Spearman correlation, which is a non-parametric measure of rank correlation, between a binary target variable and all other numerical variables in the DataFrame. 
        - The resulting correlations can be visualized in a heatmap if desired.
    '''

    # Initialize dictionaries to store correlation coefficients and heatmap data
    correlations = {}
    heatmap = {}
    
    # Iterate over all numerical columns in the DataFrame, excluding the target
    for col in df.select_dtypes(include = np.number).columns:
        if col != target:
            # Compute Spearman correlation and p-value between the target and the current numerical column
            corr, p_value = spearmanr(df[target], df[col])
            heatmap[col] = {f'{target}': corr}
            correlations[col] = {'spearman_corr': corr, 'p_value': p_value}
    
    # Convert the heatmap data to a DataFrame for easier visualization
    plot_df = pd.DataFrame(heatmap)
    
    # Convert the correlations dictionary to a DataFrame and add a significance indicator
    spearman_df = pd.DataFrame(correlations).T
    spearman_df['significant'] = spearman_df['p_value'] < 0.05
    
    # Sort the DataFrame by the correlation coefficient for better readability
    spearman_df = spearman_df.sort_values('spearman_corr')
    
    # If significant_only is True, filter to include only statistically significant correlations
    if significant_only:
        spearman_df = spearman_df[spearman_df['significant'] == True]

    # If show_heatmap is True, display a heatmap of the correlation coefficients
    if show_heatmap:
        plt.figure(figsize = (10, 1))
        if annot:
            # Create a custom annotation array
            annot = plot_df.map(lambda x: f'{x:.2f}' if isinstance(x, (int, float)) and abs(x) > threshold else '')
        
        sns.heatmap(plot_df, vmin = -1, vmax = 1, annot = annot, cmap = 'coolwarm', center = 0, fmt = '')
        plt.title(f'Spearman Correlation with {target}', y = 1.25)
        
        # Rotate the x-axis labels (column labels)
        plt.xticks(rotation = 45, ha = 'right')
    
        # # Rotate the y-axis labels (row labels)
        plt.yticks(rotation = 0)  # You can adjust this if you want to rotate the y-axis labels
        plt.show()
    
    return spearman_df


# Adds 1 space for strings written in PascalCase or camelCase
add_space = lambda text: re.sub(r'(?<!^)(?=[A-Z])', ' ', text)