import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
    

# Functions for categorical variables

def plot_categorical_distribution(df, cat_columns, n_columns = 3, *, relative = False, show_values = False, custom_labels = False, rotation = 45, palette = 'viridis'):
    '''   
    Generates bar plots to visualize the distribution of categorical variables in a given DataFrame. The function allows for plotting either the absolute or relative frequencies of the categories for each specified column. Additionally, it provides options to display frequency values directly on the bars, apply custom labels, rotate the x-axis labels, and choose a color palette. The layout of the plots is automatically adjusted based on the number of columns specified by the user.

    Parameters:
        df (pd.DataFrame): 
            DataFrame containing the data to plot. It should include the categorical columns specified in `cat_columns`.
        
        cat_columns (list of str): 
            List of column names (strings) corresponding to the categorical variables to visualize.
        
        n_columns (int, optional, default=3):
            Number of columns for the subplot grid. This value can be 1, 2, or 3. If the number of categorical columns (`cat_columns`) is fewer than `n_columns`, the grid will automatically adjust to use the exact number of columns needed. 

        relative (bool, optional, default=False): 
            If True, the function will plot the relative frequencies of the categories. If False, it will plot the absolute frequencies.

        show_values (bool, optional, default=False): 
            If True, the function will display the frequency values directly on top of the bars in the plot.

        custom_labels (dict of {str: list}, optional, default=False): 
            If provided, this dictionary should map column names to lists of custom labels to use for the x-axis ticks. The keys should be column names from `cat_columns`, and the values should be lists of labels corresponding to the categories in that column.

        rotation (int, optional, default=45): 
            Angle (in degrees) to rotate the x-axis labels for better readability.

        palette (str, optional, default='viridis'): 
            Color palette to use for the bars. If an empty string is provided, 'viridis' will be used as the default.

    Returns:
        None: 
            This function does not return any objects. It generates and displays a set of bar plots showing the frequency distributions of the specified categorical variables.
    '''

    # Validate number of specified columns
    if n_columns not in [1, 2, 3]:
        raise ValueError('n_columns must be 1, 2 or 3.')
    
    # Determine the number of columns and rows needed for the subplot grid
    n_plots = len(cat_columns)
    if n_plots in [1, 2]:
        n_columns = n_plots
    n_rows = (n_plots // n_columns) + (1 if n_plots % n_columns != 0 else 0)
    
    # Create the base figure and a grid of subplots with the specified size
    fig, axes = plt.subplots(n_rows, n_columns, figsize = (5 * n_columns, 5 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    # Set figure title and update palette if the string is empty
    plt.suptitle('Categorical Distribution', ha = 'center', y = 1, fontproperties = {'weight': 600, 'size': 14})
    
    if palette == '':
        palette = 'viridis'
    
    # Plot the frequency distribution for each categorical column
    for i, col in enumerate(cat_columns):
        ax = axes[i]
        if relative:
            # Calculate and plot the relative frequencies
            total = df[col].value_counts().sum()
            serie = df[col].value_counts().apply(lambda x: x / total)
            sns.barplot(x = serie.index, y = serie, ax = ax, palette = palette, hue = serie.index, legend = False)
            ax.set_ylabel('Relative Frequency')
        else:
            # Calculate and plot the absolute frequencies
            serie = df[col].value_counts()
            sns.barplot(x = serie.index, y = serie, ax = ax, palette = palette, hue = serie.index, legend = False)
            ax.set_ylabel('Count')

        # Set the title, ticks, grid and spine
        ax.set_title(f'{col}', ha = 'center', y = 1.025)
        ax.set_xlabel('')
        ax.tick_params(colors = '#565656')
        ax.tick_params(axis = 'x', rotation = rotation, colors = 'k')
        ax.grid(axis = 'y', color = '#CFD2D6', linewidth = 0.4)
        ax.set_axisbelow(True)
        ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        
        if custom_labels:
            # Set custom labels
            labels = custom_labels[col]
            ticks = range(len(labels))
            ax.set_xticks(ticks = ticks, labels = labels)

        if show_values:
            # Annotate each bar with its height (the frequency value)
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}' if relative else f'{height:.0f}', (p.get_x() + p.get_width() / 2., height), 
                            ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')
                
    # Hide any unused subplots if the number of categorical columns is odd
    for j in range(i + 1, n_rows * n_columns):
        axes[j].axis('off')
        
    # Adjust the layout to prevent overlap and display the plots
    plt.tight_layout(h_pad = 3, w_pad = 5)
    plt.show()

# Relationships

# Categorical-Categorical
def plot_categorical_relationship(df, cat_col1, cat_col2, *, relative = False, show_values = False, size_group = 5, rotation = 45, palette = 'viridis'):
    '''
    Generates bar plots to visualize the relationship between two categorical columns in a DataFrame. It also shows the frequency (or relative frequency) of each combination of categories in `cat_col1` and `cat_col2`. 
    If there are too many categories in `cat_col1`, the plot is divided into multiple subplots for better visualization. Additionally, it can rotate x-axis labels and display values on the bars if requested.

    Parameters:
        df (pd.DataFrame): 
            The DataFrame containing the data for the plot.
        cat_col1 (str): 
            The name of the categorical column in the DataFrame to be used for the x-axis.
        cat_col2 (str): 
            The name of the categorical column in the DataFrame to differentiate the bars in the plot (through color).
        relative (bool, optional): 
            If True, frequencies will be displayed as relative proportions instead of absolute counts. Default is False.
        show_values (bool, optional): 
            If True, values will be annotated on top of the bars. Default is False.
        size_group (int, optional): 
            Maximum number of categories to display in a single plot. If there are more categories, they will be split into multiple plots. Default is 5.
        rotation (int, optional): 
            Angle of rotation for x-axis labels. Default is 45 degrees.
        palette (str, optional): 
            Color palette to use in the plot. Default is 'viridis'. If set to an empty string, the default Seaborn palette will be used.

    Returns:
        None: The function displays the plot and does not return any value.
    '''
    
    # Prepare the data
    count_data = df.groupby([cat_col1, cat_col2]).size().reset_index(name = 'count')
    total_counts = df[cat_col1].value_counts()
    
    # Calculate relative frequencies if specified
    if relative:
        count_data['count'] = count_data.apply(lambda x: x['count'] / total_counts[x[cat_col1]], axis = 1)
    
    if palette == '':
        palette = 'viridis'

    # If there is more than size_group categories in cat_col1, they get divided into size_group groups
    unique_categories = df[cat_col1].unique()
    if len(unique_categories) > size_group:
        num_plots = int(np.ceil(len(unique_categories) / size_group))

        for i in range(num_plots):
            # Select subgroup of categories for each plot
            categories_subset = unique_categories[i * size_group:(i + 1) * size_group]
            data_subset = count_data[count_data[cat_col1].isin(categories_subset)]

            # Create plot
            plt.figure(figsize = (10, 6))
            ax = sns.barplot(x = cat_col1, y = 'count', hue = cat_col2, data = data_subset, order = categories_subset, palette = palette)
            
            # Set the title, ticks, grid and spine
            ax.set_title(f'Relationship between {cat_col1} and {cat_col2} - Group {i + 1}')
            ax.set_xlabel(cat_col1)
            ax.set_ylabel('Relative Frequency' if relative else 'Count')
            ax.tick_params(colors = '#565656')
            ax.tick_params(axis = 'x', rotation = rotation, colors = 'k')
            ax.grid(axis = 'y', color = '#CFD2D6', linewidth = 0.4)
            ax.set_axisbelow(True)
            ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)

            if show_values:
                # Annotate each bar with its height (the frequency value)
                for p in ax.patches:
                    if p.get_xy() != (0,0):
                        height = p.get_height()
                        ax.annotate(f'{height:.2f}' if relative else f'{height:.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                    ha = 'center', va = 'center', fontsize = 10, color = 'black', xytext = (0, size_group),
                                    textcoords = 'offset points')

            # Display plots
            plt.show()
    else:
        # Creates plot for less than size_group categories
        plt.figure(figsize = (10, 6))
        ax = sns.barplot(x = cat_col1, y = 'count', hue = cat_col2, data = count_data, palette = palette)
        
        # Set the title, ticks, grid and spine
        ax.set_title(f'Relationship between {cat_col1} and {cat_col2}')
        ax.set_xlabel(cat_col1)
        ax.set_ylabel('Relative Frequency' if relative else 'Count')
        ax.tick_params(colors = '#565656')
        ax.tick_params(axis = 'x', rotation = rotation, colors = 'k')
        ax.grid(axis = 'y', color = '#CFD2D6', linewidth = 0.4)
        ax.set_axisbelow(True)
        ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)

        if show_values:
            # Annotate each bar with its height (the frequency value)
            for p in ax.patches:
                if p.get_xy() != (0,0):
                    height = p.get_height()
                    ax.annotate(f'{height:.2f}' if relative else f'{height:.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha = 'center', va = 'center', fontsize = 10, color = 'black', xytext = (0, size_group),
                                textcoords = 'offset points')
        # Legend
        legend = plt.legend(title = cat_col2, title_fontsize = 'small', framealpha = 1, fontsize = 'small', edgecolor = '#565656', borderpad = 1, loc = 'best')
        legend.get_frame().set_linewidth(0.8)
  
        # Display plot
        plt.show()


# Functions for numerical variables

def plot_combined_numerical_distribution(df, columns, *, kde = True, boxplot = False, whisker_width = 1.5, bins = None):
    """
    Plots a combined visualization of histograms with optional KDE curves and boxplots for specified numerical columns in a DataFrame. It generates a grid of subplots where each row corresponds to a specified column.
    If `boxplot` is True, the first subplot in each row is a histogram (optionally with a KDE curve) and the second is a boxplot.  If `boxplot` is False, only the histogram (with optional KDE) is displayed.

    Parameters:
        df : pandas.DataFrame
            The DataFrame containing the data to plot.
        columns : list of str
            A list of column names to plot. Each column should be numerical.
        kde : bool, optional, default=True
            Whether to plot a KDE (Kernel Density Estimate) line alongside the histogram.
        boxplot : bool, optional, default=False
            Whether to include a boxplot alongside the histogram for each column.
        whisker_width : float, optional, default=1.5
            The width of the whiskers in the boxplot. Controls the extent of the whiskers relative to the IQR (Interquartile Range).
        bins : int or str, optional, default=None
            The number of bins to use for the histogram. If None, the number of bins will be determined automatically.

    Returns:
        None
            Displays the plots.
    """

    # Determine the number of columns to plot
    n_columns = len(columns)
    
    if n_columns:
        # Create the base figure and a grid of subplots. If boxplots are requested, use 2 columns, otherwise 1.
        if not boxplot:
            fig, axs = plt.subplots(n_columns, 1, figsize = (5, 5 * n_columns))
        else:
            fig, axs = plt.subplots(n_columns, 2, figsize = (10, 5 * n_columns))
        
        # Ensure axs is a 1D array, even if there's only one subplot
        axs = np.atleast_1d(axs).flatten()
        
        # Loop through each specified column to create the plots
        for i, column in enumerate(columns):
            # Check if the column is of a numerical data type
            if df[column].dtype in ['int64', 'float64']:
                # Histogram and optional KDE curve
                ax = axs[i * (2 if boxplot else 1)]
                sns.histplot(df[column], kde = kde, ax = ax, bins = 'auto' if bins is None else bins, color = '#74BBFF', alpha = 0.4)
                
                # If KDE is enabled, modify the line color for better visibility
                if kde:
                    for line in ax.lines:
                        line.set_color('#004aad')
                
                # Set title, tick parameters, and customize axis appearance
                ax.set_title(f'Histogram and KDE of {column}' if kde else f'Histogram of {column}')
                ax.tick_params(colors = '#565656')
                ax.set_axisbelow(True)
                ax.spines[['right', 'top']].set_visible(False)
                ax.spines[['left', 'bottom']].set_color('#565656')
               
                # If boxplots are requested, create a boxplot next to the histogram
                if boxplot:
                    ax = axs[i * 2 + 1]
                    sns.boxplot(x = df[column], ax = ax, whis = whisker_width, color = '#74BBFF', linecolor = 'black', linewidth = 0.8)
                    
                    # Set title, tick parameters, and customize axis appearance for the boxplot
                    ax.set_title(f'Boxplot of {column}')
                    ax.tick_params(colors = '#565656')
                    ax.spines[['right', 'top', 'left']].set_visible(False)
                    ax.spines[['bottom']].set_color('#565656')
                    
        # Set the overall title for the figure
        plt.suptitle('Histograms and Boxplots' if boxplot else 'Histograms', ha = 'center', y = 1, fontproperties = {'weight': 600, 'size': 14})
        # Adjust layout to avoid overlap
        plt.tight_layout(h_pad = 3, w_pad = 3)
        # Display the final plots
        plt.show()


def custom_scatter_plot(df, x, y, color_col = None, size_col = None, scale = 1, legend = 'auto'):
    fig, ax = plt.subplots(figsize = (20, 10))
        
    # Scatter plot
    if type(size_col) == str:
        if size_col != '':
            if color_col:
                sp = plt.scatter(x = df[x], y = df[y], c = df[color_col], s = df[size_col] * scale, cmap = 'viridis', alpha = 0.5)
            else:
                sp = plt.scatter(x = df[x], y = df[y], s = df[size_col] * scale, cmap = 'viridis', alpha = 0.5)
        elif color_col:
            sp = plt.scatter(x = df[x], y = df[y], c = df[color_col], cmap = 'viridis', alpha = 0.5)
    else:
        plt.scatter(x = df[x], y = df[y], s = df[size_col] * scale, cmap = 'viridis', alpha = 0.5)
    
    # Legend
    handles, labels = sp.legend_elements('sizes', num = 6)
    for handle in handles:
        handle.set_markerfacecolor('#CCCCCC')
        handle.set_markeredgecolor('#CCCCCC')
    
    if legend == 'colorbar':
        plt.legend(handles, labels, ncol = 3, title = size_col, title_fontsize = 'small', fontsize = 'small', frameon = False, loc = 'upper right', labelspacing = 2)
        plt.colorbar(label = f'{color_col}')
        # plt.clim(df[color_col].min(), df[color_col].max()) # No hacer esto, porque tendría en cuenta los outliers y descolocaría el resto de colores
        plt.clim(0, 7)
    else:  
        plt.legend(handles, labels, ncol = 3, title = size_col, title_fontsize = 'small', fontsize = 'small', frameon = False, loc = 'upper right', labelspacing = 2)
    
    # Figure presentation
    if legend == 'colorbar': # This fixes title horizontal alignment (not centered in colorbar option)
        plt.suptitle(f'Scatter plot: {x}, {y}{(f', {color_col}') if color_col else ''}{(f', {size_col}') if size_col else ''}', x = 0.47, y = 0.92, weight = 'bold') 
    else:
        plt.suptitle(f'Scatter plot: {x}, {y}{(f', {color_col}') if color_col else ''}{(f', {size_col}') if size_col else ''}', x = 0.5, y = 0.92, weight = 'bold')
        
    plt.xlabel(x)
    plt.ylabel(y)
    ax.spines[['top', 'right', 'bottom', 'left']].set_color('#d3d3d3')
    ax.tick_params(colors = '#565656')