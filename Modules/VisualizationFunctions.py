import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from Modules.BlahutArimotoAOS import BAiterations

"""
Set up the default theme for plots.
"""
def BAtheme():
    plt.rcParams.update({
        'lines.linewidth': 2,'font.size': 12,
        'axes.labelsize': 12,'axes.titlesize': 12,
        'xtick.labelsize': 10,'ytick.labelsize': 10,
        'legend.fontsize': 11,'legend.title_fontsize': 11,
        'axes.grid': True,'grid.linestyle': '--','grid.alpha': 0.7
    })


"""
Get a continuous color scale.
Args:
    scalename: Name of the color scale
Returns:
    Colormap object
"""
def BAcontinuouscolorscale(scalename='Blues'):
    return cm.get_cmap(scalename)


"""
Check if all elements in data are integers.
Args:
    data: Input array
Returns:
    Boolean indicating if all elements are integers
"""
def is_all_integers(data):
    return np.all(np.equal(np.mod(data,1),0))


"""
Set integer ticks on axes if data consists of integers.
Args:
    plt: Matplotlib pyplot object
    x_data: X-axis data
    y_data: Y-axis data
"""
def set_integer_ticks(plt,x_data,y_data):
    if is_all_integers(y_data):
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    if is_all_integers(x_data):
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

"""
Visualize a 2D matrix as a heatmap.
Args:
    M_xy: 2D matrix to visualize
    xvec: X-axis values
    yvec: Y-axis values
    xlabel: Label for X-axis
    ylabel: Label for Y-axis
    legendlabel: Label for the colorbar
"""
def visualize_matrix(M_xy, xvec, yvec, xlabel="x", ylabel="y", legendlabel=""):
    
    BAtheme()  # Apply the theme
    plt.figure(figsize=(8, 6))
    c = plt.pcolormesh(xvec, yvec, M_xy, shading='auto', cmap=BAcontinuouscolorscale('Blues'))
    plt.colorbar(c, label=legendlabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs. {xlabel}')
    plt.xticks(xvec)  # Only label -1 and 1 on the x-axis
    plt.yticks(yvec)  # Similarly for the y-axis
    plt.show()

"""
Plot a 2D probability distribution as a heatmap.
Args:
    pygx: 2D probability distribution
    xvec, yvec: X and Y axis values
    x_strings, y_strings: Labels for X and Y axes
    xlabel, ylabel: Axis labels
    legendlabel: Label for the colorbar
    cmap: Colormap to use
"""
def plot_scenario_xy(pygx,xvec,yvec, x_strings, y_strings,xlabel,ylabel,legendlabel,cmap='Blues'):
    
    p_df = pd.DataFrame({
        "p_ygx": 100*pygx.flatten(order='C'),
        "y": np.repeat(yvec, len(xvec)),
        "x": np.tile(xvec, len(yvec))
    })
    
    plt.figure(figsize=(8, 6))
    data = p_df.pivot(index='y', columns='x', values='p_ygx')
    c = plt.pcolormesh(xvec, yvec, data, shading='auto', cmap=cmap, vmin=0)
    plt.colorbar(c, label=legendlabel)

    # Get the colormap and normalizer
    cmap_obj = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=0, vmax=data.max().max())  # Normalizing using DataFrame values

    # Loop through data to plot text at each cell
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iloc[i, j]
            
            # Get the background color for the current cell
            bg_color = cmap_obj(norm(value))

            # Calculate brightness of the background color
            brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
            
            # Set the text color to white if background is dark, else darker
            text_color = 'white' if brightness < 0.5 else '#708090'

            # Add the text to the plot
            plt.text(xvec[j], yvec[i], f'{value:.1f}', ha='center', va='center', 
                    fontsize=6, color=text_color)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Probability of {ylabel} given {xlabel}')
    plt.xticks(ticks=xvec, labels=x_strings, rotation=45, ha='right')
    plt.yticks(ticks=yvec, labels=y_strings)
    plt.show()

"""
Plot probability distributions for a given scenario.
Args:
    scenarios: List of scenario dictionaries
    scenario_id: ID of the scenario to plot
    gamma_ao: Complexity cost parameter
"""
def plot_scenario_prob_dist(scenarios,scenario_id,gamma_ao=1/10):

    scenario=scenarios[scenario_id]
    scenario_name=scenario["name"]
    print(f"Scenario: {scenario_name}")
    plot_scenario_xy(pygx=scenario['pogs'],xvec = scenario["s_values"], yvec = scenario["o_values"], x_strings=scenario["s_strings"], y_strings=scenario["o_strings"],xlabel = "State s",ylabel = "Observation o",legendlabel = "p(o|s)",cmap='Blues')
    plot_scenario_xy(pygx=scenario['psgo'],xvec = scenario["o_values"], yvec = scenario["s_values"], x_strings=scenario["o_strings"], y_strings=scenario["s_strings"],ylabel = "State s",xlabel = "Observation o",legendlabel = "p(s|o)",cmap='Blues')
    pa, pago, pags = BAiterations(scenario, gamma_ao=gamma_ao, sample_util=0, compute_performance=False, performance_per_iteration=False, performance_as_dataframe=True, init_pago_uniformly=True)
    plot_scenario_xy(pygx=pago,xvec = scenario["o_values"], yvec = scenario["a_values"], x_strings=scenario["o_strings"], y_strings=scenario["a_strings"],xlabel = "Observation o",ylabel = "Action a",legendlabel = "p*(a|o)",cmap='Reds')
    plot_scenario_xy(pygx=pags,xvec = scenario["s_values"], yvec = scenario["a_values"], x_strings=scenario["s_strings"], y_strings=scenario["a_strings"],xlabel = "State s",ylabel = "Action a",legendlabel = "p*(a|s)",cmap='Reds')


"""
Plot results for different v values and sample sizes.
Args:
    results_v: DataFrame of results
    scenarios: List of scenario dictionaries
    x_field: Field to use for x-axis
    x_label: Label for x-axis
"""
def plot_results_v(results_v,scenarios,x_field='Var_U',x_label='Impact Volatility'):
    
    # Convert results to a dataframe for easy access and further analysis
    results_df_v = pd.DataFrame(results_v)
    results_df_v = results_df_v.sort_values(by=['scenario_id', x_field], ascending=[True, False])
    
    # Get unique n_samp values and the count
    unique_n_samp = results_df_v['n_samp'].unique()
    n_plots = len(unique_n_samp)

    # Determine the number of rows and columns for the grid
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))  # Calculate the required number of rows

    # Set up the figure for multiple subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()  # Flatten axes to easily iterate over them
    fig.suptitle(' ', fontsize=20, y=0.98) #force space between top edge and top of plots
    
    # Plot each n_samp group on the appropriate subplot
    for idx, (n_samp, n_samp_group) in enumerate(results_df_v.groupby('n_samp')):
        ax = axes[idx]
        
        # Group by 'scenario_id' within each n_samp group and plot each scenario on the same subplot
        for scenario_id, group in n_samp_group.groupby('scenario_id'):

            # Extract data for plotting
            v_values = group[x_field]
            mean_values = group['E_U_sample']
            error_values = group['standard_error_sample']
            scenario_name = group['scenario_name'].iloc[0]
            
            # Plot the central line and capture the color used
            line, = ax.plot(v_values, mean_values, label=f"{scenario_name}")
            line_color = line.get_color()  # Get the color of the line
            
            # Add the shaded area representing the standard error
            ax.fill_between(v_values, 
                            mean_values - 2*error_values, 
                            mean_values + 2*error_values, 
                            alpha=0.2)  # Adjust alpha for transparency
            
            # Highlight the maximum point
            max_idx = mean_values.argmax()  # Find index of the max value
            max_v = v_values.iloc[max_idx]  # Get corresponding v value
            max_mean = mean_values.iloc[max_idx]  # Get max mean value
            
            # Add a point at the maximum value
            ax.scatter(max_v, max_mean, color=line_color, zorder=5, s=50, label='_nolegend_')

        # Add labels, title, and legend for this subplot
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel('Expected Impact', fontsize=16)
        ax.set_title(f'Samples = {n_samp}', fontsize=17,pad=15)
        ax.grid(True)

    # Hide any unused subplots if there are fewer n_samp values than grid spaces
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    # Get handles and labels from the first subplot (borrowing legend from one plot)
    handles, labels = axes[0].get_legend_handles_labels()
    custom_handles = [Line2D([0], [0], color=h.get_color(), lw=5) for h in handles]  # Set desired line width (lw=5 here)
    
    # Create a custom handle for 'Optimal points' (dots in the plot)
    optimal_point_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Optimal Points')
    custom_handles.append(optimal_point_handle)
    labels.append('Optimal Points')

    # Create a single legend for all the plots, placing it below the entire figure
    fig.legend(custom_handles, labels, loc='lower center', ncol=len(scenarios)//2,fontsize=16, bbox_to_anchor=(0.5, -0.12))

    # Adjust layout and show the plots
    plt.tight_layout(h_pad=3.0, w_pad=3.0)  # Adjust rect to control the top margin
    plt.show()

"""
Plot results for different v values and sample sizes, organized by scenario.
Args:
    results_v: DataFrame of results
    scenarios: List of scenario dictionaries
    x_field: Field to use for x-axis
    x_label: Label for x-axis
"""
def plot_results_v_byscenario(results_v, scenarios,x_field='Var_U',x_label='Impact Volatility'):
    # Convert results to a dataframe for easy access and further analysis
    results_df_v = pd.DataFrame(results_v)
    results_df_v = results_df_v.sort_values(by=['scenario_id', x_field, 'n_samp'], ascending=[True, False, True])

    # Get unique scenarios and n_samp values, excluding scenario_id == 0
    unique_scenarios = results_df_v['scenario_id'].unique()
    unique_n_samp = results_df_v['n_samp'].unique()

    # Use the default 'tab10' color cycle
    colors = plt.cm.get_cmap('tab10').colors
    scenario_colors = {scenario_id: colors[(idx+0) % len(colors)] for idx, scenario_id in enumerate(unique_scenarios)}

    # Prepare line styles for n_samp
    line_styles = [':', '-.', '--', '-']
    n_line_styles = len(line_styles)
    n_samp_line_styles = {n_samp: line_styles[(idx) % n_line_styles] for idx, n_samp in enumerate(unique_n_samp)}

    # Set up the figure with 4 subplots (2 rows, 2 columns)
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.flatten()

    # Plot data for each scenario (excluding scenario_id == 0)
    for i, scenario_id in enumerate(unique_scenarios):
        ax = axs[i]
        scenario_name = results_df_v[results_df_v['scenario_id'] == scenario_id]['scenario_name'].iloc[0]
        for n_samp in unique_n_samp:
            group = results_df_v[(results_df_v['scenario_id'] == scenario_id) & (results_df_v['n_samp'] == n_samp)]
            if group.empty:
                continue

            # Extract data for plotting
            v_values = group[x_field]
            mean_values = group['E_U_sample']
            error_values = group['standard_error_sample']

            # Plot the central line
            ax.plot(v_values, mean_values, color=scenario_colors[scenario_id], linestyle=n_samp_line_styles[n_samp])

            # Add the shaded area representing the standard error
            ax.fill_between(v_values, 
                            mean_values - 2 * error_values, 
                            mean_values + 2 * error_values, 
                            alpha=0.1, color=scenario_colors[scenario_id], linestyle=n_samp_line_styles[n_samp])

            # Highlight the maximum point
            max_idx = mean_values.argmax()
            max_v = v_values.iloc[max_idx]
            max_mean = mean_values.iloc[max_idx]

            # Add a point at the maximum value
            ax.scatter(max_v, max_mean, color=scenario_colors[scenario_id], zorder=5, s=50)

        # Set titles to just scenario name
        ax.set_title(scenario_name, fontsize=14)
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel('Expected Impact', fontsize=16)
        ax.grid(True)

    # Hide the sixth subplot
    fig.delaxes(axs[-1])

    # Create custom legend handles for line styles, independent of colors
    n_samp_handles = [Line2D([0], [0], color='black', linestyle=n_samp_line_styles[n_samp], lw=2) for n_samp in unique_n_samp]
    n_samp_labels = [f'Samples = {n_samp}' for n_samp in unique_n_samp]

    # Add the legend into the empty space of the sixth subplot
    axs[-1] = fig.add_subplot(3, 2, 6)
    axs[-1].axis('off')  # Hide the axes

    # Place the legend in the sixth subplot
    axs[-1].legend(n_samp_handles, n_samp_labels, loc='center', fontsize=16)

    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at the bottom for the legend
    plt.show()

"""
Plot frontier curves for different scenarios.
Args:
    res_df: DataFrame of results
    scenarios: List of scenario dictionaries
    x_field: Field to use for x-axis
    y_field: Field to use for y-axis
    x_label: Label for x-axis
    y_label: Label for y-axis
    highlight_max: Whether to highlight maximum points
    display_error_bounds: Whether to display error bounds
    num_rows, num_cols: Number of rows and columns for subplots
    title: Title for the plot
"""
def plot_frontiers(res_df,scenarios,x_field='Var_U',y_field='E_U',x_label='x',y_label='y',highlight_max=False,display_error_bounds=False, num_rows=1, num_cols=1, title=None):
    scenario_colors = plt.colormaps.get_cmap('tab10').colors #get standard colors so can link to scenario_id
    res_df = res_df.sort_values(by=['scenario_id',x_field], ascending=[True,True])

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8*num_cols, 6*num_rows), squeeze=False)
    axes = axes.flatten()  # This makes it easier to iterate over axes

    if title:
        fig.suptitle(title, fontsize=18)

    ax = axes[0]
    for scenario_id, group in res_df.groupby('scenario_id'):
        x_values=group[x_field]
        y_values=group[y_field]
        ax.plot(x_values, y_values,color=scenario_colors[scenario_id], label=f"{group['scenario_name'].iloc[0]}")#, marker='x')

        # Add shaded area representing the standard error
        if display_error_bounds:
            error_values = group['standard_error_sample']
            ax.fill_between(x_values, 
                            y_values - 2 * error_values, 
                            y_values + 2 * error_values, 
                            alpha=0.21, color=scenario_colors[scenario_id])

        if highlight_max:       # Highlight the maximum point
            max_idx = y_values.argmax()
            max_x = x_values.iloc[max_idx]
            max_y = y_values.iloc[max_idx]
            ax.scatter(max_x, max_y, color=scenario_colors[scenario_id], zorder=5, s=50)

    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    # Get handles and labels from the first subplot (borrowing legend from one plot)
    handles, labels = ax.get_legend_handles_labels()
    custom_handles = [Line2D([0], [0], color=h.get_color(), lw=5) for h in handles]  # Set desired line width (lw=5 here)
    if highlight_max: # Create a custom handle for 'Optimal points' (dots in the plot)
        optimal_point_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Highest Impact Points')
        custom_handles.append(optimal_point_handle)
        labels.append('Optimal Points')

    # Create a single legend for all the plots, placing it below the entire figure
    fig.legend(custom_handles, labels, loc='lower center', ncol=len(scenarios)//2,fontsize=11, bbox_to_anchor=(0.5, -0.15))

    # plt.xlim(left=0,right=.00001)
    plt.grid(True); plt.show()

"""
Create a bar chart comparing results for different scenarios.
Args:
    results: DataFrame of results
    maxvar: Field to use for determining maximum values
    barvar: Field to use for bar heights
    x_label: Label for x-axis
    y_label: Label for y-axis
"""
def bar_by_scenario(results, maxvar='E_U_sample', barvar='mid_pa', x_label='Scenario', y_label='Percent mixed actions'):
    scenario_colors = plt.colormaps.get_cmap('tab10').colors #get standard colors so can link to scenario_id
    
    # Group by scenario_id and find the row with max E_U for each group
    max_eu_data = results.loc[results.groupby('scenario_id')[maxvar].idxmax()]
    
    # Find the corresponding rows where gamma_var is 0
    gamma_0_data = results[results['gamma_var'] == 0].groupby('scenario_id').first().reset_index()

    # Create the bar chart
    plt.figure(figsize=(15, 6))  # Increased width to accommodate more bars
    x = np.arange(len(max_eu_data['scenario_name']))
    width = 0.35  # Width of each bar

    # Create paired bars
    bars1 = plt.bar(x - width/2, gamma_0_data[barvar], width, label='gamma_var = 0')
    bars2 = plt.bar(x + width/2, max_eu_data[barvar], width, label='max E_U')

    # Customize the chart
    plt.ylabel(y_label, fontsize=16)

    # Add color and value labels on top of each bar
    for i, (index, row) in enumerate(max_eu_data.iterrows()):
        scenario_id = row['scenario_id']
        color = scenario_colors[scenario_id % len(scenario_colors)]  # Use modulo to avoid index out of range
        
        # gamma_var = 0 bar
        bar1 = bars1[i]
        bar1.set_facecolor((*color[:3], 0.5))  # Set alpha to 0.5 for gamma_var = 0 bars
        height1 = bar1.get_height()
        plt.text(bar1.get_x() + bar1.get_width()/2., height1,
                f'{height1:.2f}',
                ha='center', va='bottom', color=color, fontsize=12)
        
        # max E_U bar
        bar2 = bars2[i]
        bar2.set_facecolor((*color[:3], 0.8))  # Set alpha to 0.8 for max E_U bars
        height2 = bar2.get_height()
        plt.text(bar2.get_x() + bar2.get_width()/2., height2,
                f'{height2:.2f}',
                ha='center', va='bottom', color=color, fontsize=12)

    plt.tight_layout()
    plt.show()