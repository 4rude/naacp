# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
import pickle

# Make a dataframe holding incarceration trends in the US from 1970 to 2018
inc_tre_df = pd.read_csv("/Users/matt/Desktop/naacp/data/incarceration_trends.csv", parse_dates=['year'],
                         index_col=False)

# Use numpy to fill missing values with zero
inc_tre_df = inc_tre_df.replace(np.nan, 0)

# Separate the original csv import into the counties we need data from
hennepin_df = inc_tre_df[(inc_tre_df['county_name'] == 'Hennepin County') & (inc_tre_df['state'] == 'MN')]
ramsey_df = inc_tre_df[(inc_tre_df['county_name'] == 'Ramsey County') & (inc_tre_df['state'] == 'MN')]
dakota_df = inc_tre_df[(inc_tre_df['county_name'] == 'Dakota County') & (inc_tre_df['state'] == 'MN')]
anoka_df = inc_tre_df[(inc_tre_df['county_name'] == 'Anoka County') & (inc_tre_df['state'] == 'MN')]

# Filter out the needed variables
relevant_variables = ['year', 'county_name',
                      'total_pop',
                      'total_prison_pop',
                      'female_prison_pop',
                      'male_prison_pop',
                      'aapi_prison_pop',
                      'black_prison_pop',
                      'latinx_prison_pop',
                      'native_prison_pop',
                      'other_race_prison_pop',
                      'white_prison_pop',
                      'aapi_female_prison_pop',
                      'aapi_male_prison_pop',
                      'black_female_prison_pop',
                      'black_male_prison_pop',
                      'latinx_female_prison_pop',
                      'latinx_male_prison_pop',
                      'native_female_prison_pop',
                      'native_male_prison_pop',
                      'other_race_female_prison_pop',
                      'other_race_male_prison_pop',
                      'white_female_prison_pop',
                      'white_male_prison_pop',
                      'total_prison_pop_rate',
                      'female_prison_pop_rate',
                      'male_prison_pop_rate',
                      'aapi_prison_pop_rate',
                      'black_prison_pop_rate',
                      'latinx_prison_pop_rate',
                      'native_prison_pop_rate',
                      'white_prison_pop_rate',
                      'total_prison_adm_rate',
                      'female_prison_adm_rate',
                      'male_prison_adm_rate',
                      'aapi_prison_adm_rate',
                      'black_prison_adm_rate',
                      'latinx_prison_adm_rate',
                      'native_prison_adm_rate',
                      'white_prison_adm_rate']

# Create a list of columns that should be dropped from the dataframe to create a pie chart focused dataframe
col_list = ['county_name', 'total_pop', 'total_prison_pop',
            'female_prison_pop', 'male_prison_pop', 'latinx_female_prison_pop',
            'latinx_male_prison_pop', 'native_female_prison_pop',
            'native_male_prison_pop', 'other_race_female_prison_pop',
            'other_race_male_prison_pop', 'white_female_prison_pop',
            'white_male_prison_pop', 'total_prison_pop_rate',
            'female_prison_pop_rate', 'male_prison_pop_rate', 'total_prison_adm_rate',
            'female_prison_adm_rate', 'male_prison_adm_rate',
            'aapi_prison_adm_rate', 'black_prison_adm_rate',
            'latinx_prison_adm_rate', 'native_prison_adm_rate',
            'white_prison_adm_rate', 'aapi_female_prison_pop',
            'aapi_male_prison_pop', 'black_female_prison_pop',
            'black_male_prison_pop']

# Clean up data to reduce unneeded data points
hennepin_df = hennepin_df[relevant_variables]
ramsey_df = ramsey_df[relevant_variables]
dakota_df = dakota_df[relevant_variables]
anoka_df = anoka_df[relevant_variables]

# Reset index to year to properly concatenate the series into a new dataframe named all_prison_pop_df
hennepin_df.year = hennepin_df['year'].dt.year
hennepin_df.index = hennepin_df.year

# reset the indexes
ramsey_df.year = ramsey_df['year'].dt.year
ramsey_df = ramsey_df.set_index('year')

anoka_df.year = anoka_df['year'].dt.year
anoka_df = anoka_df.set_index('year')

dakota_df.year = dakota_df['year'].dt.year
dakota_df = dakota_df.set_index('year')

# Concatenate all the series together - All Prison Population Dataframe
app_df = pd.concat([hennepin_df['year'], hennepin_df['total_prison_pop'].rename('Hennepin')], axis=1)
app_df = app_df.set_index('year')

app_df = pd.concat([app_df, ramsey_df['total_prison_pop'].rename('Ramsey')], axis=1)
app_df = pd.concat([app_df, anoka_df['total_prison_pop'].rename('Anoka')], axis=1)
app_df = pd.concat([app_df, dakota_df['total_prison_pop'].rename('Dakota')], axis=1)

app_df = pd.concat([app_df, hennepin_df['total_prison_pop_rate'].rename('hennepinRate')], axis=1)
app_df = pd.concat([app_df, ramsey_df['total_prison_pop_rate'].rename('ramseyRate')], axis=1)
app_df = pd.concat([app_df, anoka_df['total_prison_pop_rate'].rename('anokaRate')], axis=1)
app_df = pd.concat([app_df, dakota_df['total_prison_pop_rate'].rename('dakotaRate')], axis=1)

# Drop the empty
app_df = app_df.iloc[17:-2, :]


def run_visual_one():
    """ Activates a script that shows the counties demographic populations as a data visualization. """
    # Instantiate the subplot and the respective figures
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True, sharey='col')

    # Plot the datasets
    ax1.plot(app_df.index, app_df.Hennepin, c='red', linewidth=4)
    ax1.set_title("Hennepin County")

    ax1.plot(app_df.index, app_df.hennepinRate, c='darkred')
    ax1.legend(["Population", "Rate per 100k"])

    ax2.plot(app_df.index, app_df.Ramsey, c='yellow', linewidth=4)
    ax2.set_title("Ramsey County")

    ax2.plot(app_df.index, app_df.ramseyRate, c='gold')
    ax2.legend(["Population", "Rate per 100k"])

    ax3.plot(app_df.index, app_df.Anoka, c='lime', linewidth=4)
    ax3.set_title("Anoka County")

    ax3.plot(app_df.index, app_df.anokaRate, c='darkgreen')
    ax3.legend(["Population", "Rate per 100k"])

    ax4.plot(app_df.index, app_df.Dakota, c='cyan', linewidth=4)
    ax4.set_title("Dakota County")

    ax4.plot(app_df.index, app_df.dakotaRate, c='darkblue')
    ax4.legend(["Population", "Rate per 100k"])

    # Title the plot
    fig.suptitle("Prisoner Population Rate (per 100k Minnesotans) by County", fontsize=16, va='bottom')

    # Apply these property changes to ALL ax objects
    for ax in fig.get_axes():
        ax.set_xlabel('Year')
        ax.label_outer()

    # Return the plot
    return plt


def run_visual_two():
    """ Activates a script that shows the counties demographic populations as a data visualization. """
    # Chart the population demographic for each county in a pie chart for each county
    # Create lists of columns to drop from either the demographic rate or population dataframes
    pop_drop_labels = ['aapi_prison_pop_rate', 'black_prison_pop_rate', 'latinx_prison_pop_rate',
                       'native_prison_pop_rate', 'white_prison_pop_rate']
    # Create labels for pie chart
    rate_add_labels = ['aapi_prison_pop', 'black_prison_pop', 'latinx_prison_pop', 'native_prison_pop',
                       'other_race_prison_pop', 'white_prison_pop']

    pop_demo_labels = pd.Series(["AAPI", "Black", "Latinx", "Native", "Other", "White"])

    # Transpose the labels to keep the right... shape... for the... matrix?
    pop_demo_labels = pop_demo_labels.values.T

    # --- Hennepin County Data --- #
    # Drop the columns that are unneeded
    hen_dem_c_df = hennepin_df.drop(col_list, axis=1)
    # Keep the row (year 2016) that is needed
    hen_dem_c_df = hen_dem_c_df.iloc[-3:-2, :]

    hen_dem_c_df = hen_dem_c_df.drop(pop_drop_labels, axis=1)
    hen_dem_c_df = hen_dem_c_df.iloc[0:, 1:7].T

    # --- Ramsey County Data --- #
    # Drop the columns that are unneeded
    ram_dem_c_df = ramsey_df.drop(col_list, axis=1)
    # Drop the rows that are unneeded
    ram_dem_c_df = ram_dem_c_df.iloc[-3:-2, :]

    ram_dem_c_df = ram_dem_c_df.drop(pop_drop_labels, axis=1)
    ram_dem_c_df = ram_dem_c_df.iloc[0:, 0:6].T

    # --- Anoka County Data --- #
    # Drop the columns that are unneeded
    ano_dem_c_df = anoka_df.drop(col_list, axis=1)
    # Drop the rows that are unneeded
    ano_dem_c_df = ano_dem_c_df.iloc[-3:-2, :]

    ano_dem_c_df = ano_dem_c_df.drop(pop_drop_labels, axis=1)
    ano_dem_c_df = ano_dem_c_df.iloc[0:, 0:6].T

    # --- Dakota County Data --- #
    # Drop the columns that are unneeded
    dak_dem_c_df = dakota_df.drop(col_list, axis=1)
    # Drop the rows that are unneeded
    dak_dem_c_df = dak_dem_c_df.iloc[-3:-2, :]

    dak_dem_c_df = dak_dem_c_df.drop(pop_drop_labels, axis=1)
    dak_dem_c_df = dak_dem_c_df.iloc[0:, 0:6].T

    # Instantiate figure and axis objects so we can set their properties
    # Chart the rate of demographic for each county in bar charts for each county
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    # Plot the bar graphs, relating each curated dataframe to their respective axes to plot a 2x2 graph
    hen_dem_c_df.plot(kind='pie', y=2016, colormap='gist_rainbow', ax=ax1, shadow=True, startangle=90,
                      title='Hennepin County', labeldistance=None)
    ram_dem_c_df.plot(kind='pie', y=2016, colormap='gist_rainbow', ax=ax2, shadow=True, startangle=90,
                      title='Ramsey County', labeldistance=None)
    ano_dem_c_df.plot(kind='pie', y=2016, colormap='gist_rainbow', ax=ax3, shadow=True, startangle=90,
                      title='Anoka County', labeldistance=None)
    dak_dem_c_df.plot(kind='pie', y=2016, colormap='gist_rainbow', ax=ax4, shadow=True, startangle=90,
                      title='Dakota County', labeldistance=None)

    # Apply these property changes to ALL ax objects
    for ax in fig.get_axes():
        ax.legend(labels=pop_demo_labels, loc="best")

    # Title the plot
    fig.suptitle("Prisoner Population Demographics by County", fontsize=16, va='bottom')

    # Show the plot
    return plt


def run_visual_three():
    """ Activates a script that shows the counties demographic population rates as a data visualization. """
    # Labels to drop from the copied dataframes
    rate_drop_labels = ['aapi_prison_pop', 'black_prison_pop', 'latinx_prison_pop', 'native_prison_pop',
                        'other_race_prison_pop', 'white_prison_pop']
    # Create labels for pie chart
    rate_demo_labels = pd.Series(["AAPI", "Black", "Latinx", "Native", "White"])

    rate_demo_labels = rate_demo_labels.values.T

    # Drop the columns that are unneeded
    hen_dem_c_df = hennepin_df.drop(col_list, axis=1)
    # Keep the row (year 2016) that is needed
    hen_dem_c_df = hen_dem_c_df.iloc[-3:-2, :]

    hen_dem_c_df = hen_dem_c_df.drop(rate_drop_labels, axis=1)
    hen_dem_c_df = hen_dem_c_df.iloc[0:, 1:6].T

    # Drop the columns that are unneeded
    ram_dem_c_df = ramsey_df.drop(col_list, axis=1)
    # Drop the rows that are unneeded
    ram_dem_c_df = ram_dem_c_df.iloc[-3:-2, :]

    ram_dem_c_df = ram_dem_c_df.drop(rate_drop_labels, axis=1)
    ram_dem_c_df = ram_dem_c_df.iloc[0:, 0:6].T

    # Drop the columns that are unneeded
    ano_dem_c_df = anoka_df.drop(col_list, axis=1)
    # Drop the rows that are unneeded
    ano_dem_c_df = ano_dem_c_df.iloc[-3:-2, :]

    ano_dem_c_df = ano_dem_c_df.drop(rate_drop_labels, axis=1)
    ano_dem_c_df = ano_dem_c_df.iloc[0:, 0:6].T

    # Drop the columns that are unneeded
    dak_dem_c_df = dakota_df.drop(col_list, axis=1)
    # Drop the rows that are unneeded
    dak_dem_c_df = dak_dem_c_df.iloc[-3:-2, :]

    dak_dem_c_df = dak_dem_c_df.drop(rate_drop_labels, axis=1)
    dak_dem_c_df = dak_dem_c_df.iloc[0:, 0:6].T

    # Chart the rate of demographic for each county in bar charts for each county
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    canvas = FigureCanvasAgg(fig)
    fig.suptitle("Prisoner Demographic Rate (per 100k) by County in 2016", fontsize=16, va='bottom')

    # Plot the bar graphs, relating each curated dataframe to their respective axes to plot a 2x2 graph
    hen_dem_c_df.plot.bar(grid=True, color='red', ax=ax1)
    ram_dem_c_df.plot.bar(grid=True, color='yellow', ax=ax2)
    ano_dem_c_df.plot.bar(grid=True, color='lime', ax=ax3)
    dak_dem_c_df.plot.bar(grid=True, color='cyan', ax=ax4)

    # Set the titles of the subplots on their related axis variables
    ax1.set_title('Hennepin County'), ax2.set_title('Ramsey County'), ax3.set_title('Anoka County'), ax4.set_title(
        'Dakota County')

    # Apply these property changes to ALL ax objects
    for ax in fig.get_axes():
        # ax.label_outer()
        ax.set_ylabel('Population')
        ax.get_legend().remove()
        ax.set_xticklabels(rate_demo_labels, fontsize=12, rotation=45)

    # Return the plot because we want to encapsulate the script in a function for ease of use
    return plt


def run_visual_four():
    """ Activates a script that shows the male vs. female population data visualization. """
    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(16, 10))
    # Create a copy of the Hennepin dataframe
    hennepin_c_df = hennepin_df.iloc[:-2, :]
    # Create the variables to compare, x = independent, y = dependent
    x = hennepin_c_df['year']
    y1 = hennepin_c_df['total_prison_pop']
    y2 = hennepin_c_df['female_prison_pop']
    y3 = hennepin_c_df['male_prison_pop']
    # Plot the variables and their labels as lines on a graph
    plt.plot(x, y1, label='Total Hennepin Prisoner Pop')
    plt.plot(x, y2, label='Total Female Pop')
    plt.plot(x, y3, label='Total Male Pop')
    plt.plot(kind='bar', ax=ax, colormap='gist_rainbow', title='Total VS Male/Female Populations', linewidth=10)
    # Create the labels for the x and y axis
    ax.set_xlabel("Year")
    ax.set_ylabel("Population")
    # Create the legend
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 0.5))
    leg_copy = plt.legend()
    leg_lines = leg_copy.get_lines()
    plt.setp(leg_lines, linewidth=5)

    # Return the plot because we want to encapsulate the script in a function for ease of use
    return plt


def run_visual_five(hen_blk, hen_nat, model_year=2017):
    """ Activates a script that shows the male vs. female population data visualization. """
    # Prepare dataframe subsets into new dataframe that holds democraphic populations and rates by county
    # Create a list of columns that should be dropped from the dataframe to create a pie chart focused dataframe
    col_list = ['county_name', 'total_pop', 'total_prison_pop',
                'female_prison_pop', 'male_prison_pop', 'latinx_female_prison_pop',
                'latinx_male_prison_pop', 'native_female_prison_pop',
                'native_male_prison_pop', 'other_race_female_prison_pop',
                'other_race_male_prison_pop', 'white_female_prison_pop',
                'white_male_prison_pop', 'total_prison_pop_rate',
                'female_prison_pop_rate', 'male_prison_pop_rate', 'total_prison_adm_rate',
                'female_prison_adm_rate', 'male_prison_adm_rate',
                'aapi_prison_adm_rate', 'black_prison_adm_rate',
                'latinx_prison_adm_rate', 'native_prison_adm_rate',
                'white_prison_adm_rate', 'aapi_female_prison_pop',
                'aapi_male_prison_pop', 'black_female_prison_pop',
                'black_male_prison_pop']

    rate_drop_labels = ['aapi_prison_pop', 'black_prison_pop', 'latinx_prison_pop', 'native_prison_pop', 'other_race_prison_pop', 'white_prison_pop']

    rate_demo_labels = pd.Series(["AAPI", "Black", "Latinx", "Native", "White"])

    rate_demo_labels = rate_demo_labels.values.T

    # Drop the columns that are unneeded
    hen_dem_c_df = hennepin_df.drop(col_list, axis=1)
    # Keep the row (year 2016) that is needed
    hen_dem_c_df = hen_dem_c_df.iloc[-3:-2,:]

    hen_dem_c_df = hen_dem_c_df.drop(rate_drop_labels, axis=1)
    hen_dem_c_df = hen_dem_c_df.iloc[0:, 1:6].T

    hen_model_df = hen_dem_c_df.copy()
    hen_model_df[model_year] = [0, hen_blk, 0, hen_nat, 0]
    hen_model_df.drop(columns=hen_model_df.columns[0], axis=1, inplace=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), constrained_layout = True, sharey=True)
    fig.suptitle("Prisoner Demographic Rate in Hennepin County", fontsize=16, va='bottom')

    # Plot the bar graphs, relating each curated dataframe to their respective axes to plot a 2x2 graph
    hen_dem_c_df.plot.bar(grid=True, color='red', ax=ax1)
    hen_model_df.plot.bar(grid=True, color='yellow', ax=ax2)

    # Set the titles of the subplots on their related axis variables
    ax1.set_title('Hennepin County 2016'), ax2.set_title('Hennepin County ' + str(model_year))

    # Apply these property changes to ALL ax objects
    for ax in fig.get_axes():
        # ax.label_outer()
        ax.set_ylabel('Population (per 100k)')
        ax.get_legend().remove()
        ax.set_xticklabels(rate_demo_labels, fontsize=12, rotation = 45)

    # Return the plot because we want to encapsulate the script in a function for ease of use
    return plt
#%%
