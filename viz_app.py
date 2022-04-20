# Import libraries
import pandas as pd
from tools import visual_runner as vr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import streamlit as st
from sklearn.model_selection import train_test_split

"""
Top-of-page/sidebar UI components 
"""

# Page configuration variables
padding = 0
st.set_page_config(page_title="NAACP Data Visualization & Analysis", layout="wide", page_icon="📍")

# Title the page and give it a subtitle
st.header("Incarceration Trends in Minnesota's largest counties")
st.subheader("Data Visualization and Analysis")

# Create a web app sidebar form to obtain user input for a selected year
with st.form(key='my_form'):
    st.text('Use the slider to predict future population rates of disenfranchised Hennepin prisoners.')
    year_to_predict = st.slider('Slide to pick a year', 2017, 2050)
    st.form_submit_button('Predict')

# Create a web app expander on the sidebar for Q/A
expander = st.sidebar.expander("How to you use this application?")
expander.write(
    """
This application allows researchers to view organized incarceration data on the specified Minnesota counties. The
purpose of this application is to highlight demographic inequalities in Minnesota prison populations.

The first series of visualizations on the main area of this web page highlight key demographic rates in Minnesota 
prison populations.

After the visualizations, the page hosts a dynamic component. Use the text box above to enter a number of years, and
the descriptive model will predict the demographic rates of Hennepin county prison populations! Hennepin County 
population data was chosen as a model because of its rates of inequality, size, and relevance to the Twin Cities 
(Minnesota) metro region.
"""
)

"""
Import/clean dataset
"""

# Make a dataframe holding incarceration trends in the US from 1970 to 2018
inc_tre_df = pd.read_csv("data/incarceration_trends.csv", parse_dates=['year'], index_col=False)

# Use numpy to fill missing values with zero
inc_tre_df = inc_tre_df.replace(np.nan,0)

# Separate the original csv import into the counties we need data from
hennepin_df = inc_tre_df[(inc_tre_df['county_name'] =='Hennepin County') & (inc_tre_df['state'] =='MN')]
ramsey_df = inc_tre_df[(inc_tre_df['county_name'] =='Ramsey County') & (inc_tre_df['state'] =='MN')]
dakota_df = inc_tre_df[(inc_tre_df['county_name'] =='Dakota County') & (inc_tre_df['state'] =='MN')]
anoka_df = inc_tre_df[(inc_tre_df['county_name'] =='Anoka County') & (inc_tre_df['state'] =='MN')]

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

# Clean up data to reduce unneeded data points
hennepin_df = hennepin_df[relevant_variables]
ramsey_df = ramsey_df[relevant_variables]
dakota_df = dakota_df[relevant_variables]
anoka_df = anoka_df[relevant_variables]

"""
Create the linear regression model object 
"""
# Create model objects to process data
hen_blk_model = linear_model.Lasso(alpha = 0.1)
hen_native_model = linear_model.Lasso(alpha = 0.1)
ram_blk_model = linear_model.Lasso(alpha = 0.1)

"""
Clean data
"""
# Create a dataframe holding metro area demographic data of the most unequal prison population rates
metro_blk_rate = pd.concat([hennepin_df['year'], hennepin_df['black_prison_pop_rate'].rename('hennepin')], axis=1)
metro_blk_rate = metro_blk_rate.set_index('year')

metro_blk_rate = pd.concat([metro_blk_rate, ramsey_df['black_prison_pop_rate'].rename('ramsey')], axis=1)
metro_blk_rate = pd.concat([metro_blk_rate, anoka_df['black_prison_pop_rate'].rename('anoka')], axis=1)
metro_blk_rate = pd.concat([metro_blk_rate, dakota_df['black_prison_pop_rate'].rename('dakota')], axis=1)

metro_native_rate = pd.concat([hennepin_df['year'], hennepin_df['native_prison_pop_rate'].rename('hennepin')], axis=1)
metro_native_rate = metro_native_rate.set_index('year')

metro_native_rate = pd.concat([metro_native_rate, ramsey_df['native_prison_pop_rate'].rename('ramsey')], axis=1)
metro_native_rate = pd.concat([metro_native_rate, anoka_df['native_prison_pop_rate'].rename('anoka')], axis=1)
metro_native_rate = pd.concat([metro_native_rate, dakota_df['native_prison_pop_rate'].rename('dakota')], axis=1)

# Remove empty rows from dataframes
metro_blk_rate = metro_blk_rate.iloc[24:-2,:]
metro_native_rate = metro_native_rate.iloc[24:-2,:]

# Separate out hennepin black and native rates per 100k
hen_blk_rate = metro_blk_rate.iloc[:,:1]
hen_native_rate = metro_native_rate.iloc[:,:1]

# Create X and y variables that hold formatted column data for hennepin black rate
hb_X = hen_blk_rate.index.values.reshape(-1, 1)
hb_y = hen_blk_rate.values.reshape(-1, 1)

# Create X and y variables that hold formatted column data for hennepin native rate
hn_X = hen_native_rate.index.values.reshape(-1, 1)
hn_y = hen_native_rate.values.reshape(-1, 1)

"""
Split data into train/test variables
"""
# Break up black demographic rate data
Xb_train, Xb_test, yb_train, yb_test = train_test_split(hb_X, hb_y)

# Break up native demographic rate data
Xn_train, Xn_test, yn_train, yn_test = train_test_split(hn_X, hn_y)

"""
Fit the model
"""
# Start with Hennepin
hen_blk_model.fit(X_train, y_train)

# Start with Hennepin
hen_native_model.fit(Xn_train, yn_train)

"""
Use the model to make predictions and score the predictions
"""
# We have two primary features, the year and the prisoner population.
y_hb_predictions = hen_blk_model.predict(X_test)

hennepin_black_model_score = hen_blk_model.score(X_test, y_test)
base_hen_black_rate_prediction = hen_blk_model.predict([[2017]])

# We have two primary features, the year and the prisoner population.
y_hn_predictions = hen_native_model.predict(Xn_test)

hen_native_model_score = hen_native_model.score(Xn_test, yn_test)
base_hen_native_rate_prediction = hen_native_model.predict([[2017]])

"""
Main UI components
"""
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)
col7, col8 = st.columns(2)
col9, col10 = st.columns(2)

"""
Column 1
"""
with col1:
    # Individual column linked to its st.column partner
    # Instantiate the subplot and the respective figures
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,10), constrained_layout = True, sharey='col')

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

    # Show the plot
    plt.show();

"""
Column 2
"""
with col2:
    # Individual column linked to its st.column partner
    st.write(
        """
        Column 2 text
        """
    )

"""
Column 3
"""
with col3:
    # Individual column linked to its st.column partner
    st.write(
        """
        Column 3 text
        """
    )

"""
Column 4
"""
with col4:
    # Individual column linked to its st.column partner
    st.write(
        """
        Column 4 text
        """
    )

"""
Column 5
"""
with col5:
    # Individual column linked to its st.column partner
    st.write(
        """
        Column 5 text
        """
    )

"""
Column 6
"""
with col6:
    # Individual column linked to its st.column partner
    st.write(
        """
        Column 6 text
        """
    )

"""
Column 7
"""
with col7:
    # Individual column linked to its st.column partner
    st.write(
        """
        Column 7 text
        """
    )
"""
Column 8
"""
with col8:
    # Individual column linked to its st.column partner
    st.write(
        """
        Column 8 text
        """
    )


"""
Column 9
"""
with col9:
    # Display the estimates for the Hennepin county prisoner demographic rates
    st.metric(label="Hennepin Native Population Rate", value=str(base_hen_black_rate_prediction) + " per 100k", delta="Score: " + str(hen_native_model_score))
    st.metric(label="Hennepin Native Population Rate", value=str(base_hen_native_rate_prediction) + " per 100k", delta="Score: " + str(base_hen_native_rate_prediction))

"""
Column 10
"""
with col10:
    st.title(
        """
        Future demographic estimates
        """
    )
    st.write(
        """
        The two metrics on the left represent future rates for the black and native american 
        prisoner populations.
        """
    )


"""
Add expander to explain the reasoning behind this project
"""
with st.expander("Reasoning behind this web application!"):
    st.title(
        """
        Deciding on what data to analyze
        """
    )
    # Deciding on what data to analyze
    st.write(
        """
        Since the public dataset provided by Vera has an exceptional number of columns, its best we do some research to determine which data points at most relevant to determining outcomes.
        """
    )

    st.image("img/mn_poverty_rates.png")

    st.write(
        """
        Source: https://data.web.health.state.mn.us/poverty_basic
        """
    )

    st.write(
        """
        Taken from the Minnesota department of health, this table depicts that there are serious income inequality issues in Minnesota based on race and ethnic lines.
        
        
        Education level is closely correlated with income, and in turn both are correlated to a citizens risk for incarceration. This can be seen in a comparison of levels of education and incarceration rates of white & black men from 1979 to 2009 in the table listed below (Harvard 2009). Notice how the dates seen in the table fall in range of the incarceration data from Vera for further support on this inference.
        """
    )
    st.title(
        """
        Why is income inequality relevant to incarceration in the US?
        """
    )

    st.image("img/education_and_incarceration.png")

    st.write(
        """
        Source: https://www.irp.wisc.edu/resource/connections-among-poverty-incarceration-and-inequality/#_edn5*
        """
    )

    st.write(
        """
        According to research done by the University of Wisconsin - Madison, "[Historically] mass incarceration [in the US] criminalized social problems related to racial inequality and poverty on a historically unprecedented scale..."

        This insight provided by researchers at UW Madison will help explain Minnesota's relationship with race/ethnicity and prison inequality.
        """
    )
