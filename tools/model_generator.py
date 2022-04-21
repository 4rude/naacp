 # Import libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


def generate_hen_blk_model(year_to_predict=2017):
    """Returns a prediction in a float datatype object."""
    # Make a dataframe holding incarceration trends in the US from 1970 to 2018
    inc_tre_df = pd.read_csv("data/incarceration_trends.csv", parse_dates=['year'], index_col=False)

    # Use numpy to fill missing values with zero
    inc_tre_df = inc_tre_df.replace(np.nan, 0)

    # Separate the original csv import into the counties we need data from
    hennepin_df = inc_tre_df[(inc_tre_df['county_name'] == 'Hennepin County') & (inc_tre_df['state'] == 'MN')]

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

    # reset the indexes
    hennepin_df.year = hennepin_df['year'].dt.year
    hennepin_df.index = hennepin_df.year


    """
    Create the linear regression model object
    """
    # Create model objects to process data
    hen_blk_model = linear_model.Lasso(alpha=0.1)

    """
    Clean data
    """
    # # Remove empty rows from dataframes
    hennepin_df = hennepin_df.iloc[24:-2,28:29]

    # Create X and y variables that hold formatted column data for hennepin black rate
    hb_X = hennepin_df.index.values.reshape(-1, 1)
    hb_y = hennepin_df.values.reshape(-1, 1)

    """
    Split data into train/test variables
    """
    # Break up black demographic rate data
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(hb_X, hb_y)

    """
    Fit the model
    """
    # Start with Hennepin
    hen_blk_model = hen_blk_model.fit(Xb_train, yb_train)

    """
    Use the model to make predictions plus score the predictions
    """
    # Create a score and prediction for the year (input) that the user passed in
    score = hen_blk_model.score(Xb_test, yb_test)
    # Convert the prediction to an integer, rounding in the process
    prediction = int(hen_blk_model.predict([[year_to_predict]]))

    # Return the variables
    return prediction, score


def generate_hen_nat_model(year_to_predict=2017):
    """Returns a prediction in a float datatype object."""
    inc_tre_df = pd.read_csv("/Users/matt/Desktop/naacp/data/incarceration_trends.csv", parse_dates=['year'], index_col=False)

    # Use numpy to fill missing values with zero
    inc_tre_df = inc_tre_df.replace(np.nan, 0)

    # Separate the original csv import into the counties we need data from
    hennepin_df = inc_tre_df[(inc_tre_df['county_name'] == 'Hennepin County') & (inc_tre_df['state'] == 'MN')]

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

    # reset the indexes
    hennepin_df.year = hennepin_df['year'].dt.year
    hennepin_df.index = hennepin_df.year

    """
    Create the linear regression model object
    """
    # Create model objects to process data
    hen_native_model = linear_model.Lasso(alpha=0.1)

    """
    Clean data
    """
    # Remove empty rows from dataframes
    hennepin_df = hennepin_df.iloc[24:-2,30:31]

    # Create X and y variables that hold formatted column data for hennepin native rate
    hn_X = hennepin_df.index.values.reshape(-1, 1)
    hn_y = hennepin_df.values.reshape(-1, 1)

    """
    Split data into train/test variables
    """
    # Break up native demographic rate data
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(hn_X, hn_y)

    """
    Fit the model
    """
    # Fit the Hennepin county native pop rate model
    hen_native_model.fit(Xn_train, yn_train)

    """
    Use the model to make predictions and score the predictions
    """
    score = hen_native_model.score(Xn_test, yn_test)
    # Convert the prediction to an integer, rounding in the process
    prediction = int(hen_native_model.predict([[year_to_predict]]))

    # Return the variables
    return prediction, score


#%%

#%%
