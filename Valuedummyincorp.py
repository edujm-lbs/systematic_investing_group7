#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 21:43:16 2023

@author: devanksriram
"""

import numpy as np
import statsmodels.api as sm
import pandas as pd  

from utils import calculate_dts_residuals, z_score, calculate_portfolio_returns, calculate_portfolio_weights
    
def calculate_value_signals(df):
    df['model_spread'] = np.nan
    #drop the Blank cells in N_SP & sector column 
    df['N_SP'].replace('', np.nan, inplace=True)
    df.dropna(subset=['N_SP'], inplace=True)
    df['sector'].replace('', np.nan, inplace=True)
    df.dropna(subset=['sector'], inplace=True)
    # Convert the unique words in the "N_SP" column to dummy variables
    dummies = pd.get_dummies(df["N_SP"], prefix="Rating")
    # Concatenate the original DataFrame with the new dummy variables
    dataframe_dum = pd.concat([df, dummies], axis=1)
    
    # now loop over 9 different sectors and run regressions for each sector separately
    for t in dataframe_dum['sector'].unique():
        #filter out the sector you're looping on
        dataframe_dumsharads = dataframe_dum[dataframe_dum['sector']== t]
        # Create a new dataframe with just the relevant columns
        X = pd.concat([dataframe_dumsharads[["TMT", "TMT_2", "N_SP"]].fillna(method="ffill"), dataframe_dumsharads.filter(like="Rating")], axis=1)  # Add all columns that start with "Rating"
        y = dataframe_dumsharads["SPREAD_yield"]
        X = sm.add_constant(X)
        reg_model = sm.OLS(y, X, missing='drop').fit()
        predicted_spread = reg_model.predict()
        print(t)
        print(len(df.loc[df[df['sector']== t].index.to_list(), 'model_spread'] ))
        print('&')
        print(len(pd.Series(predicted_spread, index=dataframe_dumsharads.index)))
        df.loc[df[df['sector']== t].index.to_list(), 'model_spread'] = pd.Series(predicted_spread, index=dataframe_dumsharads.index)
        #df.loc[~df[df['sector']== t].isna().any(axis=1), 'model_spread'] = predicted_spread
    df['value_reg_richness'] = df['model_spread'] - df['SPREAD_yield']
    val_richness_dtsa = calculate_dts_residuals(df, 'value_reg_richness')
    df['value_reg_richness_dts'] = np.nan
    df.loc[~df[["dts", "value_reg_richness"]].isna().any(axis=1), 'value_reg_richness_dts'] = val_richness_dtsa
        
    df['value_reg_richness_score_sa'] = df.groupby(['sector'])['value_reg_richness_dts'].apply(z_score).fillna(0) * -1
    df['value_reg_richness_score'] = z_score(df['value_reg_richness_dts']).fillna(0) * -1
    df['value_score_sa'] = df['value_reg_richness_score_sa'].values
    df['value_score'] = df['value_reg_richness_score'].values
    return df


df_final = calculate_portfolio_returns(df_final)
df_final.to_csv('final_df.csv')
