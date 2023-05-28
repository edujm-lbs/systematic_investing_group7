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

l_df = []

# Include any relevant field used for your analysis - avoid having such a large DataFrame
REL_COL = ['DATE', 'CUSIP', 'AMOUNT_OUTSTANDING', 'RET_EOM', 'SPREAD_yield',
           'return_excess_by_duration', 'datadate', 'HSICCD', 'dlc', 'dltt',
           'mib', 'upstk', 'che', 'mv', 'dt', 'gp', 'at', 'ret_var_movstd_yrl',
           'mkvalt', 'DURATION', 'TMT', 'N_SP', 'lead_EXCESS_RET', 'lead_TOT_RET']

DIR = r'bondret_data17e.sas7bdat'

df_data = pd.read_sas(DIR).loc[:,REL_COL]
df_data = df_data[df_data['datadate'].notna()]

df_data['sector'] = df_data['HSICCD'].astype(str).str[0]
df_data['dts'] = df_data['DURATION'] * df_data['SPREAD_yield']
df_data['TMT_2'] = df_data['TMT'] ** 2

    
def calculate_value_signals(df):
    df['model_spread'] = np.nan
    #drop the Blank cells in N_SP & sector column 
    df['N_SP'].replace('', np.nan, inplace=True)
    df.dropna(subset=['N_SP'], inplace=True)
    df['sector'].replace('', np.nan, inplace=True)
    df.dropna(subset=['sector'], inplace=True)
    # Convert the unique words in the "N_SP" column to dummy variables
    dummies = pd.get_dummies(df["N_SP"].unique(), prefix="Rating")
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

for date in df_data.DATE.sort_values().unique()[13:]:

    date_prev = date - np.timedelta64(500, 'D')
    # For momentum signal calculation we require trailing dates.
    df_dt = df_data[(df_data.DATE > date_prev) & (df_data.DATE <= date)].copy()
    # Value
    df_dt_m_c_q_v = calculate_value_signals(df_dt)
    # 2. Combine all Style combined Z-Scores on to one
    df_dt_m_c_q_v['combined_score'] = \
        df_dt_m_c_q_v['value_score']
    df_dt_m_c_q_v['combined_score_sa'] = \
        df_dt_m_c_q_v['value_score_sa']
    # 3. Fn to implement logic to pick top ranked scores and provide weights for all bonds in a given month
    df_dt_m_c_q_v_w = calculate_portfolio_weights(df_dt_m_c_q_v)
    # 4. Calculate market cap weight for benchmark calculations later on
    df_dt_m_c_q_v_w['benchmark_wght'] = \
        df_dt_m_c_q_v_w['AMOUNT_OUTSTANDING'] / df_dt_m_c_q_v_w['AMOUNT_OUTSTANDING'].sum()
    l_df.append(df_dt_m_c_q_v_w)

df_final = pd.concat(l_df)

df_final = calculate_portfolio_returns(df_final)
df_final.to_csv('final_df.csv')
