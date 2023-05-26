import numpy as np
import pandas as pd
import statsmodels.api as sm    

from utils import z_score


def calculate_momentum_signals(df, date):
    """
    Logic to calculate momentum signals, transform them into normalised scores,
    and combined them equally into one signal. Sub momentum factors used are:
    > 6 months spread yield difference.
    > 12 - 1 months spread yield difference.
    Missing values as transform to zero in normalised score.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe filtered out by a set of dates used for signal calculation.
   
    Returns
    -------
    pandas.DataFrame
        Same dataframe inputted but with extra columns for momentum signals calculated.
    """
    l_df = []
    for _isin in df.ISIN.unique():
        df_isin = df[df.ISIN == _isin].copy().sort_values('DATE')
        df_isin['mom_spread_6'] = df_isin['SPREAD_yield'].diff(periods=6)
        df_isin['mom_spread_12_m_1'] = df_isin['SPREAD_yield'].diff(periods=11).shift(1)
        l_df.append(df_isin)
    df_out = pd.concat(l_df)
    ##################################################################################
    # Value signal that uses historic data therefore including it here with momentum #
    ##################################################################################
    X = df[["TMT", "TMT_2", "N_SP"]].fillna(method="ffill") 
    y = df["SPREAD_yield"]
    X = sm.add_constant(X)
    reg_model = sm.OLS(y, X).fit()
    predicted_spread = np.dot(X, reg_model.params)
    df_out['value_reg_richness'] = predicted_spread - df_out['SPREAD_yield']
    df_out = df_out[df_out.DATE == date]
    df_out['value_reg_richness_score_sa'] = df_out.groupby(['sector'])['value_reg_richness'].apply(z_score).fillna(0)
    df_out['value_reg_richness_score'] = z_score(df_out['value_reg_richness']).fillna(0)
    # Back to momentum
    df_out['mom_spread_6_score_sa'] = df_out.groupby(['sector'])['mom_spread_6'].apply(z_score).fillna(0)
    df_out['mom_spread_6_score'] = z_score(df_out['mom_spread_6']).fillna(0)
    df_out['mom_spread_12_m_1_score_sa'] = df_out.groupby(['sector'])['mom_spread_12_m_1'].apply(z_score).fillna(0)
    df_out['mom_spread_12_m_1_score'] = z_score(df_out['mom_spread_12_m_1']).fillna(0)
    df_out['momentum_score_sa'] = df_out[['mom_spread_6_score_sa', 'mom_spread_12_m_1_score_sa']].mean(axis=1).values
    df_out['momentum_score'] = df_out[['mom_spread_6_score', 'mom_spread_12_m_1_score']].mean(axis=1).values
    return df_out
