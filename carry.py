import numpy as np
import statsmodels.api as sm    

from utils import z_score


def calculate_carry_signals(df):
    """
    Logic to calculate carry signal and transform it into normalised score.
    It used spread yield as a measure of carry to be realised over the life 
    of the bond. Missing values as transform to zero in normalised score.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe filtered out by a set of dates used for signal calculation.
   
    Returns
    -------
    pandas.DataFrame
        Same dataframe inputted but with extra columns for carry signal calculated.
    """
    Y = df.SPREAD_yield.to_numpy()
    X = df.dts.to_numpy()
    reg_model = sm.OLS(Y,X, missing='drop').fit()  # training the model
    residual_values = reg_model.resid # residual values
    df['carry_adj_dts'] = np.nan
    df.loc[~df.dts.isna(), 'carry_adj_dts'] = residual_values
    df['carry_score_sa'] = df.groupby(['sector'])['carry_adj_dts'].apply(z_score).fillna(0)
    df['carry_score'] = z_score(df['carry_adj_dts']).fillna(0)
    return df
