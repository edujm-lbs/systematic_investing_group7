from statistics import NormalDist

import numpy as np
import statsmodels.api as sm
import pandas as pd  

from utils import calculate_dts_residuals, z_score


def calculate_value_signals(df):
    df['debt'] = df['dt'].fillna(0)
    df['sigma_e'] = df.ret_var_movstd_yrl
    df['deleverage_factor'] = \
        df.apply(lambda x: 1 if x.debt == 0 else x.mkvalt/(x.mkvalt+x.debt), axis=1)
    df['sigma_a'] = (df.sigma_e * df.deleverage_factor).fillna(0)
    df['D2D'] = \
        df.apply(lambda x: None if ((x.sigma_a == 0) or (x.debt == 0) or (((x.mkvalt+x.debt)/x.debt)<0)) \
                                else np.log((x.mkvalt+x.debt)/x.debt)/x.sigma_a, axis=1)
    df['PD'] = df.D2D.apply(lambda x: NormalDist(mu=0, sigma=1).cdf(-x))
    # PD vs Spread yield
    Y = df.SPREAD_yield.to_numpy()
    X = df.PD.to_numpy()
    reg_model = sm.OLS(Y,X, missing='drop').fit()  # training the model
    residual_values = reg_model.resid # residual values
    df['spread_to_pd_res'] = np.nan
    df.loc[~df.PD.isna(), 'spread_to_pd_res'] = residual_values
    s2pd_dtsa = calculate_dts_residuals(df, 'spread_to_pd_res')
    df['spread_to_pd_res_dts'] = np.nan
    df.loc[~df[["dts", "spread_to_pd_res"]].isna().any(axis=1), 'spread_to_pd_res_dts'] = s2pd_dtsa
    df['spread_to_pd_res_score'] = z_score(df.spread_to_pd_res_dts).fillna(0)
    df['spread_to_pd_res_score_sa'] = df.groupby(['sector'])['spread_to_pd_res_dts'].apply(z_score).fillna(0)
    # TMT, TMT_2, and N_SP
    X = df[["TMT", "TMT_2", "N_SP"]]
    y = df["SPREAD_yield"]
    X = sm.add_constant(X)
    reg_model = sm.OLS(y, X, missing='drop').fit()
    predicted_spread = reg_model.predict()
    df['value_reg_richness'] = np.nan
    df.loc[~df[["TMT", "TMT_2", "N_SP"]].isna().any(axis=1), 'value_reg_richness'] = predicted_spread
    df['value_reg_richness'] = df['value_reg_richness'] - df['SPREAD_yield']
    val_richness_dtsa = calculate_dts_residuals(df, 'value_reg_richness')
    df['value_reg_richness_dts'] = np.nan
    df.loc[~df[["dts", "value_reg_richness"]].isna().any(axis=1), 'value_reg_richness_dts'] = val_richness_dtsa
    df['value_reg_richness_score_sa'] = df.groupby(['sector'])['value_reg_richness_dts'].apply(z_score).fillna(0) * -1
    df['value_reg_richness_score'] = z_score(df['value_reg_richness_dts']).fillna(0) * -1
    df['value_score_sa'] = df[['value_reg_richness_score_sa', 'spread_to_pd_res_score_sa']].mean(axis=1).values
    df['value_score'] = df[['value_reg_richness_score', 'spread_to_pd_res_score']].mean(axis=1).values
    return df