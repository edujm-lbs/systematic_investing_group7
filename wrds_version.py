# to be run on the jupyter server on WRDS
# https://wrds-jupyter.wharton.upenn.edu

import pandas as pd
import numpy as np
import os
from statistics import NormalDist
import statsmodels.api as sm

# this assumes you're working on the wrds server and you've already run
# the initial sample creation script
# change this to your username
username = "dont_forget_to_update_your_username"

# set working directory
os.chdir(f"/home/lbs/{username}/E499_Group_Project/data/")

def z_score(x):
    """
    used to calculate z-scores
    """
    return (x - np.mean(x)) / np.std(x)

# name of the sample file
sasdb = "bondret_data17e.sas7bdat"

df_data = pd.read_sas(sasdb)

# drop values where there's no datadate
# this is what what was done in the example SAS script
df_data = df_data[df_data['datadate'].notna()]

REL_COL = ['DATE', 'CUSIP', 'AMOUNT_OUTSTANDING', 'RET_EOM', 'SPREAD_yield',
           'return_excess_by_duration', 'datadate', 'HSICCD', 'dlc', 'dltt',
           'mib', 'upstk', 'che', 'mv', 'dt', 'gp', 'at', 'ret_var_movstd_yrl',
           'mkvalt', 'DURATION', 'TMT', 'N_SP', 'lead_EXCESS_RET', 'lead_TOT_RET','YEAR_MONTH_d',
           'MATURITY_yrs','T_Spread','YIELD','PRICE_EOM','R_MR']

df_data = df_data.loc[:,REL_COL]
df_data = df_data[df_data['datadate'].notna()]

# create sector
# directly from SAS script
df_data['sector'] = df_data['HSICCD'].astype(str).str[0]

df_data['dts'] = df_data['DURATION'] * df_data['SPREAD_yield']
df_data['TMT_2'] = df_data['TMT'] ** 2

def calculate_dts_residuals(df, Y_name):
    Y = df[Y_name].to_numpy()
    X = df.dts.to_numpy()
    reg_model = sm.OLS(Y,X, missing='drop').fit()  # training the model
    return reg_model.resid # residual values

COL_FACTORS_NAME = ['mom_spread_6_score', 'mom_spread_12_m_1_score',
                    'mom_spread_6_score_sa', 'mom_spread_12_m_1_score_sa',
                    'leverage_z', 'leverage_z_sa', 'profit_z', 'profit_z_sa',
                    'spread_to_pd_res_score', 'spread_to_pd_res_score_sa',
                    'value_reg_richness_score', 'value_reg_richness_score_sa',
                    'momentum_score', 'carry_score', 'quality_score', 'value_score',
                    'combined_score', 'momentum_score_sa', 'carry_score_sa', 'quality_score_sa', 
                    'value_score_sa', 'combined_score_sa']

def allocate_equally_weighted_exposure(df, factor_name):
    top_n_rows = df[df[factor_name] > 0].shape[0]
    df_sorted = df.copy().sort_values(factor_name, ascending=False)
    eq_w = 1 / top_n_rows
    df_sorted[f'{factor_name}_equally_weight'] = [eq_w] * top_n_rows + [0] * (df_sorted.shape[0] - top_n_rows)
    return df_sorted


def allocate_signal_weighted_exposure(df, factor_name):
    top_n_rows = df[df[factor_name] > 0].shape[0]
    df_sorted = df.copy().sort_values(factor_name, ascending=False)
    top_scores_values = df_sorted.iloc[:top_n_rows, :][factor_name].values
    signal_w = top_scores_values / top_scores_values.sum()
    df_sorted[f'{factor_name}_signal_weight'] = signal_w.tolist() + [0] * (df_sorted.shape[0] - top_n_rows)
    return df_sorted


def allocate_market_weighted_exposure(df, factor_name):
    top_n_rows = df[df[factor_name] > 0].shape[0]
    df_sorted = df.copy().sort_values(factor_name, ascending=False)
    top_market_values = df_sorted.iloc[:top_n_rows, :]['AMOUNT_OUTSTANDING'].values
    market_w = top_market_values / top_market_values.sum()
    df_sorted[f'{factor_name}_market_weight'] = market_w.tolist() + [0] * (df_sorted.shape[0] - top_n_rows)
    return df_sorted


def calculate_portfolio_weights(df):
    for factor in COL_FACTORS_NAME:
        df = allocate_equally_weighted_exposure(df, factor)
        df = allocate_signal_weighted_exposure(df, factor)
        df = allocate_market_weighted_exposure(df, factor)
    return df


def calculate_portfolio_returns(df):
    for factor in COL_FACTORS_NAME:
        df[f'{factor}_ret_ew_le'] = df[f'{factor}_equally_weight'] * df.lead_EXCESS_RET
        df[f'{factor}_ret_sw_le'] = df[f'{factor}_signal_weight'] * df.lead_EXCESS_RET
        df[f'{factor}_ret_mw_le'] = df[f'{factor}_market_weight'] * df.lead_EXCESS_RET
        df[f'{factor}_ret_ew_ltr'] = df[f'{factor}_equally_weight'] * df.lead_TOT_RET
        df[f'{factor}_ret_sw_ltr'] = df[f'{factor}_signal_weight'] * df.lead_TOT_RET
        df[f'{factor}_ret_mw_ltr'] = df[f'{factor}_market_weight'] * df.lead_TOT_RET
    df['benchmark_ret_le'] = df['benchmark_wght'] * df.lead_EXCESS_RET
    df['benchmark_ret_ltr'] = df['benchmark_wght'] * df.lead_TOT_RET
    df['EW_EXC_RET_BENCHMARK'] = df['benchmark_eq_wgt'] * df.lead_EXCESS_RET
    df['EW_TOT_RET_BENCHMARK'] = df['benchmark_eq_wgt'] * df.lead_TOT_RET
    return df

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
    df['mom_spread_6'] = df.sort_values(['DATE']).groupby(['CUSIP'])['SPREAD_yield'].diff(6)
    df['mom_spread_12_m_1'] = df.sort_values(['DATE']).groupby(['CUSIP'])['SPREAD_yield'].diff(periods=11).shift(1)
    df = df[df.DATE == date]
    # Calc for 6m spread momentum
    mom_spread_6_dtsa = calculate_dts_residuals(df, 'mom_spread_6')
    df['mom_spread_6_dts'] = np.nan
    df.loc[~df[["dts", "mom_spread_6"]].isna().any(axis=1), 'mom_spread_6_dts'] = mom_spread_6_dtsa
    df['mom_spread_6_score_sa'] = df.groupby(['sector'])['mom_spread_6_dts'].apply(z_score).fillna(0)
    df['mom_spread_6_score'] = z_score(df['mom_spread_6_dts']).fillna(0)
    # Calc for 12m spread momentum
    mom_spread_12_m_1_dtsa = calculate_dts_residuals(df, 'mom_spread_12_m_1')
    df['mom_spread_12_m_1_dts'] = np.nan
    df.loc[~df[["dts", "mom_spread_12_m_1"]].isna().any(axis=1), 'mom_spread_12_m_1_dts'] = mom_spread_12_m_1_dtsa
    df['mom_spread_12_m_1_score_sa'] = df.groupby(['sector'])['mom_spread_12_m_1_dts'].apply(z_score).fillna(0)
    df['mom_spread_12_m_1_score'] = z_score(df['mom_spread_12_m_1_dts']).fillna(0)
    # Combining scores
    df['momentum_score_sa'] = df[['mom_spread_6_score_sa', 'mom_spread_12_m_1_score_sa']].mean(axis=1).values
    df['momentum_score'] = df[['mom_spread_6_score', 'mom_spread_12_m_1_score']].mean(axis=1).values
    return df

def quality_calc(df):
    """
    calculate z-scores of leverage and profitability according to 
    https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/Common-Factors-in-Corporate-Bond-Returns.pdf
    combine them in an equal weight to arrive at your quality score
    research suggests that bonds issued by higher quality issuers tend to outperform
    bonds issued by lower quality firms, especially on a risk-adjusted basis

    params: df; dataframe containing all relevant columns

    returns: the same dataframe that was passed to the function but with new column, including quality
    """

    # first, replace all instances of short term and debt term debt that are missing with 0
    # we found that dt was missing in the db for ~30% of observations
    # we can greatly improve coverage by calculating it ourselves
    # we found factor returns were robust to either calculation
    # so we'll use the manual one since it improves coverage
    df['dlc'].fillna(0, inplace=True) # current debt
    df['dltt'].fillna(0, inplace=True) # long-term debt
    df['mib'].fillna(0, inplace=True) # minority interest
    df['upstk'].fillna(0, inplace=True) # preferred stock
    df['che'].fillna(0, inplace=True) # cash and cash equivalents
    # now calculate our revised measure of total debt
    # we'll use there measure of total debt where we have it
    # otherwise use ours
    df['total_debt'] = np.where(df.dt.isnull(), df.dlc + df.dltt, df.dt)
    # calculate leverage
    df['leverage'] = (df['total_debt'] + df['mib'] + df['upstk'] - df['che']) / (df['total_debt'] - df['che'] + df['mv'])
    leverage_dtsa = calculate_dts_residuals(df, 'leverage')
    df['leverage_dts'] = np.nan
    df.loc[~df[["dts", "leverage"]].isna().any(axis=1), 'leverage_dts'] = leverage_dtsa
    # use our z-score function to calculate a sector-neutral z-score
    # note that we multiply by -1 because we want lower leverage
    df['leverage_z_sa'] = df.groupby(['sector'])['leverage_dts'].apply(z_score).fillna(0) * -1
    df['leverage_z'] = z_score(df['leverage_dts']).fillna(0) * -1
    # there may be some obs that have zero debt after this
    # this wouldn't make sense because all these firms are issuing debt so they should have debt
    # we'll replace the z-score with 0 if the total debt is zero
    # this will slightly throw off our z-score but it shouldn't have that big of an impact because there
    # are only 1,070 obs where this happens out of ~400,000
    df['leverage_z_sa'] = np.where(df.total_debt == 0, 0, df.leverage_z_sa)
    df['leverage_z'] = np.where(df.total_debt == 0, 0, df.leverage_z)
    # calculate gross profit according to Novy-Marx
    # gross profit scaled by total assets
    df['profit'] = df['gp'] / df['at']
    # profit in z-score terms
    profit_dtsa = calculate_dts_residuals(df, 'profit')
    df['profit_dts'] = np.nan
    df.loc[~df[["dts", "profit"]].isna().any(axis=1), 'profit_dts'] = profit_dtsa
    df['profit_z_sa'] = df.groupby(['sector'])['profit_dts'].apply(z_score).fillna(0)
    df['profit_z'] = z_score(df['profit_dts']).fillna(0)
    # combine our measures into a single quality factor
    number_of_measures = 2
    df['quality_score_sa'] = (df['leverage_z_sa'] + df['profit_z_sa']) / number_of_measures
    df['quality_score'] = (df['leverage_z'] + df['profit_z']) / number_of_measures
    return(df)

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

l_df = []
for date in df_data.DATE.sort_values().unique()[13:]:
    print(date)
    date_prev = date - np.timedelta64(500, 'D')
    # For momentum signal calculation we require trailing dates.
    df_dt = df_data[(df_data.DATE > date_prev) & (df_data.DATE <= date)].copy()
    # 1. Calculation of signals (should it be raw or output directly Z-score?)
    df_dt_m = calculate_momentum_signals(df_dt, date)
    # Carry
    df_dt_m_c = calculate_carry_signals(df_dt_m)
    # Quality
    df_dt_m_c_q = quality_calc(df_dt_m_c)
    # Value
    df_dt_m_c_q_v = calculate_value_signals(df_dt_m_c_q)
    # 2. Combine all Style combined Z-Scores on to one
    df_dt_m_c_q_v['combined_score'] = \
        df_dt_m_c_q_v[['momentum_score', 'carry_score', 'quality_score', 'value_score']].mean(axis=1)
    df_dt_m_c_q_v['combined_score_sa'] = \
        df_dt_m_c_q_v[['momentum_score_sa', 'carry_score_sa', 'quality_score_sa', 'value_score_sa']].mean(axis=1)
    # 3. Fn to implement logic to pick top ranked scores and provide weights for all bonds in a given month
    df_dt_m_c_q_v_w = calculate_portfolio_weights(df_dt_m_c_q_v)
    # 4. Calculate market cap weight for benchmark calculations later on
    df_dt_m_c_q_v_w['benchmark_wght'] = \
        df_dt_m_c_q_v_w['AMOUNT_OUTSTANDING'] / df_dt_m_c_q_v_w['AMOUNT_OUTSTANDING'].sum()
    df_dt_m_c_q_v_w['benchmark_eq_one'] = 1
    df_dt_m_c_q_v_w['benchmark_eq_wgt'] = df_dt_m_c_q_v_w['benchmark_eq_one'] / df_dt_m_c_q_v_w['benchmark_eq_one'].sum()
        
    l_df.append(df_dt_m_c_q_v_w)

df_final = pd.concat(l_df)

df_final = calculate_portfolio_returns(df_final)

# these are the columns the SAS script is expecting
SAS_COLS = ["CUSIP", "sector", "combined_score_sa", "AMOUNT_OUTSTANDING", "YEAR_MONTH_d", "DATE", "combined_score_sa_market_weight", 
            "combined_score_sa_signal_weight", "combined_score_sa_equally_weight", "SPREAD_yield", "lead_TOT_RET", "lead_EXCESS_RET", "dts",
            "MATURITY_yrs", "DURATION", "T_Spread", "YIELD", "TMT", "R_MR",  "PRICE_EOM", "EW_TOT_RET_BENCHMARK",
            "EW_EXC_RET_BENCHMARK", "combined_score_sa_ret_ew_le", "combined_score_sa_ret_ew_ltr", "benchmark_ret_le",
            "benchmark_ret_ltr", "combined_score_sa_ret_mw_le", "combined_score_sa_ret_mw_ltr", "combined_score_sa_ret_sw_le",
            "combined_score_sa_ret_sw_ltr"]

sas_data = df_final.loc[:,SAS_COLS]
sas_data = sas_data.rename(columns={'sector': 'HSICCD', 
                                    'combined_score_sa': 'SIGNAL',
                                    'combined_score_sa_equally_weight': 'EQUAL_WEIGHT_P',
                                    'combined_score_sa_signal_weight': 'RANK_WEIGHT',
                                    'combined_score_sa_market_weight': 'VALUE_WEIGHT',
                                    'combined_score_sa_ret_ew_le': 'EW_EXC_RET_PORTFOLIO',
                                    'combined_score_sa_ret_ew_ltr': 'EW_TOT_RET_PORTFOLIO',
                                    'benchmark_ret_le': 'MW_EXC_RET_BENCHMARK',
                                    'benchmark_ret_ltr': 'MW_TOT_RET_BENCHMARK',
                                    'combined_score_sa_ret_mw_le': 'MW_EXC_RET_PORTFOLIO',
                                    'combined_score_sa_ret_mw_ltr': 'MW_TOT_RET_PORTFOLIO',
                                    'combined_score_sa_ret_sw_le': 'SW_EXC_RET_PORTFOLIO',
                                    'combined_score_sa_ret_sw_ltr': 'SW_TOT_RET_PORTFOLIO'})

sas_data["EQUAL_WEIGHT"] = np.where(sas_data.EQUAL_WEIGHT_P > 0, 1, 0)

sas_data.to_csv('final_data.csv', index=False)

# after running this you should be able to run the plot_creation.sas script
