import pandas as pd

from carry import calculate_carry_signals
from momentum import calculate_momentum_signals
from quality import quality_calc
from optim import get_optim_ports

# Include any relevant field used for your analysis - avoid having such a large DataFrame
REL_COL = ['DATE', 'ISIN', 'AMOUNT_OUTSTANDING', 'RET_EOM', 'SPREAD_YIELD', 'return_excess_by_duration', 'gp', 'at', 'dt', 'dlc', 'dltt', 'mib', 'upstk', 'che', 'datadate', 'HSICCD', 'lead_EXCESS_RET', 'lead_TOT_RET']

DIR = 'INPUT_FILE_DIR_FROM_YOUR_LOCAL'

df_data = pd.read_sas(DIR).loc[:,REL_COL]
df_data = df_data[df_data['datadate'].notna()]

df_data['sector'] = df_data['HSICCD'].astype(str).str[0]

# required for plots output later on
df_data["DTS"] = df_data["DURATION"] * df_data["SPREAD_yield"]

dict_mom = {}
l_df = []
for date in df_data.DATE.sort_values().unique()[12:]:
    date_prev = date - pd.DateOffset(years=1, months=3)
    df_dt = df_data[(df_data.DATE > date_prev) & (df_data.DATE <= date)].copy() # For momentum signal calculation we require trailing dates.

    # 1. Calculation of signals (should it be raw or output directly Z-score?)
    df_dt_m = calculate_momentum_signals(df_dt)
    df_dt_m = df_dt_m[df_dt_m.DATE == date]
    # Have a similar line as above but with carry fn
    df_dt_m_c = calculate_carry_signals(df_dt_m)
    # Have a similar line as above but with value fn
    df_dt_m_c_v = ...
    # Have a similar line as above but with quality fn
    df_dt_m_c_v_q = quality_calc(df_dt_m_c_v)
    # Have a similar line as above but with X creative_factor fn
    df_dt_m_c_v_q_x = ...

    # 2. Combine all Style combined Z-Scores on to one

    # 3. Fn to implement logic to pick top ranked scores and provide weights for all bonds in a given month
    df_dt_m_c_v_q_x['portfolio_wght'] = ...
    # if we're looking to optimize our combined z-scores we could use the below function
    # this should produce a dataframe will all the necessary fields for the plots we need to create later on
    # df_for_plots = get_optim_ports(data=df_dt_m_c_v_q_x, max_wgt=0.05, sector_bound=0.05, credit_bound=0.05, duration_bound=2, dts_bound=0.03)
    
    # 4. Calculate market cap weight for benchmark calculations later on
    df_dt_m_c_v_q_x['market_wght'] = \
        df_dt_m_c_v_q_x['AMOUNT_OUTSTANDING']/sum(df_dt_m_c_v_q_x['AMOUNT_OUTSTANDING'])

    l_df.append(df_dt_m_c_v_q_x)

