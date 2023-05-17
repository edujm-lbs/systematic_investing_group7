import pandas as pd
import numpy as np

from momentum import calculate_momentum_signals

# Include any relevant field used for your analysis - avoid having such a large DataFrame
REL_COL = ['DATE', 'ISIN', 'AMOUNT_OUTSTANDING', 'RET_EOM', 'return_excess_by_duration']

DIR = 'INPUT_FILE_DIR_FROM_YOUR_LOCAL'

df = pd.read_sas(DIR).loc[:,REL_COL]

dict_mom = {}
l_df = []
for date in df.DATE.sort_values().unique()[12:]:
    date_prev = date - pd.DateOffset(years=1, months=3)
    df_dt = df[(df.DATE > date_prev) & (df.DATE <= date)].copy() # For momentum signal calculation we require trailing dates.

    # 1. Calculation of signals (should it be raw or output directly Z-score?)
    df_dt_m = calculate_momentum_signals(df_dt)
    df_dt_m = df_dt_m[df_dt_m.DATE == date]
    # Have a similar line as above but with carry fn
    df_dt_m_c = ...
    # Have a similar line as above but with value fn
    df_dt_m_c_v = ...
    # Have a similar line as above but with quality fn
    df_dt_m_c_v_q = ...
    # Have a similar line as above but with X creative_factor fn
    df_dt_m_c_v_q_x = ...

    # 2. If Z-Score not calculate previous then do so. If done already, move on.

    # 3. Combine factors Z-Scores.

    # 4. Fn to implement logic to pick top ranked scores and provide weights for all bonds in a given month
    df_dt_m_c_v_q_x['portfolio_wght'] = ...
    # 5. Calculate market cap weight for benchmark calculations later on
    df_dt_m_c_v_q_x['market_wght'] = \
        df_dt_m_c_v_q_x['AMOUNT_OUTSTANDING']/df_dt_m_c_v_q_x['AMOUNT_OUTSTANDING']

    l_df.append(df_dt_m_c_v_q_x)

