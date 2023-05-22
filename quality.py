import pandas as pd
import numpy as np
from utils import z_score

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

  # use our z-score function to calculate a sector-neutral z-score
  # note that we multiply by -1 because we want lower leverage
  df['leverage_z'] = df.groupby(['DATE','sector'])['leverage'].apply(lambda x: z_score(x)).fillna(0, inplace=True) * -1
  # there may be some obs that have zero debt after this
  # this wouldn't make sense because all these firms are issuing debt so they should have debt
  # we'll replace the z-score with 0 if the total debt is zero
  # this will slightly throw off our z-score but it shouldn't have that big of an impact because there
  # are only 1,070 obs where this happens out of ~400,000
  df['leverage_z'] = np.where(df.total_debt == 0, 0, df.leverage_z)
  
  # calculate gross profit according to Novy-Marx
  # gross profit scaled by total assets
  df['profit'] = df['gp'] / df['at']

  # profit in z-score terms
  df['profit_z'] = df.groupby(['DATE','sector'])['profit'].apply(lambda x: z_score(x)).fillna(0, inplace=True)

  # combine our measures into a single quality factor
  number_of_measures = 2
  df['quality'] = (df['leverage_z'] + df['profit_z']) / number_of_measures

  return(df)
