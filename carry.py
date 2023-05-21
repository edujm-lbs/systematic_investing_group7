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
    df['carry_score'] = z_score(df['SPREAD_yield'].to_numpy()).fillna(0, inplace=True)
    return df
