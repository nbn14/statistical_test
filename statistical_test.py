import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import shapiro,anderson  ##### shapiro not good for large sample size >5k
from scipy.stats import normaltest #### this is D'Agnostino K2
from scipy.stats import pointbiserialr,pearsonr,spearmanr,kendalltau
from scipy.stats import chi2_contingency
import warnings

def normality_test(X,X_name=None,test="dagnostino",alpha=0.05):
    """
    Normality test for a continuous array
    H0 = the distribution is normally distributed
    Note: these statistical tests cannot handle NA, X must not include any NA
    Parameters:
    -----------
    X: 1D numeric non NA array
    X_name: str, default=None
        Name of input variable
    test: {"dagnostino","shapiro","anderson"}, default="dagnostino"
        shapiro is suitable for smaller datasets <5k data points
        For anderson-darling test, only a limited set of alpha is accepted. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html 
        Interpretationof anderson-darling test has a different logic from the other 2. See above link for details although it still shares the same null hypothesis
    alpha: [0,1], default=0.05
        Critical value to indicate threshold at which the null hypothesis is rejected 

    Returns:
    -------
    val: float
        Statistical test
    p: float in [0,1]
        p-value
    For test=anderson, returns None
    """

    print("H0: The population is normally distributed")
    assert ~np.isnan(X).any(), "Input array must contain no NA"
    if test=="dagnostino":
        val,p = normaltest(X)
    elif test=="shapiro":
        val,p=shapiro(X)
    else:
        warnings.warn("if test=anderson, no value is returned. Interpretation of result is different from shapiro and d'agnostino.")
        val_stat,cri_list,sig_list=anderson(X)
        assert alpha*100 in sig_list, f"alpha must be in {sig_list/100} for test=anderson"
        # For anderson-darling, if val_stat > critical_val at a particular significance => Reject H0, different from the other 2 tests
        p = cri_list[np.where(sig_list==alpha*100)]
        alpha = val_stat # Semantics of anderson-darling is different from the other 2 tests, flip these 2 values for coding consistency

    if p<= alpha:
        if X_name:
            print(f"Reject H0: the distribution of {X_name} is NOT normally distributed.")
        else:
            print(f"Reject H0: the distribution is NOT normally distributed.")
    else:
        if X_name:
            print(f"Fail to reject H0: the distribution of {X_name} is normally distributed.")
        else:
            print(f"Fail to reject H0: the distribution is normally distributed.")

    if test=="anderson":
        return None
    else:
        return val,p



def correlation(X0,X1,X_name=None,test="pearson",alpha=0.05,print_result=True):
    """
    Correlation test for a continuous array
    Parameters:
    -----------
    X0,X1: 1D numeric non NA array
    X_name: list of 2 strings
    test: {"pearson","spearman","kendall","pointbiserial"}, default="pearson"
        Note: Test assumptions need to be checked before performing the test. Otherwise, the result will not be accurate 
        pearson: measure strength and direction of linear relationship
        spearman, kendall: measure strength and directio of monotonic relationship
        pointbiserialr: give results identical to pearson. A shortcut to calculate correlation for continuous and categorical variables
    alpha: [0,1], default=0.05
        Critical value to indicate threshold at which the null hypothesis is rejected
    print_result: bool, default=True
        If True, result of test is printed
    Returns:
    --------
    val: statistic test
    p: p-value

    """
    if test=="pearson":
        val,p = pearsonr(X0,X1)
    elif test=="spearman":
        val,p = spearmanr(X0,X1)
    elif test=="kendall":
        val,p = kendalltau(X0,X1)
    elif test=="pointbiserial":
        val,p = pointbiserialr(X0,X1)

    # Interpret the level of significance
    if print_result:
        print("H0: The two variables are not correlated")
        if X_name:
            if p > alpha:
                print(f'Fail to reject H0: {X_name[0]} and {X_name[1]} are uncorrelated with p={p:.3f}')
            else:
                print(f'Reject H0: {X_name[0]} and {X_name[1]} are correlated with  p={p:.3f}')

        else:
            if p > alpha:
                print(f'Fail to reject H0: the two variables are uncorrelated with p={p:.3f}')
            else:
                print(f'Reject H0: the two variables are correlated with  p={p:.3f}')
    return val,p



def cramers_v(X0,X1,X_name=None,alpha=0.05,print_result=True):
    """
    Calculate association degrees between 2 categorical variables
    
    Parameters:
    ----------
    X0,X1: 1D numeric non NA array
    X_name: list of 2 strings
    alpha: float [0,1], default=0.05
        Thereshold significance value 
    
    Returns:
    -------
    v: statistic test
    p: p-value
    """
    crosstab = pd.crosstab(X0,X1)
    chitest, p, dof, expected = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    dof0 = min(crosstab.shape) - 1
    v = np.sqrt(chitest/(n*dof0))
    eff_size=np.nan

    # Interpretation of results
    if p <= alpha:
        if dof0 == 1:
            if v <0.3:
                eff_size ="small"
            elif v<0.5:
                eff_size = "medium"
            else:
                eff_size = "large"
        elif dof0 == 2:
            if v <0.21:
                eff_size ="small"
            elif v<0.35:
                eff_size = "medium"
            else:
                eff_size = "large"
        elif dof0 == 3:
            if v <0.17:
                eff_size ="small"
            elif v<0.29:
                eff_size = "medium"
            else:
                eff_size = "large"
        elif dof0 == 4:
            if v <0.15:
                eff_size ="small"
            elif v<0.25:
                eff_size = "medium"
            else:
                eff_size = "large"
        else:
            if v <0.15:
                eff_size ="small"
            elif v<0.25:
                eff_size = "medium"
            else:
                eff_size = "large"
        if print_result:
            print(crosstab)
            if X_name:
                print(f"{X_name[0]} is dependent on {X_name[1]} with p = {p:.3f}")
            else:
                print(f"Association relationship with p={p:.3f}")
            print(f"With a {eff_size} effect size v = {v:.2f} for dof = {dof0}")
    else:
        if print_result:
            print(crosstab)
            if X_name:
                print(f'No association between {X_name[0]} and {X_name[1]}')
            else:
                print("No association between 2 variables")
    return v,p,eff_size



def generate_sym_matrix(df,col,test="pearson",alpha=0.05,threshold=0.05):
    """
    Generate symmetric matrix of correlation coefficients
    Parameters:
    -----------
    df: original dataframe with no NA in columns to be investigated
    col: list of column names
        Columns to be compared
    test: {"pearson","spearman","kendall","cramersv","pointbiserial"}, default="pearson"
        cramersv can only be used for categorical data
    alpha: float [0,1], default=0.05
        Threshold of significance
    threshold: float [0,1], default=0.05
        Threshold of % of unique values to be considered categorical

    Returns:
    -------
    df_result: pd.DataFrame
        Symmetric matrix of coefficient
    hm: axis handle of heatmap plot
    """
    assert ~(df[col].isna().any(axis=1).any()), "Data contains NA, must be removed before performing tests"
    df_result = pd.DataFrame()
    for i in col:
        for j in col:
            if test=="cramersv":
                countval = np.array([df[k].nunique()/df[k].count() for k in col])
                assert all(countval<threshold), "Cramers only works for categorical data. Check input data or increase threshold to bypass categorical data check"
                v,p,effect_size = cramers_v(df[i].values,df[j].values,alpha=alpha,print_result=False)
            else:
                v,p = correlation(df[i].values,df[j].values,test=test,alpha=alpha,print_result=False)

            df_result.loc[i,j] = v
            
    hm = sns.heatmap(df_result,cbar=True,annot=True,square=True,fmt=".2f",yticklabels=col,xticklabels=col).set(title =f"{test} test's coefficient")

    return df_result,hm

            

