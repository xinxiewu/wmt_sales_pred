'''
util.py contains custom functions:
    1. delete_files: Delete files in given folder, with exception
    2. download_file: Download, unzip and read files
    3. vars_dist: Statistics & distribution of single variables
'''
import os
import requests
import shutil
import numpy as np
from scipy.stats import chisquare, kstest, f_oneway, chi2_contingency
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# delete_files(path, exception)
def delete_files(path=None, exception=None):
    ''' Delete files in the given folder path, with exception

    Args:
        path: path, starting with r''
        exception: file names to keep

    Returns:
        None
    '''
    for fname in os.listdir(path):
        if fname not in exception:
            os.remove(os.path.join(path, fname))
    return

# download_file(url, unzip, output, dt_col)
def download_file(url=None, unzip=None, output=None, dt_col=None):
    ''' Download, unzip and read files into DataFrame

    Args:
        url: str, data files' download link
        unzip: Y if need to unzip files, otherwise N
        output: path to store files, starting with r''
        dt_col: list, feature needed to date format

    Returns:
        DataFrame
    '''
    local_filename = os.path.join(output, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    if unzip.upper() == 'Y':
        shutil.unpack_archive(local_filename, os.path.dirname(local_filename))
        i = 0
        for fname in os.listdir(os.path.dirname(local_filename)):
            if fname.endswith('.csv'):
                res = fname
                i += 1
        if i == 1:
            if dt_col == None:
                return pd.read_csv(os.path.join(os.path.dirname(local_filename), res))
            else:
                return pd.read_csv(os.path.join(os.path.dirname(local_filename), res), parse_dates=dt_col)
        else:
            print(f"There are {i} csv files!")
            return
    else:
        if dt_col == None:
            return pd.read_csv(local_filename)
        else:
            return pd.read_csv(local_filename, parse_dates=dt_col)
        
# vars_dist(df, catex, ft_catex, cont, ft_cont, output, fname, fpath, subplots, figsize)
def vars_dist(df=None, catex=False, ft_catex=None, cont=False, ft_cont=None, output=False, fname=None, fpath=None, subplots=None, figsize=None):
    ''' Generate single variable's statistics & distribution, and save

    Args:
        df: DataFrame, input data for analysis
        catex: True if analyze categorical features, otherwise False
        ft_catex: List, categorical features
        cont: True if analyze continuous features, otherwise False
        ft_cont: List, continuous features
        output: True if generate file and save, otherwise False
        fname: str, file name if need to output
        fpath: path to store files, starting with r''
        subplots: List, # of plots
        figsize: tuple, figure size

    Returns:
        DataFrame
    '''
    sns.set()
    fig, axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)
    ax, i, cols = axes.flatten(), 0, list(ft_catex) if catex == True else list(ft_cont)
    if catex == True:
        res = pd.DataFrame(columns=['Variable', 'Type', 'Records', 'Unique', 'Mean', 'Std', 'Median', 'Pctile_25', 'Pctile_75', 'Chi-square', 'P-value', 'Conclusion'])
        for col in cols:
            temp = sns.countplot(data=df, x=col, ax=ax[i])
            temp.set(xlabel=col, ylabel=f"Count of {col}", title=f"Count Plot of {col}")
            if df.loc[:, col].nunique() > 10:
                temp.set(xticklabels=[])
            else:
                tol = len(getattr(df, col))
                for p in ax[i].patches:
                    txt = str((100*p.get_height()/tol).round(2)) + '%'
                    x, y = p.get_x(), p.get_height()
                    ax[i].annotate(txt, (x, y), fontsize=15)
            col_cnt = getattr(df, col).value_counts()
            chi_sq = chisquare(np.array(col_cnt), np.repeat(len(getattr(df, col))/len(col_cnt), len(col_cnt)))
            if chi_sq[1] < 0.05:
                conclus = 'Imbalanced'
            else:
                conclus = 'Uniformly distributed'
            df_temp = pd.DataFrame({'Variable': [col], 'Type': ['Categorical'], 'Records': [len(getattr(df, col))], 
                                    'Unique': [getattr(df, col).nunique()], 'Mean': [round(col_cnt.mean(), 2)], 'Std': [round(col_cnt.std(), 2)], 
                                    'Median': [round(col_cnt.median(), 2)], 'Pctile_25': [round(col_cnt.quantile(q=0.25), 2)], 'Pctile_75': [round(col_cnt.quantile(q=0.75), 2)], 
                                    'Chi-square': [round(chi_sq[0], 2)], 'P-value': [round(chi_sq[1], 2)],
                                    'Conclusion': [conclus]})
            res = pd.concat([res, df_temp])
            i += 1
            col_cnt.mean(), col_cnt.std(), col_cnt.median(), col_cnt.quantile(q=0.25), col_cnt.quantile(q=0.75)
        plt.suptitle('Distribution of Categorical Variables')
    else:
        res = pd.DataFrame(columns=['Variable', 'Type', 'Records', 'Mean', 'Std', 
                                    'Median', 'Pctile_25', 'Pctile_75', 'IQR', 'R_Lo', 'R_Hi', 
                                    'Skewness', 'Kurtosis', 'KS_stat', 'KS_P_val'])
        for col in cols:
            temp = sns.histplot(getattr(df, col), kde=True, color='purple', ax=ax[i])
            temp.set(xlabel=col, ylabel='Density', title=f"Histogram of {col}")
            i += 1
            temp = sns.boxplot(getattr(df, col), width=0.3, palette=['m'], ax=ax[i])
            temp.set(xlabel=col, xticklabels=[], ylabel='Value', title=f"Boxplot of {col}")
            ks_res = kstest(getattr(df, col), 'norm', (getattr(df, col).mean(), getattr(df, col).std()))
            conclus = 'Unknown'
            if ks_res[1] >= 0.05:
                conclus = 'Normally distributed'
            elif getattr(df, col).skew() < 0 and getattr(df, col).kurt() < 0:
                conclus = 'Left-skewed & low-peak'
            elif getattr(df, col).skew() < 0 and getattr(df, col).kurt() > 0:
                conclus = 'Left-skewed & high-peak'
            elif getattr(df, col).skew() > 0 and getattr(df, col).kurt() < 0:
                conclus = 'Right-skewed & low-peak'
            elif getattr(df, col).skew() > 0 and getattr(df, col).kurt() > 0:
                conclus = 'Right-skewed & high-peak'
            df_temp = pd.DataFrame({'Variable': [col], 'Type': ['Continuous'], 'Records': [len(getattr(df, col))], 
                                    'Mean': [round(getattr(df, col).mean(), 2)], 'Std': [round(getattr(df, col).std(), 2)], 'Median': [round(getattr(df, col).median(), 2)], 
                                    'Pctile_25': [round(getattr(df, col).quantile(q=0.25), 2)], 'Pctile_75': [round(getattr(df, col).quantile(q=0.75), 2)], 
                                    'IQR': [round(getattr(df, col).quantile(q=0.75)-getattr(df, col).quantile(q=0.25), 2)], 
                                    'R_Lo': [round(getattr(df, col).quantile(q=0.25)-1.5*(getattr(df, col).quantile(q=0.75)-getattr(df, col).quantile(q=0.25)), 2)], 
                                    'R_Hi': [round(getattr(df, col).quantile(q=0.75)+1.5*(getattr(df, col).quantile(q=0.75)-getattr(df, col).quantile(q=0.25)), 2)],
                                    'Skewness': [round(getattr(df, col).skew(), 2)], 'Kurtosis': [round(getattr(df, col).kurt(), 2)],
                                    'KS_stat': [round(ks_res[0], 2)], 'KS_P_val': [ks_res[1]],
                                    'Conclusion': [conclus]})
            res = pd.concat([res, df_temp])
            i += 1
        plt.suptitle('Distribution of Continuous Variables')
        
    plt.tight_layout()
    
    if output == True:
        plt.savefig(os.path.join(fpath, fname))

    return res.reset_index().drop(columns=['index'])

# vars_relatsh(df, cols_1, cols_2, method, output, fpath, fname, subplots, figsize)
def vars_relatsh(df=None, cols_1=None, cols_2=None, method=None, output=False, fpath=None, fname=None, subplots=None, figsize=None):
    ''' Generate statistics & relationships b/w variables, and save
    
    Args:
        df: DataFrame, input data for analysis
        cols_1: List, variables to use
        cols_2: List, variable to use
        method: str, method for analysis. 'cont-catex', 'cont-cont', 'catex-catex'
        output: True if generate file and save, otherwise False
        fname: str, file name if need to output
        fpath: path to store files, starting with r''
        subplots: List, # of plots
        figsize: tuple, figure size

    Returns:
        DataFrame
    '''
    if method.lower() == 'cont-cont':
        sns.set_theme(style="white")
        res = df[cols_1].corr()
        ax = sns.heatmap(data=res, annot=True, fmt='.2f',
                 vmin=-1.0, vmax=1.0, center=0.0,
                 mask=np.triu(np.ones_like(res, dtype=bool)),
                 cmap=sns.diverging_palette(230, 20, n=200, as_cmap=True),
                 square=True, linewidths=0.5, cbar_kws={"shrink":.5}
                )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45, horizontalalignment='right', fontsize=8)
        plt.suptitle('Correlation w/ Continuous Variables')
    else:
        sns.set()
        fig, axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)
        if subplots[0] + subplots[1] > 2:
            ax = axes.flatten()
        i = 0

        if method.lower() == 'cont-catex':
            res = pd.DataFrame(columns=['var_pair', 'f_val_anova', 'p_val_anova'])

            for col_2 in cols_2:
                arr_temp = getattr(df, col_2).unique()
                for col_1 in cols_1:
                    f_val, p_val = f_oneway(df.query(f'{col_2} == {arr_temp[0]}')[[col_1]], df.query(f'{col_2} == {arr_temp[1]}')[[col_1]])
                    df_temp = pd.DataFrame({'var_pair': [col_2 + '-' + col_1], 'f_val_anova': [round(f_val[0], 2)], 'p_val_anova': [round(p_val[0], 2)]})
                    res = pd.concat([res, df_temp])

            for col_catex in cols_2:
                for col in cols_1:
                    temp = sns.boxplot(data=df, x=col_catex, y=col, width=0.3, palette=['m'], ax=ax[i])
                    i += 1
                    
            plt.suptitle(f'Boxplot of {cols_2} w/ Continuous Variables')
        elif method.lower() =='catex-catex':
            res = pd.DataFrame(columns=['var_pair', 'chi-square', 'p_val', 'df'])

            for m in range(len(cols_1)):
                col_1 = cols_1[m]
                for j in range(m+1, len(cols_1)):
                    col_2 = cols_1[j]
                    col_1_distinct, col_1_distinct_len = getattr(df, col_1).unique(), len(getattr(df, col_1).unique())
                    col_1_dict = dict(zip(col_1_distinct, range(col_1_distinct_len)))
                    col_2_distinct, col_2_distinct_len = getattr(df, col_2).unique(), len(getattr(df, col_2).unique())
                    col_2_dict = dict(zip(col_2_distinct, range(col_2_distinct_len)))
                    M = np.zeros((col_1_distinct_len, col_2_distinct_len))
                    for z in range(df.shape[0]):
                        M[col_1_dict[df.iloc[z,:][col_1]]][col_2_dict[df.iloc[z,:][col_2]]] += 1
                    chi2, p, defre, expected = chi2_contingency(M, correction=False)
                    df_temp = pd.DataFrame({'var_pair': [cols_1[m] + '-' + cols_1[j]], 'chi-square': [round(chi2, 2)], 'p_val': [round(p, 2)], 'df': [defre]})
                    res = pd.concat([res, df_temp])

            for m in range(len(cols_1)):
                col_1 = cols_1[m]
                for j in range(m+1, len(cols_1)):
                    col_2 = cols_1[j]
                    temp = sns.scatterplot(data=df, x=col_1, y=col_2, hue=col_2)
                    i += 1
            plt.suptitle(f'Scatterplot of {cols_1}')

    plt.tight_layout()

    if output == True:
        plt.savefig(os.path.join(fpath, fname))

    return res.reset_index().drop(columns=['index'])