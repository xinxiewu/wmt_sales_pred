'''
util.py contains custom functions:
    1. delete_files: Delete files in given folder, with exception
    2. download_file: Download, unzip and read files
    3. vars_dist: Statistics & distribution of single variables
'''
import os
import requests
import shutil
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
        
# vars_dist(df)
def vars_dist(df=None):
    pass