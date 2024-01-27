'''
config.py contains global variables
    1. REPO_NAME: str, Repository name
    2. OUTPUT_PATH: Path to store outputs
    3. STATIC_FNAME: List to store static file names
    4. DOWNLOAD_URL: String of url to download data files
'''
import os

REPO_NAME = 'wmt_sales_pred'
ROOT_PATH = os.path.realpath(os.getcwd())[:os.path.realpath(os.getcwd()).find(REPO_NAME)+len(REPO_NAME)]
OUTPUT_PATH = os.path.join(ROOT_PATH, r'doc')

STATIC_FNAME = ['README.md']

DOWNLOAD_URL = 'https://raw.githubusercontent.com/xinxiewu/datasets/main/walmart_sales/walmart_sales.zip'