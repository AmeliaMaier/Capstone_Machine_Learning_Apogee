import requests
from bs4 import BeautifulSoup
from urllib.error import HTTPError
from urllib.error import URLError
import pandas as pd
import pickle
import numpy as np
import sys

MAX_RETRIES = 20
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES)
session.mount('https://', adapter)
session.mount('http://', adapter)

def create_unique_urls_list():
    columns = ['DBM_URL']

    activity_df = pd.read_csv('data/activity_dsi_april.csv', dtype=str, usecols=columns, squeeze=True, skip_blank_lines=True)
    activity = activity_df.fillna('https://9gag.com/').unique()
    click_df = pd.read_csv('data/click_dsi_april.csv', usecols=columns, dtype=str, squeeze=True, skip_blank_lines=True)
    click = click_df.fillna('https://9gag.com/').unique()
    imp0_df = pd.read_csv('data/impressions_dsi_april-000000000000.csv', usecols=columns, dtype=str, squeeze=True, skip_blank_lines=True)
    imp0 = imp0_df.fillna('https://9gag.com/').unique()
    imp1_df = pd.read_csv('data/impressions_dsi_april-000000000001.csv', usecols=columns, dtype=str, squeeze=True, skip_blank_lines=True)
    imp1 = imp1_df.fillna('https://9gag.com/').unique()
    imp2_df = pd.read_csv('data/impressions_dsi_april-000000000002.csv', usecols=columns, dtype=str, squeeze=True, skip_blank_lines=True)
    imp2 = imp2_df.fillna('https://9gag.com/').unique()
    imp3_df = pd.read_csv('data/impressions_dsi_april-000000000003.csv', usecols=columns, dtype=str, squeeze=True, skip_blank_lines=True)
    imp3 = imp3_df.fillna('https://9gag.com/').unique()

    urls_series = [activity, click, imp0, imp1, imp2, imp3]
    pickle.dump(urls_series, open( "data/url_series.p", "wb" ) )

def read_descriptions_simple():
    urls_series = pickle.load( open( "data/url_series.p", "rb" ) )

    urls = []
    failed_urls = []
    descriptions = []
    for series in urls_series:
        for url in series:
            print(url)
            try:
                response = session.get(url)
            except HTTPError as e:
                print(e)
                failed_urls.append(url)
            except URLError:
                print("Server down or incorrect domain")
                failed_urls.append(url)
            except: # catch *all* exceptions
                e = sys.exc_info()[0]
                print( "<p>Error: %s</p>" % e )
                failed_urls.append(url)
            else:
                try:
                    soup = BeautifulSoup(response.text, "lxml")
                    metas = soup.find_all('meta')
                    descriptions.append([ meta.attrs['content'] for meta in metas if 'name' in meta.attrs and meta.attrs['name'] == 'description' ])
                    urls.append(url)
                except: # catch *all* exceptions
                    e = sys.exc_info()[0]
                    print( "<p>Error: %s</p>" % e )
                    failed_urls.append(url)
    url_description_dict = dict(zip(urls, descriptions))
    pickle.dump(url_description_dict, open( "data/url_desription_dict.p", "wb" ) )
    print(failed_urls.append(url))
