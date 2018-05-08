import requests
import psycopg2
from bs4 import BeautifulSoup
# from sqlalchemy import create_engine
# # from sqlalchemy import Table, Column, String, MetaData
# from sqlalchemy.dialects.postgresql import insert
import urllib
from urllib.error import HTTPError
from urllib.error import URLError
import pandas as pd
import pickle
import numpy as np
import sys
import re
import os

psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')

POOL_SIZE = 2
MAX_RETRIES = 20
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES)
session.mount('https://', adapter)
session.mount('http://', adapter)
failed_urls = []

#create list of all urls
urls_starting_points = pickle.load(open( "data/url_series.p", "rb" ))


# def scrape_parallel_concurrent(urls, pool_size):
#     """
#     Uses multiple processes to make url requests.
#
#     Parameters
#     Y----------
#     pool_size: number of worker processes
#     ulrs: list of urls to request
#
#     Returns
#     -------
#     None
#     """
#     coll.remove({})
#     pool = multiprocessing.Pool(pool_size)
#
#     pool.map(url_search_parallel, urls)
#     pool.close()
#     pool.join()
#
# def url_search_parallel(url):
#     """
#     Retrieves the html for the url provided.
#
#     Parameters
#     ----------
#     url: string, full site url
#
#     Returns
#     -------
#     None
#     """
#     try:
#         html_page = urllib2.urlopen(url)
#     except HTTPError as e:
#         print(e)
#         failed_urls.append(url)
#         return
#     except URLError:
#         print("Server down or incorrect domain")
#         failed_urls.append(url)
#         return
#     except: # catch *all* exceptions
#         e = sys.exc_info()[0]
#         print( "<p>Error: %s</p>" % e )
#         failed_urls.append(url)
#         return
#     url_info_concurrent(url, html_page)
#
# def url_info_concurrent(url, response):
#     """
#     Extracts the links from the html and
#     retrieves the url data for each link concurrently.
#
#     Parameters
#     ----------
#     url: originating url
#     response: html response from the originating url.
#
#     Returns
#     None
#     """
#     soup = BeautifulSoup(response)
#     links = [link.get('href') soup.findAll('a', attrs={'href': re.compile("^http://")})]
#
#     threads = len(links)  # Number of threads to create
#
#     jobs = []
#     for i in range(0, threads):
#         thread = threading.Thread(target=scrape_url_info, args=(links[i],originating_url=url))
#         jobs.append(thread)
#         thread.start()
#     for j in jobs:
#         j.join()

def get_url_html(url):
    #print(url)
    try:
        html_page = session.get(url)
    except HTTPError as e:
        print(e)
        failed_urls.append(url)
        return
    except URLError:
        print("Server down or incorrect domain")
        failed_urls.append(url)
        return
    except: # catch *all* exceptions
        e = sys.exc_info()[0]
        print( "<p>Error: %s</p>" % e )
        print('ln 120')
        failed_urls.append(url)
        return
    return html_page.text

def get_links(html_page):
    soup = BeautifulSoup(html_page, "lxml")
    links = [link.get('href') for link in soup.findAll('a', attrs={'href': re.compile("^http://")})]
    #print(links)
    return links

def get_description(html_page):
    soup = BeautifulSoup(html_page, "lxml")
    metas = soup.find_all('meta')
    desc = [meta.attrs['content'] for meta in metas if 'name' in meta.attrs and meta.attrs['name'] == 'description']
    if len(desc) < 1:
        return 'not_available'
    return desc[0]

def write_to_tables(url, links, html_page):
    # engine = create_engine(f'postgresql+psycopg2://{psql_user}:{psql_password}@localhost:5432/website_link_mapping',echo=False)
    conn = psycopg2.connect(dbname='website_link_mapping', user=psql_user, password=psql_password, host='localhost')
    c = conn.cursor()

    #add originating link html
    sql_statement = '''
    INSERT INTO urls (url_raw, site_description, html_raw, linked)
        VALUES (%(url_raw)s, %(site_description)s, %(html_raw)s, %(linked)s)
        ON CONFLICT (url_raw)
        DO UPDATE SET (site_description, html_raw, linked)
            =(EXCLUDED.site_description, EXCLUDED.html_raw, EXCLUDED.linked);
    '''
    description = get_description(html_page)
    var_dict = {'url_raw':url, 'html_raw':html_page, 'site_description':description, 'linked': True}

#    add links to urls
    sql_statement += ' INSERT INTO urls (url_raw) VALUES '
    link_vars = []
    link_string_urls = ' '
    for ind in range(len(links)):
        if ind == 0:
            link_string_urls += f'( %({"link" + str(ind)})s )'
            link_vars.append(f'{"link" + str(ind)}')
        else:
            link_string_urls += f', ( %({"link" + str(ind)})s )'
            link_vars.append(f'{"link" + str(ind)}')
    var_dict.update(dict(zip(link_vars, links)))
    sql_statement += link_string_urls
    sql_statement += ' ON CONFLICT (url_raw) DO NOTHING; '

    #add links to website_links
    sql_statement += ' INSERT INTO website_links (from_url_ID, to_url_ID) VALUES '
    link_string_website_links = ' '
    for ind in range(len(link_vars)):
        if ind == 0:
            link_string_website_links += '((SELECT url_ID FROM urls WHERE url_raw = %(url_raw)s), (SELECT url_ID FROM urls WHERE url_raw = '
            link_string_website_links += f'%({link_vars[ind]})s))'
        else:
            link_string_website_links += ', ((SELECT url_ID FROM urls WHERE url_raw = %(url_raw)s), (SELECT url_ID FROM urls WHERE url_raw = '
            link_string_website_links += f'%({link_vars[ind]})s))'
    sql_statement += link_string_website_links
    sql_statement += ' ON CONFLICT ON CONSTRAINT website_links_pkey DO NOTHING;'

    c.execute(sql_statement, var_dict)
    conn.commit()
    conn.close()

def get_url_layer():
    conn = psycopg2.connect(dbname='website_link_mapping', user=psql_user, password=psql_password, host='localhost')
    c = conn.cursor()

    sql_statement = '''
    SELECT urls.url_raw FROM urls
        LEFT JOIN website_links
            ON urls.url_ID = website_links.from_url_ID
        WHERE urls.linked = False
        AND website_links.from_url_ID is NULL;
    '''

    c.execute(sql_statement)
    links = c.fetchall()
    links = [link[0] for link in links]
    conn.commit()
    conn.close()
    print(links)
    return links

def initial_load_depth_one(urls_starting_points, limit=None):
    #initial load that will do depth of 1
    count = 0
    for series in urls_starting_points: #each series contains unique urls from an different csv
        for url in series: # loops through the urls from one csv
            count += 1
            html_page = get_url_html(url)
            if html_page is None:
                continue
            links = get_links(html_page)
            if len(links) < 1:
                print(f'no links found, url: {url}')
                continue
            write_to_tables(url, links, html_page)
            if not count is None and count >= limit:
                break
        if not count is None:
            break

def crawl(depth=10):
    for layer in range(depth):
        print(f'starting layer {layer}')
        urls = get_url_layer()
        print(f'urls in layer {len(urls)}')
        for url in urls: # loops through the urls from one csv
            html_page = get_url_html(url)
            if html_page is None:
                continue
            links = get_links(html_page)
            if len(links) < 1:
                print(f'no links found, url: {url}')
                continue
            write_to_tables(url, links, html_page)
        print(f'ending layer {layer}')

initial_load_depth_one(urls_starting_points, limit=4)
crawl(depth=2)

'''
sudo code for planning web link scraping:
    pull list of urls (starting points)
    multi-process across starting urls (width)
    multi-thread across processes (depth)
        per url:
            set depth to 1 at beginning, have maxdepth set to default
            get_links(url, depth)
                if url in website_links table from_url_ID column, return
                if url in starter list and depth != 1, return
                get html from url
                get list of links
                write out to db tables(urls and website_links)
                depth +1
                if depth >= max depth, return
                for link in list of links:
                    get_links(link, depth)
start with 2 starting urls and max depth 5 just to get idea of how long it will take and test code
'''





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
