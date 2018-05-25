import requests
import psycopg2
from bs4 import BeautifulSoup
import urllib
import lxml.html
from urllib import parse as urlparse
import urllib
from urllib.error import HTTPError
from urllib.error import URLError
import pandas as pd
import pickle
import numpy as np
import sys
import re
import os
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import logging
import db_connection as conn
import sql_statements

psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')

MAX_RETRIES = 20
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES)
session.mount('https://', adapter)
session.mount('http://', adapter)
failed_urls = []


def get_url_html(url):
    #print(url)
    try:
        html_page = session.get(url)
    except HTTPError as e:
        logging.warning(e)
        failed_urls.append(url)
        return
    except URLError:
        logging.warning("Server down or incorrect domain")
        failed_urls.append(url)
        return
    except: # catch *all* exceptions
        e = sys.exc_info()[0]
        logging.warning( "<p>Error: %s</p>" % e )
        failed_urls.append(url)
        return
    return html_page.text

def get_links(html_page):
    soup = BeautifulSoup(html_page, "lxml")
    links = [link['href'] for link in soup.findAll('a') if link.has_attr('href')]
    #print(links)
    return links


def guess_root(links):
    ''' original found from a stacked overflow page
    https://stackoverflow.com/questions/1080411/retrieve-links-from-web-page-using-python-and-beautifulsoup'''
    for link in links:
        if link.startswith('http'):
            parsed_link = urlparse.urlparse(link)
            scheme = parsed_link.scheme + '://'
            netloc = parsed_link.netloc
            return scheme + netloc

def fill_missing_roots():
    missing_roots = pd.read_csv('data/nodes.csv')
    missing_roots['root_url'] = guess_root(missing_roots['url_raw'])
    missing_roots.to_csv('data/nodes.csv')

def resolve_links(links):
    ''' original found from a stacked overflow page
    https://stackoverflow.com/questions/1080411/retrieve-links-from-web-page-using-python-and-beautifulsoup'''
    root = guess_root(links)
    resolved = []
    for link in links:
        if not link.startswith('http'):
            new_link = urlparse.urljoin(root, link)
            if link == new_link:
                continue #was not actually fixed and will be skipped
            resolved.append(new_link)
        else:
            resolved.append(link)
    return (root, resolved)

def get_description(html_page):
    try:
        soup = BeautifulSoup(html_page, "lxml")
        metas = soup.find_all('meta')
        desc = [meta.attrs['content'] for meta in metas if 'name' in meta.attrs and meta.attrs['name'] == 'description']
        if len(desc) < 1:
            return 'not_available'
    except:
        return 'not_available'
    return desc[0]

def check_url_already_seen(url):
    match_count = 0
    sql_statement = sql_statements.check_url_already_seen()
    var_dict = {'url_raw':url, 'linked': True}
    db_conn = conn((psql_user, psql_password))
    match_count = query_for_all_w_vars(query_str, var_dict)
    return (match_count[0][0] > 0)

def bulk_check_url_already_seen():
    '''
    had to stop the initial load early, used this to cut down on the urls sent to threading
    '''
    match_count = 0
    sql_statement = sql_statements.bulk_check_url_already_seen()
    var_dict = {'linked': True}
    db_conn = conn((psql_user, psql_password))
    seen = query_for_all_w_vars(query_str, var_dict)
    seen = set([x[0] for x in seen])
    return seen

def write_to_tables(url, links, root, html_page):

    #add originating link html
    sql_statement = sql_statements.update_urls_table_from()

    description = get_description(html_page)

    if root is None:
        root = 'not_available'
    if description is None:
        description = 'not_available'

    var_dict = {'url_raw':url, 'html_raw':html_page, 'site_description':description, 'linked': True, 'root_url': root}

    if (not links is None) and len(links) > 0:
    #    add links to urls
        sql_temp, var_dict_addition = sql_statements.update_urls_table_tos(links)
        sql_statement += sql_temp
        var_dict.update(var_dict_addition)

    db_conn = conn((psql_user, psql_password))
    db_conn.insert_into_db_with_vars(var_dict, sql_statement)

def get_url_layer():
    sql_statement = sql_statments.get_url_layer()
    db_conn = conn((psql_user, psql_password))
    links = db_conn.query_for_all(sql_statement)
    links = [link[0] for link in links]
    return links

def initial_load_depth_one(urls_starting_points, limit=None):
    #initial load that will do depth of 1
    count = 0
    if not count is None:
        urls_starting_points = urls_starting_points[:limit]
    for url in urls_starting_points: #list of urls
        if check_url_already_seen(url):
            print(f'url already linked, skipped: {url}')
            continue #the url has already been linked and doesn't need to be looked at again
        count += 1
        html_page = get_url_html(url)
        if html_page is None:
            root = 'not_available'
            links = []
            html_page = 'not_available'
        else:
            root, links = resolve_links(get_links(html_page))
        if len(links) < 1:
            print(f'no links found, url: {url}')
        write_to_tables(url, links, root, html_page)


def initial_load_threaded(urls_starting_points, limit=None):
    print(f'starting initial load')
    if not limit is None:
        urls_starting_points = urls_starting_points[:limit]
    print(f'urls in intial load {len(urls_starting_points)}')
    pool = ThreadPool(min(50, len(urls_starting_points)))
    pool.map(pooled_url_to_db, urls_starting_points)
    pool.close()
    pool.join()
    print(f'ending initial load')

def pooled_url_to_db(url):
    if check_url_already_seen(url):
        logging.warning(f'url already linked, skipped. {url}')
        return
    html_page = get_url_html(url)
    if html_page is None:
        root = 'not_available'
        links = []
        html_page = 'not_available'
    else:
        root, links = resolve_links(get_links(html_page))
    if len(links) < 1:
        logging.warning(f'no links found, url: {url}')
    write_to_tables(url, links, root, html_page)

def crawl_thread(depth=5, limit=None):
    for layer in range(depth):
        print(f'starting layer {layer}')
        urls = get_url_layer()
        if not limit is None:
            urls = urls[:limit]
        print(f'urls in layer {len(urls)}')
        pool = ThreadPool(min(50, len(urls)))
        pool.map(pooled_url_to_db, urls)
        pool.close()
        pool.join()
        print(f'ending layer {layer}')

def crawl(depth=5, limit=None):
    for layer in range(depth):
        print(f'starting layer {layer}')
        urls = get_url_layer()
        if not limit is None:
            urls = urls[:limit]
        print(f'urls in layer {len(urls)}')
        for url in urls: # loops through the urls from one csv
            if check_url_already_seen(url):
                continue #the url has already been linked and doesn't need to be looked at again
            html_page = get_url_html(url)
            if html_page is None:
                root = 'not_available'
                links = []
                html_page = 'not_available'
            else:
                root, links = resolve_links(get_links(html_page))
            if len(links) < 1:
                print(f'no links found, url: {url}')
            write_to_tables(url, links, root, html_page)
        print(f'ending layer {layer}')

if __name__ == '__main__':

    fill_missing_roots()


#create_unique_urls_list()

#create list of all urls
# urls_starting_points = pickle.load(open("data/url_series.p","rb"))
# urls_starting_points = urls_starting_points - bulk_check_url_already_seen()
# urls_starting_points = list(urls_starting_points)
# initial_load_threaded(urls_starting_points)
'''
urls in intial load 538302
started with about 11000 in the initial load
'''
# crawl_thread(depth=2)
# print(failed_urls)
'''

urls in layer 1: 697741

'''
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


multi-threading info:
https://jeffknupp.com/blog/2012/03/31/pythons-hardest-problem/
http://chriskiehl.com/article/parallelism-in-one-line/

'''
