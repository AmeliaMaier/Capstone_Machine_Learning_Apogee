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

psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')

MAX_RETRIES = 20
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES)
session.mount('https://', adapter)
session.mount('http://', adapter)
failed_urls = []

#create list of all urls
urls_starting_points = pickle.load(open( "data/url_series.p", "rb" ))


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

def write_to_tables(url, links, root, html_page):
    # engine = create_engine(f'postgresql+psycopg2://{psql_user}:{psql_password}@localhost:5432/website_link_mapping',echo=False)
    conn = psycopg2.connect(dbname='website_link_mapping', user=psql_user, password=psql_password, host='localhost')
    c = conn.cursor()

    #add originating link html
    sql_statement = '''
    INSERT INTO urls (url_raw, site_description, html_raw, linked, root_url)
        VALUES (%(url_raw)s, %(site_description)s, %(html_raw)s, %(linked)s, %(root_url)s)
        ON CONFLICT (url_raw)
        DO UPDATE SET (site_description, html_raw, linked, root_url)
            =(EXCLUDED.site_description, EXCLUDED.html_raw, EXCLUDED.linked, EXCLUDED.root_url);
    '''
    description = get_description(html_page)

    if root is None:
        root = 'not_available'
    if description is None:
        description = 'not_available'

    var_dict = {'url_raw':url, 'html_raw':html_page, 'site_description':description, 'linked': True, 'root_url': root}

    if (not links is None) and len(links) > 0:
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

    try:
        c.execute(sql_statement, var_dict)
    except ValueError:
        e = sys.exc_info()[0]
        logging.warning( "<p>Error Writing to table: %s</p>" % e )
        for k in var_dict:
            if isinstance(var_dict[k], str) and '\x00' in var_dict[k]:
                #needed because some descriptions, urls, or html have a '\x00' which psql thinks is null
                var_dict[k] = 'not_available'
        try:
            c.execute(sql_statement, var_dict)
        except:
            e = sys.exc_info()[0]
            logging.warning( "<p>Error Writing to table for url %s: %s</p>" % url, e )
    except:
        e = sys.exc_info()[0]
        logging.warning( "<p>Error Writing to table for url %s: %s</p>" % url, e )
    conn.commit()
    conn.close()

def get_url_layer():
    conn = psycopg2.connect(dbname='website_link_mapping', user=psql_user, password=psql_password, host='localhost')
    c = conn.cursor()

    sql_statement = '''
    SELECT urls.url_raw FROM urls
        LEFT JOIN website_links
            ON urls.url_ID = website_links.from_url_ID
        WHERE NOT urls.linked
        AND website_links.from_url_ID is NULL;
    '''

    c.execute(sql_statement)
    links = c.fetchall()
    links = [link[0] for link in links]
    conn.commit()
    conn.close()
    return links

def initial_load_depth_one(urls_starting_points, limit=None):
    #initial load that will do depth of 1
    count = 0
    for series in urls_starting_points: #each series contains unique urls from an different csv
        for url in series: # loops through the urls from one csv
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
            if not count is None and count >= limit:
                break
        if not count is None:
            break

def pooled_url_to_db(url):
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


#initial_load_depth_one(urls_starting_points, limit=4)
crawl_thread(depth=1, limit=1000)

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
