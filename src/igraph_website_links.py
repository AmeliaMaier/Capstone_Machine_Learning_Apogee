import igraph
import pandas as pd
import numpy as np
import boto3
# from sqlalchemy import create_engine
import pickle
from io import StringIO
# import psycopg2
import os
import sys
# import pandas.io.sql as psql
# import psycopg2.extras

client = boto3.client('s3') #low-level functional API

# psql_user = os.environ.get('PSQL_USER')
# psql_password = os.environ.get('PSQL_PASSWORD')
#
# def create_starting_points():
#     print('attempting to connect')
#     conn = psycopg2.connect(dbname='website_link_mapping', user=psql_user, password=psql_password, host='localhost')
#     cur = conn.cursor()
#     print('connected to db')
#     query_starting_points = '''
#         SELECT DISTINCT(from_url_ID) FROM website_links
#     '''
#     print('attempting to load starting points')
#     cur.execute(query_starting_points)
#     # starting_points = psql.read_sql(query_starting_points, conn)
#     starting_points = []
#     for record in cur:
#         starting_points.append(record[0])
#     print('starting points loaded')
#     conn.commit()
#     conn.close()
#     pd.DataFrame(data = starting_points, columns = ['starting_points']).to_csv('subgraphs/starting_points.csv')
#
#
# def create_map_dfs_in_file_structure():
#     starting_points = pd.read_csv('subgraphs/starting_points.csv')
#
#     conn = psycopg2.connect(dbname='website_link_mapping', user=psql_user, password=psql_password, host='localhost')
#     cur = conn.cursor()
#
#     query_temp_table = '''
#         DROP TABLE IF EXISTS limited_links;
#         CREATE TABLE IF NOT EXISTS limited_links AS
#         SELECT from_url_ID, to_url_ID	FROM (
#         	SELECT ROW_NUMBER() OVER(PARTITION BY from_url_ID) AS row, website_links.* FROM	website_links) AS temp_grouping
#         WHERE temp_grouping.row <= 25;
#
#     '''
#     query_subgraphs = '''
#         WITH RECURSIVE first_level_elements AS (
#         		(
#         		SELECT from_url_ID, to_url_ID, array[from_url_ID] AS link_path, 0 depth_limit FROM limited_links
#         			WHERE from_url_ID = %(starting_point)s
#         		)
#         		UNION
#         			SELECT nle.from_url_ID, nle.to_url_ID, (fle.link_path || nle.from_url_ID), fle.depth_limit+1 FROM first_level_elements as fle
#         				JOIN limited_links as nle
#         					ON fle.to_url_ID = nle.from_url_ID
#         			WHERE NOT (nle.from_url_ID = any(link_path))
#         				AND fle.depth_limit < 20
#         	)
#         	SELECT from_url_ID, to_url_ID, (link_path || to_url_ID)as link_path, depth_limit  from first_level_elements;
#     '''
#
#     cur.execute(query_temp_table)
#
#     for starting_point in starting_points.starting_points:
#         print(f'working on {starting_point}')
#         cur.execute(query_subgraphs,{'starting_point': starting_point})
#         subgraph_list = []
#         while True:
#             row = cur.fetchone()
#             if row == None:
#                 break
#             subgraph_list.append(list(row))
#         subgraph = pd.DataFrame(data=subgraph_list, columns=['from_url_id', 'to_url_id', 'link_path', 'depth'])
#         if len(subgraph.from_url_id.values) < 2:
#             print('skipped')
#             continue
#         depth = max(subgraph.depth)
#         bredth = len(subgraph.from_url_id.values)
#         directory = f'subgraphs/depth_{depth}/bredth_{bredth}/'
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         print(f'{directory}starting_point_{starting_point}.csv')
#         subgraph.to_csv(f'{directory}starting_point_{starting_point}.csv')
#     conn.commit()
#     conn.close()
#
def create_node_edge_csvs():
    node_query = 'SELECT url_ID, url_raw from urls ORDER BY url_ID;'
    edge_query = 'SELECT from_url_ID, to_url_ID from website_links;'
    nodes = run_query(node_query)
    nodes.to_csv('data/nodes.csv')
    edges = run_query(edge_query)
    edges.to_csv('data/edges.csv')

def run_query(query):
    conn = psycopg2.connect(dbname='website_link_mapping', user=psql_user, password=psql_password, host='localhost')
    results = pd.read_sql_query(query, con=conn)
    conn.commit()
    conn.close()
    return results

def load_graph(path):
    obj = client.get_object(Bucket='websitelinksigraph', Key=path)
    body = obj['Body']
    csv_string = body.read().decode('utf-8')
    edges = pd.read_csv(StringIO(csv_string))
    # edges = pd.read_csv('data/edges.csv', nrows=10000)
    print edges.info()
    return list(zip(edges['from_url_id'], edges['to_url_id']))
    # edges = pd.read_csv('data/edges.csv') about 13.5 million
    # nodes = pd.read_csv('data/nodes.csv') about 4 million


def install_test():
    print igraph.__version__

if __name__=='__main__':
    install_test()
    #create_node_edge_csvs()
    paths = ['subgraphs/depth_0/bredth_25/starting_point_43211.csv', 'subgraphs/depth_0/bredth_5/starting_point_8320.csv','subgraphs/depth_0/bredth_25/starting_point_7602.csv', 'subgraphs/depth_0/bredth_25/starting_point_5029.csv', 'subgraphs/depth_1/bredth_50/starting_point_677194.csv']
    #paths = ['subgraphs/depth_4/bredth_142/starting_point_3747.csv']
    starting_points = ['43211', '8320', '7602', '5029', '677194']
    #starting_points = [3747]
    for path, sp in zip(paths, starting_points):
        print 'loading graph'
        g = igraph.Graph(edges=load_graph(path), directed=True)
        print 'printing simple graph'
        igraph.plot(g, "fast_test_"+sp+".pdf")
        print 'laying out'
        layout = g.layout_kamada_kawai()
        igraph.plot(g, "fast_test_"+sp+"_kk.pdf", layout=layout, directed=True)
        # my_bucket.upload_file(f"fast_test_{sp}.pdf",Key=f"fast_test_{sp}.pdf")

    # create_map_dfs_in_file_structure()
