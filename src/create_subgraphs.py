import pandas as pd
import numpy as np
import boto3
import pickle
import os
import sys
import sql_statements
import db_connection as conn

psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')

client = boto3.client('s3') #low-level functional API

def create_starting_points():
    query_starting_points = sql_statements.starting_points()
    db_conn = conn((psql_user, psql_password))
    results = db_conn.query_for_all(query_starting_points)
    # starting_points = psql.read_sql(query_starting_points, conn)
    starting_points = []
    for record in results:
        starting_points.append(record[0])
    pd.DataFrame(data = starting_points, columns = ['starting_points']).to_csv('subgraphs/starting_points.csv')


def create_map_dfs_in_file_structure():
    starting_points = pd.read_csv('subgraphs/starting_points.csv')

    query_temp_table = sql_statements.create_limited_links_temp()
    query_subgraphs = sql_statements.recursive_subgraphs()
    db_conn = conn((psql_user, psql_password))
    db_conn.simple_execute(query_temp_table)

    columns=['from_url_id', 'to_url_id', 'link_path', 'depth']

    for starting_point in starting_points.starting_points:
        print(f'working on {starting_point}')
        var_dict = {'starting_point': starting_point}
        subgraph = db_conn.query_for_df_w_vars(self, query_str, var_dict, columns)
        if len(subgraph.from_url_id.values) < 2:
            print('skipped')
            continue
        depth = max(subgraph.depth)
        bredth = len(subgraph.from_url_id.values)
        directory = f'subgraphs/depth_{depth}/bredth_{bredth}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        print(f'{directory}starting_point_{starting_point}.csv')
        subgraph.to_csv(f'{directory}starting_point_{starting_point}.csv')
    conn.commit()
    conn.close()


def create_node_edge_csvs():
    node_query = sql_statements.all_nodes()
    edge_query = sql_statments.all_edges()
    db_conn = conn((psql_user, psql_password))
    nodes = db_conn.query_for_all(node_query)
    edges = db_conn.query_for_all(edge_query)
    nodes.to_csv('data/nodes.csv')
    edges.to_csv('data/edges.csv')


if __name__=='__main__':
    create_node_edge_csvs()
