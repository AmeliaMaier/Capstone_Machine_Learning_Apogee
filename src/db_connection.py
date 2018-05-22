import psycopg2
import pandas as pd
import pandas.io.sql as psql
import psycopg2.extras
from io import StringIO
import os
from sqlalchemy import create_engine

class ReadFromDB:
    def __init__(self, credentials):
        psql_user, psql_password = credentials
        self.conn = psycopg2.connect(dbname='website_link_mapping', user=psql_user, password=psql_password, host='localhost')
        self.c = conn.cursor()

    def query_for_df_w_vars(self, query_str, var_dict, columns):
        cur.execute(query_str,var_dict)
        result_list = []
        while True:
            row = cur.fetchone()
            if row == None:
                break
            result_list.append(list(row))
        conn.commit()
        return pd.DataFrame(data=result_list, columns=columns)

    def query_for_df(self, query_str, columns):
        cur.execute(query_str)
        result_list = []
        while True:
            row = cur.fetchone()
            if row == None:
                break
            result_list.append(list(row))
        conn.commit()
        return pd.DataFrame(data=result_list, columns=columns)

    def query_for_all_w_vars(self, query_str, var_dict):
        c.execute(query_str, var_dict)
        result = c.fetchall()
        conn.commit()
        return result

    def query_for_all(self, query_str,):
        c.execute(query_str)
        result = c.fetchall()
        conn.commit()
        return result

    def insert_into_db_with_vars(self, var_dict, query_str):
        try:
            c.execute(query_str, var_dict)
        except ValueError:
            e = sys.exc_info()[0]
            logging.warning( "<p>Error Writing to table: %s</p>" % e )
            for k in var_dict:
                if isinstance(var_dict[k], str) and '\x00' in var_dict[k]:
                    #needed because some descriptions, urls, or html have a '\x00' which psql thinks is null
                    var_dict[k] = 'not_available'
            try:
                c.execute(query_str, var_dict)
            except:
                e = sys.exc_info()[0]
                logging.warning( "<p>Error Writing to table for url %s: %s</p>" % url, e )
        except:
            e = sys.exc_info()[0]
            logging.warning( "<p>Error Writing to table for url %s: %s</p>" % url, e )
        conn.commit()

    def simple_execute(self, query_str):
        c.execute(query_str)
        conn.commit()

    def __del__(self):
        conn.commit()
        conn.close()
