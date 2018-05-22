import psycopg2
import pandas as pd
import os
from sqlalchemy import create_engine

class ReadFromDB:
    def __init__(self, credentials):
        psql_user, psql_password = credentials
        self.conn = psycopg2.connect(dbname='website_link_mapping', user=psql_user, password=psql_password, host='localhost')
        self.c = conn.cursor()

    def query_for_df(self, query_str):
        pass

    def query_for_all_w_vars(self, query_str, var_dict):
        c.execute(query_str, var_dict)
        result = c.fetchall()
        return result

    def query_for_all(self, query_str,):
        c.execute(query_str)
        result = c.fetchall()
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

    def __del__(self):
        conn.commit()
        conn.close()
