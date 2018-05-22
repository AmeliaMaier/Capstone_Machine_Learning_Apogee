'''
The primary unit test file for my trackwell capstone.
'''

import time
import unittest
from src.db_connection import ReadFromDB
import pandas as pd
import numpy as np
import os

SLOW_TEST_THRESHOLD = 0.1
psql_user = os.environ.get('PSQL_USER')
psql_password = os.environ.get('PSQL_PASSWORD')
psql_db_name = os.environ.get('PSQL_TEST')

class TestReadFromDB(unittest.TestCase):
    def setUp(self):
        self._started_at = time.time()

    def tearDown(self):
        elapsed = time.time() - self._started_at
        if elapsed > SLOW_TEST_THRESHOLD:
            print(f'{self.id()}: {round(elapsed,2)}s')

    def test_db_connection(self):
        db_conn = ReadFromDB(f'{psql_user}:{psql_password}@{psql_db_name}')
        df_test = db_conn.query_for_df(f'SELECT * FROM user_table;', ['_id'])
        df_expected = pd.DataFrame(data = {'_id': ['1', '2', '3']})
        self.assertEqual(tuple(df_expected.columns), tuple(df_test.columns))
        self.assertEqual(df_expected.shape, df_test.shape)
        self.assertEqual(df_expected['_id'][1], df_test['_id'][1])
