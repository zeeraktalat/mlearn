import unittest
from readers import MongoDB

class MongoTest(unittest.TestCase):

    def test_conn(self):
        conn = MongoDB('localhost', 27017)
        self.assertEqual(conn, None)

    def test_database(self):
        conn = MongoDB('localhost', 27017)
        self.assertEqual(conn.database, None)
        conn.database = 'Twitter'
        self.assertEqual('Twitter', conn.database)
