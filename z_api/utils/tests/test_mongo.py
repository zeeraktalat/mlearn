import unittest
import pymongo
from z_api.utils.mongo import MongoDB
from z_api.utils.mongo import MongoRetrieveManyIter


class MongoTest(unittest.TestCase):
    """Tests MongoDB methods."""

    @classmethod
    def setUpClass(cls):
        """Create database and setup database and collection variables."""
        cls.conn = MongoDB('localhost', 27017)

    @classmethod
    def setup_database(cls):
        """Create database and setup database and collection variables."""
        cls.conn.database   = 'TestDB'
        cls.conn.collection = 'testCollection'

    @classmethod
    def tearDownClass(cls):
        """Tear down after each test."""
        cls.conn.drop_database('TestDB')

    @classmethod
    def setup_fill_collection(cls):
        first  = {'_id': 1, 'text': 'This is a test', 'id': 1}
        second = {'_id': 2, 'text': 'This is not a test', 'id': 2}
        cls.conn.store_one_record(first, 'testCollection')
        cls.conn.store_one_record(second, 'testCollection')

    def test_successful_connection(self):
        """Test that connection is established
        and that correct exceptions are raised if error occurs.
        """
        self.assertIsInstance(self.conn, MongoDB)

    def test_created_database(self):
        """Test that database exists once specified."""
        self.setup_database()
        self.assertEqual('TestDB', self.conn.database.name)

    def test_created_collection(self):
        """Test that collection exists once specified."""
        self.setup_database()
        self.assertEqual('testCollection', self.conn.col.name)

    def test_created_indices(self):
        """Test that indices exist once created."""
        self.setup_database()
        self.conn.indices    = [('id', pymongo.DESCENDING)]
        self.assertEqual(self.conn.indices['id_-1']['key'], [('id', -1)])

    def test_create_record(self):
        """Test that insertion of record works."""
        self.setup_database()

        first  = {'_id': 1, 'text': 'This is a test', 'id': 1}
        second = {'_id': 2, 'text': 'This is not a test', 'id': 2}

        self.assertTrue(self.conn.store_one_record(first, 'testCollection'))
        self.assertTrue(self.conn.store_one_record(second, 'testCollection'))
        self.assertFalse(self.conn.store_one_record(second, 'testCollection'))
        self.assertFalse(self.conn.store_one_record('id', 'testCollection'))

    def test_update_record(self):
        """Test that record is updated."""
        self.setup_database()
        self.setup_fill_collection()

        update = {'_id': 1, 'text': 'This is also not a test', 'id': 3}

        self.assertFalse(self.conn.update_one_record(1, 'id'))
        self.assertTrue(self.conn.update_one_record(1, update, op = '$set'))
        self.assertIsNone(self.conn.update_one_record(1, update, op = '$set'))

    def test_retrieve_one_record(self):
        """Test that record is retrieved."""
        self.setup_database()
        self.setup_fill_collection()

        first = {'_id': 1, 'text': 'This is a test', 'id': 1}
        self.assertEqual(self.conn.retrieve_one_record({'_id': 1}), first)

    def test_retrieve_many_records(self):
        """Test that many records are retrieved."""
        self.setup_database()
        self.setup_fill_collection()

        first  = {'_id': 1, 'text': 'This is a test', 'id': 1}
        second = {'_id': 2, 'text': 'This is not a test', 'id': 2}

        self.assertListEqual(list(self.conn.retrieve_many_records({}, 'testCollection')),
                                                                  [first, second])

    def test_reusable_iterator(self):
        """Test that iterator is in fact reusable."""
        self.setup_database()
        self.setup_fill_collection()

        res        = MongoRetrieveManyIter(self.conn, {})
        first_gen  = [record for record in res]
        second_gen = [record for record in res]

        self.assertListEqual(first_gen, second_gen)

    def test_delete_record(self):
        """Test that record is deleted."""
        self.setup_database()
        self.setup_fill_collection()

        self.assertIsNone(self.conn.drop_one_record({'_id': 1}))
        self.assertIsNone(self.conn.drop_one_record({'_id': 2}))

    def test_delete_database(self):
        """Test that record is deleted."""
        self.assertIsNone(self.conn.drop_database('TestDB'))


if __name__ == "__main__":
    unittest.main()
