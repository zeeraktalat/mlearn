import unittest
from mlapi.utils.readers import MongoDB

class MongoTest(unittest.TestCase):
    """Tests MongoDB methods."""
    conn = MongoDB('localhost', 27017)

    def test_successful_connection(self):
        """Test that connection is established
        and that correct exceptions are raised if error occurs.
        """
        self.assertIsInstance(MongoDB, self.conn)

    def test_no_created_database(self):
        """Test that database does not exist unless specified."""
        self.assertRaise(AttributeError, self.conn.database)

    def test_created_database(self):
        """Test that database exists once specified."""
        self.conn.database = 'TestDB'
        self.assertEqual('TestDB', self.conn.database.name)

    def test_no_created_collection(self):
        """Test that collection does not exist unless specified."""
        self.assertRaise(AttributeError, self.conn.collection)

    def test_created_collection(self):
        """Test that collection exists once specified."""
        self.conn.database   = 'TestDB'
        self.conn.collection = 'testCollection'
        self.assertEqual('testCollection', self.conn.coll.name)

    def test_no_created_indices(self):
        """Test that indices do not exist prior to creation."""
        self.assertRaises(AttributeError, self.conn.indices)

        self.conn.database   = 'TestDB'
        self.conn.collection = 'testCollection'
        self.assertEqual(self.conn.indices, {})

    def test_created_indices(self):
        """Test that indices exist once created."""
        self.conn.database   = 'TestDB'
        self.conn.collection = 'testCollection'
        self.conn.indices    = ('id', pymongo.DESCENDING)
        self.assertEqual(self.conn.indices['id_-1']['key'], [('id', -1)])

    def test_create_record(self):
        """Test that insertion of record works."""
        self.conn.database   = 'TestDB'
        self.conn.collection = 'testCollection'
        first                = {'_id': 1, 'text': 'This is a test', 'id': 1}
        second               = {'_id': 2, 'text': 'This is a test', 'id': 2}
        third                = {'_id': 3, 'text': 'This is a test', 'id': 3}
        self.assertTrue(self.conn.store_one_record(first))
        self.assertTrue(self.conn.store_one_record(second))
        self.assertTrue(self.conn.store_one_record(third))
        self.assertFalse(self.onn.store_one_record(third))
        self.assertFalse(self.conn.store_one_record('id'))

    def test_update_record(self):
        """Test that record is updated."""
        third = {'_id': 1, 'text': 'This is a test', 'id': 3}
        self.assertTrue(self.conn.store_one_record(1, third))
        self.assertFalse(self.conn.store_one_record('id'))

    def test_retrieve_one_record(self):
        """Test that record is retrieved."""
        pass

    def test_retrieve_many_records(self):
        """Test that many records are retrieved."""
        pass

    def test_reusable_iterator(self):
        """Test that iterator is in fact reusable."""
        pass

    def test_delete_record(self):
        """Test that record is deleted."""
        pass

    def test_delete_database(self):
        """Test that record is deleted."""
        pass

if __name__ == "__main__":
    unittest.main()
