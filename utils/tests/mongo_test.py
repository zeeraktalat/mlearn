import unittest
from readers import MongoDB

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
        self.assertEqual(self.conn.database, None)

    def test_created_database(self)
        """Test that database exists once specified."""
        self.conn.database = 'TestDB'
        self.assertEqual('TestDB', self.conn.database.name)

    def test_no_created_collection(self):
        """Test that collection does not exist unless specified."""
        self.assertEqual(self.conn.collection, None)

    def test_created_collection(self)
        """Test that collection exists once specified."""
        self.conn.database = 'TestDB'
        self.conn.collection = 'testCollection'
        self.assertEqual('testCollection', self.conn.coll.name)

    def test_no_created_indices(self):
        """Test that indices do not exist prior to creation."""
        self.assertEqual(self.conn.indices, None)

    def test_created_indices(self):
        """Test that indices exist once created."""
        pass

    def test_create_record(self):
        """Test that insertion of record works."""
        pass

    def test_update_record(self):
        """Test that record is updated."""
        pass

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

