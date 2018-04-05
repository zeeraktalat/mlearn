"""Module for overloading __iter__ to create reusable generators as needed."""

class MongoRetrieveManyIter(object):
    """Iterates over many search results allowing for multiple iterations."""

    def __init__(self,
                 conn,
                 query: dict,
                 collection: str = None,
                 **kwargs):
        """Allow for reusable iterator for retrieval of documents.

        :param conn: Fully initialised connection to database
        :param query: Query to be executed.
        """
        self.conn = conn
        self.query = query
        self.kwargs = kwargs
        self.collection = collection

    def __iter__(self):
        for doc in self.conn.retrieve_many_records(self.query, self.collection, **self.kwargs):
            yield doc
