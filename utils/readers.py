"""Module for data reading."""

import pymongo
import logger
from pymongo import MongoClient
# from nltk import word_tokenize


class MongoDB(object):
    """MongoDB connector. Everything connnected to database side."""

    def __init__(self, hostname, port):
        """Initialise methods."""
        # Set up logging
        self.log = logger.initialise_loggers('mongodb', '../log/db.log')

        try:
            # Connect to DB
            self.conn  = MongoClient(host = hostname, port = port)
            self.conn.admin.command("ismaster")
        except Exception as e:
            self.log.error("Connection Error:\n\t{0}".format(e))
            raise(e)

    def __iter__(self):
        """Overload __iter__ so that reusable iterators can be used.

        TODO: Unfinished.
        """
        # TODO Unfinished
        for record in self.retrieve_many_records(self.collection, {}, no_cursor_timeout = False):
            yield record

    @property
    def database(self):
        """Obtain and set the database used. Setter accepts a string."""
        return self.db

    @database.setter
    def database(self, db):
        self.db = self.conn[db]

    @property
    def collection(self):
        """Obtain and set the collection used. Setter accepts a string."""
        return self.coll

    @collection.setter
    def collection(self, collection):
        self.coll = self.db[collection]

    @property
    def indices(self):
        """Retrieve indices for a given collection.

        :returns indices: Dictionary of indices
        """
        return self.coll.index_information()

    @indices.setter
    def indices(self, indices):
        """Create indices.

        :param indices: Fields to be used as indices, see pymongo API for format.
        """
        try:
            self.coll.create_index(indices)
        except Exception as e:
            self.logger.error("Could not create index:\n\t{0}".format(e))
            raise(e)

    def retrieve_one_record(self, query, *args):
        """Retrieve or store single record.

        :query: Query sought
        :*args: Other limitations or inputs to query
        """
        return self.coll.find_one(query, *args)

    def store_one_record(self, record, collection):
        """Insert a recording into collection. Only writes if _id does not exist already.

        :param record: Record to insert
        :param collection: Collection to write to
        :param *args: Additional arguments to specify
        :param **kwargs: **kwargs additional keyword arguments to specify
        :returns boolean: True if write suceeds, False if not.
        """
        try:
            assert(isinstance(record, dict))
        except AssertionError as e:
            self.log.error("Input is not a dict. Type is: {0} Error:\n{1}".format(type(record), e))
            return False

        try:
            self.db[collection].insert_one(record)
        except pymongo.errors.DuplicateKeyError as e:
            self.log.warning("Duplicate Key found {0}".format(e))
            return -1
        except pymongo.errors.WriteError as e:
            self.store_one_record(record, collection)
        except Exception as e:
            self.log.error("WriteError (collection: {}): {}".format(collection, e))
            return 0
        return 1

    def update_one_record(self, _id, updates, collection, *args, **kwargs):
        """Update one document in DB.

        :param _id: ID to filter by
        :param updates: Dict containing information on which fields are updated and their values.
        :param collection: Collection to find document in.
        """
        try:
            assert(isinstance(updates, dict))
        except AssertionError as e:
            self.log.error("Input is not a dict. Type is: {0} Error:\n{1}".format(type(updates), e))
            return False

        try:
            self.db[collection].update({'_id': _id}, updates, *args, upsert = True, **kwargs)
        except Exception as e:
            self.log.error("WriteError (collection: {}): {}".format(collection, e))
            return False
        return True

    def retrieve_many_records(self, collection, query, **kwargs):
        """Return all records that satisfy query.

        :param query: Filtering query
        :param *args: Optional arguments
        :returns resultset: Mongo ResultSet
        """
        return self.db[collection].find(query, **kwargs)


# def ReadMongo(object):
#     def __init__(self, conn, dbs):
#         self.conn = conn
#         self.databses = dbs
#
#     def __iter__(self):
#         for db in self.databases:
#             self.conn.database = db
#             for coll in self.conn.get_collection_names():
#                 self.conn.collection = coll
#
#                 for doc in self.conn.retrieve_many_records(coll, {}, no_cursor_timeout = False):
#                     if 'Posts' in coll:
#                         if db == 'VoatDB':
#                             tokenised = word_tokenize(doc['title'].lower())
#                         else:
#                             tokenised = word_tokenize(doc['selftext'].lower() +
#                                                       ' ' +
#                                                       doc['title'].lower())
#                     else:
#                         if db == 'VoatDB':
#                             tokenised = word_tokenize(doc['text'].lower())
#                         else:
#                             tokenised = word_tokenize(doc['comment'].lower())
#                     yield tokenised
