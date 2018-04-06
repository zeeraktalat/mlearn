"""Module for data reading."""

import pymongo
from mlapi.utils import logger
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, WriteError
from pymongo.collection import Collection
from pymongo.database import Database
from typing import Generator


class MongoDB(object):
    """MongoDB connector. Everything connnected to database side."""

    def __init__(self, hostname: str = 'localhost', port: int = 27017) -> None:
        """Initialise methods."""
        # Set up logging
        # TODO Update logging to global + local log
        self.log = logger.initialise_loggers('mongodb', '../log/db.log')

        try:
            # Connect to DB
            self.conn  = MongoClient(host = hostname, port = port)
            self.conn.admin.command("ismaster")
        except Exception as e:
            self.log.error("Connection Error:\n\t{0}".format(e))
            raise(e)

    @property
    def database(self) -> Database:
        """Obtain and set the database used. Setter accepts a string."""
        return self.db

    @database.setter
    def database(self, db: str) -> None:
        self.db = self.conn[db]

    def drop_database(self, db: str) -> None:
        """Drop Database.

        :param db: Database to be dropped.
        """
        self.conn.drop_database(db)

    @property
    def collection(self) -> Collection:
        """Obtain and set the collection used. Setter accepts a string."""
        return self.col

    @collection.setter
    def collection(self, collection: str) -> None:
        self.col = self.db[collection]

    @property
    def indices(self) -> dict:
        """Retrieve indices for a given collection.

        :returns indices: Dictionary of indices
        """
        return self.col.index_information()

    @indices.setter
    def indices(self, indices: list) -> None:
        """Create indices.

        :param indices: Fields to be used as indices, see pymongo API for format.
        """
        try:
            self.col.create_index(indices)
        except Exception as e:
            self.log.error("Could not create index:\n\t{0}".format(e))
            raise(e)

    def drop_index(self, index: str) -> None:
        """Drop created index. Must be string or dict.

        :param index: String or Dict with index information.
        """
        self.col.drop_index(index)

    def drop_all_indices(self) -> None:
        """Drop all indices for collection."""
        self.col.drop_indexes()

    def retrieve_one_record(self, query: dict, *args) -> dict:
        """Retrieve or store single record.

        :query: Query sought
        :*args: Other limitations or inputs to query
        """
        return self.col.find_one(query, *args)

    def store_one_record(self, record: dict, collection: str = None) -> bool:
        """Insert a recording into collection. Only writes if _id does not exist already.

        :param record: Record to insert
        :param collection: Collection to write to
        :param *args: Additional arguments to specify
        :param **kwargs: **kwargs additional keyword arguments to specify
        :returns boolean: True if write suceeds, False if not.
        """
        if not collection:
            collection = self.col
        if isinstance(collection, pymongo.collection.Collection):
            collection = collection.name
        try:
            assert(isinstance(record, dict))
        except AssertionError as e:
            self.log.error("Input is not a dict. Type is: {0} Error:\n{1}".format(type(record), e))
            return False

        try:
            self.db[collection].insert_one(record)
        except DuplicateKeyError as e:
            self.log.warning("Duplicate Key found {0}".format(e))
            return False
        except WriteError as e:
            self.store_one_record(record, collection)
        except Exception as e:
            self.log.error("WriteError (collection: {}): {}".format(collection, e))
            return False
        return True

    def update_one_record(self, _id: str, updates: dict, collection: str = None,
                          op: str = '$set', *args, **kwargs) -> bool:
        """Update one document in DB.

        :param _id: ID to filter by
        :param updates: Dict containing information on which fields are updated and their values.
        :param collection: Collection to find document in.
        """
        try:
            if not collection:
                collection = self.col
            if isinstance(collection, pymongo.collection.Collection):
                collection = collection.name

            assert(isinstance(updates, dict))
            updated = self.db[collection].update_one({'_id': _id},
                                                     {'$set': updates},
                                                     *args,
                                                     **kwargs)
        except AssertionError as e:
            self.log.error("Input is not a dict. Type is: {0} Error:\n{1}".format(type(updates), e))
            return False
        except Exception as e:
            self.log.error("WriteError (collection: {}): {}".format(collection, e))
            return False

        return True if updated.modified_count != 0 else None

    def drop_one_record(self, query: dict, collection: str = None) -> None:
        """Drop record from collection.

        :param query: Query retrieving document to remove.
        """
        if not collection:
            collection = self.col
        if isinstance(collection, Collection):
            collection = collection.name

        self.db[collection].delete_one(query)
        self.log.info("Record deleted: {}".format(query))

    def retrieve_many_records(cls, query: dict, collection: str = None, **kwargs) -> Generator:
        """Return all records that satisfy query.

        :param query: Filtering query
        :param **kwargs: Optional keyword arguments
        :returns resultset: Mongo ResultSet
        """
        if not collection:
            collection = cls.col
        if isinstance(collection, Collection):
            collection = collection.name

        for record in cls.db[collection].find(query, **kwargs):
            yield record