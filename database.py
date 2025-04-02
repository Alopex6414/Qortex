#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Iterator, Tuple
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _convert_positional_placeholders(where_clause: str) -> str:
    """replace position with placeholder"""
    parts = []
    param_index = 0
    for token in where_clause.split():
        if token == "?":
            param_name = f":param_{param_index}"
            parts.append(param_name)
            param_index += 1
        else:
            parts.append(token)
    return " ".join(parts)


def build_where_clause(conditions: Union[str, Dict[str, Any], List[tuple]], operator: str = "AND") -> Tuple[str, Dict[str, Any]]:
    params = {}
    clauses = []
    # construct where clause
    if isinstance(conditions, dict):
        for col, val in conditions.items():
            param_name = f"where_{col}"
            clauses.append(f"{col} = :{param_name}")
            params[param_name] = val
    elif isinstance(conditions, list):
        for i, (col, op, val) in enumerate(conditions):
            param_name = f"where_{col}_{i}"
            clauses.append(f"{col} {op} :{param_name}")
            params[param_name] = val
    else:
        return conditions, params
    return f" {operator} ".join(clauses), params


class SQLite3:
    def __init__(self, database, autocommit=True, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False):
        self.database = database
        self.autocommit = autocommit
        self.detect_types = detect_types
        self.check_same_thread = check_same_thread
        self.conn: Optional[sqlite3.Connection] = None
        self.in_transaction = False
        sqlite3.register_adapter(list, lambda x: json.dumps(x).encode('utf-8'))
        sqlite3.register_converter("LIST", lambda x: json.loads(x.decode('utf-8')))

    def __enter__(self):
        self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        """establish a connection to the database"""
        try:
            self.conn = sqlite3.connect(self.database, detect_types=self.detect_types, check_same_thread=self.check_same_thread)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.database}")
        except sqlite3.Error as e:
            logger.error(f"Connection failed: {e}")
            raise Exception(f"Database connection failed: {e}") from e

    def close(self):
        """close the connection to the database"""
        if self.conn is not None:
            try:
                if self.in_transaction:
                    self.rollback()
                self.conn.close()
                logger.info("Database connection closed")
            except sqlite3.Error as e:
                logger.error(f"Error closing connection: {e}")
                raise Exception("Failed to close connection") from e
            finally:
                self.conn = None

    @contextmanager
    def transaction(self):
        """transaction context manager"""
        self.begin()
        try:
            yield
            self.commit()
        except Exception as e:
            self.rollback()
            raise

    def begin(self):
        """begin transaction"""
        if self.conn is None:
            self.connect()
        self.in_transaction = True
        self.conn.execute("BEGIN")

    def commit(self):
        """commit transaction"""
        if self.conn:
            try:
                self.conn.commit()
                self.in_transaction = False
                logger.debug("Transaction committed")
            except sqlite3.Error as e:
                logger.error(f"Commit failed: {e}")
                raise Exception("Commit failed") from e

    def rollback(self):
        """rollback transaction"""
        if self.conn:
            try:
                self.conn.rollback()
                self.in_transaction = False
                logger.debug("Transaction rolled back")
            except sqlite3.Error as e:
                logger.error(f"Rollback failed: {e}")
                raise Exception("Rollback failed") from e

    def execute(self, query: str, params: Optional[Union[tuple, dict]] = None, commit: Optional[bool] = None) -> sqlite3.Cursor:
        """
        execute a query and return a cursor
        :param query: SQL query
        :param params: parameters (tuple, dict)
        :param commit: commit transaction (None means autocommit)
        """
        # check database connect status
        if self.conn is None:
            self.connect()
        # get cursor before execute query
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            # handle commit
            should_commit = commit if commit is not None else self.autocommit
            if should_commit and not self.in_transaction:
                self.conn.commit()
            return cursor
        except sqlite3.Error as e:
            logger.error(f"Query failed: {e}\nQuery: {query}\nParams: {params}")
            if self.in_transaction:
                self.rollback()
            raise Exception(f"Query execution failed: {e}") from e

    def table_exists(self, table_name: str) -> bool:
        """check table exists"""
        query = """
            SELECT count(*) FROM sqlite_master
            WHERE type='table' AND name=?
        """
        cursor = self.execute(query, (table_name,))
        return cursor.fetchone()[0] > 0

    def create_table(self, table_name: str, schema: Dict[str, str], if_not_exists: bool = True):
        """
        create table
        :param table_name: name
        :param schema: define dictionary name
        :param if_not_exists: whether add IF NOT EXISTS condition
        """
        if self.table_exists(table_name):
            logger.info(f"Table {table_name} already exists")
            return

        columns = [f"{name} {definition}" for name, definition in schema.items()]
        if_not_exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        query = f"""
            CREATE TABLE {if_not_exists_clause}{table_name} (
                {', '.join(columns)})
        """
        self.execute(query, commit=True)
        logger.info(f"Table {table_name} created")

    def create_index(self, table_name: str, column: str, unique: bool = False):
        """create index"""
        index_name = f"idx_{table_name}_{column}"
        unique_clause = "UNIQUE" if unique else ""
        query = f"""
            CREATE {unique_clause} INDEX {index_name}
            ON {table_name} ({column})
        """
        self.execute(query, commit=True)
        logger.info(f"Index {index_name} created")

    def insert(self, table_name: str, data: Dict[str, Any]) -> int:
        """
        insert row
        :return: identity
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join([f":{k}" for k in data.keys()])
        query = f"""
            INSERT INTO {table_name} ({columns})
            VALUES ({placeholders})
        """
        cursor = self.execute(query, data, commit=not self.in_transaction)
        return cursor.lastrowid

    def batch_insert(self, table_name: str, data: List[Dict[str, Any]]):
        """batch insert"""
        # if not insert data directly return back
        if not data:
            return
        columns = data[0].keys()
        placeholders = ', '.join([f":{col}" for col in columns])
        query = f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES ({placeholders})
        """
        # batch execute insert
        with self.transaction():
            self.conn.executemany(query, data)

    def query(self, table_name: str, columns: Union[str, List[str]] = "*", where: Optional[str] = None, params: Optional[Union[tuple, dict]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict]:
        """
        query method
        :param columns: query columns
        :param where: where clause
        :param params: query parameters
        :param limit: result limit
        :param offset: result offset
        """
        # join columns
        if isinstance(columns, list):
            columns = ', '.join(columns)
        # query condition
        query = f"SELECT {columns} FROM {table_name}"
        # query judgement
        if where:
            query += f" WHERE {where}"
        if limit is not None:
            query += f" LIMIT {limit}"
        if offset is not None:
            query += f" OFFSET {offset}"
        # query execute
        cursor = self.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def paginate(self, table_name: str, page: int = 1, per_page: int = 10, **query_kwargs) -> Dict[str, Any]:
        """query paginate"""
        offset = (page - 1) * per_page
        results = self.query(table_name, limit=per_page, offset=offset, **query_kwargs)
        total = self.count(table_name, query_kwargs.get('where'), query_kwargs.get('params'))
        return {
            "items": results,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page
        }

    def count(self, table_name: str, where: Optional[str] = None, params=None) -> int:
        """query count"""
        query = f"SELECT COUNT(*) FROM {table_name}"
        if where:
            query += f" WHERE {where}"
        cursor = self.execute(query, params)
        return cursor.fetchone()[0]

    def update(self, table_name: str, updates: Dict[str, Any], where: Optional[str] = None, params: Optional[Union[tuple, dict]] = None) -> int:
        """
        update database
        :param table_name: table name
        :param updates: update dict
        :param where: WHERE condition
        :param params: parameters
        :return: rows
        """
        if not updates:
            logger.warning("Update called with empty updates dict")
            return 0
        # construct SET clause
        set_clause = ", ".join([f"{k} = :set_{k}" for k in updates.keys()])
        # construct parameters
        update_params = {f"set_{k}": v for k, v in updates.items()}
        # merge condition parameters
        if params:
            if isinstance(params, dict):
                update_params.update(params)
            else:
                # replace tuple with parameters
                where_params = {f"param_{i}": p for i, p in enumerate(params)}
                update_params.update(where_params)
                where = _convert_positional_placeholders(where)
        # construct query condition
        query = f"UPDATE {table_name} SET {set_clause}"
        if where:
            query += f" WHERE {where}"
        # update database
        try:
            cursor = self.execute(query, update_params, commit=not self.in_transaction)
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Update failed: {str(e)}")
            raise

    def delete(self, table_name: str, where: Optional[str] = None, params: Optional[Union[tuple, dict]] = None) -> int:
        """
        delete database
        :param table_name: tabel name
        :param where: WHERE condition
        :param params: parameters
        :return: rows
        """
        if not where:
            logger.warning("Delete called without WHERE clause")
            raise Exception("Delete requires WHERE clause for safety")
        # construct query condition
        query = f"DELETE FROM {table_name}"
        if where:
            query += f" WHERE {where}"
        # delete database
        try:
            cursor = self.execute(query, params, commit=not self.in_transaction)
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Delete failed: {str(e)}")
            raise


if __name__ == '__main__':
    pass