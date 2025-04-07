#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import sqlite3
import redis
import mysql.connector
from mysql.connector import Error
from typing import Optional, List, Dict, Any, Union, Tuple
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


class MySQL:
    def __init__(self, host: str, user: str, password: str, database: str, port: int = 3306):
        """
        database init
        :param host: database host
        :param user: database user
        :param password: database password
        :param database: database name
        :param port: port number, default: 3306
        """
        self.config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database,
            'port': port
        }
        self.connection: Optional[mysql.connector.MySQLConnection] = None

    def __enter__(self):
        """enter"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """exit"""
        self.close()

    def connect(self) -> bool:
        """establish database connection"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            logger.info("Connected to database successfully")
            return True
        except Error as e:
            logger.error(f"Connected to database error: {e}")
            return False

    def close(self) -> None:
        """close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database already been closed")

    def execute_query(self, query: str, params: tuple = None, dictionary: bool = True) -> List[Dict[str, Any]]:
        """
        execute query
        :param query: SQL query
        :param params: query parameters
        :param dictionary: whether return dictionary or not
        :return: result
        """
        results = []
        try:
            with self.connection.cursor(dictionary=dictionary) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
        except Error as e:
            logger.error(f"Query failed: {e}")
        return results

    def execute_command(self, query: str, params: tuple = None) -> int:
        """
        execute command
        :param query: SQL query
        :param params: query parameters
        :return: result
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                self.connection.commit()
                return cursor.rowcount
        except Error as e:
            self.connection.rollback()
            logger.error(f"Execute error: {e}")
            return 0

    def batch_execute(self, query: str, params_list: List[tuple]) -> int:
        """
        batch execute command
        :param query: SQL query
        :param params_list: query parameter list
        :return: result
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.executemany(query, params_list)
                self.connection.commit()
                return cursor.rowcount
        except Error as e:
            self.connection.rollback()
            logger.error(f"Batch execute error: {e}")
            return 0

    def create_users_table(self) -> None:
        """create users table"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            age INT,
            email VARCHAR(100)
        )
        """
        self.execute_command(create_table_query)
        logger.info("Users table created successfully")

    def insert_user(self, name: str, age: int, email: str) -> int:
        """insert user"""
        insert_query = """
        INSERT INTO users (name, age, email)
        VALUES (%s, %s, %s)
        """
        rowcount = self.execute_command(insert_query, (name, age, email))
        if rowcount > 0:
            logger.info(f"User {name} insert successfully")
        return rowcount

    def batch_insert_users(self, users: List[tuple]) -> int:
        """batch insert users"""
        insert_query = """
        INSERT INTO users (name, age, email)
        VALUES (%s, %s, %s)
        """
        rowcount = self.batch_execute(insert_query, users)
        if rowcount > 0:
            logger.info(f"Batch insert users successfully，total {rowcount} records inserted")
        return rowcount

    def get_all_users(self) -> List[Dict[str, Any]]:
        """get all users"""
        query = "SELECT * FROM users"
        results = self.execute_query(query)
        logger.info(f"find {len(results)} users records")
        return results

    def update_user_age(self, user_id: int, new_age: int) -> int:
        """update user age"""
        update_query = "UPDATE users SET age = %s WHERE id = %s"
        rowcount = self.execute_command(update_query, (new_age, user_id))
        if rowcount > 0:
            logger.info(f"user {user_id} age updated successfully")
        return rowcount

    def delete_user(self, user_id: int) -> int:
        """delete user"""
        delete_query = "DELETE FROM users WHERE id = %s"
        rowcount = self.execute_command(delete_query, (user_id,))
        if rowcount > 0:
            logger.info(f"user {user_id} deleted successfully")
        return rowcount


class RedisClient:
    """Redis Client"""

    _pool = None  # 连接池实例

    def __init__(self, host: str = 'localhost', port: int = 6379,
                 password: str = None, db: int = 0,
                 max_connections: int = 20):
        """
        init Redid
        :param host: Redis host address
        :param port: Redis port
        :param password: authentication password
        :param db: database number
        :param max_connections: maximum connections
        """
        if not RedisClient._pool:
            RedisClient._pool = redis.ConnectionPool(
                host=host,
                port=port,
                password=password,
                db=db,
                max_connections=max_connections,
                decode_responses=True
            )
        self._client = redis.Redis(connection_pool=RedisClient._pool)

    def __enter__(self):
        """enter"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """exit"""
        self.close()

    def close(self):
        """close"""
        self._client.close()

    def set(self, key: str, value: Any, ex: int = None) -> bool:
        """store set"""
        try:
            return self._client.set(key, value, ex=ex)
        except redis.RedisError as e:
            logger.error(f"Redis set error: {str(e)}")
            raise

    def get(self, key: str) -> Optional[str]:
        """get key"""
        try:
            return self._client.get(key)
        except redis.RedisError as e:
            logger.error(f"Redis get error: {str(e)}")
            return None

    def delete(self, *keys: str) -> int:
        """delete key"""
        try:
            return self._client.delete(*keys)
        except redis.RedisError as e:
            logger.error(f"Redis delete error: {str(e)}")
            return 0

    def hset(self, name: str, mapping: Dict[str, Any]) -> int:
        """hash set"""
        try:
            return self._client.hset(name, mapping=mapping)
        except redis.RedisError as e:
            logger.error(f"Redis hset error: {str(e)}")
            return 0

    def hgetall(self, name: str) -> Dict[str, str]:
        """hash get all"""
        try:
            return self._client.hgetall(name)
        except redis.RedisError as e:
            logger.error(f"Redis hgetall error: {str(e)}")
            return {}

    def expire(self, key: str, seconds: int) -> bool:
        """set expire"""
        try:
            return self._client.expire(key, seconds)
        except redis.RedisError as e:
            logger.error(f"Redis expire error: {str(e)}")
            return False


if __name__ == '__main__':
    pass