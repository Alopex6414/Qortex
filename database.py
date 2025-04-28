#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import sqlite3
import redis
import mysql.connector
import subprocess
from datetime import datetime
from mysql.connector import Error, pooling
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
    def __init__(self, database: str, autocommit: bool = True, detect_types: int = sqlite3.PARSE_DECLTYPES, check_same_thread: bool = False):
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
    def __init__(self, host: str, user: str, password: str, database: str, pool_size: int = 5, autocommit: bool = False, enable_logging: bool = True):
        """
        database init
        :param pool_size: pool size
        :param enable_logging: enable logging
        """
        self.config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database,
            "pool_size": pool_size,
            "autocommit": autocommit
        }
        self.pool = None
        self.connection = None
        self.in_transaction = False
        self.logger = logging.getLogger('MySQL')
        self.enable_logging = enable_logging
        if enable_logging:
            self._setup_logging()

    def _setup_logging(self):
        """setup logging"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def connect(self) -> None:
        """创建连接池并获取连接"""
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name="mypool",
                pool_size=self.config["pool_size"],
                **self.config
            )
            self.connection = self.pool.get_connection()
            self.logger.info(f"成功连接到数据库 {self.config['database']}")
        except Error as e:
            self.logger.error(f"连接数据库失败: {e}")
            raise

    def close(self) -> None:
        """关闭所有连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            self.logger.info("数据库连接已关闭")

        if self.pool:
            self.pool.close_all()
            self.logger.info("连接池已关闭")

    def _execute(self,
                 sql: str,
                 params: Optional[Union[tuple, dict]] = None,
                 commit: bool = False) -> tuple:
        """
        执行SQL语句通用方法
        :return: (受影响行数, 结果集)
        """
        cursor = None
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(sql, params or ())

            if commit and not self.config["autocommit"]:
                self.connection.commit()

            if cursor.with_rows:
                return cursor.rowcount, cursor.fetchall()
            return cursor.rowcount, None

        except Error as e:
            self.logger.error(f"SQL执行失败: {e}\nSQL: {sql}\nParams: {params}")
            if commit and not self.config["autocommit"]:
                self.connection.rollback()
            raise
        finally:
            if cursor: cursor.close()

    # ---------- ORM风格操作 ----------
    def find_one(self,
                 table: str,
                 where: Optional[dict] = None,
                 fields: List[str] = ["*"]) -> Optional[dict]:
        """查询单条记录"""
        where_clause, params = self._parse_where(where)
        sql = f"SELECT {','.join(fields)} FROM {table} {where_clause} LIMIT 1"
        rowcount, result = self._execute(sql, params)
        return result[0] if result else None

    def insert(self,
               table: str,
               data: dict,
               commit: bool = True) -> int:
        """插入单条记录"""
        columns = ",".join(data.keys())
        placeholders = ",".join(["%s"] * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        rowcount, _ = self._execute(sql, tuple(data.values()), commit)
        return rowcount

    def update(self,
               table: str,
               data: dict,
               where: dict,
               commit: bool = True) -> int:
        """更新记录"""
        set_clause = ",".join([f"{k}=%s" for k in data.keys()])
        where_clause, where_params = self._parse_where(where)
        sql = f"UPDATE {table} SET {set_clause} {where_clause}"
        params = tuple(data.values()) + where_params
        rowcount, _ = self._execute(sql, params, commit)
        return rowcount

    def delete(self,
               table: str,
               where: dict,
               commit: bool = True) -> int:
        """删除记录"""
        where_clause, params = self._parse_where(where)
        sql = f"DELETE FROM {table} {where_clause}"
        rowcount, _ = self._execute(sql, params, commit)
        return rowcount

    # ---------- 高级功能 ----------
    def paginate(self,
                 table: str,
                 page: int = 1,
                 per_page: int = 10,
                 where: Optional[dict] = None,
                 order_by: str = "id DESC") -> dict:
        """分页查询"""
        offset = (page - 1) * per_page
        where_clause, params = self._parse_where(where)
        sql = f"SELECT * FROM {table} {where_clause} ORDER BY {order_by} LIMIT %s OFFSET %s"
        params += (per_page, offset)
        rowcount, result = self._execute(sql, params)
        return {
            "data": result,
            "page": page,
            "per_page": per_page,
            "total": self.count(table, where)
        }

    def bulk_insert(self,
                    table: str,
                    data: List[dict],
                    commit: bool = True) -> int:
        """批量插入数据"""
        if not data:
            return 0

        columns = ",".join(data[0].keys())
        placeholders = ",".join(["%s"] * len(data[0]))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        values = [tuple(item.values()) for item in data]

        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.executemany(sql, values)
            if commit:
                self.connection.commit()
            return cursor.rowcount
        except Error as e:
            self.logger.error(f"批量插入失败: {e}")
            if commit:
                self.connection.rollback()
            raise
        finally:
            if cursor: cursor.close()

    # ---------- 实用工具方法 ----------
    @staticmethod
    def _parse_where(where: Optional[dict]) -> tuple:
        """解析WHERE条件"""
        if not where:
            return "", ()

        conditions = []
        params = []
        for k, v in where.items():
            if isinstance(v, list):
                conditions.append(f"{k} IN ({','.join(['%s'] * len(v))})")
                params.extend(v)
            else:
                conditions.append(f"{k}=%s")
                params.append(v)

        return "WHERE " + " AND ".join(conditions), tuple(params)

    def backup(self,
               output_file: str,
               options: str = "--single-transaction") -> bool:
        """执行数据库备份"""
        try:
            cmd = f"mysqldump -h {self.config['host']} -u {self.config['user']} " \
                  f"-p{self.config['password']} {self.config['database']} {options} > {output_file}"
            subprocess.run(cmd, shell=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"备份失败: {e}")
            return False

    def restore(self, input_file: str) -> bool:
        """执行数据库恢复"""
        try:
            cmd = f"mysql -h {self.config['host']} -u {self.config['user']} " \
                  f"-p{self.config['password']} {self.config['database']} < {input_file}"
            subprocess.run(cmd, shell=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"恢复失败: {e}")
            return False

    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        sql = """
        SELECT COUNT(*) AS count 
        FROM information_schema.tables 
        WHERE table_schema = %s AND table_name = %s
        """
        params = (self.config['database'], table_name)
        rowcount, result = self._execute(sql, params)
        return result[0]['count'] > 0 if result else False

    # ---------- 事务管理增强 ----------
    def begin(self) -> None:
        """开启事务"""
        if not self.in_transaction:
            self.connection.start_transaction()
            self.in_transaction = True
            self.logger.info("事务已开启")

    def savepoint(self, name: str) -> None:
        """创建保存点"""
        self._execute(f"SAVEPOINT {name}")
        self.logger.info(f"保存点 {name} 已创建")

    def rollback_to(self, name: str) -> None:
        """回滚到保存点"""
        self._execute(f"ROLLBACK TO SAVEPOINT {name}")
        self.logger.info(f"已回滚到保存点 {name}")

    # ---------- 类型转换 ----------
    @staticmethod
    def convert_types(row: dict) -> dict:
        """自动转换数据类型"""
        converted = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                converted[key] = value.isoformat()
            elif isinstance(value, bytes):
                converted[key] = value.decode('utf-8')
            elif isinstance(value, float):
                converted[key] = float(value)
            else:
                converted[key] = value
        return converted

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.in_transaction:
            self.connection.rollback()
            self.logger.warning("发生异常，事务已回滚")
        self.close()


class RedisClient:
    """Redis Client"""

    _pool = None

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