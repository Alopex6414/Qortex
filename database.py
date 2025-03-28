#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sqlite3

class Database:
    def __init__(self, database, autocommit=True, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False):
        self.database = database
        self.autocommit = autocommit
        self.detect_types = detect_types
        self.check_same_thread = check_same_thread
        self.connection: sqlite3.Connection
        self.cursor: sqlite3.Cursor
        self.in_transaction = False

    def connect(self):
        conn = sqlite3.connect(self.database)



if __name__ == '__main__':
    pass