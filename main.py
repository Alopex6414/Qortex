#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from database import SQLite3
from utils import load_json, clean_data

if __name__ == '__main__':
    # load json file
    data = load_json("./data/test.json")
    # clean json data
    dataset = clean_data(data)
    # create database
    sqlite = SQLite3("./database/qortex.db")
    sqlite.create_table("fst_customer", {
        "key": "TEXT PRIMARY KEY",
        "issuetype": "TEXT",
        "summary": "TEXT",
        "teamName": "TEXT",
        "created": "DATETIME",
        "reporter": "TEXT",
        "priority": "TEXT",
        "customers": "LIST",
        "labels": "LIST",
        "importance": "TEXT",
        "assignee": "TEXT",
        "updated": "DATETIME",
        "status": "TEXT",
        "issuelinks": "LIST",
        "answerCategory": "TEXT",
        "includedInBuild": "TEXT",
        "closedDate": "DATETIME",
        "trOverdueDate": "DATETIME",
        "investigatedDate": "DATETIME",
        "testedInBuildDate": "DATETIME",
        "includedInReleaseDate": "DATETIME",
        "whyWasTheFaultIntroduced": "TEXT",
        "whyWasTheFaultNotFoundBeforeDelivery": "TEXT",
        "whatCanBeImprovedInTheNextSprint": "TEXT",
        "assignedDate": "DATETIME",
        "resultingChanges": "TEXT",
        "answerTextB2": "TEXT",
        "faultCodeB2": "TEXT",
        "answerCodeFaultWillBeCorrected": "TEXT",
        "fixedVersions": "TEXT"
    })
    result = sqlite.query(table_name="fst_customer", where="")
    print(result)
    pass