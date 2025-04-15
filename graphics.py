#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from typing import Any, List, Tuple
from database import SQLite3
from statistics import *


def graphics_plot_pie(dataset:List[Tuple[str, Any]]) -> None:
    # extract keys and values from dataset
    keys = list()
    values = list()
    for k, v in enumerate(dataset):
        keys.append(v[0])
        values.append(v[1])
    # plot pie graphics
    plt.pie(values, labels=keys)
    plt.show()

def graphics_plot_bar(dataset:List[Tuple[str, Any]]) -> None:
    # extract keys and values from dataset
    keys = list()
    values = list()
    for k, v in enumerate(dataset):
        keys.append(v[0])
        values.append(v[1])
    # plot bar graphics
    plt.bar(keys, values)
    plt.show()

if __name__ == '__main__':
    # create database
    sqlite = SQLite3("./database/qortex.db")
    # query dataset
    dataset = sqlite.query(table_name="internal", where="investigatedDate BETWEEN '2021-01-01' AND '2025-12-31'")
    # sort dataset
    sorted_dataset = statistics_sort_by_importance(dataset)
    graphics_plot_pie(sorted_dataset)
    graphics_plot_bar(sorted_dataset)
    pass
