#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Counter
from typing import Any, Dict, List, Tuple

def statistics_sort_by_priority(dataset:List[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    # create list
    subset = list()
    for k, v in enumerate(dataset):
        if v.get('priority') is not None:
            subset.append(v.get('priority'))
    # statistics result
    results = Counter(subset).most_common()
    return results

def statistics_sort_by_importance(dataset:List[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    # create list
    subset = list()
    for k, v in enumerate(dataset):
        if v.get('importance') is not None:
            subset.append(v.get('importance'))
    # statistics result
    results = Counter(subset).most_common()
    return results

def statistics_sort_by_customers(dataset:List[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    # create list
    subset = list()
    for k, v in enumerate(dataset):
        if v.get('customers') is not None:
            for i, j in enumerate(v.get('customers')):
                subset.append(j)
    # statistics result
    results = Counter(subset).most_common()
    return results

def statistics_sort_by_team_name(dataset:List[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    # create list
    subset = list()
    for k, v in enumerate(dataset):
        if v.get('teamName') is not None:
            subset.append(v.get('teamName'))
    # statistics result
    results = Counter(subset).most_common()
    return results

def statistics_sort_by_status(dataset:List[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    # create list
    subset = list()
    for k, v in enumerate(dataset):
        if v.get('status') is not None:
            subset.append(v.get('status'))
    # statistics result
    results = Counter(subset).most_common()
    return results

def statistics_sort_by_fixed_versions(dataset:List[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    # create list
    subset = list()
    for k, v in enumerate(dataset):
        if v.get('fixedVersions') is not None:
            subset.append(v.get('fixedVersions'))
    # statistics result
    results = Counter(subset).most_common()
    return results

def statistics_sort_by_fault_code_b2(dataset:List[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    # create list
    subset = list()
    for k, v in enumerate(dataset):
        if v.get('faultCodeB2') is not None:
            subset.append(v.get('faultCodeB2'))
    # statistics result
    results = Counter(subset).most_common()
    return results


if __name__ == '__main__':
    pass