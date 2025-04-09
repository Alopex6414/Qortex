#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from typing import List, Dict, Any


def load_json(path):
    with open(path, "r", encoding="utf-8") as fr:
        return json.load(fr)

def formalize(data:List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dataset = list()
    for item in data:
        if not "includedInBuild" in item.keys():
            item.update({"includedInBuild": ""})
        if not "testedInBuildDate" in item.keys():
            item.update({"testedInBuildDate": ""})
        if not "includedInReleaseDate" in item.keys():
            item.update({"includedInReleaseDate": ""})
        if not "whyWasTheFaultIntroduced" in item.keys():
            item.update({"whyWasTheFaultIntroduced": ""})
        if not "whyWasTheFaultNotFoundBeforeDelivery" in item.keys():
            item.update({"whyWasTheFaultNotFoundBeforeDelivery": ""})
        if not "resultingChanges" in item.keys():
            item.update({"resultingChanges": ""})
        if not "answerTextB2" in item.keys():
            item.update({"answerTextB2": ""})
        if not "faultCodeB2" in item.keys():
            item.update({"faultCodeB2": ""})
        if not "answerCodeFaultWillBeCorrected" in item.keys():
            item.update({"answerCodeFaultWillBeCorrected": ""})
        if not "fixedVersions" in item.keys():
            item.update({"fixedVersions": ""})
        if not "whatCanBeImprovedInTheNextSprint" in item.keys():
            item.update({"whatCanBeImprovedInTheNextSprint": ""})
        if not "customers" in item.keys():
            item.update({"customers": []})
        if not "teamName" in item.keys():
            item.update({"teamName": ""})
        if not "closedDate" in item.keys():
            item.update({"closedDate": ""})
        if not "assignedDate" in item.keys():
            item.update({"assignedDate": ""})
        if not "trOverdueDate" in item.keys():
            item.update({"trOverdueDate": ""})
        if not "assignee" in item.keys():
            item.update({"assignee": ""})
        if not "answerCategory" in item.keys():
            item.update({"answerCategory": ""})
        dataset.append(item)
    return dataset


if __name__ == '__main__':
    pass