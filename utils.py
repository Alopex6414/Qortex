#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json


def load_json(path):
    with open(path, "r", encoding="utf-8") as fr:
        return json.load(fr)

if __name__ == '__main__':
    pass