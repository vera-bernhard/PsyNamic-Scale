#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: analyse_data.py
Description: ...
Author: Vera Bernhard
Date: 27-05-2025
"""

# TODO
# Classification
# - Number of labels per task
# - Frequency of labels per task and per split
# - Total number of samples per task and per split
# - Check if split is stratified (label distribution consistency across splits)
# - (Label entropy per task (to measure imbalance))
# - Average/median input length (tokens or characters) per split
# - Rare class analysis (e.g., labels with <10 examples)
# - Plot input length distributions per split
# - Plot label distribution per task and per split

# NER
# - Total number of entities per label
# - Number of entities per label and per split
# - Average number of entities per sample
# - Average number of entities per sample and per split
# - Average length of entities (in tokens)
# - Number of abstracts without entities
# - Frequency of each BIO tag (B-*, I-*, O)
# - Plot distribution of entity span lengths
# - Rare entity type detection
# - (Label entropy over entity types)
# - Check if split is stratified by entity types
# - Check for malformed BIO sequences (e.g., I-* without preceding B-*)
