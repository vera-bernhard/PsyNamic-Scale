#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: helper.py
Description: ...
Author: Vera Bernhard
"""
import pandas as pd

def add_full_text_to_bioner_files(file: str):

    with open('data/all_studies_relevant.csv', 'r', encoding='utf-8') as f, open(file, 'r', encoding='utf-8') as f2:

        df_full = pd.read_csv(f)
        df_bioner = pd.read_csv(f2)
        df_bioner_columns = df_bioner.columns.tolist()
        
        # add 'text' from df_full to df_bioner, match on id
        df_bioner = df_bioner.merge(df_full[['id', 'text']], on='id', how='left')
        
        # save the updated df_bioner old columns and new 'text' column
        df_bioner = df_bioner[df_bioner_columns + ['text']]
        df_bioner.to_csv(file, index=False, encoding='utf-8')

def main():
    add_full_text_to_bioner_files('data/ner_bio/test.csv')
    add_full_text_to_bioner_files('data/ner_bio/train.csv')
    add_full_text_to_bioner_files('data/ner_bio/val.csv')

if __name__ == "__main__":
    main()
        