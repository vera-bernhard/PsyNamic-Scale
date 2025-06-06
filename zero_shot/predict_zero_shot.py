#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: analyse_data.py
Description: ...
Author: Vera Bernhard
"""

import os
from dotenv import load_dotenv
from datetime import datetime
from prompts.build_prompts import build_class_prompt, system_role_class, system_role_ner, build_ner_prompt
from openai import OpenAI
import pandas as pd
import json

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def gpt_prediction(prompt: str, model: str = "gpt-4o-mini", system_role: str = '') -> tuple[str, str]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    response_content = response.choices[0].message.content.strip()
    model_spec = response.model
    return response_content, model_spec


def make_class_predictions(task: str, model: str, outfile: str):
    task_lower = task.lower().replace(' ', '_')
    file = os.path.join(os.path.dirname(__file__), '..',
                        'data', task_lower, 'test.csv')
    label2int = get_label2int(task)

    if not os.path.exists(file):
        raise FileNotFoundError(
            f"The file {file} does not exist. Check the task name.")

    df = pd.read_csv(file)

    predictor_function = None
    if 'gpt' in model:
        predictor_function = gpt_prediction

    # Some cleaning up
    # df_pred = pd.read_csv('zero_shot/study_type_gpt-4o-mini_05-06-05_old.csv')
    # df['prompt'] = df['text'].apply(lambda text: build_prompt(task, text))
    # df['prediction_text'] = df_pred['prediction_text']
    # df['model'] = df_pred['model']

    # df['pred_labels'] = df['prediction_text'].apply(
    #     lambda x: parse_prediction(x, label2int))

    prompts = []
    predictions = []
    model_specs = []

    for _, row in df.iterrows():
        prompt = build_class_prompt(task, row['text'])
        prediction, model_spec = predictor_function(
            prompt, model=model, system_role=system_role_class)
        prompts.append(prompt)
        predictions.append(prediction)
        model_specs.append(model_spec)

    df['prompt'] = prompts
    df['prediction_text'] = predictions
    df['model'] = model_specs
    df['pred_labels'] = df['prediction_text'].apply(
        lambda x: parse_class_prediction(x, label2int))

    df_out = df[['id', 'text', 'prompt', 'prediction_text',
                 'model', 'labels',  'pred_labels']]
    df_out.to_csv(outfile, index=False, encoding='utf-8')


def make_ner_prediction(model: str, outfile: str):
    file = os.path.join(os.path.dirname(__file__), '..',
                        'data', 'ner_bio', 'test.csv')
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"The file {file} does not exist. Check the task name.")

    df = pd.read_csv(file)

    predictor_function = None
    if 'gpt' in model:
        predictor_function = gpt_prediction

    for i, row in df.iterrows():
        prompt = build_ner_prompt(row['text'])
        prediction, model_spec = predictor_function(
            prompt, model=model, system_role=system_role_ner)
        df.at[i, 'prompt'] = prompt
        df.at[i, 'prediction_text'] = prediction
        df.at[i, 'model'] = model_spec

    # df['pred_labels'] = df['prediction_text'].apply(
    #     lambda x: parse_prediction(x, label2int))

    # df_out = df[['id', 'text', 'prompt', 'prediction_text',
    #              'model', 'labels',  'pred_labels']]
    df_out = df[['id', 'text', 'prompt', 'prediction_text', 'model']]
    df_out.to_csv(outfile, index=False, encoding='utf-8')


def parse_class_prediction(pred: str, label2int: dict) -> str:
    pred = pred.replace('\\n', '\n').replace('""', '"')

    start = pred.find('{')
    end = pred.rfind('}') + 1
    json_str = pred[start:end]
    try:
        pred = json.loads(json_str)
    except json.JSONDecodeError:
        json_str = pred.replace(': ",', ': "0",')
        json_str = json_str.replace(': "\n', ': "0"\n')
        try:
            pred = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {json_str}")
            return ""

    # convert to string
    onehot_list = [0] * len(label2int)
    for label, value in pred.items():
        pos = label2int[label]
        # insert at position
        onehot_list[pos] = int(value)
    return str(onehot_list)


def parse_ner_prediction(pred: str) -> str:
    return ''


def get_label2int(task: str) -> dict:
    task_lower = task.lower().replace(' ', '_')
    file = os.path.join(os.path.dirname(__file__), '..',
                        'data', task_lower, 'meta.json')

    if not os.path.exists(file):
        raise FileNotFoundError(
            f"The file {file} does not exist. Check the task name.")

    with open(file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    int2label = meta.get('Int_to_label', {})
    label2int = {v: int(k) for k, v in int2label.items()}
    return label2int


def main():
    task = "Study Type"
    model = "gpt-4o-mini"
    date = datetime.today().strftime('%d-%m-%d')
    # outfile_class = f"zero_shot/{task.lower().replace(' ', '_')}_{model}_{date}.csv"
    # make_class_predictions(task, model, outfile_class)

    outfile_ner = f"zero_shot/ner_{model}_{date}.csv"
    make_ner_prediction(model, outfile_ner)


if __name__ == "__main__":
    main()
