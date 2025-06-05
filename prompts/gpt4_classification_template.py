# -*- coding: utf-8 -*-
"""
Filename: gpt4_classification_template.py
Description: Template for GPT-4 classification tasks.
Author: Vera Bernhard
"""
# Prompts based on Chen et al. 2025

import json
import os

system_role = "You are a helpful medical expert who is helping to classify medical abstracts."
user_prompt = '''***TASK***

{TASK_DESCRIPTION}

***INPUT***

The input is the title and abstract text.

***DOCUMENTATION***

There are {NUMBER_OF_TASKS} {TASK} options. The followings are the options and their definitions:

{TASK_OPTIONS}

***OUTPUT***

The output should be in a json format, with relevant value for each option: {VALUES}

Put value 1 if the option applies to the research paper, 0 if it does not apply.

Please note again that {IS_MULTIPLE} can be selected for each research paper.

Example output format:
{OUPUT_EXAMPLE}

    
INPUT: {TITLE_ABBSTRACT}


OUTPUT: '''


def build_prompt(task: str, title_abstract: str):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'classification_description.json')

    with open(file_path, 'r', encoding='utf-8') as f:
        task_descriptions = json.load(f)
    
    if task not in task_descriptions:
        raise ValueError(f"Task '{task}' not found in the task descriptions.")

    task_user_prompt = user_prompt.format(
        TASK_DESCRIPTION=task_descriptions[task]['Task_descripton'],
        NUMBER_OF_TASKS=len(task_descriptions[task]['Options']),
        TASK=task,
        TASK_OPTIONS='\n\n'.join(
            [f"{option}: {desc}" for option, desc in task_descriptions[task]['Options'].items()]),
        VALUES=', '.join(task_descriptions[task]['Options'].keys()),
        IS_MULTIPLE='multiple options' if task_descriptions[
            task]['Is_multilabel'] else 'a single option',
        OUPUT_EXAMPLE=json.dumps(
            {key: "" for key in task_descriptions[task]['Options'].keys()},
            indent=4
        ),
        TITLE_ABBSTRACT=title_abstract
    )
    return task_user_prompt
