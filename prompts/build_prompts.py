# -*- coding: utf-8 -*-
"""
Filename: gpt4_classification_template.py
Description: Template for GPT-4 classification tasks.
Author: Vera Bernhard
"""
# Prompts based on Chen et al. 2025

import json
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


system_role_class = "You are a helpful medical expert who is helping to classify medical abstracts."
user_prompt_class = '''***TASK***

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


# based on Hu et al. 2024





system_role_ner = "You are a helpful medical expert who is helping to extract named entities from medical abstracts."

user_prompt_ner = '''###Task

Your task is to generate an HTML version of an input text, marking up specific entities. The entities to be identified are: {ENTITIES}. Use HTML <span> tags to highlight these entities. Each <span> should have a class attribute indicating the type of entity.

###Entity Markup Guide
{ENTITY_MARKUP_GUIDE}

###Entity Definitions
{ENTITY_DEFINITIONS}

###Annotation Guidelines
{ANNOTATION_GUIDELINES}

INPUT: {TITLE_ABSTRACT}

OUTPUT: '''


def build_class_prompt(task: str, title_abstract: str):

    file_path = os.path.join(SCRIPT_DIR, 'classification_description.json')

    with open(file_path, 'r', encoding='utf-8') as f:
        task_descriptions = json.load(f)
    
    if task not in task_descriptions:
        raise ValueError(f"Task '{task}' not found in the task descriptions.")

    task_user_prompt = user_prompt_class.format(
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


def build_ner_prompt(title_abstract: str):
    file_path = os.path.join(SCRIPT_DIR, 'ner_description.json')

    with open(file_path, 'r', encoding='utf-8') as f:
        task_descriptions = json.load(f)

    entity_markup_guide = 'Use '
    annotation_guidelines = ''
    definitions = ''

    for entity, det in task_descriptions.items():
        entity_markup_guide += f'<span class="{entity.lower().replace(" ", "-")}"> to denote {entity}, '

        definitions += f'{entity} is defined as: {det['Definition']}\n\n'

        annotation_guidelines += f'{entity} should be annotated according to the following criteria:\n'
        for crit in det['Criteria']:
            annotation_guidelines += f'* {crit}\n'
        annotation_guidelines += '\n'   

    entity_markup_guide = entity_markup_guide[:-2] + '.'
    definitions = definitions[:-2]
    entities = ', '.join([entity for entity, _ in task_descriptions.items()])
    entities.rstrip(', ')
    annotation_guidelines = annotation_guidelines.rstrip('\n')

                    
    return user_prompt_ner.format(
        ENTITIES=entities,
        ENTITY_MARKUP_GUIDE=entity_markup_guide,
        ENTITY_DEFINITIONS=definitions,
        ANNOTATION_GUIDELINES=annotation_guidelines,
        TITLE_ABSTRACT=title_abstract
    )


