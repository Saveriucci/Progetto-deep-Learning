# Fine-Tuning Small Language Models for Structured Information Extraction

## Project 2 - Structuring: "Scrivimelo come dico io"

## Project Overview

This project corresponds to Project 2 (Structuring) within the theme *"Fine Tuning - SLM: Fare bene con poco"*.

The objective is to train Small Language Models (SLMs) to extract relevant information from natural language text and convert it into a rigid, machine-readable format, specifically JSON.

The project aims to demonstrate that model specialization through fine-tuning can outperform generalization on vertical tasks. In particular, it shows how compact models, when properly adapted, can approach the performance of much larger Large Language Models (LLMs) while significantly reducing computational costs.

## Dataset

The experiments are based on the RecipeNLG dataset, which contains approximately 2 million recipes.  
For computational feasibility, a subset of 1000 recipes was selected.

Each original recipe includes the following fields:
- id
- title
- directions
- source
- link
- NER

The reduced dataset, referred to as D1, contains only:
- id
- title
- directions

## Dataset Versions

### D1 - Structured Dataset
Subset of RecipeNLG containing only id, title and directions.

### D2 - Textual Dataset
Each recipe from D1 rewritten in English as a free-form textual recipe, simulating natural language input.

### D3 - Paired Dataset
A supervised fine-tuning dataset composed of pairs where:
- input: textual recipe from D2
- output: structured JSON with id, title and directions

## Models

The following Small Language Models were evaluated:
- Phi-3 (3.5B parameters)
- Mistral (7B parameters)
- Qwen 2.5 (1.5B parameters)

All models use the same system prompt and the same output constraints to ensure a fair comparison.

As a baseline Large Language Model, the following model was used:
- LLaMA 3.3 (70B parameters), inference only

## Experimental Pipeline

### Inference (Pre-Fine-Tuning)
- Input: first 50 recipes from D2
- Task: generate a structured JSON representation
- Purpose: establish baseline performance for each SLM

### Fine-Tuning
- Training dataset: D3
- Excluded data:
  - first 50 recipes (used in inference)
  - last 100 recipes (reserved for evaluation)
- Fine-tuning techniques:
  - LoRA for Phi-3 and Qwen 2.5
  - QLoRA for Mistral

### Evaluation
- Evaluation data:
  - first 50 recipes (baseline comparison)
  - last 100 recipes (generalization test)
- All evaluation inputs are taken from D2

### LLM Baseline Comparison
- Model: LLaMA 3.3 (70B)
- Mode: inference only
- Input: same recipes from D2 used in the evaluation phase
- Goal: compare fine-tuned SLMs against a large-scale LLM

## Evaluation Metrics

Model outputs are evaluated using the following metrics:
- Percentage of syntactically correct JSON
- Percentage of JSON outputs compliant with the predefined schema
- Percentage of semantically correct JSON

## Results Summary

All Small Language Models show consistent improvements after fine-tuning across all evaluation metrics.

A particularly notable result is observed for Qwen 2.5, which performs worst during the initial inference phase but achieves the largest relative improvement after fine-tuning. This highlights the effectiveness of vertical domain specialization, especially for very compact models.

Overall, the results confirm the main objective of the project: specialization beats generalization on vertical tasks.  
Fine-tuned SLMs can approach LLM-level performance on domain-specific structured extraction tasks while requiring significantly fewer computational resources.
