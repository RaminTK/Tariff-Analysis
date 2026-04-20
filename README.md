# Tariff Analysis under Uncertainty and Misinformation

## Overview

This repository implements a two-stage stochastic optimization framework for supply chain planning under tariff uncertainty and misinformation.

In many real-world settings, firms make operational decisions based not on realized policies, but on **signals** about potential policy changes. These signals may be incomplete, delayed, or misleading. This project models the resulting gap between **perceived tariff conditions** and **actual tariff realizations**, and quantifies its operational impact.

---

## Problem Description

We consider a cross-border supply chain in which:
- suppliers are located internationally,
- warehouses and demand zones are domestic,
- tariff policies affect procurement decisions.

The decision-maker observes a **tariff-related signal** and makes first-stage decisions accordingly. After the true tariff is realized, recourse actions are taken to satisfy demand.

The key modeling feature is the explicit distinction between:
- **perceived tariffs** (used in planning),
- **actual tariffs** (determining realized costs).

---

## Model Structure

The framework is formulated as a two-stage stochastic program:

### First Stage (Anticipatory Decisions)
- Procurement and import decisions
- Inventory positioning at warehouses
- Decisions are made under **perceived tariff signals**

### Second Stage (Recourse Decisions)
- Regular distribution
- Emergency shipments
- Unmet demand penalties
- Inventory carryover

These decisions are evaluated under **actual tariff scenarios**.

---

## Key Features

- Scenario-based stochastic optimization
- Explicit modeling of misinformation through signal–scenario mapping
- Separation of perception and realization in decision-making
- Gurobi-based exact optimization
- Reproducible data architecture using structured input files

---

## Repository Structure

```text
Tariff-Analysis/
│
├── scenario_generator.py            # Generates tariff and misinformation scenarios
├── gurobi_tariff_misinfo_model.py   # Optimization model (Gurobi)
├── dataset_template.xlsx            # Structured input data
├── results/                         # Computational outputs and experiment results
├── README.md
└── .gitignore
```
---

## Data and Scenario Design

The dataset is organized as a structured workbook with separate sheets for:
- suppliers, warehouses, and demand zones
- transportation and cost parameters
- tariff scenarios and prior probabilities
- signal states and perceived tariffs
- conditional probabilities linking signals to true scenarios

Misinformation is modeled through a **conditional probability matrix**, which captures how signals influence beliefs about actual tariff outcomes.

---

## Computational Experiments

The model is solved using Gurobi as a deterministic equivalent linear program.

Experiments evaluate:
- total system cost
- inventory levels
- emergency shipments
- unmet demand

Different scenarios are tested, including:
- accurate signals
- false-positive tariff expectations
- underestimated tariff signals
- policy reversals

---

## Key Insights

The results show that misinformation can significantly degrade supply chain performance:

- **False-positive signals** lead to overstocking and increased holding costs  
- **False-negative signals** result in shortages and costly emergency shipments  
- **Policy reversals** amplify inefficiencies by invalidating anticipatory decisions  

Overall, inaccurate expectations about tariffs can generate substantial operational inefficiencies.

---

## Requirements

- Python 3.x
- Gurobi Optimizer
- pandas
- numpy
- openpyxl

Install dependencies:

pip install -r requirements.txt

---

## Contribution

This repository provides:
- a reproducible implementation of a stochastic supply chain model under policy uncertainty,
- a structured approach to modeling misinformation,
- a case study aligned with real-world tariff dynamics.

---

## Author

Ramin TK
