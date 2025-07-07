---
title: "Justice Inequality Simulator — A Generative & Probabilistic Approach to Judicial Bias"
date: 2025-07-06
author: "Nicolas Cozzarin (@ncozzarin)"
---

[Share on Twitter](#) | [Share on LinkedIn](#)

## HFU Deep Generative Models — Project Report

### 1. Motivation

Judicial outcomes shape lives, yet historical data show that race, gender, or socioeconomic status sometimes correlate with harsher sentences.  
**Goal**: reveal and quantify those correlations through counterfactual generation — not to automate justice, but to explain bias.

---

### 2. Datasets

| Source                      | Rows     | Key Features                                |
|----------------------------|----------|---------------------------------------------|
| COMPAS recidivism          | ≈ 7,000  | age, priors, race, charge                   |
| U.S. Supreme Court (SCDB)  | ≈ 28,000 | issue, petitioner, lower-court direction    |
| Justice CSV (course data)  | ≈ 5,000  | free-text facts, first_party_winner         |

**Figure 1 – Class imbalance before resampling**  
![Class imbalance](assets/2025-07-06-class-imbalance.png)

