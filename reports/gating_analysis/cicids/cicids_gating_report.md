# MoE Gating Analysis: CICIDS

## Overview

This report analyzes how the Mixture-of-Experts (MoE) model distributes weights between the **Tabular Expert** and **Temporal Expert** for different attack types in the CICIDS dataset.

## Attack Type Statistics

| Attack Type | Count | Mean Tabular | Mean Temporal | Std Tabular | Std Temporal |
|------------|-------|--------------|---------------|-------------|--------------|
| BENIGN | 50,000 | 0.5220 | 0.4780 | 0.3431 | 0.3431 |

## Key Findings

### Top 3 Attacks Using Temporal Expert:

1. **BENIGN**: 47.80% temporal, 52.20% tabular (50,000 samples)

### Top 3 Attacks Using Tabular Expert:

1. **BENIGN**: 52.20% tabular, 47.80% temporal (50,000 samples)

## Hypothesis Validation


## Overall Summary

- **Mean Tabular Weight**: 52.20%
- **Mean Temporal Weight**: 47.80%
- **Expert Preference**: Balanced
- **Total Samples Analyzed**: 50,000
