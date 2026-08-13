# Overcooked-AI Explanation Dissertation

A Python-based evaluation framework investigating whether concise natural-language explanations of an AI teammate's goals and actions can improve coordination in cooperative Overcooked-AI tasks.

The project combines a scripted AI teammate, goal/action-level explanation generation, an explanation-aware proxy human simulator, per-timestep logging, matched seeded A/B experiments, statistical aggregation and automated visualisation.

## Project Overview

Cooperative AI agents may perform effectively while still being difficult for a teammate to interpret. This project investigates whether exposing an AI teammate's current goal and action through concise natural-language explanations can improve cooperative coordination.

The system uses the Overcooked-AI benchmark and compares explanation-enabled and no-explanation conditions through repeated, seeded experiments. Evaluation considers both task performance and coordination measures such as reward, delivery timing, collisions, idle behaviour and partner blocking.

This repository contains the main implementation artefacts for my final-year dissertation:

**Explaining AI Actions in Cooperative Human-AI Tasks Using the Overcooked Benchmark**

## Main files

| File | Purpose |
|---|---|
| `my_scripts/human_simulator_agent.py` | Implements the explanation-aware proxy human simulator and scripted teammate logic |
| `my_scripts/run_simulator_ab.py` | Runs repeated seeded A/B experiments for explanation-enabled and no-explanation conditions |
| `my_scripts/aggregate_results.py` | Aggregates per-run summary JSON files into condition-level results tables |
| `my_scripts/plot_results.py` | Generates plots from aggregated results |
| `my_scripts/human_study_runner.py` | Interactive participant-facing runner implemented but not deployed |

## Main evaluation

The final evaluation compares:

- `noexp`: no explanation text provided to the proxy human simulator
- `exp`: explanation text provided to the proxy human simulator

The primary layout is `cramped_room`.

## Results

The `results/` folder contains the main evaluation and secondary pilot outputs used for the dissertation tables and figures.

## Notes

This repository excludes virtual environment files, cache files, unnecessary intermediate development outputs and personal files.
