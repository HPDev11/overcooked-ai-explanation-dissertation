# Overcooked-AI Explanation Dissertation

A Python-based evaluation framework investigating whether concise natural-language explanations of an AI teammate's goals and actions can improve coordination in cooperative Overcooked-AI tasks.

The project combines a scripted AI teammate, goal/action-level explanation generation, an explanation-aware proxy human simulator, per-timestep logging, matched seeded A/B experiments, statistical aggregation and automated visualisation.

## Project Overview

Cooperative AI agents may perform effectively while still being difficult for a teammate to interpret. This project investigates whether exposing an AI teammate's current goal and action through concise natural-language explanations can improve cooperative coordination.

The system uses the Overcooked-AI benchmark and compares explanation-enabled and no-explanation conditions through repeated, seeded experiments. Evaluation considers both task performance and coordination measures such as reward, delivery timing, collisions, idle behaviour and partner blocking.

This repository contains the main implementation artefacts for my final-year dissertation:

**Explaining AI Actions in Cooperative Human-AI Tasks Using the Overcooked Benchmark**
## What I Built

- **Natural-language explanation generation** – maps the scripted AI teammate's current goal and action into concise action-level, goal-level or combined explanations.
- **Explanation-aware proxy human simulator** – models a cooperative partner that reacts to the current environment state and, when available, the AI teammate's communicated intent.
- **Automated A/B experiment runner** – executes matched explanation-enabled and no-explanation trials across configurable random seeds and experiment settings.
- **Per-timestep logging pipeline** – records actions, positions, held-object transitions, inferred events, rewards, explanations and coordination metrics to structured CSV files.
- **Results aggregation pipeline** – converts run-level JSON summaries into condition-level statistics including means, standard deviations and 95% confidence intervals.
- **Automated visualisation tooling** – generates report-ready plots from aggregated and run-level experimental results.
- **Interactive human-study prototype** – provides a participant-facing Overcooked-AI interface with live explanation display and structured session logging.

## Tech Stack

- **Language:** Python 3.10
- **AI / Simulation:** Overcooked-AI
- **Data:** CSV, JSON
- **Experimentation:** Seeded A/B evaluation, automated batch execution
- **Analysis:** Statistical aggregation, mean, standard deviation and 95% confidence intervals
- **Visualisation:** Automated result plotting
- **Development Environment:** Windows, Python virtual environment
## System Architecture

The project is structured as an end-to-end experimental pipeline:

```text
Scripted AI Teammate
        ↓
Goal / Action State
        ↓
Explanation Generator
        ↓
Explanation-Aware Proxy Partner
        ↓
Overcooked-AI Environment
        ↓
Per-Timestep CSV Logs + Run Summary JSON
        ↓
Results Aggregation
        ↓
Statistical Analysis + Visualisation
```
## Experimental Design

The main evaluation used a controlled matched A/B experiment on the `cramped_room` layout.

| Setting | Value |
| --- | --- |
| Primary layout | `cramped_room` |
| Episode horizon | 800 timesteps |
| Seeds | 30 |
| Conditions | `exp`, `noexp` |
| Total main runs | 60 |
| Explanation mode | Goal + action |
| Goal-change trigger | Enabled |
| Event trigger | Enabled |

Each random seed was evaluated under both conditions using the same environment configuration and scripted AI teammate.

- **`exp`** – explanation information was available to the proxy partner.
- **`noexp`** – no explanation information was provided through that channel.

Each run generated a detailed per-timestep CSV trace and a run-level JSON summary for later aggregation and analysis.
## Main files

| File | Purpose |
|---|---|
| `my_scripts/human_simulator_agent.py` | Implements the explanation-aware proxy human simulator and scripted teammate logic |
| `my_scripts/run_simulator_ab.py` | Runs repeated seeded A/B experiments for explanation-enabled and no-explanation conditions |
| `my_scripts/aggregate_results.py` | Aggregates per-run summary JSON files into condition-level results tables |
| `my_scripts/plot_results.py` | Generates plots from aggregated results |
| `my_scripts/human_study_runner.py` | Interactive participant-facing runner implemented but not deployed |

## Key Results

The main evaluation consisted of 30 matched seeded runs per condition on the `cramped_room` layout. Values below are reported as mean ± 95% confidence interval.

| Metric | Explanation (`exp`) | No Explanation (`noexp`) |
| --- | ---: | ---: |
| Total game reward | **382.00 ± 2.18** | 370.67 ± 3.63 |
| Soups delivered | **19.10 ± 0.11** | 18.53 ± 0.18 |
| Time to first delivery | **44.10 ± 0.59** | 46.77 ± 1.04 |
| Mean delivery interval | **40.40 ± 0.23** | 41.71 ± 0.26 |
| Blocked-partner events | **2.10 ± 0.57** | 9.17 ± 3.02 |
| Average idle rate | **0.638 ± 0.001** | 0.650 ± 0.002 |
| Collisions | **0.70 ± 0.25** | 1.90 ± 1.28 |
| Productive actions | **178.97 ± 0.90** | 173.17 ± 1.28 |

The explanation-enabled condition produced modest improvements in task throughput and stronger improvements in coordination quality. The clearest result was the reduction in blocked-partner events from 9.17 to 2.10 on average, alongside fewer collisions and slightly lower idle behaviour.

These findings apply to the controlled proxy-simulator evaluation on `cramped_room` and should not be interpreted as direct evidence of improved coordination with real human participants.
## Running the Project

### Prerequisites

The project was developed using **Python 3.10**. To run the simulation pipeline, the environment must have:

- Python 3.10
- Overcooked-AI installed and importable as `overcooked_ai_py`
- Matplotlib for result visualisation
- OpenPyXL (optional) for XLSX result export

The scripts should be run from the repository root.
Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```
### Run the Main A/B Experiment

The following command reproduces the configuration used for the main `cramped_room` evaluation:

```bash
python my_scripts/run_simulator_ab.py --layout cramped_room --horizon 800 --n_seeds 30 --seed_start 0 --mode both --goal_change_only --event_trigger --logs_dir logs/final_cramped_room
```

For each seed, the experiment runner automatically executes both the `noexp` and `exp` conditions.

Each run produces:

- A per-timestep CSV trace
- A run-level summary JSON file

### Aggregate Results

After the experiment completes, aggregate the run-level summaries using:

```bash
python my_scripts/aggregate_results.py --input_dirs logs/final_cramped_room --out_dir results/final_cramped_room --write_xlsx
```

This generates:

- `run_level_results.csv`
- `results_table.csv`
- `results_table.xlsx` if OpenPyXL is installed

### Generate Plots

Generate the result figures using:

```bash
python my_scripts/plot_results.py --results_csv results/final_cramped_room/results_table.csv --run_level_csv results/final_cramped_room/run_level_results.csv --layout cramped_room --mode both --goal_change_only True --event_trigger True --fig_dir figures
```

The plotting script generates condition-level comparison figures and selected per-run scatter plots.
## Limitations & Future Work

The main evaluation uses a rule-based proxy human simulator rather than real human participants. This enables reproducible, controlled experiments but means the results should not be interpreted as direct evidence of how real users would respond to AI explanations.

The strongest positive results were observed on the `cramped_room` layout. Secondary pilot experiments showed that the current hand-crafted policies do not generalise equally well across all Overcooked-AI environments. In particular, `asymmetric_advantages` produced different performance trends, while the current agents were unable to make meaningful progress on `forced_coordination`.

The explanation mechanism is also template-based and uses a fixed set of goal/action labels and heuristic explanation triggers.

Future work could include:

- Conducting a formal human-participant study using the implemented interactive runner
- Developing more general or learned partner policies that transfer across layouts
- Comparing goal-only, action-only and combined explanation strategies
- Investigating adaptive explanation timing based on uncertainty or coordination risk
- Evaluating the framework across additional cooperative environments and partner behaviours
## Notes

This repository excludes virtual environment files, cache files, unnecessary intermediate development outputs and personal files.
