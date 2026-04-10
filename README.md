# Credit Risk Assessment: Multi-Agent System

## Overview
This system automates credit risk assessment using a multi-agent LangGraph-style pipeline that validates loan applications, predicts default risk, and generates explainable decision outcomes. It is designed for credit underwriting in financial services, with explicit guardrails and a full-system runner for end-to-end execution.

## Architecture
INPUT → validate_raw_input → validate_input_schema → data_validator → alt_credit → risk_model → decision_engine → explanation → OUTPUT

- data_validator: Input validation, fraud detection, and application gating
- alt_credit: Alternative credit scoring using income-to-credit ratio
- risk_model: ML-based default probability prediction
- decision_engine: Business-rule decision making for APPROVE / REJECT / MANUAL_REVIEW
- explanation: LLM-generated explanation with output guardrails

## Guardrails

### Input Guardrails
- Raw input validation: required fields, object shape, maximum size
- Schema validation: numeric types, null rejection, unexpected fields
- Prompt injection detection: blocks malicious text patterns
- Fraud watchlist lookup: detects known high-risk applicant IDs

### Output Guardrails
- Agent output validation: ensures each node returns expected keys
- Explanation validation: non-empty, string output
- Grounding checks: explanation aligns with decision intent
- PII redaction: masks emails, phone numbers, SSNs
- Harmful content filtering: blocks unsafe language in explanations

## Project Structure
```
agents/
├── data_validator.py
├── alt_credit.py
├── risk_model.py
└── decision_engine.py
orchestrator/
├── langgraoh_pipline.py
└── explanation.py
utils/
└── guardrails.py
prompts/
└── explanation.md
tests/
└── test_guardrails.py
evaluation/
└── evaluate_alt_credit.py
deliverables/
├── scenario_test_results.json
├── scenario_test_results_table.md
└── run_full_system_output.json
docs/
└── final_design_document.md
```

## Setup & Installation
1. Clone the repository and switch to branch main
2. Install dependencies: `pip3 install -r requirements.txt`
3. Optionally set environment variables: `export MANUAL_REVIEW_OVERRIDE=y`
4. Ensure Ollama is installed and available for LLM explanation execution
5. Run the full project with: `python3 run_full_system.py`

## Running the Full Project
The preferred end-to-end runner is `run_full_system.py`. It trains the model, runs the main pipeline, executes scenario tests, evaluates the alt credit agent, and writes consolidated outputs.

### Execute the complete flow
```bash
python3 run_full_system.py
```

### Expected runner behavior
- trains the LightGBM risk model
- executes the main pipeline via `orchestrator/langgraoh_pipline.py`
- runs scenario tests and saves results to `deliverables/`
- evaluates the alt credit sub-agent
- writes final summary output to `deliverables/run_full_system_output.json`

## Running Tests
Run the guardrail test suite only:
```bash
python -m pytest tests/test_guardrails.py -v
```

Expected output: all tests pass, confirming input validation, output guardrail wiring, and end-to-end pipeline behavior.


## LLM Setup (Ollama)
Linux / Codespaces:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull llama3
```

## Outputs
- Model artifact: `models/risk_model.pkl`
- Scenario outputs: `deliverables/scenario_test_results.json`, `deliverables/scenario_test_results_table.md`
- Full run output: `deliverables/run_full_system_output.json`
- Output data includes decisions, explanations, and trace logs.

## Scenario Testing
The project includes scenario coverage for normal, edge, adversarial, invalid, and manual-review cases. Scenario outputs are stored under `deliverables/`.

## Sub-Agent Evaluation
The alt credit sub-agent is evaluated using sample application rows via `evaluation/evaluate_alt_credit.py`, producing precision, recall, and F1 metrics.

## Key Features
- multi-agent orchestration
- full-system runner via `run_full_system.py`
- robust input/output guardrails
- explainable LLM-powered explanations
- traceable pipeline execution


