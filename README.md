Agentic Credit Underwriting System
A multi-agent loan-underwriting pipeline built with LangGraph. Each application is validated, scored for affordability, scored for default risk by a LightGBM model, and routed to Approve, Reject or Manual Review. A locally hosted LLM then explains each decision in plain language.

Tech stack: Python · LangGraph · LightGBM · scikit-learn · pandas · Llama 2 (via Ollama)


How it works
flowchart LR

    A[Loan application] --> B[Data Validator]

    B -- FAIL --> X[Stop: rejected at validation]

    B -- PASS --> C[Alt Credit Agent]

    C --> D[Risk Model Agent]

    D --> E[Decision Engine]

    E -- Approve / Reject --> F[Final decision]

    E -- Manual Review --> H[Analyst review]

    H --> F

    F --> G[LLM explanation]

Each agent has one job and passes its output to the next. Every step is recorded in an execution trace so any decision can be audited end to end.

Agent
What it does
Data Validator
Checks that required fields are present, blocks prompt-injection text in application fields, and screens the applicant against a fraud watchlist. A failed check stops the pipeline before any scoring.
Alt Credit Agent
Computes an income-to-credit affordability ratio and turns it into an alternative credit score from 0 to 1.
Risk Model Agent
Uses a trained LightGBM model to predict the applicant's probability of default.
Decision Engine
Applies business rules to the default probability and the alternative credit score.
LLM Explanation
Llama 2, prompted with few-shot examples, writes a plain-language reason for the decision. It only explains the decision and never changes it.

Decision rules
Condition
Decision
Default probability < 5% and alt credit score > 0.6
Approve
Default probability > 20%
Reject
Anything in between
Manual Review (the pipeline pauses for an analyst to approve or reject)



Risk model
Data: Home Credit Default Risk (application_train.csv)
Features: loan amount, income, annuity, age, employment length, external bureau scores (EXT_SOURCE_1–3), and an engineered income-to-credit ratio
Model: LightGBM classifier (150 trees, learning rate 0.05, max depth 6), trained on an 80/20 train/test split
Evaluation: accuracy and ROC-AUC on the held-out test set, printed when load_data.py runs


Scenario testing
The full pipeline was run against five representative applicants:

Scenario
Outcome
Why
Happy path
Approve
Low risk, strong affordability
High risk
Blocked at validation
Fraud watchlist score above threshold
Medium risk
Manual Review
Default risk between thresholds
Fraud case
Manual Review
Passed validation, flagged as borderline
Prompt injection
Blocked at validation
Injection text found in an input field


Full results are in deliverables/scenario_test_results.json.


Getting started
git clone https://github.com/anmolisb/CR.git

cd CR

python -m venv venv

source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

Add the data. Download application_train.csv from Kaggle and place it in data/.
Train the model. This saves models/risk_model.pkl and prints accuracy and AUC.

python load_data.py

Run the pipeline on a sample application.

python main.py

Run the scenario tests. Results are written to deliverables/.

python run_scenarios.py

(Optional) Generate LLM explanations. Install Ollama and run ollama pull llama2 first.


Project structure
CR/

├── agents/            # Validator, alt credit, risk model and decision engine agents

├── orchestrator/      # Sequential pipeline, LangGraph graph and LLM explanation layer

├── tools/             # Fraud watchlist lookup and alt credit scoring helpers

├── prompts/           # Few-shot prompt for decision explanations

├── models/            # Trained LightGBM model

├── evaluation/        # Alt credit agent evaluation script

├── deliverables/      # Scenario test results

├── docs/              # Design document, run guide and sample trace

├── data/              # Sample input (place the Kaggle dataset here)

├── load_data.py       # Data preparation and model training

├── main.py            # Runs the full pipeline on a sample application

└── run_scenarios.py   # Runs the scenario test suite


Limitations and next steps
Fraud watchlist: It is a small in-memory sample. In production it would connect to a real fraud database.
Injection checks: They match a short keyword list. A classifier-based input guard would be more robust.
Decision thresholds: They are fixed in code. Moving them into configs/rules.yaml would make them easier to tune.
Model tuning: The model uses a small feature set and light tuning. Adding more application and bureau features, plus probability calibration, would improve risk estimates.
