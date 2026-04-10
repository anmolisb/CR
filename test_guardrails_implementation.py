#!/usr/bin/env python3
"""
Quick test to verify guardrails are implemented in the pipeline.
"""

from orchestrator.langgraoh_pipline import run_langgraoh_pipline

def test_guardrails():
    print("Testing guardrails implementation...")

    # Test data for happy path
    test_data = {
        'SK_ID_CURR': 100002,
        'AMT_CREDIT': 400000,
        'AMT_INCOME_TOTAL': 250000,
        'AMT_ANNUITY': 20000,
        'DAYS_BIRTH': -16000,
        'DAYS_EMPLOYED': -3000,
        'EXT_SOURCE_1': 0.5,
        'EXT_SOURCE_2': 0.5,
        'EXT_SOURCE_3': 0.5
    }

    print("\nRunning pipeline with test data...")
    result = run_langgraoh_pipline(test_data, interactive=False, manual_override='n')

    print("\nPipeline completed successfully!")
    print(f"Decision: {result['decision']['decision']}")
    print(f"Explanation: {result['explanation']}")

if __name__ == "__main__":
    test_guardrails()</content>
<parameter name="filePath">/workspaces/CR/test_guardrails_implementation.py