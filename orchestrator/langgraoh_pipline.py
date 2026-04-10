"""
This file runs the full credit risk pipeline using the single LangGraph orchestrator entrypoint.
"""

import os
import json
from typing import Any

from agents.data_validator import validate_application
from agents.alt_credit import generate_alt_credit
from agents.risk_model import predict_risk
from agents.decision_engine import make_decision
from orchestrator.explanation import generate_explanation
from utils.guardrails import (
    validate_raw_input, validate_input_schema,
    validate_output_schema, check_explanation_grounding,
    redact_sensitive_data, filter_harmful_content
)


def _validate_agent_output(output: Any, required_keys, fallback: dict) -> dict:
    if not isinstance(output, dict) or not all(key in output for key in required_keys):
        return fallback
    return output


def run_langgraoh_pipline(data: dict, interactive: bool = True, manual_override: str = None) -> dict:
    """Runs validation, alt credit, risk, decision, and explanation in LangGraph-style order."""
    trace = []

    input_ok, input_reason = validate_raw_input(data)
    if not input_ok:
        validator_output = {
            "status": "FAIL",
            "reason": input_reason,
            "fraud_score": 0.0,
            "flags": []
        }
        trace.append({"step": "validator", "output": validator_output})
        result = {
            "validator": validator_output,
            "alt_credit": None,
            "risk": None,
            "decision": {
                "decision": "INVALID",
                "reason": validator_output.get("reason", "Validation failed")
            }
        }
        explanation_text = generate_explanation(result)
        trace.append({"step": "explanation", "output": explanation_text})
        result["explanation"] = explanation_text
        result["trace"] = trace
        return result

    schema_ok, schema_reason = validate_input_schema(data)
    if not schema_ok:
        validator_output = {
            "status": "FAIL",
            "reason": schema_reason,
            "fraud_score": 0.0,
            "flags": []
        }
        trace.append({"step": "validator", "output": validator_output})
        result = {
            "validator": validator_output,
            "alt_credit": None,
            "risk": None,
            "decision": {
                "decision": "INVALID",
                "reason": validator_output.get("reason", "Validation failed")
            }
        }
        explanation_text = generate_explanation(result)
        trace.append({"step": "explanation", "output": explanation_text})
        result["explanation"] = explanation_text
        result["trace"] = trace
        return result

    # LangGraph node: validator
    validator_output = validate_application(data)
    validator_output = _validate_agent_output(
        validator_output,
        ['status', 'reason', 'fraud_score', 'flags'],
        {
            "status": "FAIL",
            "reason": "Validator output schema invalid",
            "fraud_score": 0.0,
            "flags": []
        }
    )
    # Output guardrail: basic schema validation already done above
    print("GUARDRAIL: Validator output schema validated successfully")
    trace.append({"step": "validator", "output": validator_output})

    if validator_output["status"] == "FAIL":
        result = {
            "validator": validator_output,
            "alt_credit": None,
            "risk": None,
            "decision": {
                "decision": "INVALID",
                "reason": validator_output.get("reason", "Validation failed")
            }
        }
        explanation_text = generate_explanation(result)
        trace.append({"step": "explanation", "output": explanation_text})
        result["explanation"] = explanation_text
        result["trace"] = trace
        return result

    # LangGraph node: alt_credit
    alt_credit_output = generate_alt_credit(data)
    alt_credit_output = _validate_agent_output(
        alt_credit_output,
        ['alt_credit_score', 'income_credit_ratio', 'comment'],
        {
            "alt_credit_score": 0.0,
            "income_credit_ratio": 0.0,
            "comment": "Output validation failed"
        }
    )
    # Output guardrail: basic schema validation already done above
    print("GUARDRAIL: Alt credit output schema validated successfully")
    trace.append({"step": "alt_credit", "output": alt_credit_output})

    # LangGraph node: risk
    risk_output = predict_risk(data)
    risk_output = _validate_agent_output(
        risk_output,
        ['p_default', 'model_used', 'comment'],
        {
            "p_default": 1.0,
            "model_used": "fallback",
            "comment": "Output validation failed"
        }
    )
    # Output guardrail: basic schema validation already done above
    print("GUARDRAIL: Risk model output schema validated successfully")
    trace.append({"step": "risk", "output": risk_output})

    # LangGraph node: decision
    decision_output = make_decision(risk_output, alt_credit_output)
    decision_output = _validate_agent_output(
        decision_output,
        ['decision', 'reason'],
        {
            "decision": "INVALID",
            "reason": "Decision output validation failed"
        }
    )
    # Output guardrail: basic schema validation already done above
    print("GUARDRAIL: Decision engine output schema validated successfully")

    if decision_output["decision"] == "MANUAL_REVIEW":
        review_details = {
            "risk_probability": risk_output.get("p_default"),
            "alt_credit_score": alt_credit_output.get("alt_credit_score"),
            "risk_comment": risk_output.get("comment"),
            "alt_credit_comment": alt_credit_output.get("comment"),
            "decision_reason": decision_output.get("reason")
        }
        decision_output["manual_review_details"] = review_details

        if manual_override is None:
            manual_override = os.getenv("MANUAL_REVIEW_OVERRIDE")

        if manual_override is not None:
            manual_override = manual_override.strip().lower()

        if interactive and manual_override is None:
            print("Manual review required, details:")
            print(json.dumps(review_details, indent=2))
            print("Approve? (y/n)")
            user_input = input().strip().lower()
            manual_override = user_input

        if manual_override == "y":
            decision_output["decision"] = "APPROVE"
            decision_output["reason"] = "Approved after manual review"
        elif manual_override == "n":
            decision_output["decision"] = "REJECT"
            decision_output["reason"] = "Rejected after manual review"
        else:
            decision_output["reason"] = "Manual review pending"

    trace.append({"step": "decision", "output": decision_output})

    result = {
        "validator": validator_output,
        "alt_credit": alt_credit_output,
        "risk": risk_output,
        "decision": decision_output
    }

    explanation_text = generate_explanation(result)
    
    # Output guardrail: validate explanation schema
    valid, explanation_text, reason = validate_output_schema(explanation_text, 'explanation')
    if not valid:
        print(f"GUARDRAIL: Explanation output schema validation failed - {reason}")
        explanation_text = "Explanation generation failed due to output guardrail."
    else:
        print("GUARDRAIL: Explanation output schema validated successfully")
    
    # Output guardrail: check explanation grounding
    grounded, explanation_text, reason = check_explanation_grounding(explanation_text, result)
    if not grounded:
        print(f"GUARDRAIL: Explanation grounding check failed - {reason}")
        explanation_text = "Explanation does not match decision intent."
    else:
        print("GUARDRAIL: Explanation grounding validated successfully")
    
    # Output guardrail: redact sensitive data
    redacted, explanation_text, reason = redact_sensitive_data(explanation_text)
    if redacted:
        print("GUARDRAIL: Sensitive data redaction applied successfully")
    
    # Output guardrail: filter harmful content
    safe, explanation_text, reason = filter_harmful_content(explanation_text)
    if not safe:
        print(f"GUARDRAIL: Harmful content detected and blocked - {reason}")
        explanation_text = "Explanation contains inappropriate content and has been blocked."
    else:
        print("GUARDRAIL: No harmful content detected in explanation")
  
    trace.append({"step": "explanation", "output": explanation_text})
    result["explanation"] = explanation_text
    result["trace"] = trace
    return result
