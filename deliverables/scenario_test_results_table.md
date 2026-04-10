| Scenario | Decision | Notes |
|----------|----------|-------|
| happy_path | APPROVE | Passed pipeline |
| high_risk | INVALID | Failed validation: High fraud score |
| medium_risk | REJECT | Passed pipeline |
| fraud_case | APPROVE | Passed pipeline |
| injection_case | INVALID | Failed validation: Prompt injection pattern detected. |
| off_domain_case | INVALID | Failed validation: Unexpected field(s): ['query']. |
