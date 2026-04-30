rule_base = [
    {"min": 0.80, "max": 1.01, "msg": "High confidence prediction"},
    {"min": 0.50, "max": 0.80, "msg": "Medium confidence prediction"},
    {"min": 0.00, "max": 0.50, "msg": "Low confidence prediction"},
]

def apply_rules(conf):
    for r in rule_base:
        if r["min"] <= conf < r["max"]:
            return r["msg"]
    return "No rule matched"
