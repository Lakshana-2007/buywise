"""Read and pretty-print diagnostic records for specific request IDs."""
import json, sys

TARGET = {"request_02","request_03","request_04","request_06",
          "request_11","request_15","request_17","request_21"}

with open("code/evaluation/regression_diagnostics.jsonl", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

print(f"Total records: {len(records)}\n")

for rec in records:
    rid = rec["request_id"]
    if rid not in TARGET:
        continue

    twin   = rec.get("financial_twin", {})
    proof  = rec.get("safety_proof") or {}
    sel    = rec.get("selected_strategy") or {}
    facts  = rec.get("evidence_facts", [])
    evts   = rec.get("normalized_events", [])
    cands  = rec.get("candidate_strategies", [])

    # Identify recurring income / expense events
    rec_income  = [e for e in evts if e["recurring"] and e["direction"] == "credit"]
    rec_expense = [e for e in evts if e["recurring"] and e["direction"] == "debit"]
    future_non_rec = [e for e in evts if not e["recurring"] and e["direction"] in ("credit","debit")]

    print("=" * 68)
    print(f"REQUEST: {rid}  |  user={rec['user_id']}")
    print(f"  evidence_facts       : {len(facts)}")
    print(f"  normalized_events    : {len(evts)}  "
          f"(rec_inc={len(rec_income)}, rec_exp={len(rec_expense)}, "
          f"non_rec_future={len(future_non_rec)})")
    print(f"  financial_twin:")
    print(f"    amount_safe_to_pay : {twin.get('amount_safe_to_pay')}")
    print(f"    earliest_full_pay  : {twin.get('earliest_full_payment_date')}")
    print(f"    min_bal (proof)    : {twin.get('minimum_balance_for_full_payment')}")
    print(f"  candidate_strategies : {len(cands)}")
    for c in cands:
        print(f"    [{c.get('method')}] plan={c.get('payment_plan','')[:55]}  "
              f"valid={c.get('validity')}  min_bal={c.get('minimum_projected_balance')}")
    print(f"  selected_strategy    : {sel.get('method')}  "
          f"plan={sel.get('payment_plan','')[:55]}")
    print(f"    spending_changes   : {sel.get('spending_changes')}")
    print(f"    rank               : {sel.get('rank')}  reason={sel.get('rank_reason','')[:60]}")
    if proof:
        chk = proof.get("checks", {})
        print(f"  safety_proof:")
        print(f"    min_projected    : {proof.get('minimum_projected_balance')}")
        print(f"    required_min     : {proof.get('required_minimum_balance')}")
        print(f"    all_pass         : {proof.get('all_constraints_pass')}")
        print(f"    checks           : {chk}")
        print(f"    spending_changes : {proof.get('spending_changes')}")
    if facts:
        print(f"  evidence_facts detail:")
        for f_ in facts[:5]:
            print(f"    {f_['source_id']} [{f_['source_type']}] "
                  f"type={f_.get('evidence_type','')}  "  # not stored — source only
                  f"amt={f_['amount']} {f_['currency']}  "
                  f"confidence={f_['confidence']}  "
                  f"event={f_['related_event_id']}  "
                  f"injection={f_['contains_instruction']}")
    if rec.get("errors"):
        print(f"  errors: {rec['errors']}")
    if rec.get("warnings"):
        print(f"  warnings: {rec['warnings'][:3]}")
    print()
