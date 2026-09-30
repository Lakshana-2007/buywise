"""Drill into normalized events and evidence for specific requests."""
import json

TARGET_DETAILS = {
    "request_04": "Why does our simulator say safe (12693000) but sample says wait?",
    "request_06": "Why does our simulator say affordable_now but sample needs stop:event_476?",
    "request_11": "Why does our simulator say affordable_now but sample needs reduce_to:event_989?",
    "request_15": "Why amount_safe=0 but sample says 83.05?",
    "request_17": "Why 0 candidate strategies but sample says installments?",
    "request_21": "Why 0 candidates but sample says full_payment with spending changes?",
}

with open("code/evaluation/regression_diagnostics.jsonl", encoding="utf-8") as f:
    records = {json.loads(line)["request_id"]: json.loads(line) for line in f}

# Re-read because the dict comprehension consumed the iterator
with open("code/evaluation/regression_diagnostics.jsonl", encoding="utf-8") as f:
    records = {}
    for line in f:
        rec = json.loads(line)
        records[rec["request_id"]] = rec

for rid, question in TARGET_DETAILS.items():
    rec = records.get(rid)
    if not rec:
        print(f"{rid}: NOT FOUND\n")
        continue

    twin  = rec["financial_twin"]
    evts  = rec["normalized_events"]
    cands = rec["candidate_strategies"]

    print(f"\n{'='*68}")
    print(f"{rid}: {question}")
    print(f"  amount_safe_to_pay = {twin['amount_safe_to_pay']}")
    print(f"  earliest           = {twin['earliest_full_payment_date']}")
    print(f"  min_projected      = {twin['minimum_balance_for_full_payment']}")

    # Recurring income events (projected forward)
    rec_inc = [(e["event_id"], e["amount_home"], e["recurrence_day"],
                e["effective_date"], e.get("amount_from_message"), e.get("amount_from_image"))
               for e in evts if e["recurring"] and e["direction"] == "credit"]
    print(f"  RECURRING INCOME ({len(rec_inc)}):")
    for ei, amt, day, dt, frm_msg, frm_img in sorted(rec_inc, key=lambda x: (x[3], x[0])):
        print(f"    {ei}  day={day}  amt={amt}  last_date={dt}  "
              f"from_msg={frm_msg}  from_img={frm_img}")

    # Recurring expenses (projected forward)
    rec_exp = [(e["event_id"], e["amount_home"], e["recurrence_day"], e["effective_date"])
               for e in evts if e["recurring"] and e["direction"] == "debit"]
    print(f"  RECURRING EXPENSES ({len(rec_exp)}) [sample of latest per recurrence_day]:")
    # Show unique (category proxy = recurrence_day) latest entry
    latest_per_day: dict = {}
    for ei, amt, day, dt in rec_exp:
        if day not in latest_per_day or dt > latest_per_day[day][3]:
            latest_per_day[day] = (ei, amt, day, dt)
    for ei, amt, day, dt in sorted(latest_per_day.values(), key=lambda x: x[2]):
        print(f"    day={day}  amt={amt}  last_date={dt}")

    # Future non-recurring events (confirmed obligations)
    non_rec = [(e["event_id"], e["direction"], e["amount_home"], e["status"], e["effective_date"])
               for e in evts if not e["recurring"]]
    future_credits = [(ei, a, s, d) for ei, di, a, s, d in non_rec
                      if di == "credit" and s == "scheduled"]
    future_debits  = [(ei, a, s, d) for ei, di, a, s, d in non_rec
                      if di == "debit" and s in ("scheduled","pending")]
    print(f"  CONFIRMED FUTURE CREDITS (non-recurring, scheduled): {len(future_credits)}")
    for ei, a, s, d in sorted(future_credits, key=lambda x: x[3])[:5]:
        print(f"    {ei}  {d}  {s}  amt={a}")
    print(f"  CONFIRMED FUTURE DEBITS  (non-recurring, sched/pend): {len(future_debits)}")
    for ei, a, s, d in sorted(future_debits, key=lambda x: x[3])[:5]:
        print(f"    {ei}  {d}  {s}  amt={a}")

    print(f"  candidate_strategies: {len(cands)}")
    for c in cands:
        print(f"    [{c['method']}] valid={c['validity']} "
              f"min_bal={c['minimum_projected_balance']}  "
              f"violations={c.get('validation_violations',[])[:2]}")
