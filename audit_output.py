"""Validate output.csv format and content."""
import csv, re
with open('output.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

valid_statuses = {'affordable_now','affordable_with_plan','affordable_later','not_affordable'}
valid_methods = {'full_payment','partial_payment','installments','wait','not_recommended'}
plan_re = re.compile(r'^\d{4}-\d{2}-\d{2}:[0-9]+(\.[0-9]+)?$')
change_re = re.compile(r'^(stop:[a-zA-Z0-9_]+|reduce_to:[a-zA-Z0-9_]+:[0-9]+(\.[0-9]+)?)$')
date_re = re.compile(r'^\d{4}-\d{2}-\d{2}$')

issues = []
for r in rows:
    rid = r['request_id']
    status = r['affordability_status']
    method = r['recommended_payment_method']
    plan = r['payment_plan']
    changes = r['spending_changes_needed']
    earliest = r['earliest_date_for_full_payment']
    expl = r['decision_explanation']

    if status not in valid_statuses:
        issues.append(rid + ': bad status ' + status)
    if method not in valid_methods:
        issues.append(rid + ': bad method ' + method)
    if plan != 'none':
        for part in plan.split('|'):
            if not plan_re.match(part.strip()):
                issues.append(rid + ': bad plan entry: ' + part)
    if changes != 'none':
        parts = changes.split('|')
        if len(parts) > 3:
            issues.append(rid + ': too many changes: ' + str(len(parts)))
        for p in parts:
            if not change_re.match(p.strip()):
                issues.append(rid + ': bad change: ' + p)
    if earliest and not date_re.match(earliest):
        issues.append(rid + ': bad date format: ' + earliest)
    if status == 'affordable_now' and not earliest:
        issues.append(rid + ': affordable_now missing earliest_date')
    if status == 'not_affordable' and earliest:
        issues.append(rid + ': not_affordable should have empty earliest, got: ' + earliest)
    if not expl.strip():
        issues.append(rid + ': empty explanation')

print("Output.csv validation issues:", len(issues))
for i in issues:
    print(" ", i)

# Distribution
from collections import Counter
print("\nStatus distribution:", dict(Counter(r['affordability_status'] for r in rows)))
print("Method distribution:", dict(Counter(r['recommended_payment_method'] for r in rows)))
print("Rows with spending changes:", sum(1 for r in rows if r['spending_changes_needed'] != 'none'))
print("Rows with partial payment:", sum(1 for r in rows if r['recommended_payment_method'] == 'partial_payment'))
print("Rows with earliest empty:", sum(1 for r in rows if not r['earliest_date_for_full_payment']))
print("\nAll OK" if not issues else "\n*** ISSUES FOUND ***")
