"""Full sample accuracy audit — produces the regression table."""
import sys, os, csv
sys.path.insert(0, '.')
os.environ['MOCK_LLM'] = 'true'

from buywise.data.loader import DatasetBundle, load_requests, load_sample_outputs
from buywise.data.joins import build_request_context
from buywise.graph.state import FinancialState
from buywise.graph.workflow import build_pipeline
from buywise.ai.provider import MockProvider

bundle = DatasetBundle()
samples = load_sample_outputs()
sample_reqs = load_requests(sample=True)
pipeline = build_pipeline(MockProvider())

results = []
for req in sample_reqs:
    exp = samples[req.request_id]
    ctx = build_request_context(req, bundle)
    state = FinancialState(context=ctx)
    state = pipeline.run(state)
    row = state.output_row

    got_status = row.affordability_status
    got_method = row.recommended_payment_method
    got_safe   = round(float(row.amount_safe_to_pay), 2)
    got_earliest = row.earliest_date_for_full_payment
    got_changes = row.spending_changes_needed
    got_plan = row.payment_plan

    exp_status   = exp['affordability_status']
    exp_method   = exp['recommended_payment_method']
    exp_safe     = round(float(exp['amount_safe_to_pay']), 2) if exp['amount_safe_to_pay'] else 0
    exp_earliest = exp['earliest_date_for_full_payment']
    exp_changes  = exp['spending_changes_needed']
    exp_plan     = exp['payment_plan']

    s_match = (got_status   == exp_status)
    m_match = (got_method   == exp_method)
    a_match = (abs(got_safe - exp_safe) < 1.0)
    e_match = (got_earliest == exp_earliest)
    c_match = (got_changes  == exp_changes)

    results.append({
        'rid': req.request_id,
        'exp_status': exp_status, 'got_status': got_status, 'S': 'Y' if s_match else 'N',
        'exp_method': exp_method, 'got_method': got_method, 'M': 'Y' if m_match else 'N',
        'exp_safe': exp_safe, 'got_safe': got_safe, 'A': 'Y' if a_match else 'N',
        'exp_earliest': exp_earliest, 'got_earliest': got_earliest, 'E': 'Y' if e_match else 'N',
        'exp_changes': exp_changes, 'got_changes': got_changes, 'C': 'Y' if c_match else 'N',
    })

# Print table
print("\n=== SAMPLE ACCURACY REPORT (MOCK MODE — no LLM calls) ===\n")
print(f"{'RID':<12} {'ExpStat':<22} {'GotStat':<22} S  {'ExpMethod':<16} {'GotMethod':<16} M  {'ExpSafe':>14} {'GotSafe':>14} A  E  C")
print("-"*150)

s_tot=m_tot=a_tot=e_tot=c_tot=0
mismatches = []
for r in results:
    s_tot += r['S']=='Y'; m_tot += r['M']=='Y'; a_tot += r['A']=='Y'; e_tot += r['E']=='Y'; c_tot += r['C']=='Y'
    line = (f"{r['rid']:<12} {r['exp_status']:<22} {r['got_status']:<22} {r['S']}  "
            f"{r['exp_method']:<16} {r['got_method']:<16} {r['M']}  "
            f"{r['exp_safe']:>14.2f} {r['got_safe']:>14.2f} {r['A']}  {r['E']}  {r['C']}")
    print(line)
    if 'N' in [r['S'], r['M']]:
        mismatches.append(r)

n = len(results)
print(f"\nTotals: Status={s_tot}/{n}  Method={m_tot}/{n}  Amount(±1)={a_tot}/{n}  Earliest={e_tot}/{n}  Changes={c_tot}/{n}")

print("\n=== MISMATCH ROOT CAUSES ===")
for r in mismatches:
    print(f"\n{r['rid']}:")
    if r['S'] == 'N':
        print(f"  STATUS: expected={r['exp_status']} got={r['got_status']}")
    if r['M'] == 'N':
        print(f"  METHOD: expected={r['exp_method']} got={r['got_method']}")
    if r['C'] == 'N':
        print(f"  CHANGES: expected={r['exp_changes']} got={r['got_changes']}")
