"""Trace root causes of each sample mismatch."""
import sys, os
sys.path.insert(0, '.')
os.environ['MOCK_LLM'] = 'true'

from buywise.data.loader import DatasetBundle, load_requests
bundle = DatasetBundle()
sample_reqs = load_requests(sample=True)

MISMATCHES = ['request_04', 'request_05', 'request_06', 'request_11', 'request_17', 'request_21']

for rid in MISMATCHES:
    req = next(r for r in sample_reqs if r.request_id == rid)
    profile = bundle.get_profile(req.user_id)
    msgs = bundle.get_messages_for_user(req.user_id) + bundle.get_messages_for_request(req.request_id)
    imgs = [img for imgs_list in [bundle.get_images_for_event(e.event_id)
                                   for e in bundle.get_user_events(req.user_id) if e.amount is None]
            for img in imgs_list if img.request_id == req.request_id or img.user_id == req.user_id]

    print(f"\n{'='*60}")
    print(f"REQUEST: {rid} | user={req.user_id}")
    print(f"  requested={req.requested_amount} {profile.home_currency}")
    print(f"  balance={profile.current_available_balance}  min={profile.minimum_balance_to_keep}")
    print(f"  methods={profile.payment_methods_user_will_consider}")
    print(f"  messages={len(msgs)}")

    seen_msg_ids = set()
    for m in bundle.messages.get(req.user_id, []) + bundle.messages.get(req.request_id, []):
        if m.message_id not in seen_msg_ids:
            seen_msg_ids.add(m.message_id)
            print(f"    MSG {m.message_id} [{m.source_type}]: {m.message_text[:150]}")

    # Check blank-amount events for this user
    blank_evts = [e for e in bundle.get_user_events(req.user_id) if e.amount is None]
    if blank_evts:
        for e in blank_evts:
            print(f"    BLANK EVENT {e.event_id}: {e.event_type} {e.category}")
