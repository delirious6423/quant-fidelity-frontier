"""Queue ranking and discovery rules: pure checks, no network."""
import datetime as dt, json, os, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE, "pipeline"))
import qcl_rank as R
T = dt.date(2026, 10, 20)
m = lambda q, u, rel: {"quants": q, "quants_1k": u, "released": rel}
cfg = json.load(open(os.path.join(CODE, "discovery.json")))
assert R.HALF_LIFE == cfg["ranking"]["half_life_days"] and R.USED_W == cfg["ranking"]["used_weight"]
# same demand: newer ranks first; same age: more used quants ranks first
assert R.score(m(60, 5, "2026-08-20"), T)[0] > R.score(m(60, 5, "2026-04-20"), T)[0]
assert R.score(m(60, 20, "2026-08-20"), T)[0] > R.score(m(60, 5, "2026-08-20"), T)[0]
# half-life: a model one half-life older with the same demand scores exactly half (both past the early-boost window)
a = R.score(m(60, 5, "2026-07-20"), T)[0]; b = R.score(m(60, 5, (T - dt.timedelta(days=(T - dt.date(2026, 7, 20)).days + R.HALF_LIFE)).isoformat()), T)[0]
assert abs(b / a - 0.5) < 1e-9, (a, b)
# early boost is capped, and a 3-day-old model with fewer quants can still lead
assert R.score(m(10, 0, "2026-10-19"), T)[0] <= 10 * R.EARLY_MAX
order = R.rank([{"model": "old", **m(80, 30, "2026-05-01")}, {"model": "new", **m(60, 3, "2026-10-17")}, {"model": "bare"}], T)
assert [x["model"] for x in order] == ["new", "old", "bare"], order
wl = json.load(open(os.path.join(CODE, "watchlist.json")))
assert wl["min_hours_between_ingests"] == 0
print("rank ok")
