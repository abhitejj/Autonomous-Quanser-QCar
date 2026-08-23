"""
analyse_run.py

Reads a --log CSV from taxi_control.py and reports how well the car held its
lane, plus the steering trim that would centre it.

    python3 python/analyse_run.py run1.csv
"""
import argparse
import csv
import numpy as np

parser = argparse.ArgumentParser(prog='Run analysis')
parser.add_argument('csv')
parser.add_argument('--lane-width', type=float, default=0.27)
args = parser.parse_args()

rows = list(csv.DictReader(open(args.csv)))
column = lambda k: np.array([float(r[k]) for r in rows])
v = column('v')
crossTrack = column('crossTrack')
delta = column('delta')
state = np.array([r['state'] for r in rows])
nodes = np.array([r['nodes'] for r in rows])

driving = np.isin(state, ['NAV_TO_PICKUP', 'NAV_TO_DROPOFF',
                          'RETURN_TO_HUB']) & (v > 0.4)
sane = driving & (np.abs(crossTrack) < 0.5)
half = args.lane_width / 2.0

print('samples driving: %d (%.0f s)' % (sane.sum(), sane.sum()/200.0))
print()
print('Lane keeping   (+ = left of the lane centre)')
print('  mean         : %+.4f m' % crossTrack[sane].mean())
print('  median       : %+.4f m' % np.median(crossTrack[sane]))
print('  95th abs     : %.4f m' % np.percentile(np.abs(crossTrack[sane]), 95))
print('  out of lane  : %.1f%% of the time (|err| > %.3f m)'
      % (100*np.mean(np.abs(crossTrack[sane]) > half), half))
print('  time left of centre : %.0f%%' % (100*np.mean(crossTrack[sane] > 0)))
print()

# Straight sections: the controller should need no steering there, so whatever
# it does need is the disturbance to trim out.
straight = sane & (np.abs(delta) < np.radians(8))
if straight.sum() > 200:
    held = delta[straight].mean()
    print('Straight-line behaviour (%d samples)' % straight.sum())
    print('  mean steering held : %+.4f rad (%+.2f deg)'
          % (held, np.degrees(held)))
    print('  mean cross-track   : %+.4f m' % crossTrack[straight].mean())
    print()
    # The car holds `held` to sit at its steady offset, so the disturbance is
    # -held and the trim that cancels it is +(-(-held)) = held itself.
    # Sanity: riding left (+cross-track) must give a negative trim, which
    # steers right.
    # Stanley: delta = psi + atan(k*ect/v), and ect = -crossTrack.
    # Recover the heading error the controller is seeing in steady state.
    k, speed = 1.0, max(v[straight].mean(), 0.2)
    e = crossTrack[straight].mean()
    psi = held + np.arctan2(k*e, speed)
    print('  implied heading error: %+.2f deg' % np.degrees(psi))
    print()
    if abs(np.degrees(psi)) > 1.5:
        print('  The car points %.1f deg at its lane yet holds a %.3f m offset.'
              % (abs(np.degrees(psi)), abs(e)))
        print('  A car cannot crab that much, so the heading estimate is biased.')
        print('  Suggested   : --heading-trim %+.4f  (%+.2f deg)'
              % (psi, np.degrees(psi)))
    else:
        trim = held
        print('  Heading looks sound; the residual is a steering offset.')
        print('  Suggested   : --steer-trim %+.4f  (%+.2f deg)'
              % (trim, np.degrees(trim)))
    print('  (re-run and re-measure; cross-track should fall toward zero)')
print()
print('per leg:')
print('  %-10s %7s %9s %9s' % ('leg', 'n', 'mean', 'p95 abs'))
for leg in sorted(set(nodes[sane])):
    s = sane & (nodes == leg)
    if s.sum() < 200:
        continue
    print('  %-10s %7d %+9.4f %9.4f'
          % (leg, s.sum(), crossTrack[s].mean(),
             np.percentile(np.abs(crossTrack[s]), 95)))
