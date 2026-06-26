"""
low_source_probe.py — what is the real low source? (DIAGNOSTIC, freeze-safe)
===========================================================================
Two questions decide the entire V5.30 lows architecture:

  Q1  Can we reconstruct the SETTLEMENT min (Iowa CLI calendar-day minimum)
      from wethr's OBSERVATION HISTORY? wethr observations.php with the mode
      param omitted returns history. If min(temperature) over a local calendar
      day matches the CLI min for that day, then wethr's obs history IS a usable
      low source (and an obs-min tracker becomes buildable).

  Q2  What does wethr's FORECAST low represent — today's calendar-day min
      (mostly already occurred), or the upcoming overnight low (a different,
      ~1-day-shifted quantity)? We compare today's forecast low to the
      min actually observed so far today.

Method, per city:
  - CLI min for the last 2 settled days   (ground truth, Iowa)
  - wethr obs-history calendar-day min for those same days  (compare → Q1)
  - today's forecast low + today's observed-min-so-far       (characterize → Q2)

Needs WETHR_API_KEY. Read-only, no Supabase, no model code. Freeze-safe.
"""

import os
from datetime import datetime, timedelta

import requests
import pytz

WETHR_API_KEY = os.environ.get('WETHR_API_KEY', '')
WETHR_HEADERS = {'Authorization': f'Bearer {WETHR_API_KEY}', 'Accept': 'application/json'}
HEADERS = {'User-Agent': 'low-source-probe/1.0', 'Accept': 'application/json'}

# tight-low leads + one desert + one cold city for contrast
CITIES = {
    'Miami': ('KMIA', 'America/New_York'),
    'New Orleans': ('KMSY', 'America/Chicago'),
    'Los Angeles': ('KLAX', 'America/Los_Angeles'),
    'Houston': ('KHOU', 'America/Chicago'),
    'Phoenix': ('KPHX', 'America/Phoenix'),
    'Chicago': ('KMDW', 'America/Chicago'),
}

_CLI = {}


def cli_min(station, date_str):
    year = date_str[:4]
    key = station + '_' + year
    if key not in _CLI:
        try:
            r = requests.get('https://mesonet.agron.iastate.edu/json/cli.py',
                             params={'station': station, 'year': year},
                             headers=HEADERS, timeout=20)
            r.raise_for_status()
            d = {}
            for e in r.json().get('results', []):
                if e.get('valid') and e.get('low') is not None:
                    try:
                        d[e['valid']] = float(e['low'])
                    except Exception:
                        pass
            _CLI[key] = d
        except Exception:
            _CLI[key] = {}
    return _CLI[key].get(date_str)


def fcst_low(station):
    today = datetime.utcnow().strftime('%Y-%m-%d')
    try:
        r = requests.get('https://wethr.net/api/v2/nws_forecasts.php',
                         params={'station_code': station, 'date': today, 'mode': 'latest'},
                         headers=WETHR_HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json().get('low')
    except Exception:
        pass
    return None


def obs_history(station):
    """observations.php with mode omitted = history. Requires start_time/end_time
    (UTC). Pull a 3.5-day window so we cover the last 2 settled days + today."""
    now = datetime.utcnow()
    start = (now - timedelta(days=3, hours=12)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    # try a few likely param spellings; return the first that returns a list
    param_variants = [
        {'station_code': station, 'start_time': start, 'end_time': end},
        {'station_code': station, 'start_time': start[:10], 'end_time': end[:10]},
    ]
    last = None
    for params in param_variants:
        try:
            r = requests.get('https://wethr.net/api/v2/observations.php',
                             params=params, headers=WETHR_HEADERS, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return data
                last = data
            else:
                last = {'__http__': r.status_code, '__body__': r.text[:200]}
        except Exception as e:
            last = {'__exc__': f'{type(e).__name__}: {str(e)[:150]}'}
    return last


def temp_of(rec):
    for k in ('temperature_f', 'temperature_display'):
        v = rec.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return None


def daily_mins_from_history(hist, tz_name):
    """Group history records by LOCAL calendar date → min temperature_f."""
    if not isinstance(hist, list):
        return None, 0, None
    tz = pytz.timezone(tz_name)
    by_date = {}
    for rec in hist:
        if not isinstance(rec, dict):
            continue
        t = temp_of(rec)
        ot = rec.get('observation_time') or rec.get('valid_time') or rec.get('last_updated')
        if t is None or not ot:
            continue
        try:
            dt = datetime.fromisoformat(str(ot).replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = pytz.utc.localize(dt)
            local_date = dt.astimezone(tz).strftime('%Y-%m-%d')
        except Exception:
            continue
        by_date.setdefault(local_date, []).append(t)
    mins = {d: round(min(v), 1) for d, v in by_date.items() if v}
    span = (min(mins), max(mins)) if mins else None
    return mins, len(hist), span


def main():
    print(f'=== low-source probe | {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC | '
          f'key: {bool(WETHR_API_KEY)} ===\n')

    today = datetime.utcnow().date()
    d1 = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    d2 = (today - timedelta(days=2)).strftime('%Y-%m-%d')

    # First, dump the shape of the history payload for one city so we learn it
    sample_st = CITIES['Miami'][0]
    sample = obs_history(sample_st)
    if isinstance(sample, list):
        print(f'obs-history shape: LIST of {len(sample)} records'
              + (f' | sample keys: {list(sample[0].keys())[:8]}…' if sample and isinstance(sample[0], dict) else ''))
    else:
        print(f'obs-history shape: {type(sample).__name__} → {str(sample)[:200]}')
    print()

    q1_matches = []
    for city, (st, tz) in CITIES.items():
        hist = obs_history(st)
        mins, nrec, span = daily_mins_from_history(hist, tz)
        f_low = fcst_low(st)

        cli_d1, cli_d2 = cli_min(st, d1), cli_min(st, d2)
        obs_d1 = mins.get(d1) if mins else None
        obs_d2 = mins.get(d2) if mins else None
        obs_today = mins.get(today.strftime('%Y-%m-%d')) if mins else None

        print(f'[{city}]  hist_records={nrec}  date_span={span}')
        print(f'    {d2}:  CLI={cli_d2}  obs-hist-min={obs_d2}')
        print(f'    {d1}:  CLI={cli_d1}  obs-hist-min={obs_d1}')
        print(f'    today: fcst_low={f_low}  obs-min-so-far={obs_today}')

        for cli_v, obs_v in ((cli_d1, obs_d1), (cli_d2, obs_d2)):
            if cli_v is not None and obs_v is not None:
                q1_matches.append(abs(cli_v - obs_v))
        print()

    # ── verdicts ──────────────────────────────────────────────────────────────
    print('=== Q1: does wethr obs-history reconstruct the CLI settlement min? ===')
    if q1_matches:
        avg = sum(q1_matches) / len(q1_matches)
        worst = max(q1_matches)
        ok = avg <= 1.0
        print(f'  compared {len(q1_matches)} city-days | avg |diff|={avg:.2f}F | worst={worst:.1f}F')
        print('  ✅ obs-history MATCHES CLI — usable low source.' if ok
              else '  ⚠️ obs-history DIVERGES from CLI — needs investigation before use.')
    else:
        print('  ⚠️ no overlapping city-days to compare (history window too short or '
              'no records). Check obs-history shape above — may need a date-range param.')
    print('\n=== Q2: forecast low = calendar-day min, or upcoming overnight? ===')
    print('  Compare today\'s fcst_low to obs-min-so-far per city above. If fcst_low '
          'sits well BELOW the min already observed today, it is predicting the upcoming '
          'overnight low (wrong target for KXLOWT). If close, it tracks the calendar-day min.')


if __name__ == '__main__':
    main()
