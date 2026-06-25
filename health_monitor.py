"""
health_monitor.py — daily heartbeat for the Kalshi weather pipeline
====================================================================
Runs once a day and FAILS LOUDLY when something is broken, instead of letting
the model degrade silently (the dead-wethr-key-for-six-weeks problem).

It checks the exact things that died unnoticed:
  CHECK 1  Data freshness   — is the fetch pipeline still writing rows?
  CHECK 2  obs_high feed     — is wethr authenticating? (the dead-key catcher)
  CHECK 3  ensemble feed     — is Open-Meteo GFS coming back?
  CHECK 4  settlement lag    — are settlements getting actuals from Iowa CLI?
  CHECK 5  bet settlement lag — are paper bets resolving?

If ANY hard check fails, the script exits non-zero. A failed GitHub Actions run
emails you automatically (no SMTP setup needed) — that's the alert. Green days
exit 0 and stay quiet.

Needs SUPABASE_URL + SUPABASE_KEY (already in repo secrets). Read-only. No model
code touched — pure monitoring, freeze-safe.

Recommended schedule: 05:00 UTC daily (~1am ET). At that hour the latest fully
populated settlement date is YESTERDAY (today's fetch windows haven't run yet),
so obs_high etc. are complete and we avoid intraday false alarms.
"""

import os
import sys
from datetime import datetime, timedelta

import requests
import pytz

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')

# Hard-fail thresholds
OBS_HIGH_MIN_FILL   = 0.01   # latest day must have >0% obs_high (0 = feed dead)
ENSEMBLE_MIN_FILL   = 0.01   # >0% ensemble (0 = Open-Meteo dead)
SETTLE_LAG_DAYS     = 2      # settlements older than this with null actual = broken
BET_LAG_DAYS        = 2      # bets older than this still pending = broken
EXPECTED_CITIES     = 18


def sb_headers():
    return {'apikey': SUPABASE_KEY, 'Authorization': 'Bearer ' + SUPABASE_KEY,
            'Accept': 'application/json'}


def sb_get(table, params):
    r = requests.get(SUPABASE_URL + '/rest/v1/' + table,
                     headers=sb_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def et_today():
    return datetime.now(pytz.timezone('America/New_York')).date()


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print('❌ SUPABASE_URL / SUPABASE_KEY missing from env — cannot run monitor.')
        sys.exit(1)

    today = et_today()
    print(f'=== Health monitor | ET {today} | {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC ===\n')

    results = []  # (ok: bool, name, detail, action)

    # ── latest settlement date ────────────────────────────────────────────────
    try:
        latest = sb_get('settlements', {'select': 'date', 'order': 'date.desc', 'limit': '1'})
        latest_date = latest[0]['date'] if latest else None
    except Exception as e:
        print(f'❌ Could not reach Supabase: {e}')
        sys.exit(1)

    # CHECK 1 — data freshness
    if latest_date is None:
        results.append((False, 'Data freshness', 'settlements table is EMPTY',
                        'Fetch pipeline never wrote — check Weather Data Fetch workflow.'))
    else:
        ld = datetime.strptime(latest_date, '%Y-%m-%d').date()
        age = (today - ld).days
        ok = age <= 1
        results.append((ok, 'Data freshness',
                        f'latest settlement row = {latest_date} ({age}d old)',
                        'Fetch pipeline stalled — check Weather Data Fetch workflow / cron.' if not ok else ''))

    # rows for the latest date (feed checks run against a fully-populated day)
    rows = []
    if latest_date:
        rows = sb_get('settlements', {'date': 'eq.' + latest_date,
                                      'select': 'city,obs_high,ensemble_mean'})
    n = len(rows) if rows else 0

    # CHECK 2 — obs_high feed (the dead-key catcher)
    obs_filled = sum(1 for r in rows if r.get('obs_high') is not None)
    obs_fill = (obs_filled / n) if n else 0.0
    ok2 = obs_fill > OBS_HIGH_MIN_FILL
    results.append((ok2, 'obs_high feed (wethr)',
                    f'{obs_filled}/{n} cities filled ({obs_fill*100:.0f}%) on {latest_date}',
                    'wethr key likely DEAD or disabled. Check wethr.net → API Keys, '
                    'generate a live key, update WETHR_API_KEY secret.' if not ok2 else ''))

    # CHECK 3 — ensemble feed (Open-Meteo)
    ens_filled = sum(1 for r in rows if r.get('ensemble_mean') is not None)
    ens_fill = (ens_filled / n) if n else 0.0
    ok3 = ens_fill > ENSEMBLE_MIN_FILL
    results.append((ok3, 'ensemble feed (Open-Meteo)',
                    f'{ens_filled}/{n} cities filled ({ens_fill*100:.0f}%) on {latest_date}',
                    'Open-Meteo GFS down or blocked — check ensemble endpoint / rate limits.' if not ok3 else ''))

    # CHECK 4 — settlement lag (Iowa CLI actuals)
    settle_cutoff = (today - timedelta(days=SETTLE_LAG_DAYS)).strftime('%Y-%m-%d')
    stale_settles = sb_get('settlements', {'actual': 'is.null', 'date': 'lt.' + settle_cutoff,
                                           'select': 'date,city', 'limit': '500'})
    ok4 = len(stale_settles) == 0
    results.append((ok4, 'settlement lag (Iowa CLI)',
                    f'{len(stale_settles)} settlement row(s) older than {SETTLE_LAG_DAYS}d still have no actual',
                    'CLI settlement pass not resolving — check fetch_cli_max_temp / run_settlement_pass.' if not ok4 else ''))

    # CHECK 5 — bet settlement lag
    bet_cutoff = (today - timedelta(days=BET_LAG_DAYS)).strftime('%Y-%m-%d')
    stale_bets = sb_get('bets', {'or': '(result.eq.Pending,result.is.null)',
                                 'date': 'lt.' + bet_cutoff,
                                 'select': 'date,city', 'limit': '500'})
    ok5 = len(stale_bets) == 0
    results.append((ok5, 'bet settlement lag',
                    f'{len(stale_bets)} bet(s) older than {BET_LAG_DAYS}d still Pending',
                    'Bet settlement not resolving — check settlement pass bet-matching.' if not ok5 else ''))

    # ── report ────────────────────────────────────────────────────────────────
    any_fail = False
    for ok, name, detail, action in results:
        mark = '✅' if ok else '❌'
        print(f'{mark} {name}: {detail}')
        if not ok:
            any_fail = True
            print(f'     → {action}')

    print()
    if any_fail:
        print('🚨 HEALTH CHECK FAILED — see ❌ above. This run exits with an error so '
              'GitHub emails you. Fix the flagged item.')
        sys.exit(1)
    else:
        print(f'✅ All checks passed. Pipeline healthy ({n}/{EXPECTED_CITIES} cities '
              f'on {latest_date}).')


if __name__ == '__main__':
    main()
