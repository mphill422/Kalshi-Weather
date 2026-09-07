"""
fetch_obs_live.py — running daily max from the station Kalshi settles on.

WHY THIS EXISTS, AND WHAT IT CANNOT DO
=======================================
On 2026-09-06 San Antonio displayed "97" for roughly ninety minutes, then
printed 99. The bracket flipped and a 58c entry that had reached 97c settled a
loser. Two days were spent hunting for a feed with finer resolution.

That feed does not exist. Here is what the data actually looks like — Synoptic,
KSAT, ten consecutive 5-minute observations on 2026-09-07:

    96.8, 95.0, 95.0, 95.0, 96.8, 96.8, 96.8, 95.0, 96.8, 96.8

Two distinct values. 95.0F is exactly 35C. 96.8F is exactly 36C.

⚠️ THE 5-MINUTE ASOS FEED TRANSMITS WHOLE DEGREES CELSIUS. The Fahrenheit
decimals are a unit conversion artifact, not precision. The only readings
possible near a hot afternoon are:

    35C = 95.0F     36C = 96.8F     37C = 98.6F     38C = 100.4F

There is NOTHING between 96.8 and 98.6. The step is 1.8F.

So the 97 -> 99 jump was the station stepping 36C to 37C. Nothing was hidden.
There was genuinely no intermediate value to see, and no amount of polling or
paying would have produced one.

Verified against BOTH feeds independently:
  - api.weather.gov 5-minute rows: temperature.value = 35, 35, 35, 36, 35 (C)
  - Synoptic same window: 95.0, 95.0, 95.0, 96.8, 95.0 (F)
Same numbers. Synoptic offers nothing NWS does not on resolution.

⚠️ ONLY the hourly :51 METAR carries true tenths, in the T-group of the raw
message: `T03500194` = 35.0C / 19.4C dewpoint. Once an hour, not every 5 min.

WHAT THIS DOES BUY, WHICH IS STILL WORTH HAVING
------------------------------------------------
  1. TRUE OBSERVATION AGE. The Streamlit panel showed "0s old" next to an
     observation that was 48 minutes stale — "0s" meant the FETCH was fresh,
     not the reading. This reports the age of the reading itself.
  2. A RUNNING MAX COMPUTED FROM RAW OBSERVATIONS at the station Kalshi
     settles on, rather than trusting a vendor's summary field. On 2026-09-06
     Wethr reported an obs high of 79.0F on a day the actual high was 78 —
     which eliminated "78 or below" from the model and was simply wrong.
     The punchlist also records Denver at 24.0F against a CLI of 56.0F with
     every quality flag reading clean.
  3. 5-minute cadence on that max, all 20 stations in ONE API call.
  4. Sky condition from the SAME station. Wunderground showed high cloud cover
     for San Antonio on 9/6 while KSAT itself read CLR through FEW075 all
     afternoon. Second-hand source, wrong answer.

WHAT IT IS NOT
--------------
NOT a forecast. It reports what the high IS so far and has no opinion about
where it ends up. Do not let a rising obs floor talk you out of a band entry —
the 58-69c band is measured on the ENTRY price, not on how the afternoon feels.

NOT a settlement source. Kalshi settles on the official CLI. These are pre-QC
preliminary observations. Use this to SEE the day, never to score it.

NOT relevant to FAV V1 at all. That strategy never looks at a temperature.

NEAR-EDGE IS MEASURED IN CELSIUS, DELIBERATELY
-----------------------------------------------
An earlier draft flagged when the running max came within 0.5F of the next
whole Fahrenheit degree. That is meaningless when the underlying values move in
1.8F steps — the max is ALWAYS sitting on a value like 96.8 and can only ever
jump to 98.6.

So the flag tracks what actually matters: how close the running max is to a
Kalshi BRACKET BOUNDARY, and what the next possible reading would be. If the
max is 96.8 and the bracket is 96-97, the next transmittable value (98.6)
breaks it. That is real information. "0.2F from 97" is not.

STATIONS
--------
Kalshi's OWN settlement stations, read off weather.com/kalshi.
Chicago is MIDWAY (KMDW) and Houston is HOBBY (KHOU) — O'Hare and Bush are
SEPARATE Kalshi markets and run several degrees apart (on 2026-09-06 the list
showed HOU 93 and IAH 92). Getting these wrong produces settlement surprises
that look like model error.

SETUP
-----
1. Synoptic account -> Credentials -> Public tokens -> Create token.
   (A private KEY is not a token. Keys manage tokens; you cannot make data
   requests with a key.)
2. GitHub secret: SYNOPTIC_TOKEN
3. Table (Supabase SQL editor, safe to re-run):

  CREATE TABLE IF NOT EXISTS public.obs_live (
    id            BIGSERIAL PRIMARY KEY,
    city          TEXT NOT NULL,
    station       TEXT NOT NULL,
    local_date    DATE NOT NULL,
    temp_f        NUMERIC(6,2),
    temp_time_utc TIMESTAMPTZ,
    obs_age_min   NUMERIC(6,1),
    day_max_f     NUMERIC(6,2),
    day_max_time  TIMESTAMPTZ,
    next_step_f   NUMERIC(6,2),
    trend_30min   NUMERIC(5,2),
    n_obs_today   INTEGER,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (city, local_date)
  );
  ALTER TABLE public.obs_live ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "Allow all access" ON public.obs_live
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);
  CREATE INDEX IF NOT EXISTS idx_obs_live_date ON public.obs_live (local_date);

4. cron-job.org -> workflow_dispatch on obs_live.yml,
   America/New_York, every 5 min 9am-9pm ET.

   ⚠️ NOT GitHub's `schedule`. GitHub delayed this repo's scheduled runs by
   ~3 HOURS on 2026-09-03. For a job whose entire purpose is freshness that is
   disqualifying.

STATELESS BY DESIGN. Every run recomputes the daily max from the raw 24h array
rather than incrementing a stored value. A missed run cannot corrupt the max and
a bad reading cannot poison it permanently.

⚠️ LICENSING. Confirm your Synoptic account tier covers this use. The free Open
Access program is for academic and non-profit research.
"""

import os
import requests
import datetime as dt
from zoneinfo import ZoneInfo

SYNOPTIC_TOKEN = os.environ["SYNOPTIC_TOKEN"]
SB_URL = os.environ["SUPABASE_URL"].rstrip("/")
SB_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]

SYNOPTIC = "https://api.synopticdata.com/v2/stations/timeseries"

# Kalshi settlement stations. Chicago = MIDWAY. Houston = HOBBY.
STATIONS = {
    "KATL": ("Atlanta",        "America/New_York"),
    "KAUS": ("Austin",         "America/Chicago"),
    "KBOS": ("Boston",         "America/New_York"),
    "KDCA": ("Washington DC",  "America/New_York"),
    "KDEN": ("Denver",         "America/Denver"),
    "KDFW": ("Dallas",         "America/Chicago"),
    "KHOU": ("Houston",        "America/Chicago"),
    "KLAS": ("Las Vegas",      "America/Los_Angeles"),
    "KLAX": ("Los Angeles",    "America/Los_Angeles"),
    "KMDW": ("Chicago",        "America/Chicago"),
    "KMIA": ("Miami",          "America/New_York"),
    "KMSP": ("Minneapolis",    "America/Chicago"),
    "KMSY": ("New Orleans",    "America/Chicago"),
    "KNYC": ("New York",       "America/New_York"),
    "KOKC": ("Oklahoma City",  "America/Chicago"),
    "KPHL": ("Philadelphia",   "America/New_York"),
    "KPHX": ("Phoenix",        "America/Phoenix"),
    "KSAT": ("San Antonio",    "America/Chicago"),
    "KSEA": ("Seattle",        "America/Los_Angeles"),
    "KSFO": ("San Francisco",  "America/Los_Angeles"),
}

TREND_WINDOW_MIN = 30

# An observation older than this is called out. The 5-minute feed means a
# healthy station is never more than ~6 minutes stale; 20 means something is
# wrong with the station or the feed, not with the weather.
STALE_WARN_MIN = 20


def sb_headers(prefer="return=minimal"):
    return {
        "apikey": SB_KEY,
        "Authorization": "Bearer " + SB_KEY,
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


def next_celsius_step_f(temp_f):
    """The next value the station could actually transmit, in F.

    The feed sends whole degrees C, so from 96.8F (36C) the only possible next
    reading up is 98.6F (37C). Reporting "0.2F from 97" would be nonsense —
    97.0F is not a value this station can produce.
    """
    try:
        c = (float(temp_f) - 32.0) * 5.0 / 9.0
        return round((round(c) + 1) * 9.0 / 5.0 + 32.0, 1)
    except Exception:
        return None


def fetch_all():
    """One call for all 20 stations, 24h of air_temp at native resolution."""
    params = {
        "stid": ",".join(STATIONS.keys()),
        "vars": "air_temp",
        "recent": 1440,              # minutes
        "units": "english",          # degrees F (converted from native C)
        "obtimezone": "utc",
        "qc": "on",                  # surface QC rather than silently passing
        "token": SYNOPTIC_TOKEN,
    }
    try:
        r = requests.get(SYNOPTIC, params=params, timeout=45)
    except Exception as e:
        print(f"  Synoptic request failed: {type(e).__name__}: {str(e)[:150]}")
        return None
    if r.status_code != 200:
        print(f"  Synoptic HTTP {r.status_code}: {r.text[:200]}")
        return None
    data = r.json()
    summary = data.get("SUMMARY") or {}
    if summary.get("RESPONSE_CODE") != 1:
        print(f"  Synoptic error: {summary.get('RESPONSE_MESSAGE')}")
        return None
    return data


def parse_station(entry, tzname, now_utc):
    """Rows for the station's LOCAL calendar day, midnight to now.

    Local date matters: a 20:40 UTC observation is the same calendar day in
    New York and in Phoenix, but the daily max must reset at LOCAL midnight
    because that is what Kalshi settles on.
    """
    obs = entry.get("OBSERVATIONS") or {}
    times = obs.get("date_time") or []
    temps = (obs.get("air_temp_set_1")
             or obs.get("air_temp_set_1d")
             or [])
    if not times or not temps:
        return [], None

    tz = ZoneInfo(tzname)
    today_local = now_utc.astimezone(tz).date()

    rows = []
    for ts, t in zip(times, temps):
        if t is None:
            continue
        try:
            when = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        if when.astimezone(tz).date() != today_local:
            continue
        try:
            rows.append((when, float(t)))
        except Exception:
            continue

    rows.sort(key=lambda x: x[0])
    return rows, today_local


def trend_over(rows, minutes):
    """Change in F across the last `minutes`. None if not enough history.

    Note this will read 0.0 a lot — the feed steps in 1.8F increments, so a
    genuinely warming afternoon shows flat until it jumps a full step.
    """
    if len(rows) < 2:
        return None
    latest_t, latest_v = rows[-1]
    cutoff = latest_t - dt.timedelta(minutes=minutes)
    older = [(t, v) for t, v in rows if t <= cutoff]
    if not older:
        return None
    return round(latest_v - older[-1][1], 2)


def upsert(row):
    try:
        r = requests.post(
            f"{SB_URL}/rest/v1/obs_live?on_conflict=city,local_date",
            headers=sb_headers("return=minimal,resolution=merge-duplicates"),
            json=row, timeout=20)
        if r.status_code not in (200, 201, 204):
            print(f"    upsert HTTP {r.status_code}: {r.text[:120]}")
            return False
        return True
    except Exception as e:
        print(f"    upsert failed: {type(e).__name__}: {str(e)[:120]}")
        return False


def main():
    now_utc = dt.datetime.now(dt.timezone.utc)
    print(f"OBS LIVE | {now_utc:%Y-%m-%d %H:%M:%S} UTC | {len(STATIONS)} stations")
    print("running daily max from Kalshi's own settlement stations")
    print("⚠️ feed transmits WHOLE DEGREES C — F values step 1.8 at a time\n")

    data = fetch_all()
    if not data:
        print("no data returned — aborting")
        return

    stations = data.get("STATION") or []
    print(f"{'CITY':<15} {'NOW':>7} {'AGE':>5}  {'DAY MAX':>8} {'AT':>7}  "
          f"{'NEXT':>7}  {'30m':>5}   n")
    print("-" * 68)

    seen = set()
    written = 0
    for entry in stations:
        stid = (entry.get("STID") or "").upper()
        if stid not in STATIONS:
            continue
        city, tzname = STATIONS[stid]
        seen.add(stid)

        rows, local_date = parse_station(entry, tzname, now_utc)
        if not rows:
            print(f"{city:<15} {'—':>7}  no observations today")
            continue

        tz = ZoneInfo(tzname)
        last_t, last_v = rows[-1]
        age_min = round((now_utc - last_t).total_seconds() / 60.0, 1)

        max_t, max_v = max(rows, key=lambda x: x[1])
        trend = trend_over(rows, TREND_WINDOW_MIN)
        next_step = next_celsius_step_f(max_v)

        flag = ""
        if age_min > STALE_WARN_MIN:
            flag += f"  ⚠️ STALE {age_min:.0f}m"
        if trend is not None and trend > 0:
            flag += f"  ↑ +{trend:.1f}/30m"

        print(f"{city:<15} {last_v:>7.1f} {age_min:>4.0f}m  {max_v:>8.1f} "
              f"{max_t.astimezone(tz):%H:%M}  {next_step:>7.1f}  "
              f"{(f'{trend:+.1f}' if trend is not None else '—'):>5} "
              f"{len(rows):>4}{flag}")

        if upsert({
            "city": city,
            "station": stid,
            "local_date": local_date.isoformat(),
            "temp_f": round(last_v, 2),
            "temp_time_utc": last_t.isoformat(),
            "obs_age_min": age_min,
            "day_max_f": round(max_v, 2),
            "day_max_time": max_t.isoformat(),
            "next_step_f": next_step,
            "trend_30min": trend,
            "n_obs_today": len(rows),
            "updated_at": now_utc.isoformat(),
        }):
            written += 1

    missing = set(STATIONS) - seen
    if missing:
        print(f"\n  no data returned for: {', '.join(sorted(missing))}")
    print(f"\n  wrote {written} rows")

    print("\nNEXT is the next value the station can actually transmit.")
    print("If DAY MAX is 96.8 and your bracket is 96-97, NEXT (98.6) breaks it.")
    print("\n  select city, temp_f, obs_age_min, day_max_f, next_step_f, trend_30min")
    print("  from obs_live where local_date = current_date")
    print("  order by day_max_f desc;")


if __name__ == "__main__":
    main()
