"""
fetch_obs_live.py — running daily max from the station Kalshi settles on.

V2.2 (2026-09-09): THE STALE ROW FIX.
=================================================================
On 2026-09-09 the panel showed San Antonio at a DAY MAX of 81.0F, in a green
box, at 5:19pm ET. The station had actually reached 97.0F at 3:51pm local. The
panel was not wrong about the data it had — it was showing a database row
written at about 13:24Z (8:24am local) and never updated since. Nine hours
stale, displayed as if it were live.

Boston showed the same signature (an off-grid 81.0F) the week before. Same
failure, and it will keep happening until the three causes below are closed.

WHAT WAS ACTUALLY BROKEN
------------------------
1. THE TOTAL ABORT. `main()` did:

       data = fetch_synoptic()
       if not data:
           print("no Synoptic data — aborting")
           return

   One failed Synoptic call — expired token, quota exhausted, a 500, a timeout
   — and the ENTIRE run returned before writing a single row. Not one city
   updated. The previous row stayed in the table looking perfectly healthy,
   because nothing in it records WHEN it was written relative to now. Every
   subsequent run hit the same wall and did the same nothing.

   V2.2 degrades instead of aborting. If Synoptic is down the run continues on
   METARs alone and still writes rows, because a row that says "METAR only,
   feed unavailable" is infinitely better than a row from this morning.

2. THE NWS TRUNCATION — the mirror image of the V2.0 bug.
   V2.0 used `limit=24` and got the 24 most RECENT observations, so it caught
   only the last hour or two. V2.1 fixed that with `start=local_midnight` but
   passed NO limit. api.weather.gov applies its own default cap and, with a
   start bound, returns the OLDEST records in the window. So the day was cut
   off from the FRONT: San Antonio returned 11 METARs ending at 13:24Z, and
   `precise_max_f` became the max of the MORNING — 81.0F — while the afternoon
   climb to 97.0F was never in the array at all.

   V2.2 passes an explicit `limit`, bounds with `end`, and follows pagination
   if the API offers it. It also asserts the newest row it got back is roughly
   current, and flags the station if it isn't.

3. NOTHING MEASURED FRESHNESS AT READ TIME. `obs_age_min` was computed when
   the row was WRITTEN. A row written at 13:24Z saying "obs_age_min: 4.2" still
   says 4.2 at 22:00Z. The panel had no way to know.

   V2.2 writes `updated_at` (it already did), plus `is_stale`, `stale_reason`,
   `synoptic_ok` and `metar_ok`. The panel must compare `updated_at` to NOW
   itself — see the note at the bottom of this file.

⚠️ THE DISPLAY IS STILL HALF THE BUG. This file cannot stop streamlit_app.py
from painting a nine-hour-old number green. The panel MUST refuse to show a
DAY MAX at all when now() - updated_at exceeds ~15 minutes. Fields are provided
below; wiring them up is a change to streamlit_app.py.

--- everything below is the V2.1 documentation, unchanged and still true ---

THE T-GROUP
-----------
On 2026-09-08 the panel showed New York at 79.0F. The station had actually
transmitted 25.6C — which is 78.1F. A full degree lower. That difference sat
directly on a bracket boundary and it changed a live decision.

Both hourly METARs that afternoon read the same:

    KNYC 081851Z AUTO 28006KT 10SM CLR 26/13 A3023 RMK AO2 SLP229 T02560128
    KNYC 081951Z AUTO VRB03KT 10SM CLR 26/13 A3023 RMK AO2 SLP228 T02560133
                                                                  ^^^^^^
The `26/13` is the rounded pair everyone displays. The `T0256` is the real
number: 25.6C, precise to a tenth. That group is in every ASOS METAR and every
consumer source throws it away.

WHY THE 5-MINUTE FEED CANNOT GIVE YOU THIS
-------------------------------------------
Verified independently against api.weather.gov and Synoptic: the 5-minute ASOS
observations transmit WHOLE DEGREES CELSIUS. Near a hot afternoon the only
values that exist are:

    35C = 95.0F   36C = 96.8F   37C = 98.6F   38C = 100.4F

There is nothing between 96.8 and 98.6 — the step is 1.8F. So a 5-minute
reading of "79.0F" actually means "somewhere in 25.5C to 26.4C", which is
77.9F to 79.5F. On a 79-or-below bracket that range spans the boundary.

The T-group collapses that range to a single number. It is available ONCE AN
HOUR, at :51, and it is the only precise reading the station publishes.

⚠️ THE PRECISE MAX IS HOURLY, THE FEED MAX IS 5-MINUTELY. They answer different
questions and neither is strictly better:

  - day_max_f     may MISS the true peak (whole-degree C, up to 0.9F low) but
                  samples every 5 minutes so it rarely misses the peak HOUR
  - precise_max_f is exact to a tenth but only samples 12-14 times a day, so a
                  peak that occurs between :51 reports is invisible to it

Kalshi settles on CLI, which is built from the precise record. So
precise_max_f is closer to what settles, but it is a FLOOR — the true daily max
is at least that, possibly higher if the peak fell between hourly reports.

⚠️ SOME STATIONS REPORT LESS OFTEN THAN OTHERS. KNYC (Central Park) produced
14 observations by 2:38pm on 2026-09-08 while the airport ASOS sites had 244+.
It also SKIPPED or delayed its 19:51 report by more than ten minutes that day.
n_metars_today makes that visible per city.

NOT a forecast. NOT a settlement source. NOT relevant to FAV V1, which never
looks at a temperature.

STATIONS
--------
Kalshi's OWN settlement stations, read off weather.com/kalshi.
Chicago is MIDWAY (KMDW) and Houston is HOBBY (KHOU) — O'Hare and Bush are
SEPARATE Kalshi markets and run several degrees apart.

SETUP
-----
1. Synoptic public token -> GitHub secret SYNOPTIC_TOKEN
2. Table columns (safe to re-run):

  ALTER TABLE public.obs_live
    ADD COLUMN IF NOT EXISTS metar_temp_f     NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS metar_time_utc   TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS metar_age_min    NUMERIC(6,1),
    ADD COLUMN IF NOT EXISTS precise_max_f    NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS precise_max_time TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS n_metars_today   INTEGER,
    ADD COLUMN IF NOT EXISTS is_stale         BOOLEAN,
    ADD COLUMN IF NOT EXISTS stale_reason     TEXT,
    ADD COLUMN IF NOT EXISTS synoptic_ok      BOOLEAN,
    ADD COLUMN IF NOT EXISTS metar_ok         BOOLEAN;

3. cron-job.org -> workflow_dispatch on obs_live.yml, America/New_York,
   */5 9-21 * * *.  NOT GitHub's `schedule` — it delayed this repo's runs by
   ~3 hours on 2026-09-03.

STATELESS BY DESIGN. Every run recomputes both maxima from the raw arrays
rather than incrementing stored values. A missed run cannot corrupt the max and
a bad reading cannot poison it permanently.
"""

import os
import re
import requests
import datetime as dt
from zoneinfo import ZoneInfo

SYNOPTIC_TOKEN = os.environ["SYNOPTIC_TOKEN"]
SB_URL = os.environ["SUPABASE_URL"].rstrip("/")
SB_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]

SYNOPTIC = "https://api.synopticdata.com/v2/stations/timeseries"
NWS_OBS = "https://api.weather.gov/stations/{stid}/observations"
NWS_HEADERS = {"User-Agent": "kalshi-obs/2.2", "Accept": "application/geo+json"}

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
STALE_WARN_MIN = 20          # 5-minute feed considered stale past this
METAR_STALE_MIN = 75         # hourly METAR considered stale past this
NWS_LIMIT = 500              # ⚠️ V2.2: explicit. Absent, the API truncates.

# T-group: T + sign + 3 digits (temp in tenths C) + sign + 3 digits (dewpoint).
# Sign digit is 0 for positive, 1 for negative.
#   T02560128 -> +25.6C / +12.8C
#   T10061033 -> -00.6C /  -3.3C
T_GROUP = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")


def sb_headers(prefer="return=minimal"):
    return {
        "apikey": SB_KEY,
        "Authorization": "Bearer " + SB_KEY,
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


def parse_t_group(raw_message):
    """Precise temperature in F from a METAR remark T-group, or None.

    `26/13` in the body is rounded; `T02560128` carries tenths. Every ASOS
    METAR has it; every consumer display drops it.
    """
    if not raw_message:
        return None
    m = T_GROUP.search(raw_message)
    if not m:
        return None
    try:
        sign, tenths = m.group(1), m.group(2)
        c = int(tenths) / 10.0
        if sign == "1":
            c = -c
        return round(c * 9.0 / 5.0 + 32.0, 2)
    except Exception:
        return None


def next_celsius_step_f(temp_f):
    """Next value the 5-minute feed could transmit, in F.

    The feed sends whole degrees C, so from 96.8F (36C) the only possible next
    reading up is 98.6F (37C). "0.2F from 97" would be nonsense — 97.0F is not
    a value this station can produce.
    """
    try:
        c = (float(temp_f) - 32.0) * 5.0 / 9.0
        return round((round(c) + 1) * 9.0 / 5.0 + 32.0, 1)
    except Exception:
        return None


def fetch_synoptic():
    """One call, all 20 stations, 24h of air_temp at native (~5 min) cadence.

    Returns None on ANY failure. ⚠️ V2.2: a None here no longer aborts the run —
    see main(). That abort is what froze every row for nine hours on 09-09.
    """
    params = {
        "stid": ",".join(STATIONS.keys()),
        "vars": "air_temp",
        "recent": 1440,
        "units": "english",
        "obtimezone": "utc",
        "qc": "on",
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
    try:
        data = r.json()
    except Exception as e:
        print(f"  Synoptic returned non-JSON: {type(e).__name__}")
        return None
    summary = data.get("SUMMARY") or {}
    if summary.get("RESPONSE_CODE") != 1:
        print(f"  Synoptic error: {summary.get('RESPONSE_MESSAGE')}")
        return None
    return data


def fetch_metars(stid, tzname, now_utc):
    """Hourly METARs for today, parsed for T-group precision.

    ⚠️ THE WINDOW HAS BEEN WRONG TWICE. Get this right or the daily max is a
    lie in whichever direction the truncation falls.

      V2.0 used `limit=24`  -> the 24 most RECENT obs. On a station reporting
                               every 5 minutes that is ~2 hours. Caught 1-2 of
                               the :51 METARs. Max was of the last hour.
      V2.1 used `start=` only -> no explicit limit, so the API's default cap
                               applied and returned the OLDEST rows in the
                               window. San Antonio 2026-09-09 returned 11
                               METARs ending 13:24Z; the 97.0F afternoon peak
                               was never in the array. Max was of the MORNING.
      V2.2 uses start + end + explicit limit + pagination.

    Also asserts the newest row returned is actually recent. If the API hands
    back a window that ends hours ago, that is reported as truncation rather
    than silently believed.

    Returns (rows, n, truncated) where rows is [(when_utc, temp_f), ...] for
    TODAY in the station's LOCAL calendar day, sorted oldest first.
    """
    tz = ZoneInfo(tzname)
    today_local = now_utc.astimezone(tz).date()
    local_midnight = dt.datetime.combine(
        today_local, dt.time(0, 0), tzinfo=tz).astimezone(dt.timezone.utc)

    params = {
        "start": local_midnight.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": NWS_LIMIT,
    }

    features = []
    url = NWS_OBS.format(stid=stid)
    pages = 0
    try:
        while url and pages < 5:
            r = requests.get(url, params=params if pages == 0 else None,
                             headers=NWS_HEADERS, timeout=30)
            if r.status_code != 200:
                print(f"    {stid} METAR HTTP {r.status_code}")
                break
            body = r.json() or {}
            got = body.get("features") or []
            features.extend(got)
            pages += 1
            # api.weather.gov may expose a next link; follow it if present.
            nxt = ((body.get("pagination") or {}).get("next")) or None
            if not nxt or not got:
                break
            url = nxt
    except Exception as e:
        print(f"    {stid} METAR fetch failed: {type(e).__name__}")
        return [], 0, True

    rows = []
    for f in features:
        props = f.get("properties") or {}
        raw = props.get("rawMessage")
        ts = props.get("timestamp")
        if not raw or not ts:
            continue
        t_f = parse_t_group(raw)
        if t_f is None:
            continue
        try:
            when = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        if when.astimezone(tz).date() != today_local:
            continue
        rows.append((when, t_f))

    rows.sort(key=lambda x: x[0])

    # ⚠️ Truncation check. If the newest METAR we hold is far older than the
    # newest one that SHOULD exist, we did not get the whole day.
    truncated = False
    if rows:
        newest_age_min = (now_utc - rows[-1][0]).total_seconds() / 60.0
        if newest_age_min > METAR_STALE_MIN:
            truncated = True
    return rows, len(rows), truncated


def parse_station(entry, tzname, now_utc):
    """5-minute rows for the station's LOCAL calendar day.

    Local date matters: the daily max must reset at LOCAL midnight, because
    that is the day Kalshi settles.
    """
    obs = entry.get("OBSERVATIONS") or {}
    times = obs.get("date_time") or []
    temps = (obs.get("air_temp_set_1") or obs.get("air_temp_set_1d") or [])
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
    """Change in F across the last `minutes`. Reads 0.0 often — the 5-minute
    feed steps 1.8F at a time, so a warming afternoon looks flat until it
    jumps a whole Celsius degree."""
    if len(rows) < 2:
        return None
    latest_t, latest_v = rows[-1]
    cutoff = latest_t - dt.timedelta(minutes=minutes)
    older = [(t, v) for t, v in rows if t <= cutoff]
    if not older:
        return None
    return round(latest_v - older[-1][1], 2)


def expected_metars_by_now(tzname, now_utc):
    """Roughly how many hourly METARs should exist since local midnight.

    Used only to flag suspiciously sparse returns. KNYC is legitimately sparse,
    so this warns, it does not reject.
    """
    tz = ZoneInfo(tzname)
    local_now = now_utc.astimezone(tz)
    return max(1, local_now.hour)


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
    print(f"OBS LIVE v2.2 | {now_utc:%Y-%m-%d %H:%M:%S} UTC | {len(STATIONS)} stations")
    print("5-min feed = whole degrees C (1.8F steps) | hourly METAR T-group = tenths\n")

    data = fetch_synoptic()
    synoptic_ok = data is not None

    if not synoptic_ok:
        # ⚠️ V2.2: THIS NO LONGER ABORTS.
        # V2.1 returned here. Every city kept its previous row, the panel had
        # no way to tell, and San Antonio displayed an 8:24am number in a green
        # box at 5:19pm while the station was 16 degrees hotter.
        print("⚠️ SYNOPTIC UNAVAILABLE — continuing on METARs only.")
        print("   Rows WILL still be written, flagged synoptic_ok=false.\n")
        by_stid = {}
    else:
        stations = data.get("STATION") or []
        by_stid = {(e.get("STID") or "").upper(): e for e in stations}

    print(f"{'CITY':<15} {'FEED':>6} {'AGE':>5}  {'T-GRP':>6} {'AGE':>5}  "
          f"{'FEEDMAX':>7} {'PRECMAX':>7} {'NEXT':>6}  {'30m':>5}  n/m")
    print("-" * 92)

    written = 0
    stale_cities = []

    for stid, (city, tzname) in STATIONS.items():
        tz = ZoneInfo(tzname)
        local_date = now_utc.astimezone(tz).date()

        # ---- 5-minute feed (may be entirely absent if Synoptic is down) ----
        rows = []
        entry = by_stid.get(stid)
        if entry:
            rows, parsed_date = parse_station(entry, tzname, now_utc)
            if parsed_date:
                local_date = parsed_date

        if rows:
            last_t, last_v = rows[-1]
            age_min = round((now_utc - last_t).total_seconds() / 60.0, 1)
            max_t, max_v = max(rows, key=lambda x: x[1])
            trend = trend_over(rows, TREND_WINDOW_MIN)
            next_step = next_celsius_step_f(max_v)
        else:
            last_t = last_v = age_min = max_t = max_v = trend = next_step = None

        # ---- hourly METAR T-groups (the precise record) ----
        metars, n_metars, truncated = fetch_metars(stid, tzname, now_utc)
        if metars:
            m_last_t, m_last_v = metars[-1]
            m_age = round((now_utc - m_last_t).total_seconds() / 60.0, 1)
            pm_t, pm_v = max(metars, key=lambda x: x[1])
        else:
            m_last_t = m_last_v = m_age = pm_t = pm_v = None

        metar_ok = bool(metars) and not truncated

        # ---- staleness verdict -------------------------------------------
        reasons = []
        if not synoptic_ok:
            reasons.append("synoptic down")
        elif not rows:
            reasons.append("no feed obs today")
        elif age_min > STALE_WARN_MIN:
            reasons.append(f"feed {age_min:.0f}m old")

        if not metars:
            reasons.append("no METARs today")
        elif truncated:
            reasons.append(f"METAR window truncated (newest {m_age:.0f}m old)")
        elif m_age > METAR_STALE_MIN:
            reasons.append(f"METAR {m_age:.0f}m old")

        exp = expected_metars_by_now(tzname, now_utc)
        if n_metars and n_metars < exp * 0.5:
            reasons.append(f"sparse: {n_metars} METARs, expected ~{exp}")

        is_stale = bool(reasons)
        stale_reason = "; ".join(reasons) if reasons else None
        if is_stale:
            stale_cities.append(city)

        if not rows and not metars:
            print(f"{city:<15} NO DATA — {stale_reason}")
            # still write the row so the panel can see the failure
            upsert({
                "city": city, "station": stid,
                "local_date": local_date.isoformat(),
                "is_stale": True, "stale_reason": stale_reason,
                "synoptic_ok": synoptic_ok, "metar_ok": False,
                "n_obs_today": 0, "n_metars_today": 0,
                "updated_at": now_utc.isoformat(),
            })
            continue

        flag = ""
        if is_stale:
            flag = f"  ⚠️ {stale_reason}"
        # the gap that cost a decision on 2026-09-08
        if pm_v is not None and max_v is not None and abs(max_v - pm_v) >= 0.8:
            flag += f"  ⚠️ feed/T-grp gap {max_v - pm_v:+.1f}F"

        print(f"{city:<15} "
              f"{(f'{last_v:.1f}' if last_v is not None else '—'):>6} "
              f"{(f'{age_min:.0f}m' if age_min is not None else '—'):>5}  "
              f"{(f'{m_last_v:.1f}' if m_last_v is not None else '—'):>6} "
              f"{(f'{m_age:.0f}m' if m_age is not None else '—'):>5}  "
              f"{(f'{max_v:.1f}' if max_v is not None else '—'):>7} "
              f"{(f'{pm_v:.1f}' if pm_v is not None else '—'):>7} "
              f"{(f'{next_step:.1f}' if next_step is not None else '—'):>6}  "
              f"{(f'{trend:+.1f}' if trend is not None else '—'):>5}  "
              f"{len(rows)}/{n_metars}{flag}")

        if upsert({
            "city": city,
            "station": stid,
            "local_date": local_date.isoformat(),
            "temp_f": round(last_v, 2) if last_v is not None else None,
            "temp_time_utc": last_t.isoformat() if last_t else None,
            "obs_age_min": age_min,
            "day_max_f": round(max_v, 2) if max_v is not None else None,
            "day_max_time": max_t.isoformat() if max_t else None,
            "next_step_f": next_step,
            "trend_30min": trend,
            "n_obs_today": len(rows),
            "metar_temp_f": round(m_last_v, 2) if m_last_v is not None else None,
            "metar_time_utc": m_last_t.isoformat() if m_last_t else None,
            "metar_age_min": m_age,
            "precise_max_f": round(pm_v, 2) if pm_v is not None else None,
            "precise_max_time": pm_t.isoformat() if pm_t else None,
            "n_metars_today": n_metars,
            "is_stale": is_stale,
            "stale_reason": stale_reason,
            "synoptic_ok": synoptic_ok,
            "metar_ok": metar_ok,
            "updated_at": now_utc.isoformat(),
        }):
            written += 1

    print(f"\n  wrote {written} rows")
    if stale_cities:
        print(f"  ⚠️ STALE OR DEGRADED: {', '.join(stale_cities)}")
    else:
        print("  all stations fresh")

    print("\n  FEEDMAX is 5-minutely but quantized to whole degrees C.")
    print("  PRECMAX is exact to a tenth but only ~12-14 samples a day.")
    print("  PRECMAX is a FLOOR — a peak between :51 reports is invisible to it.")
    print("\n  ⚠️ THE PANEL MUST STILL CHECK updated_at ITSELF.")
    print("     obs_age_min is computed at WRITE time and does not age.")
    print("     If now() - updated_at > 15 min, show STALE, not a number.")
    print("\n  select city, temp_f, metar_temp_f, day_max_f, precise_max_f,")
    print("         obs_age_min, metar_age_min, n_metars_today,")
    print("         is_stale, stale_reason, updated_at")
    print("  from obs_live where local_date = current_date")
    print("  order by precise_max_f desc nulls last;")


if __name__ == "__main__":
    main()
