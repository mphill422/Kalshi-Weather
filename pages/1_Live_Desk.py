"""
Live Desk — V1.6 (2026-09-20)  ·  V1.6: CLI as-of stamp is local clock time, not standard (was an hour ahead)  ·  V1.5: official-high fix — overnight partial report no longer read as FINAL  ·  V1.4: Synoptic + aviationweather fast feeds, STEPPED UP / NEW HIGH flash, :51 countdown  ·  V1.3: official NWS high (whole °F) card; locks the call when posted  ·  V1.2: KBOS/KMSP on-grid readings shown as ranges  ·  V1.1: 📊 Mac / 📱 iPhone toggle; Celsius straddle cards

What this page is for: the last hour of a bet. Hold, or cash out?

It shows, for every settlement station, in °F only:
  * the temperature right now, and the range it could really be
  * the high so far, and the lowest number the day can still settle at
  * whether the station looks past its peak
  * the Kalshi ladder with the model's % beside each bracket
  * a HOLD / CASH OUT call for the bracket you are holding

It refreshes itself every 60 seconds. It is a separate page and changes
nothing in streamlit_app.py.

HOW THE NUMBERS ARE BUILT (so the display never has to mention Celsius)
-----------------------------------------------------------------------
Most stations send their 5-minute reading rounded to a whole degree Celsius,
so a 5-minute "84.2" really means "somewhere from 83.3 to 85.1". This page
shows that as a plain °F range. Two readings are exact:
  * the hourly report (its remarks carry the temperature to a tenth)
  * the 6-hour maximum group in the 1751Z / 2351Z / 0551Z reports
KBOS and KMSP send exact tenths on every reading.

The settle odds use THIS ACCOUNT'S measured error: every (CLI actual − feed
max) from the max_vs_settled view in Supabase. The same numbers the decision
board uses. If Supabase is unreachable it falls back to that measured
summary (mean +0.15, sd 0.6, range −0.9 to +1.9).

⚠️ The odds only mean "where it settles IF today's high is already in".
Before the peak, the high can still climb. The page says so on screen.

⚠️ Data comes from api.weather.gov and typically lags the sensor by 5-15
minutes. It is as live as free public data gets.
"""

import math
import re
import datetime as dt
from statistics import NormalDist
from concurrent.futures import ThreadPoolExecutor

import pytz
import requests
import streamlit as st

st.set_page_config(page_title="Live Desk", page_icon="🌡️", layout="wide")


# ── Same password gate as the main app ───────────────────────────────────────
def check_password():
    try:
        correct = st.secrets.get('app_password', None)
    except Exception:
        correct = None
    if not correct or st.session_state.get('_authed'):
        return
    st.markdown('### 🌡️ MPH Weather')
    pw = st.text_input('Password', type='password',
                       label_visibility='collapsed', placeholder='Password')
    if pw:
        if pw == correct:
            st.session_state['_authed'] = True
            st.rerun()
        else:
            st.error('Incorrect.')
    st.stop()


check_password()

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.stApp { background: #0a0e1a; }
.card { background:#0d1b2a; border:1px solid #1e3a5f; border-radius:10px;
        padding:12px 16px; margin-bottom:8px; }
.lbl  { color:#64748b; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
.big  { color:#fff; font-size:36px; font-weight:700; line-height:1.1;
        font-family:'JetBrains Mono',monospace; }
.sm   { color:#94a3b8; font-size:12px; font-family:'JetBrains Mono',monospace; }
.verdict { border-radius:12px; padding:16px 20px; margin:10px 0; font-size:26px;
           font-weight:800; font-family:'JetBrains Mono',monospace; }
.v-hold { background:#06301f; border:2px solid #00ff88; color:#00ff88; }
.v-sell { background:#2a0d12; border:2px solid #ef4444; color:#ef4444; }
.v-wait { background:#2a2208; border:2px solid #fbbf24; color:#fbbf24; }
.v-why  { color:#cbd5e1; font-size:14px; font-weight:400; margin-top:6px; }
</style>
""", unsafe_allow_html=True)


# ── Stations ─────────────────────────────────────────────────────────────────
# city: (settlement station, Kalshi series, station time zone)
CITIES = {
    "New York":      ("KNYC", "KXHIGHNY",     "America/New_York"),
    "Miami":         ("KMIA", "KXHIGHMIA",    "America/New_York"),
    "Atlanta":       ("KATL", "KXHIGHTATL",   "America/New_York"),
    "Philadelphia":  ("KPHL", "KXHIGHPHIL",   "America/New_York"),
    "Washington DC": ("KDCA", "KXHIGHTDC",    "America/New_York"),
    "Boston":        ("KBOS", "KXHIGHTBOS",   "America/New_York"),
    "Chicago":       ("KMDW", "KXHIGHCHI",    "America/Chicago"),
    "Austin":        ("KAUS", "KXHIGHAUS",    "America/Chicago"),
    "Dallas":        ("KDFW", "KXHIGHTDAL",   "America/Chicago"),
    "Houston":       ("KHOU", "KXHIGHTHOU",   "America/Chicago"),
    "Oklahoma City": ("KOKC", "KXHIGHTOKC",   "America/Chicago"),
    "Minneapolis":   ("KMSP", "KXHIGHTMIN",   "America/Chicago"),
    "San Antonio":   ("KSAT", "KXHIGHTSATX",  "America/Chicago"),
    "New Orleans":   ("KMSY", "KXHIGHTNOLA",  "America/Chicago"),
    "Denver":        ("KDEN", "KXHIGHDEN",    "America/Denver"),
    "Phoenix":       ("KPHX", "KXHIGHTPHX",   "America/Phoenix"),
    "Las Vegas":     ("KLAS", "KXHIGHTLV",    "America/Los_Angeles"),
    "Los Angeles":   ("KLAX", "KXHIGHLAX",    "America/Los_Angeles"),
    "Seattle":       ("KSEA", "KXHIGHTSEA",   "America/Los_Angeles"),
    "San Francisco": ("KSFO", "KXHIGHTSFO",   "America/Los_Angeles"),
}
TENTHS_F = {"KBOS", "KMSP"}          # the only two that send exact tenths
NWS_HDR = {"User-Agent": "MPH-Weather-LiveDesk/1.0 (github.com/mphill422)",
           "Accept": "application/geo+json"}
KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets"
FEE = 0.07


def c_to_f(c):
    return c * 1.8 + 32.0


def settle_round(x):
    """CLI whole degrees, half rounds up."""
    return int(math.floor(x + 0.5))


# ── Climate day ──────────────────────────────────────────────────────────────
def climate_day_start(tz_name, now_utc):
    """CLI's day runs midnight-to-midnight LOCAL STANDARD time, so in summer
    it starts at 1:00am on the clock. Returns (start_utc, local_date)."""
    tz = pytz.timezone(tz_name)
    now_local = now_utc.astimezone(tz)
    d = now_local.date()
    mid = tz.localize(dt.datetime(d.year, d.month, d.day, 0, 0))
    start = mid + (mid.dst() or dt.timedelta(0))
    if now_local < start:                       # 12:00-12:59am in summer
        d = d - dt.timedelta(days=1)
        mid = tz.localize(dt.datetime(d.year, d.month, d.day, 0, 0))
        start = mid + (mid.dst() or dt.timedelta(0))
    return start.astimezone(pytz.utc), d


# ── Observations ─────────────────────────────────────────────────────────────
TGROUP = re.compile(r"\bT([01])(\d{3})[01]\d{3}\b")
MAX6 = re.compile(r"(?<![\d/])1([01])(\d{3})(?![\d/])")


def parse_reading(stn, ts, c, raw):
    """One observation -> °F value plus the range the true temp can be in."""
    raw = raw or ""
    m = TGROUP.search(raw)
    if m:
        cc = int(m.group(2)) / 10.0 * (-1 if m.group(1) == "1" else 1)
        f = c_to_f(cc)
        return dict(ts=ts, f=f, lo=f, hi=f, exact=True, kind="hourly — exact")
    ci = round(c)
    # KBOS/KMSP can send real tenths — but a value sitting exactly on a whole
    # Celsius step (62.6 = 17C) is treated as rounded. Safer to show a range.
    if stn in TENTHS_F and abs(c - ci) >= 0.05:
        f = c_to_f(c)
        return dict(ts=ts, f=f, lo=f, hi=f, exact=True, kind="exact")
    if abs(c - ci) < 0.05:
        f = c_to_f(ci)
        return dict(ts=ts, f=f, lo=c_to_f(ci - 0.5), hi=c_to_f(ci + 0.5),
                    exact=False, kind="5-min — rounded")
    f = c_to_f(c)                        # off the grid: suspect, not precise
    return dict(ts=ts, f=f, lo=f - 0.9, hi=f + 0.9, exact=False,
                kind="unverified")


def six_hour_max(ts, raw, day_start):
    """The 1sTTT group — an exact max over the previous 6 hours. Only used
    when that whole 6-hour window sits inside today's climate day."""
    if not raw or "RMK" not in raw:
        return None
    if ts - dt.timedelta(hours=6) < day_start - dt.timedelta(minutes=10):
        return None
    rmk = raw.split("RMK", 1)[1]
    for tok in rmk.split():
        mm = MAX6.fullmatch(tok)
        if mm:
            cc = int(mm.group(2)) / 10.0 * (-1 if mm.group(1) == "1" else 1)
            return c_to_f(cc)
    return None


def fetch_obs(stn, day_start, now_utc):
    """NWS observations for today. ⚠️ start + end + explicit limit + paging —
    start-only returns the OLDEST rows under a default cap (obs_live V2.1 bug:
    San Antonio's afternoon peak never came back)."""
    url = f"https://api.weather.gov/stations/{stn}/observations"
    params = {"start": day_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "end": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 500}
    feats, pages = [], 0
    while url and pages < 4:
        r = requests.get(url, params=params if pages == 0 else None,
                         headers=NWS_HDR, timeout=15)
        r.raise_for_status()
        body = r.json() or {}
        got = body.get("features") or []
        feats.extend(got)
        pages += 1
        url = (body.get("pagination") or {}).get("next")
        if not got:
            break
    rows, max6 = [], []
    for feat in feats:
        p = feat.get("properties", {})
        c = (p.get("temperature") or {}).get("value")
        tsr = p.get("timestamp")
        if c is None or not tsr:
            continue
        ts = dt.datetime.fromisoformat(tsr.replace("Z", "+00:00"))
        if ts < day_start:
            continue
        raw = p.get("rawMessage")
        rd = parse_reading(stn, ts, float(c), raw)
        rd["src"] = "NWS"
        rows.append(rd)
        m6 = six_hour_max(ts, raw, day_start)
        if m6 is not None:
            max6.append(m6)
    return rows, max6


# ── Fast feeds ───────────────────────────────────────────────────────────────
# NWS runs 5-15 min behind. These two close most of that gap:
#   Synoptic         — the 5-minute readings, usually within a few minutes
#   aviationweather  — the hourly report, usually within 1-2 min of :51-:53
def synoptic_token():
    for path in (("SYNOPTIC_TOKEN",), ("synoptic_token",), ("synoptic", "token")):
        try:
            v = st.secrets
            for k in path:
                v = v[k]
            if v:
                return str(v)
        except Exception:
            continue
    return None


def fetch_synoptic(stids, token):
    """One call, every station, last 3 hours. {stid: [(ts, °F)]} or None."""
    r = requests.get("https://api.synopticdata.com/v2/stations/timeseries",
                     params={"stid": ",".join(stids), "vars": "air_temp",
                             "recent": 180, "units": "english",
                             "obtimezone": "utc", "qc": "on", "token": token},
                     timeout=20)
    r.raise_for_status()
    data = r.json()
    if (data.get("SUMMARY") or {}).get("RESPONSE_CODE") != 1:
        return None
    out = {}
    for ent in data.get("STATION") or []:
        ob = ent.get("OBSERVATIONS") or {}
        times = ob.get("date_time") or []
        temps = ob.get("air_temp_set_1") or ob.get("air_temp_set_1d") or []
        rows = []
        for tsr, f in zip(times, temps):
            if f is None:
                continue
            try:
                rows.append((dt.datetime.fromisoformat(tsr.replace("Z", "+00:00")),
                             float(f)))
            except Exception:
                continue
        out[ent.get("STID")] = rows
    return out


def fetch_awc(stids):
    """Hourly reports straight from aviationweather.gov. {stid: [(ts, raw)]}."""
    r = requests.get("https://aviationweather.gov/api/data/metar",
                     params={"ids": ",".join(stids), "format": "json",
                             "hours": 30},
                     headers={"User-Agent": NWS_HDR["User-Agent"]}, timeout=15)
    r.raise_for_status()
    out = {}
    for m in r.json() or []:
        raw = m.get("rawOb") or ""
        ot = m.get("obsTime")
        try:
            ts = (dt.datetime.fromtimestamp(int(ot), pytz.utc) if ot is not None
                  else dt.datetime.fromisoformat(
                      str(m.get("reportTime")).replace("Z", "+00:00")
                      .replace(" ", "T")))
        except Exception:
            continue
        if ts.tzinfo is None:
            ts = pytz.utc.localize(ts)
        out.setdefault(m.get("icaoId"), []).append((ts, raw))
    return out


def merge_rows(stn, day_start, nws_rows, nws_max6, syn, awc):
    """One timeline. Same minute from two sources -> keep the exact one."""
    rows = list(nws_rows)
    max6 = list(nws_max6)
    for ts, f in (syn or []):
        if ts >= day_start:
            rd = parse_reading(stn, ts, (f - 32) / 1.8, "")
            rd["src"] = "Synoptic"
            rows.append(rd)
    for ts, raw in (awc or []):
        if ts < day_start or not TGROUP.search(raw or ""):
            continue
        rd = parse_reading(stn, ts, 0.0, raw)
        rd["src"] = "AWC"
        rows.append(rd)
        m6 = six_hour_max(ts, raw, day_start)
        if m6 is not None:
            max6.append(m6)
    best = {}
    for r in rows:
        k = r["ts"].replace(second=0, microsecond=0)
        cur = best.get(k)
        if cur is None or (r["exact"] and not cur["exact"]):
            best[k] = r
    out = sorted(best.values(), key=lambda x: x["ts"])
    return out, sorted(set(round(v, 1) for v in max6))


def summarize(rows, max6, now_utc):
    if not rows:
        return None
    last = rows[-1]
    feed_max = max(r["f"] for r in rows)
    exact_vals = [r["f"] for r in rows if r["exact"]] + list(max6)
    exact_max = max(exact_vals) if exact_vals else None
    # An exact reading (hourly tenths / 6-hr max) above the 5-min max wins.
    base = max(feed_max, exact_max) if exact_max is not None else feed_max
    max_row = max(rows, key=lambda r: (r["f"], r["ts"]))
    first_at_max = min(r["ts"] for r in rows if r["f"] >= feed_max - 0.01)
    true_lo = max([r["lo"] for r in rows] + ([exact_max] if exact_max else []))
    true_hi = max(max(r["hi"] for r in rows), true_lo)
    floor_settle = settle_round(true_lo)
    hour_ago = [r for r in rows if r["ts"] <= last["ts"] - dt.timedelta(minutes=55)]
    trend = last["f"] - hour_ago[-1]["f"] if hour_ago else None
    mins_since_max = (now_utc - first_at_max).total_seconds() / 60.0
    past_peak = (mins_since_max >= 60 and last["f"] <= feed_max - 1.5)
    prior = [r for r in rows[:-1] if not r["exact"]]
    stepped_up = bool(prior and not last["exact"]
                      and last["f"] - prior[-1]["f"] >= 1.5
                      and (now_utc - last["ts"]).total_seconds() <= 20 * 60)
    new_high = bool(len(rows) > 1 and last["f"] >= feed_max - 0.01
                    and last["f"] > max(r["f"] for r in rows[:-1]) + 0.01
                    and (now_utc - last["ts"]).total_seconds() <= 20 * 60)
    return dict(last=last, feed_max=feed_max, base=base, exact_max=exact_max,
                max_row=max_row, first_at_max=first_at_max, true_lo=true_lo,
                true_hi=true_hi, floor_settle=floor_settle, trend=trend,
                mins_since_max=mins_since_max, past_peak=past_peak,
                stepped_up=stepped_up, new_high=new_high,
                prev_f=(prior[-1]["f"] if prior else None),
                age_min=(now_utc - last["ts"]).total_seconds() / 60.0,
                n=len(rows))


# ── Official high: the NWS climate report (CLI) ─────────────────────────────
# The station's own daily max in whole °F — the number Kalshi settles on.
# Most offices post a preliminary "high so far" in the late afternoon and the
# final one after midnight. When it is posted, it replaces every range above.
CLI_LOC = {"KNYC": "NYC", "KMIA": "MIA", "KATL": "ATL", "KPHL": "PHL",
           "KDCA": "DCA", "KBOS": "BOS", "KMDW": "MDW", "KAUS": "AUS",
           "KDFW": "DFW", "KHOU": "HOU", "KOKC": "OKC", "KMSP": "MSP",
           "KSAT": "SAT", "KMSY": "MSY", "KDEN": "DEN", "KPHX": "PHX",
           "KLAS": "LAS", "KLAX": "LAX", "KSEA": "SEA", "KSFO": "SFO"}
MONTHS = {m: i for i, m in enumerate(
    ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
     "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"], 1)}
_CLI_CACHE = {}


def parse_cli(text, tz_name):
    """-> dict(date, max_f, prelim, asof_utc) or None."""
    t = (text or "").upper()
    m = re.search(r"CLIMATE SUMMARY FOR\s+([A-Z]+)\s+(\d{1,2})\s+(\d{4})", t)
    if not m or m.group(1) not in MONTHS:
        return None
    day = dt.date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
    mx = re.search(r"TEMPERATURE \(F\).*?MAXIMUM\s+(-?\d+)", t, re.S)
    if not mx:
        return None
    # ⚠️ Any "VALID ... AS OF" wording means partial. The overnight report says
    # "VALID AS OF 1259 AM" and carries a max of whatever it hit by midnight —
    # reading that as final locked San Antonio to 79 on a 94-degree day.
    prelim = "AS OF" in t
    asof_utc = None
    a = re.search(r"VALID(?:\s+TODAY)?\s+AS OF\s+(\d{1,2})(\d{2})\s*(AM|PM)", t)
    if a:
        hh = int(a.group(1)) % 12 + (12 if a.group(3) == "PM" else 0)
        tz = pytz.timezone(tz_name)
        # ⚠️ "VALID TODAY AS OF ... LOCAL TIME" is the local CLOCK time. Only
        # the observation-time column inside the report is standard time.
        # Shifting this by the DST hour put the as-of stamp in the future.
        naive = dt.datetime(day.year, day.month, day.day, hh, int(a.group(2)))
        asof_utc = tz.localize(naive).astimezone(pytz.utc)
    return dict(date=day, max_f=int(mx.group(1)), prelim=prelim,
                asof_utc=asof_utc)


def fetch_official(stn, tz_name, local_date):
    """Best CLI for today: read the recent products, keep the HIGHEST max for
    today's date (a later report can only be equal or warmer). Cached 5 min."""
    key = (stn, local_date)
    hit = _CLI_CACHE.get(key)
    if hit and (dt.datetime.now(pytz.utc) - hit[0]).total_seconds() < 300:
        return hit[1]
    loc = CLI_LOC.get(stn)
    best = None
    try:
        r = requests.get(
            f"https://api.weather.gov/products/types/CLI/locations/{loc}",
            headers=NWS_HDR, timeout=12)
        r.raise_for_status()
        for prod in (r.json().get("@graph") or [])[:4]:
            pr = requests.get(f"https://api.weather.gov/products/{prod['id']}",
                              headers=NWS_HDR, timeout=12)
            pr.raise_for_status()
            c = parse_cli(pr.json().get("productText", ""), tz_name)
            if not c or c["date"] != local_date:
                continue
            if best is None or c["max_f"] > best["max_f"]:
                best = c
    except Exception:
        best = None
    _CLI_CACHE[key] = (dt.datetime.now(pytz.utc), best)
    return best


def apply_official(s, off):
    """Fold the official high into the summary. Once it's posted and the
    station is past peak after the as-of time, the answer is locked."""
    s["official"] = off
    s["locked"] = False
    s["official_stale"] = False
    if not off:
        return s
    M = off["max_f"]
    # ⚠️ An official high BELOW what the station has already measured is a
    # partial report, not the answer. Never lock on it.
    if M < settle_round(s["true_lo"]):
        s["official_stale"] = True
        s["official"] = None
        return s
    s["floor_settle"] = max(s["floor_settle"], M)
    if not off["prelim"]:
        s["locked"] = True
    elif (s["past_peak"] and off["asof_utc"] is not None
          and off["asof_utc"] >= s["first_at_max"]):
        s["locked"] = True
    return s


# ── Settle odds from this account's own calibration ─────────────────────────
def fallback_diffs():
    nd = NormalDist(0.153, 0.60)
    return [min(1.9, max(-0.8, nd.inv_cdf((i + 0.5) / 200))) for i in range(200)]


@st.cache_data(ttl=3600, show_spinner=False)
def calibration_diffs():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        r = requests.get(f"{url}/rest/v1/max_vs_settled",
                         params={"select": "diff"},
                         headers={"apikey": key, "Authorization": "Bearer " + key},
                         timeout=10)
        r.raise_for_status()
        d = [float(x["diff"]) for x in r.json() if x.get("diff") is not None]
        if len(d) >= 30:
            return d, f"measured, n={len(d)}"
    except Exception:
        pass
    return fallback_diffs(), "fallback summary (Supabase unreachable)"


def settle_odds(s, diffs):
    """{whole degree: probability} for where CLI prints, IF the high is in."""
    if s.get("locked"):
        return {s["official"]["max_f"]: 1.0}
    counts = {}
    for d in diffs:
        k = max(settle_round(s["base"] + d), s["floor_settle"])
        counts[k] = counts.get(k, 0) + 1
    n = float(len(diffs))
    return {k: v / n for k, v in sorted(counts.items())}


# ── Kalshi ───────────────────────────────────────────────────────────────────
def cents(m, key):
    v = m.get(key + "_dollars")
    if v not in (None, ""):
        try:
            return int(round(float(v) * 100))
        except Exception:
            pass
    v = m.get(key)
    if v is not None:
        try:
            return int(v)
        except Exception:
            pass
    return None


def bracket_bounds(label):
    """Same parser as fetch_favorites.py."""
    if not label:
        return None, None
    s = label.replace("\u00b0", "").replace("deg", "").strip()
    low = s.lower()
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if not nums:
        return None, None
    if "below" in low or "under" in low:
        return None, nums[0]
    if "above" in low or "over" in low:
        return nums[0], None
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None, None


def fetch_ladder(series, local_date):
    et = series + "-" + local_date.strftime("%y%b%d").upper()
    r = requests.get(KALSHI, params={"event_ticker": et, "limit": 40},
                     headers={"Accept": "application/json"}, timeout=15)
    r.raise_for_status()
    out = []
    for m in r.json().get("markets", []):
        label = ""
        for f in ("yes_sub_title", "subtitle", "title"):
            label = (m.get(f) or "").replace("\u00b0", "").strip()
            if label:
                break
        lo, hi = bracket_bounds(label)
        out.append(dict(label=label, lo=lo, hi=hi, ask=cents(m, "yes_ask"),
                        bid=cents(m, "yes_bid")))
    out.sort(key=lambda b: (b["lo"] if b["lo"] is not None else -999))
    return out


def in_bracket(k, b):
    if b["lo"] is None and b["hi"] is None:
        return False
    return ((b["lo"] is None or k >= b["lo"]) and
            (b["hi"] is None or k <= b["hi"]))


def bracket_prob(b, odds):
    return sum(p for k, p in odds.items() if in_bracket(k, b))


def band_split(lo, hi):
    """Whole degrees a sensor band covers, and each one's share (uniform).
    A whole-°C reading is 1.8°F wide, so it always touches 2 numbers, often 3.
    e.g. 102.2 (39C) -> 101.3-103.1 -> 101 11% / 102 56% / 103 33%."""
    if hi - lo <= 1e-9:
        return [(settle_round(lo), 1.0)]
    out, k = [], settle_round(lo)
    while k - 0.5 < hi:
        ov = min(hi, k + 0.5) - max(lo, k - 0.5)
        if ov > 1e-9:
            out.append((k, ov / (hi - lo)))
        k += 1
    return out


def straddle(lo, hi, ladder):
    """[(bracket label, share)] for the Kalshi brackets a sensor band touches."""
    split = band_split(lo, hi)
    acc = {}
    for k, sh in split:
        lab = next((b["label"] for b in ladder if in_bracket(k, b)), f"{k}")
        acc[lab] = acc.get(lab, 0) + sh
    return split, list(acc.items())


def straddle_html(title, lo, hi, ladder):
    split, brs = straddle(lo, hi, ladder)
    nums = " · ".join(f"<b>{k}°</b> {p*100:.0f}%" for k, p in split)
    if len(brs) > 1:
        b_txt = " &nbsp;|&nbsp; ".join(f"<b>{l}</b> {p*100:.0f}%" for l, p in brs)
        tag = "<span style='color:#fbbf24'>STRADDLES</span>"
    else:
        b_txt = f"<b>{brs[0][0]}</b> only" if brs else "—"
        tag = "<span style='color:#00ff88'>ONE BRACKET</span>"
    return (f"<div class='card'><div class='lbl'>{title} · {tag}</div>"
            f"<div style='color:#fff;font-size:18px'>{b_txt}</div>"
            f"<div class='sm'>range {lo:.1f}–{hi:.1f}° covers {nums}</div></div>")


def is_dead(b, s):
    return b["hi"] is not None and b["hi"] < s["floor_settle"]


def exit_fee_cents(bid):
    p = bid / 100.0
    return FEE * p * (1 - p) * 100


def verdict(b, s, odds):
    """(css class, headline, reason) for someone holding YES on bracket b."""
    bid = b.get("bid")
    if is_dead(b, s):
        return ("v-sell", "DEAD — SELL FOR ANY BID",
                f"The high is already at least {s['floor_settle']}°, above "
                f"this bracket. It cannot win.")
    if s.get("locked"):
        won = in_bracket(s["official"]["max_f"], b)
        return (("v-hold", "WINNER — HOLD TO SETTLE",
                 f"Official high {s['official']['max_f']}° is in this bracket.")
                if won else
                ("v-sell", "LOST — SELL FOR ANY BID",
                 f"Official high {s['official']['max_f']}° is outside this bracket."))
    if not s["past_peak"]:
        if b["lo"] is not None and b["lo"] > s["true_hi"]:
            return ("v-wait", "STILL WARMING — NEEDS MORE HEAT",
                    f"High so far tops out around {s['true_hi']:.1f}°. This "
                    f"bracket needs {b['lo']}°+. Watch the next hour's trend.")
        return ("v-wait", "TOO EARLY TO CALL",
                "The station hasn't clearly peaked. Odds below assume the "
                "high is already in, and it may not be.")
    p = bracket_prob(b, odds) * 100
    if bid is None or bid <= 0:
        return ("v-hold" if p >= 50 else "v-wait",
                f"HOLD — MODEL {p:.0f}%" if p >= 50 else f"MODEL {p:.0f}%",
                "No bid on the book, so there is nothing to cash out into.")
    cash = bid - exit_fee_cents(bid)
    edge = p - cash
    why = f"Holding is worth ~{p:.0f}¢. Cashing out pays {cash:.0f}¢ after fee."
    if edge >= 5:
        return ("v-hold", f"HOLD  (+{edge:.0f}¢)", why)
    if edge <= -5:
        return ("v-sell", f"CASH OUT  ({edge:.0f}¢ to hold)", why)
    return ("v-wait", "CLOSE CALL — EITHER IS FINE", why)


# ── Fetch everything, in parallel, once a minute ─────────────────────────────
def load_city(city, now_utc, syn_all, awc_all):
    stn, series, tz = CITIES[city]
    day_start, local_date = climate_day_start(tz, now_utc)
    out = dict(city=city, stn=stn, tz=tz, rows=[], s=None, ladder=[],
               obs_err=None, k_err=None)
    try:
        try:
            nws_rows, nws_max6 = fetch_obs(stn, day_start, now_utc)
        except Exception as e:
            nws_rows, nws_max6 = [], []
            out["obs_err"] = "NWS: " + str(e)[:80]
        rows, max6 = merge_rows(stn, day_start, nws_rows, nws_max6,
                                (syn_all or {}).get(stn), (awc_all or {}).get(stn))
        out["rows"] = rows
        out["s"] = summarize(rows, max6, now_utc)
        if out["s"]:
            out["obs_err"] = None
            apply_official(out["s"], fetch_official(stn, tz, local_date))
    except Exception as e:
        out["obs_err"] = str(e)[:120]
    try:
        out["ladder"] = fetch_ladder(series, local_date)
    except Exception as e:
        out["k_err"] = str(e)[:120]
    return out


@st.cache_data(ttl=55, show_spinner=False)
def load_all(minute_key):
    now_utc = dt.datetime.now(pytz.utc)
    stids = [v[0] for v in CITIES.values()]
    feeds = {"Synoptic": "no token in secrets", "AWC": "ok"}
    syn_all, awc_all = None, None
    tok = synoptic_token()
    if tok:
        try:
            syn_all = fetch_synoptic(stids, tok)
            feeds["Synoptic"] = "ok" if syn_all is not None else "error"
        except Exception as e:
            feeds["Synoptic"] = "error: " + str(e)[:60]
    try:
        awc_all = fetch_awc(stids)
    except Exception as e:
        feeds["AWC"] = "error: " + str(e)[:60]
    with ThreadPoolExecutor(max_workers=10) as ex:
        res = list(ex.map(lambda c: load_city(c, now_utc, syn_all, awc_all),
                          CITIES))
    return {r["city"]: r for r in res}, now_utc, feeds


def local_hm(ts, tz):
    return ts.astimezone(pytz.timezone(tz)).strftime("%-I:%M%p").lower()


# ── Page ─────────────────────────────────────────────────────────────────────
try:
    st.page_link("streamlit_app.py", label="← Decision board")
except Exception:
    st.markdown("[← Decision board](/)")
st.markdown("## 🌡️ Live Desk")
st.caption("Settlement stations · °F only · refreshes every 60 seconds · "
           "public NWS data runs ~5-15 min behind the sensor")

view_mode = st.radio("View", ["📊 Mac", "📱 iPhone"], horizontal=True,
                     label_visibility="collapsed", key="desk_view")
is_mobile = view_mode == "📱 iPhone"
if is_mobile:
    st.markdown("<style>.big{font-size:28px}.card{padding:10px 12px}"
                ".verdict{font-size:22px}</style>", unsafe_allow_html=True)

if "desk_city" not in st.session_state:
    st.session_state["desk_city"] = "Miami"


@st.fragment(run_every=60)
def desk():
    minute_key = dt.datetime.now(pytz.utc).strftime("%Y%m%d%H%M")
    data, now_utc, feeds = load_all(minute_key)
    diffs, diff_src = calibration_diffs()

    city = st.selectbox("City", list(CITIES), key="desk_city")
    d = data[city]
    s = d["s"]

    if d["obs_err"] or not s:
        st.error(f"No readings for {d['stn']} yet today. {d['obs_err'] or ''}")
        return

    last = s["last"]
    tzl = pytz.timezone(d["tz"])
    nowl = now_utc.astimezone(tzl)
    nxt = nowl.replace(minute=51, second=0, microsecond=0)
    if nowl.minute >= 51:
        nxt += dt.timedelta(hours=1)
    mins_to = int((nxt - nowl).total_seconds() // 60)
    if s.get("new_high") and not s.get("locked"):
        st.markdown(
            f"<div class='verdict v-wait'>⬆ NEW HIGH — {last['f']:.1f}° at "
            f"{local_hm(last['ts'], d['tz'])}"
            f"<div class='v-why'>High is now {last['lo']:.1f}–{last['hi']:.1f}°. "
            f"Brackets above just got more likely.</div></div>",
            unsafe_allow_html=True)
    elif s.get("stepped_up") and not s.get("locked"):
        st.markdown(
            f"<div class='verdict v-wait'>⬆ STEPPED UP — {s['prev_f']:.1f}° → "
            f"{last['f']:.1f}° at {local_hm(last['ts'], d['tz'])}"
            f"<div class='v-why'>Now {last['lo']:.1f}–{last['hi']:.1f}°.</div></div>",
            unsafe_allow_html=True)
    st.markdown(
        f"<div class='sm'>⏱ next hourly report in <b style='color:#fff'>{mins_to} min</b> "
        f"({nxt.strftime('%-I:%M%p').lower()} local, exact reading usually posts "
        f"by :55) · latest reading via {last.get('src', 'NWS')}, "
        f"{s['age_min']:.0f} min old · feeds: Synoptic {feeds['Synoptic']}, "
        f"aviationweather {feeds['AWC']}</div>", unsafe_allow_html=True)
    odds = settle_odds(s, diffs)
    top = [kv for kv in sorted(odds.items(), key=lambda kv: -kv[1])[:3]
           if kv[1] >= 0.01]
    top_txt = " · ".join(f"<b>{k}°</b> {p*100:.0f}%" for k, p in top)

    now_rng = ("exact" if last["exact"]
               else f"really {last['lo']:.1f}–{last['hi']:.1f}°")
    trend = s["trend"]
    trend_txt = ("—" if trend is None else
                 f"{'▲' if trend > 0.4 else '▼' if trend < -0.4 else '▶'} "
                 f"{trend:+.1f}° last hour")
    peak_txt = ("PAST PEAK" if s["past_peak"] else "STILL IN PLAY")
    peak_col = "#00ff88" if s["past_peak"] else "#fbbf24"
    stale = s["age_min"] > 25

    off = s.get("official")
    if off:
        state = ("FINAL" if not off["prelim"] else
                 "LOCKED — this is the settle" if s["locked"] else
                 "can still go up if it warms again")
        asof = (f" · as of {local_hm(off['asof_utc'], d['tz'])} local"
                if off.get("asof_utc") else "")
        st.markdown(
            f"<div class='card' style='border:2px solid #00ff88'>"
            f"<div class='lbl'>Official high so far · NWS climate report{asof}</div>"
            f"<div class='big' style='color:#00ff88'>{off['max_f']}°F</div>"
            f"<div class='sm'>{state}</div></div>", unsafe_allow_html=True)
    else:
        note = ("Latest NWS report is an earlier partial one, already below "
                "what the station has measured — ignoring it until the "
                "afternoon report posts."
                if s.get("official_stale") else
                "Not posted yet — NWS usually posts it late afternoon. "
                "Until then, the ranges below are the best available.")
        st.markdown(
            f"<div class='card'><div class='lbl'>Official high so far</div>"
            f"<div class='sm'>{note}</div></div>", unsafe_allow_html=True)

    if is_mobile:
        r1 = st.columns(2)
        r2 = st.columns(2)
        c1, c2, c3, c4 = r1[0], r1[1], r2[0], r2[1]
    else:
        c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f"<div class='card'><div class='lbl'>Now · {local_hm(last['ts'], d['tz'])}"
        f" local{' · STALE' if stale else ''}</div>"
        f"<div class='big'>{last['f']:.0f}°</div>"
        f"<div class='sm'>{now_rng} · {trend_txt}</div></div>",
        unsafe_allow_html=True)
    c2.markdown(
        f"<div class='card'><div class='lbl'>High so far · set "
        f"{local_hm(s['first_at_max'], d['tz'])}</div>"
        f"<div class='big'>{s['base']:.0f}°</div>"
        f"<div class='sm'>true high {s['true_lo']:.1f}–{s['true_hi']:.1f}°</div></div>",
        unsafe_allow_html=True)
    c3.markdown(
        f"<div class='card'><div class='lbl'>Settles at least</div>"
        f"<div class='big'>{s['floor_settle']}°</div>"
        f"<div class='sm'>can only go up from here</div></div>",
        unsafe_allow_html=True)
    c4.markdown(
        f"<div class='card'><div class='lbl'>Peak</div>"
        f"<div class='big' style='color:{peak_col};font-size:26px'>{peak_txt}</div>"
        f"<div class='sm'>high set {s['mins_since_max']:.0f} min ago</div></div>",
        unsafe_allow_html=True)

    st.markdown(
        f"<div class='card'><span class='lbl'>If the high is in, CLI prints</span>"
        f"<br><span style='color:#fff;font-size:20px'>{top_txt}</span>"
        f"<div class='sm'>odds from {diff_src}"
        f"{'' if s['past_peak'] else ' · ⚠️ not past peak — high can still rise'}"
        f"</div></div>", unsafe_allow_html=True)

    # ── Celsius straddle: which brackets the readings can really be in ──
    ladder = d["ladder"]
    if not s.get("locked"):       # once the official high locks, ranges are moot
        sc1, sc2 = (st.container(), st.container()) if is_mobile else st.columns(2)
        sc1.markdown(straddle_html("High so far can be", s["true_lo"],
                                   s["true_hi"], ladder), unsafe_allow_html=True)
        sc2.markdown(straddle_html("Right now can be", last["lo"], last["hi"],
                                   ladder), unsafe_allow_html=True)

    # ── Position call ──
    if d["k_err"] or not ladder:
        st.warning(f"Kalshi ladder unavailable. {d['k_err'] or ''}")
    else:
        labels = ["—"] + [b["label"] for b in ladder]
        pos = st.selectbox("I'm holding YES on…", labels, key=f"pos_{city}")
        if pos != "—":
            b = next(x for x in ladder if x["label"] == pos)
            cls, head, why = verdict(b, s, odds)
            st.markdown(f"<div class='verdict {cls}'>{head}"
                        f"<div class='v-why'>{why}</div></div>",
                        unsafe_allow_html=True)

        rows = []
        for b in ladder:
            p = bracket_prob(b, odds) * 100
            dead = is_dead(b, s)
            rows.append({
                "Bracket": b["label"],
                "Ask": f"{b['ask']}¢" if b["ask"] else "—",
                "Bid": f"{b['bid']}¢" if b["bid"] else "—",
                "Model": "DEAD" if dead else f"{p:.0f}%",
                "Model − Ask": ("" if dead or not b["ask"] or not s["past_peak"]
                                else f"{p - b['ask']:+.0f}"),
            })
        if is_mobile:
            rows = [{k: r[k] for k in ("Bracket", "Bid", "Model")} for r in rows]
        st.dataframe(rows, hide_index=True)
        if not s["past_peak"]:
            st.caption("Model − Ask is hidden until the station is past peak.")

    # ── Recent readings ──
    with st.expander(f"Recent readings at {d['stn']} ({s['n']} today)"):
        rec = []
        for r in reversed(d["rows"][-15:]):
            rec.append({"Local": local_hm(r["ts"], d["tz"]),
                        "°F": f"{r['f']:.1f}",
                        "True temp": ("exact" if r["exact"]
                                      else f"{r['lo']:.1f}–{r['hi']:.1f}"),
                        "Type": r["kind"], "Source": r.get("src", "")})
        st.dataframe(rec, hide_index=True)
        if s["exact_max"]:
            st.caption(f"Highest exact reading today: {s['exact_max']:.1f}°")

    # ── Every city at a glance ──
    st.markdown("#### All cities")
    board = []
    for c in CITIES:
        dd = data[c]
        ss = dd["s"]
        if not ss:
            board.append({"City": c, "Now": "—", "High": "—", "At least": "—",
                          "Peak": "no data", "Favorite": "—"})
            continue
        fav = max((b for b in dd["ladder"] if b["ask"]),
                  key=lambda b: b["ask"], default=None)
        board.append({
            "City": c,
            "Now": f"{ss['last']['f']:.0f}°",
            "High": f"{ss['base']:.0f}° ({ss['true_lo']:.0f}–{ss['true_hi']:.0f})",
            "Official": (f"{ss['official']['max_f']}°" if ss.get("official")
                         else "—"),
            "At least": f"{ss['floor_settle']}°",
            "Peak": "past" if ss["past_peak"] else "in play",
            "Favorite": f"{fav['label']} @ {fav['ask']}¢" if fav else "—",
            "Updated": f"{ss['age_min']:.0f}m ago",
        })
    if is_mobile:
        board = [{k: r.get(k, "—") for k in ("City", "Now", "Official", "Peak")}
                 for r in board]
    st.dataframe(board, hide_index=True)
    st.caption(f"Last refresh {now_utc.astimezone(pytz.timezone('America/New_York')).strftime('%-I:%M:%S %p')} ET")


desk()
