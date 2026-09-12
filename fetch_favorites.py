"""
fetch_favorites.py — FAV V1.5. Buy the market's own favorite, in band, at set times.

WHAT THIS IS
============
No forecasting. Never fetches NWS, GFS, or NBM. No consensus, no sigma, no
trust score, no opinion about tomorrow's temperature.

    1. pull the Kalshi ladder
    2. take the highest-priced bracket (the market's own favorite)
    3. if the ask is inside the band, log it
    4. next day, settle against Kalshi's own result

That is the entire strategy. It is the automated version of the Daily Capture
Grid — same three times, same bands, same pick.

V1.5 CHANGES (2026-09-11) — SIX COLUMNS, NO CHANGE TO ANY BET
==============================================================
Six fields, all from the SAME markets call already being made. Nothing is
filtered on any of them. Every one exists because a question was asked this
week that the stored data could not answer.

  yes_bid_cents / spread_cents
      Only the ask was stored, so a 70c favorite quoted 69/70 and one quoted
      58/70 were recorded as the same number. They are not the same market.

      ⚠️ SPREAD IS NOT SLIPPAGE. This file is a PAPER logger — it sends no
      order, so there is no fill to compare against. A wide spread is a hint
      that a real fill might land badly; it is not a measurement of one.
      Slippage still requires real orders and Kalshi credentials.

  volume
      Whether the edge sits in thin markets or liquid ones. Volume is heavily
      city-correlated, and per-city selection has failed FOUR times, so any
      result here needs a city control before it means anything.

  third_bracket / third_ask / top3_share
      gap_cents was supposed to answer "is this a dominant favorite" and
      cannot. Measured 2026-09-11 on the 44 settled bets that carry it, the
      favorite's lead over rank 2 ranged 16 to 67 cents and NEVER fell below
      16 — the band selects out narrow favorites, so there is no low-dominance
      population to compare against. Bucketing it gave 50.0 / 80.0 / 63.3 with
      n of 4 / 10 / 30: no gradient, middle bucket best, i.e. noise.
      top3_share measures ladder concentration instead, which gap_cents
      structurally cannot see.

⚠️ WHAT THESE COLUMNS CANNOT FIX. The binding constraint is sample, not
fields. As of 2026-09-11 the record is 126 settled bets over six days, and
the V1.2 columns (gap_cents, bracket bounds) exist on only 41-44 of them.
Every structural hypothesis tested this week returned "not enough data":

    price bands       pooled showed +8.4 margin at 65-69c; the by-day split
                      put $18.59 of the $23.09 on a single Tuesday
    dominance         no usable variation (see above)
    bracket width     ZERO variation — every parsed bracket is 2F wide, so
                      the hypothesis has no dimension to test on
    per-city          seven cities at exactly 100%, five at 33% or below,
                      nothing in between: the signature of tiny cells

These columns are cheap and they start accumulating immediately. They are
not expected to produce an answer for weeks. Resist re-running these cuts
every evening — the analysis is what generates false positives, the waiting
is what generates the sample.

V1.4 CHANGES (2026-09-10) — TWO COLLECTORS, NO CHANGE TO ANY BET
=================================================================
Nothing in this version alters which bets are placed. Both additions are
instrumentation for questions that could not be answered from stored data.

1. SNAPSHOTS — the hour-of-day question, and the price-free baseline.

   Two questions came up that the existing tables cannot answer:

     (a) Is there a better entry time between 12:00 and 16:00? The record has
         only three points (10:30 / 12:00 / 16:00) because those are the only
         times a bet was ever logged.

     (b) How often does the market's favorite win REGARDLESS of price? The
         only figure available is 74.3% on 626 city-days from the consensus
         comparison, pooled, with no split by hour or price.

   `kalshi_candles` was the obvious place to look and it is USELESS for this.
   Checked 2026-09-09: 1,207,943 rows, Aug 2 - Sep 3, 24 cities — and ZERO
   rows with yes_ask between 30 and 80. Every captured price is a market
   already resolved to near-0 or near-100. The interior brackets that carried
   the action were never captured. Do not go back to that table.

   So V1.4 captures forward. At each SNAPSHOT_HOUR it writes the favorite and
   runner-up for all 20 cities to `favorites_snapshots` with:
     - NO BAND FILTER. Every favorite is recorded at any price. That is what
       makes (b) answerable — win rate by price decile across the whole range,
       not just 58-79.
     - NO BET. Nothing is staked, nothing enters favorites_bets, and none of
       this pools with the FAV V1 record. Separate table, separate analysis.

   ⚠️ SNAPSHOTS ARE NOT BETS. Never UNION these tables. The standing rule is
   never pool strategy tags, and this is a different instrument entirely — it
   has no stake, no fee, and no band.

   Sample math, so expectations are calibrated: 20 cities x 5 snapshot hours is
   100 rows/day. The hour-of-day question (~600 rows/hour) is answerable in
   about three weeks. A per-city-per-hour breakdown is 100 cells and needs
   months — collect at fine grain, analyse pooled first.

2. OBS AT ENTRY — how far the temperature has to travel.

   When a bet is logged at 12:00 on a 97-98 bracket, is the station currently
   at 85 or at 95? Nothing recorded that. V1.4 reads obs_live at write time
   and stores:

       temp_at_entry        current 5-min feed reading
       day_max_at_entry     running max so far today
       degrees_to_lo        bracket_lo - day_max_at_entry
       obs_age_min_at_entry so a stale obs row can be excluded later

   ⚠️ THESE ARE COLUMNS, NOT A FILTER. No bet is skipped for being far from
   its bracket. The point is to have the data in three months to test whether
   distance predicts anything. Filtering on it now, on 126 settled bets, is
   how the last four filters died.

   ⚠️ obs_live CAN BE STALE. On 2026-09-09 it froze at 08:24 local and served
   an 81.0F reading until 17:39 while San Antonio was at 98.6. The row is
   written with whatever obs_live holds plus its age; ALWAYS filter on
   obs_age_min_at_entry before using these fields.

WHAT IS NOT HERE, AND WHY
=========================
SLIPPAGE / ACTUAL FILL PRICE — cannot be built into this file.

    FAV V1 is a PAPER logger. It never sends an order; it writes a row at the
    displayed ask and settles against Kalshi's `result` field. There is no fill
    to capture because there is no trade.

    Slippage is real and it is the biggest unmeasured risk to the record — a
    displayed 58c showed as 61c on a manual fill, and the whole edge is 2-3
    points. But measuring it requires REAL orders and Kalshi API credentials
    (portfolio/fills), neither of which this script has. Until then it has to
    be logged by hand off the manual trades: displayed ask at click, actual
    average fill from the confirmation. Ten pairs is enough.

V1.3 CHANGES (2026-09-08) — THE FEE WAS WRONG BY MORE THAN HALF
================================================================
Every net figure produced before this version used FEE_CENTS = 3.6, a flat
per-contract number backed out from a single ticket. It was wrong in BOTH its
value and its shape.

Kalshi's published schedule:

    fee = round_up_to_cent( 0.07 * C * P * (1-P) )
        C = contracts, P = price in DOLLARS

CONFIRMED against a real fill: Las Vegas 95-96, Sep 7, 30 contracts at 65c.
Formula gives $19.50 + $0.48 = $19.98. The ticket read $19.99. 1.60c per
contract, not 3.6c.

Two things the flat figure got wrong:

  1. IT IS A PARABOLA, not a rate. P*(1-P) peaks at 50c and collapses toward
     the extremes. Across the band: 58c pays 1.71c, 69c pays 1.50c, 79c pays
     1.16c. At 99c it is 0.07c — which is why selling a winner at 99 costs
     almost nothing in fees (it costs you the last cent of value instead).

  2. THERE IS NO SETTLEMENT FEE. A winner held to resolution pays ONLY the
     entry fee. Selling before settlement pays the fee a second time. FAV V1
     holds to settlement, so it pays once. The 3.6c figure was most likely a
     round-trip number from a manual trade that was sold.

WHAT THAT DOES TO THE RECORD (97 settled bets, 4 days):

    window      n    win%   old BE   NEW BE    old net    NEW net
    AFTERNOON  30    73.3    72.5     70.4     +$2.83     +$7.30
    MIDDAY     40    67.5    66.5     64.5     +$3.38     +$9.52
    MORNING    27    63.0    65.8     63.9     -$7.09     -$2.94
    TOTAL      97    68.0    68.2     66.1     -$0.89    +$13.88

The strategy has been profitable the entire time. A bad fee estimate was
hiding it. Afternoon and midday clear break-even by about three points each;
morning is still marginally under.

⚠️ THE KILL LINE CHANGED SHAPE, NOT JUST VALUE.
The old line — "below 66% win rate at n=50, retire the window" — was set before
the fee curve was known and is wrong in both directions. Break-even is a
function of ENTRY PRICE:

    58c -> 59.7%    62c -> 63.7%    69c -> 70.5%    79c -> 80.2%

A 79c afternoon bet winning 75% of the time is LOSING. A 58c morning bet
winning 62% is WINNING. One threshold cannot express that.

    NEW KILL LINE: a window retires when its win rate sits below its OWN
    average break-even (avg ask + fee at that ask) at n=50 settled.

Current standing: afternoon +2.9 points, midday +3.0, morning -0.9.

V1.2 CHANGES (2026-09-08)
=========================
REMOVED: the consensus agreement tag added in V1.1.

    It read settlements.consensus and stored whether the weather model's
    temperature fell inside the market's bracket. Backtest said AGREE won 80%
    vs 76% overall, and turned midday and afternoon from negative to positive.

    First full day of live data: 4 AGREE, 20 DISAGREE. Inspecting the
    afternoon disagreements showed what it was actually measuring:

        Austin       100-101   consensus 99.20   short 0.80
        Denver        86-87    consensus 85.20   short 0.80
        San Antonio   99-100   consensus 98.50   short 0.50
        New York      80-81    consensus 81.50   over  0.50
        Minneapolis   83-84    consensus 81.60   short 1.40
        Las Vegas    100-101   consensus 97.50   short 2.50

    Five of six within 1.4F, four within 0.8F, and FIVE OF SIX with consensus
    running LOW. That is not the model disagreeing with the market. That is the
    model's documented cold bias landing a fraction under a bracket edge, and
    strict containment calling it conflict.

    The tag was measuring bias, not agreement. Removed rather than retuned —
    the backtest was run at strict containment, so changing the tolerance would
    have invalidated the comparison anyway.

    The consensus_f and agrees_with_consensus columns are left in the table
    with the 09-07/09-08 rows in them. Nothing reads them.

ADDED: the runner-up bracket and its price.

    Not a filter and not an improvement — it makes the one directional finding
    in the data testable. Measured on 97 settled bets:

        losses where settlement came in ABOVE the bracket : 18
        losses where settlement came in BELOW the bracket :  7
        average miss                                      : 1.39F

    72/28 split, and the misses are small — most losses are ONE BRACKET HIGH,
    not a blown call. This matches the market's favorite running -0.44F vs
    actual across 72 days in the older snapshot data.

    But with only the favorite stored, there is no way to tell whether the
    settlement landed on the ADJACENT bracket or somewhere else. Storing
    rank 2 answers that directly: if losses consistently settle in the
    runner-up bracket AND the runner-up is consistently the one above, that is
    a systematic mispricing. If they scatter, it is noise.

    Also stored: bracket_lo / bracket_hi parsed at write time, so miss analysis
    is a subtraction instead of a regex over label strings, and gap_cents (the
    favorite's lead over rank 2) which has never been tested forward.

    ⚠️ NONE OF THIS CHANGES A BET. Every in-band favorite is still taken,
    exactly as before. These are columns, not logic.

WHERE THE STRATEGY STOOD UNDER THE OLD (WRONG) FEE — kept for the record
========================================================================
        window      n    win%    break-even    net
        AFTERNOON  30    73.3      72.5      +$2.83
        MIDDAY     40    67.5      66.5      +$3.38
        MORNING    27    63.0      65.8      -$7.09
        TOTAL      97    68.0      68.2      -$0.89

Gross +$26.32, net -$0.89. Sixty-eight percent of bets win and the fee takes
all of it. Afternoon and midday clear break-even by under a point; morning is
below it.

Day by day: +0.87, +21.65, -13.75, -9.64. The entire four-day P&L is one day.
Three of four days finished under the 66% break-even.

⚠️ Those are the OLD numbers, computed with the wrong fee. See the V1.3 block
at the top for the corrected table: net is +$13.88, not -$0.89.

⚠️ KILL LINE: a window retires when its win rate sits below its OWN average
break-even at n=50 settled. Morning is at 63.0% against a 63.9% break-even on
n=27 — under the line but not yet at the sample size that would retire it.

BAND DEFINITIONS
================
FLOOR = 58, not 60, not 55. Measured 10am-1pm ET:
    55-57c   n=246   53.7%   -2.38
    58-59c   n=174   64.9%   +6.44
    60-64c   n=392   68.6%   +6.69
58-59 behaves like 60-64; 55-57 is a different population and loses. Confirmed
independently at 16:00 ET. Forward data agrees there is no ordering within the
band — the price tiers scatter with no monotonic trend, which is what you get
if price inside the band carries no information.

⚠️ THE 55-57 TEST USED THE WRONG FEE. It was run when the fee was believed to
be a flat 3.6c. At 56c the true fee is 1.72c, so break-even is ~57.7% and not
the ~59.6% the old figure implied — a two-point handicap that test never
removed. -2.38 at n=246 may be closer to flat than to losing. NOT a reason to
reopen the floor; it IS a reason to re-run the band analysis against
break_even_pct() before treating 58 as settled. The snapshots collected by
V1.4 record every price with no band filter, which is the clean way to redo it.

The floor survived two further tests. City-days where the favorite NEVER
cleared 58c all day (Seattle 59.4% of days, SF 46.9%, Denver 42.9%) scored
45.3% at 47.8c ask, -6.11 net. Persistent cheapness is correctly priced.

CEILING = 69 morning and midday, 79 afternoon. 71-72 looked good at +8.62 but
sits between +0.97 and -2.20 on cells of 31-45 — noise, not a ceiling worth
extending. The 70-79 extension was tested across all 19 cities and CLOSED:
n=90, ~44% win against a ~75.5% break-even, roughly -27c/contract.

NEVER an open-ended floor. "58 and over" drags in the 80+ tier, which wins
85-96% and pays nothing at 87-94c.

NO BETS ARE STRUCTURALLY DEAD. Every cell with n>100 is negative across all
ranks and price bands. At ranks 4-12 in the 80+ band the break-even required
exceeds 100% after fees — arithmetic impossibility, not a bad bet.

TIMES
=====
58-69 band by half hour (n=89-101 each):
    08:00 +1.54   08:30 +0.92   09:00 +0.73   09:30 +7.61
    10:00 +4.73   10:30 +18.45  11:00 +13.77  11:30 +11.04
The count of qualifying picks is FLAT across all of these — the same picks are
available at 8am and are simply wrong more often. Do not add an earlier window.

⚠️ Those figures are from the exact-minute filter and are STALE. Corrected with
ladder reconstruction (626 city-days): MORNING 58-69 ran 77.4% / +10.87 in
August and 74.3% / +8.14 in September. Read the corrected numbers.

⚠️ A 17:00 window was considered and not built. By 4pm many cities are already
above 79c; by 5pm more are. It would likely produce FEWER qualifying bets, not
more, and the +10.41 figure for that hour carries the same ladder bug.
V1.4 SNAPSHOTS a 17:00 reading without betting it, which settles the question
from data rather than argument.

⚠️ DST. Windows are EASTERN LOCAL TIME through pytz. From early November the
same ET times are UTC-5. Hardcoding UTC would silently shift every window.

⚠️ SEASONALITY. Every band figure is from summer days. Summer highs are
solar-driven and predictable; winter highs are driven by frontal timing. Re-run
the band analysis each season.

THE LATCH
=========
Scheduled runs start LATE and never early, so a symmetric tolerance discards
the entire late half of a distribution that is entirely late — silently.

⚠️ THE REAL CAUSE WAS WORSE THAN QUEUE DELAY. GitHub delayed this repo's
scheduled runs by ~3 HOURS on 2026-09-03. No latch width fixes that. Scheduling
runs through cron-job.org -> workflow_dispatch, America/New_York. favorites.yml
has `workflow_dispatch` ONLY. Do not re-add `on: schedule:`.

The latch stays because it is correct for ordinary delay and makes a stray
manual dispatch a safe no-op. minutes_late is stored on every row — it has read
0 on all four days so far. A window that already has rows for today is skipped.

CITIES
======
No forecast means no per-city calibration. Seattle and San Francisco are fine
here despite being dropped from the weather model — this never forecasts them.

EXCLUDED: San Diego (KXHIGHTSAN, 11.8% win at 90.4c — not a real market's
behavior), Louisville (KXHIGHTSDF, 98c average with a 14.3% win rate on 7
city-days), Trenton and Newark (~150 candle rows total).

⚠️ NO PER-CITY FILTER IS SUPPORTED. City selection has failed four separate
tests now: the tier-3 scan, lock-in timing, persistent uncertainty, and the
per-city breakdown of this table. The band is the edge.

⚠️ THE PER-CITY TABLE LOOKS COMPELLING AND IS NOT. Run 2026-09-09 on 126
settled bets: seven cities at exactly 100% (Minneapolis 3/3, DC 5/5, Denver
8/8, LA 4/4, Vegas 5/5, Atlanta 5/5, Houston 2/2) and five at 33% or below
(Boston, SF, Miami 1/3, New Orleans 2/9, Austin 0/6). Almost nothing in
between. That bimodal shape with an empty middle is the signature of small
samples, not skill — and every city's average ask sits between 62 and 69c, so
the market prices them all the same. Denver's 8 bets are ~4 city-days, since
the three windows on one day are not independent.

SETTLEMENT
==========
Against KALSHI'S OWN `result` field, not Iowa CLI. We are scoring a contract,
not a temperature, so any bracket-boundary disagreement between our arithmetic
and Kalshi's settlement is removed entirely.

CREATE THE TABLE ONCE:

  CREATE TABLE IF NOT EXISTS public.favorites_bets (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL, city TEXT NOT NULL, series TEXT,
    event_ticker TEXT, market_ticker TEXT, bracket TEXT,
    window_label TEXT NOT NULL, yes_ask_cents INTEGER,
    sigma_p NUMERIC, n_brackets INTEGER, strategy_tag TEXT NOT NULL,
    amount NUMERIC NOT NULL DEFAULT 5.0,
    result TEXT NOT NULL DEFAULT 'Pending',
    profit NUMERIC, settled_at TIMESTAMPTZ,
    placed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, city, window_label)
  );
  ALTER TABLE public.favorites_bets ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "Allow all access" ON public.favorites_bets
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);
  CREATE INDEX IF NOT EXISTS idx_fav_date ON public.favorites_bets (date);
  CREATE INDEX IF NOT EXISTS idx_fav_tag ON public.favorites_bets (strategy_tag);

RUN THIS BEFORE DEPLOYING V1.2 (safe to re-run):

  ALTER TABLE public.favorites_bets
    ADD COLUMN IF NOT EXISTS minutes_late   INTEGER,
    ADD COLUMN IF NOT EXISTS fee_dollars    NUMERIC(8,4),
    ADD COLUMN IF NOT EXISTS net_profit     NUMERIC(10,4),
    ADD COLUMN IF NOT EXISTS bracket_lo     INTEGER,
    ADD COLUMN IF NOT EXISTS bracket_hi     INTEGER,
    ADD COLUMN IF NOT EXISTS runner_bracket TEXT,
    ADD COLUMN IF NOT EXISTS runner_ask     INTEGER,
    ADD COLUMN IF NOT EXISTS runner_lo      INTEGER,
    ADD COLUMN IF NOT EXISTS runner_hi      INTEGER,
    ADD COLUMN IF NOT EXISTS gap_cents      INTEGER;

⚠️ RUN THIS BEFORE DEPLOYING V1.4 (safe to re-run). Without it every insert
returns PGRST204 "Could not find the column" and writes ZERO rows — that
exact failure cost a full day of obs_live on 2026-09-09.

  ALTER TABLE public.favorites_bets
    ADD COLUMN IF NOT EXISTS temp_at_entry        NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS day_max_at_entry     NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS degrees_to_lo        NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS obs_age_min_at_entry NUMERIC(6,1);

  CREATE TABLE IF NOT EXISTS public.favorites_snapshots (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    city TEXT NOT NULL,
    snap_label TEXT NOT NULL,
    series TEXT,
    event_ticker TEXT,
    market_ticker TEXT,
    bracket TEXT,
    bracket_lo INTEGER,
    bracket_hi INTEGER,
    yes_ask_cents INTEGER,
    runner_bracket TEXT,
    runner_ask INTEGER,
    runner_lo INTEGER,
    runner_hi INTEGER,
    gap_cents INTEGER,
    sigma_p NUMERIC,
    n_brackets INTEGER,
    in_band BOOLEAN,
    temp_at_snap NUMERIC(6,2),
    day_max_at_snap NUMERIC(6,2),
    degrees_to_lo NUMERIC(6,2),
    obs_age_min_at_snap NUMERIC(6,1),
    result TEXT NOT NULL DEFAULT 'Pending',
    settled_at TIMESTAMPTZ,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    minutes_late INTEGER,
    UNIQUE (date, city, snap_label)
  );
  ALTER TABLE public.favorites_snapshots ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "Allow all access" ON public.favorites_snapshots
    FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);
  CREATE INDEX IF NOT EXISTS idx_snap_date ON public.favorites_snapshots (date);
  CREATE INDEX IF NOT EXISTS idx_snap_label ON public.favorites_snapshots (snap_label);

⚠️ RUN THIS BEFORE DEPLOYING V1.5 (safe to re-run). Both tables. Skip it and
every insert returns PGRST204 and writes ZERO rows — that exact failure cost
a full day of obs_live on 2026-09-09 and a full evening of favorites_snapshots
on 2026-09-10.

  ALTER TABLE public.favorites_bets
    ADD COLUMN IF NOT EXISTS yes_bid_cents  INTEGER,
    ADD COLUMN IF NOT EXISTS spread_cents   INTEGER,
    ADD COLUMN IF NOT EXISTS volume         INTEGER,
    ADD COLUMN IF NOT EXISTS third_bracket  TEXT,
    ADD COLUMN IF NOT EXISTS third_ask      INTEGER,
    ADD COLUMN IF NOT EXISTS top3_share     NUMERIC(6,4);

  ALTER TABLE public.favorites_snapshots
    ADD COLUMN IF NOT EXISTS yes_bid_cents  INTEGER,
    ADD COLUMN IF NOT EXISTS spread_cents   INTEGER,
    ADD COLUMN IF NOT EXISTS volume         INTEGER,
    ADD COLUMN IF NOT EXISTS third_bracket  TEXT,
    ADD COLUMN IF NOT EXISTS third_ask      INTEGER,
    ADD COLUMN IF NOT EXISTS top3_share     NUMERIC(6,4);

THE QUERY V1.2 EXISTS FOR (run at ~60 settled losses):

  -- when the favorite loses, does the settlement land on the runner-up?
  select f.window_label,
         case when f.runner_lo > f.bracket_lo then 'runner is ABOVE'
              when f.runner_lo < f.bracket_lo then 'runner is BELOW'
              else 'same/unparsed' end as runner_side,
         count(*) n,
         count(*) filter (where s.actual::numeric between f.runner_lo and f.runner_hi)
           as settled_in_runner,
         round(avg(f.gap_cents),1) avg_gap
  from favorites_bets f
  join settlements s on s.city = f.city and s.date = f.date::text
  where f.result = 'Lost' and s.actual is not null and f.runner_lo is not null
  group by f.window_label, runner_side
  order by f.window_label, runner_side;

THE QUERIES V1.4 EXISTS FOR (run at ~3 weeks of snapshots):

  -- (a) is there a better hour? in-band only, so it is comparable to the bets
  select snap_label, count(*) n,
         round(avg(yes_ask_cents),1) avg_ask,
         round(100.0*avg((result='Won')::int),1) win_pct,
         round(100.0*avg((result='Won')::int)
               - avg(yes_ask_cents
                 + 0.07*yes_ask_cents*(100-yes_ask_cents)/100.0),1) margin
  from favorites_snapshots
  where result in ('Won','Lost') and in_band
  group by snap_label order by snap_label;

  -- (b) the price-free baseline: how good is the favorite at ANY price?
  select width_bucket(yes_ask_cents, 30, 100, 7)*10+30 as price_bucket,
         count(*) n,
         round(100.0*avg((result='Won')::int),1) win_pct,
         round(avg(yes_ask_cents
               + 0.07*yes_ask_cents*(100-yes_ask_cents)/100.0),1) break_even
  from favorites_snapshots
  where result in ('Won','Lost')
  group by 1 order by 1;

  ⚠️ Read n before any percentage in either. Most cells will be thin for weeks.

Secrets: SUPABASE_URL, and SUPABASE_SERVICE_KEY or SUPABASE_KEY.
No Kalshi credentials needed — the markets endpoint is public.
"""

import math
import os
import re
import time
import requests
import datetime as dt

import pytz

SB_URL = os.environ["SUPABASE_URL"].rstrip("/")
SB_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]

KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets"
HEADERS = {"User-Agent": "kalshi-favorites/1.5", "Accept": "application/json"}

ET = pytz.timezone("America/New_York")

STAKE = 5.0
TAG_PREFIX = "FAV_V1"

# Kalshi's PUBLISHED fee formula, confirmed against a real ticket 2026-09-08:
#
#     fee = round_up_to_cent( 0.07 * C * P * (1-P) )
#         C = number of contracts, P = price in DOLLARS (65c = 0.65)
#
# It is a parabola peaking at 50c and falling to nearly nothing at the extremes.
# It is NOT a flat rate, and the 3.6c figure used through 2026-09-08 was wrong
# by more than half. See the FEE section in the docstring.
FEE_COEFFICIENT = 0.07

# Windows in EASTERN LOCAL TIME. pytz resolves DST.
#   (hour, minute, label, band_low_cents, band_high_cents_exclusive)
WINDOWS = [
    (10, 30, "MORNING",   58, 70),
    (12,  0, "MIDDAY",    58, 70),
    (16,  0, "AFTERNOON", 58, 80),
]

# ⚠️ SNAPSHOT HOURS PLACE NO BETS. They fill the gaps between the three betting
# windows so the hour-of-day question can be answered from data instead of
# argument. 17:00 is included precisely because a 17:00 WINDOW was rejected on
# reasoning alone — this measures it without risking anything.
#   (hour, minute, label)
SNAPSHOT_HOURS = [
    (11, 0, "T1100"),
    (13, 0, "T1300"),
    (14, 0, "T1400"),
    (15, 0, "T1500"),
    (17, 0, "T1700"),
]

# Band used ONLY to tag snapshots as in_band for comparison with the bets.
# It filters nothing — every favorite is recorded at every price.
SNAP_BAND_LO, SNAP_BAND_HI = 58, 80

# One-sided latch, minutes AFTER the target. Delay is always late, never early.
WINDOW_LATCH_MIN = 25

SERIES = {
    "New York":       "KXHIGHNY",
    "Miami":          "KXHIGHMIA",
    "Atlanta":        "KXHIGHTATL",
    "Philadelphia":   "KXHIGHPHIL",
    "Washington DC":  "KXHIGHTDC",
    "Boston":         "KXHIGHTBOS",
    "Chicago":        "KXHIGHCHI",
    "Austin":         "KXHIGHAUS",
    "Dallas":         "KXHIGHTDAL",
    "Houston":        "KXHIGHTHOU",
    "Oklahoma City":  "KXHIGHTOKC",
    "Minneapolis":    "KXHIGHTMIN",
    "San Antonio":    "KXHIGHTSATX",
    "New Orleans":    "KXHIGHTNOLA",
    "Los Angeles":    "KXHIGHLAX",
    "Phoenix":        "KXHIGHTPHX",
    "Seattle":        "KXHIGHTSEA",
    "San Francisco":  "KXHIGHTSFO",
    "Denver":         "KXHIGHDEN",
    "Las Vegas":      "KXHIGHTLV",
}


# ── Supabase ─────────────────────────────────────────────────────────────────
def sb_headers(prefer="return=representation"):
    return {
        "apikey": SB_KEY,
        "Authorization": "Bearer " + SB_KEY,
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


def sb_url(table):
    return f"{SB_URL}/rest/v1/{table}"


def insert_bet(row):
    """UNIQUE (date, city, window_label) makes a repeat run a no-op."""
    try:
        r = requests.post(
            sb_url("favorites_bets") + "?on_conflict=date,city,window_label",
            headers=sb_headers("return=minimal,resolution=ignore-duplicates"),
            json=row, timeout=15)
        if r.status_code not in (200, 201, 204):
            print(f"    insert HTTP {r.status_code}: {r.text[:140]}")
            return False
        return True
    except Exception as e:
        print(f"    insert failed: {type(e).__name__}: {str(e)[:120]}")
        return False


def insert_snapshot(row):
    """UNIQUE (date, city, snap_label) makes a repeat run a no-op."""
    try:
        r = requests.post(
            sb_url("favorites_snapshots") + "?on_conflict=date,city,snap_label",
            headers=sb_headers("return=minimal,resolution=ignore-duplicates"),
            json=row, timeout=15)
        if r.status_code not in (200, 201, 204):
            print(f"    snap insert HTTP {r.status_code}: {r.text[:140]}")
            return False
        return True
    except Exception as e:
        print(f"    snap insert failed: {type(e).__name__}: {str(e)[:120]}")
        return False


def window_already_logged(date_str, label):
    """True if this (date, window) already has rows.

    Fails OPEN on error: a missed window loses a day of sample, a duplicate
    insert is ignored by the constraint. The safer default is to proceed.
    """
    try:
        r = requests.get(
            sb_url("favorites_bets"),
            headers=sb_headers(),
            params={"date": f"eq.{date_str}", "window_label": f"eq.{label}",
                    "select": "id", "limit": "1"},
            timeout=15)
        if r.status_code == 200:
            return len(r.json()) > 0
    except Exception:
        pass
    return False


def snapshot_already_logged(date_str, label):
    try:
        r = requests.get(
            sb_url("favorites_snapshots"),
            headers=sb_headers(),
            params={"date": f"eq.{date_str}", "snap_label": f"eq.{label}",
                    "select": "id", "limit": "1"},
            timeout=15)
        if r.status_code == 200:
            return len(r.json()) > 0
    except Exception:
        pass
    return False


def fetch_obs_today(date_str):
    """obs_live rows for today, keyed by city.

    ⚠️ THE ROW CAN BE HOURS OLD AND LOOK FINE. On 2026-09-09 obs_live froze at
    08:24 local and kept serving 81.0F for San Antonio until 17:39, while the
    station was at 98.6. obs_age_min is stamped at WRITE time and does not age,
    so it is stored here as-is and MUST be filtered on before the distance
    fields are trusted. Returns {} on any failure — these are optional columns
    and a missing obs row must never block a bet.
    """
    try:
        r = requests.get(
            sb_url("obs_live"),
            headers=sb_headers(),
            params={"local_date": f"eq.{date_str}",
                    "select": "city,temp_f,day_max_f,obs_age_min",
                    "limit": "60"},
            timeout=15)
        if r.status_code != 200:
            return {}
        return {row.get("city"): row for row in (r.json() or []) if row.get("city")}
    except Exception:
        return {}


def obs_fields(obs_row, bracket_lo, suffix):
    """Distance-to-bracket columns. All None if obs is missing — never blocks."""
    out = {
        f"temp_at_{suffix}": None,
        f"day_max_at_{suffix}": None,
        "degrees_to_lo": None,
        f"obs_age_min_at_{suffix}": None,
    }
    if not obs_row:
        return out
    try:
        t = obs_row.get("temp_f")
        dm = obs_row.get("day_max_f")
        out[f"temp_at_{suffix}"] = round(float(t), 2) if t is not None else None
        out[f"day_max_at_{suffix}"] = round(float(dm), 2) if dm is not None else None
        age = obs_row.get("obs_age_min")
        out[f"obs_age_min_at_{suffix}"] = round(float(age), 1) if age is not None else None
        if dm is not None and bracket_lo is not None:
            out["degrees_to_lo"] = round(float(bracket_lo) - float(dm), 2)
    except Exception:
        pass
    return out


def fetch_pending(table, order_col="date"):
    try:
        r = requests.get(
            sb_url(table),
            headers=sb_headers(),
            params={"result": "eq.Pending", "order": f"{order_col}.asc",
                    "limit": "1000"},
            timeout=20)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def update_row(table, row_id, updates):
    try:
        r = requests.patch(
            sb_url(table) + "?id=eq." + str(row_id),
            headers=sb_headers(), json=updates, timeout=15)
        return r.status_code in (200, 204)
    except Exception:
        return False


# ── Kalshi ───────────────────────────────────────────────────────────────────
def event_ticker_for(series, when_et):
    return series + "-" + when_et.strftime("%y%b%d").upper()


def kalshi_markets(params):
    for attempt in (1, 2, 3):
        try:
            r = requests.get(KALSHI, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.json().get("markets", [])
            if r.status_code in (429, 500, 502, 503):
                time.sleep(1.5 * attempt)
                continue
            return []
        except Exception:
            if attempt == 3:
                return []
            time.sleep(1.5 * attempt)
    return []


def label_of(m):
    for f in ("yes_sub_title", "subtitle", "title"):
        s = (m.get(f) or "").replace("\u00b0", "").strip()
        if s:
            return s
    return ""


def ask_cents(m):
    """Ask price in cents. ASK, not mid, not bid — you pay the ask.

    An earlier analysis used mid and produced 55-59 numbers that did not
    reproduce; the band a contract falls into changes with which price you
    classify by, so this is not a 1-2c adjustment.
    """
    v = m.get("yes_ask_dollars")
    if v:
        try:
            return int(round(float(v) * 100))
        except Exception:
            pass
    v = m.get("yes_ask")
    if v is not None:
        try:
            return int(v)
        except Exception:
            pass
    return None


def bid_cents(m):
    """Bid price in cents, from the SAME markets call that gives the ask.

    ⚠️ V1.5. Until now only the ask was stored, so a 70c favorite quoted
    69/70 was indistinguishable from one quoted 58/70 — a tight two-sided
    market and a thin one-sided one recorded as the same number.

    spread = ask - bid is a columns-only addition. No bet is filtered on it.
    """
    v = m.get("yes_bid_dollars")
    if v:
        try:
            return int(round(float(v) * 100))
        except Exception:
            pass
    v = m.get("yes_bid")
    if v is not None:
        try:
            return int(v)
        except Exception:
            pass
    return None


def volume_of(m):
    """Contracts traded on this market, if the endpoint reports it.

    ⚠️ V1.5. Logged to test whether the edge concentrates in thin markets or
    in liquid ones. COLUMNS ONLY — per-city selection has failed four times
    and volume is heavily city-correlated, so any result here needs the
    half-split and a city control before it means anything.
    """
    for f in ("volume", "volume_24h"):
        v = m.get(f)
        if v is not None:
            try:
                return int(v)
            except Exception:
                continue
    return None


def bracket_bounds(label):
    """Parse a Kalshi bracket label into (lo, hi). None means unbounded.

        "94 to 95" / "94-95"  -> (94, 95)
        "97 or below"         -> (None, 97)
        "106 or above"        -> (106, None)
        unparseable           -> (None, None)

    Stored at write time so miss analysis is a subtraction rather than a regex
    over label strings months later.

    ⚠️ Despite the range naming, Kalshi brackets settle EXCLUSIVE — verified by
    counting yes-per-event, which came back exactly 1 of 6 on every event
    checked. A temperature belongs to exactly one bracket.
    """
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


def top_three(markets):
    """The market's favorite, runner-up, and rank 3.

    Returns (m1, a1, m2, a2, m3, a3, sigma_p, n_priced, top3_share) or None.
    m3/a3 are None when the ladder has only two priced brackets.

    sigma_p is the sum of implied probabilities across the ladder — logged as
    context, NOT used as a filter. The weather model gated on sigma_p > 1.15;
    whether that helps THIS strategy is untested. Forward data so far shows no
    ordering by sigma tier.

    ⚠️ V1.5 ADDS RANK 3. gap_cents (favorite minus runner-up) turned out to
    carry almost no variation in the band: measured 2026-09-11 on 44 settled
    bets it ranged 16 to 67 cents and NEVER below 16. The band selects out
    narrow favorites entirely, so "is this a dominant favorite" could not be
    asked — every in-band favorite already dominates rank 2.

    top3_share (the top three asks as a fraction of the whole ladder) measures
    something gap_cents cannot: whether the remaining probability is
    concentrated in a couple of neighbours or smeared across the ladder. Two
    markets with an identical 70c favorite and an identical 20c runner-up can
    still differ in how the last 10c is distributed.

    COLUMNS ONLY. No bet is filtered on any of this.
    """
    priced = [(m, ask_cents(m)) for m in markets]
    priced = [(m, a) for m, a in priced if a is not None and 0 < a < 100]
    if len(priced) < 2:
        return None
    total = sum(a for _, a in priced)
    sigma_p = round(total / 100.0, 4)
    priced.sort(key=lambda x: x[1], reverse=True)
    (m1, a1), (m2, a2) = priced[0], priced[1]
    if len(priced) >= 3:
        m3, a3 = priced[2]
    else:
        m3, a3 = None, None
    top3 = a1 + a2 + (a3 or 0)
    top3_share = round(top3 / total, 4) if total else None
    return m1, a1, m2, a2, m3, a3, sigma_p, len(priced), top3_share


def ladder_for(city, series, now_et):
    """One city's priced ladder, with the fallback path. Shared by both passes."""
    et_ticker = event_ticker_for(series, now_et)
    markets = kalshi_markets({"event_ticker": et_ticker, "limit": 40})
    if not markets:
        markets = kalshi_markets(
            {"series_ticker": series, "status": "open", "limit": 40})
        markets = [m for m in markets
                   if et_ticker.upper() in (m.get("event_ticker") or "").upper()]
    return markets


def fee_for(amount, price_cents):
    """Entry fee in dollars, per Kalshi's published schedule.

        fee = round_up_to_cent( 0.07 * C * P * (1-P) )

    CONFIRMED 2026-09-08 against a real fill: Las Vegas 95-96 on Sep 7, 30
    contracts at 65c. Formula gives $19.50 + $0.48 = $19.98; the ticket read
    $19.99. That is 1.60c per contract.

    ⚠️ ONE FEE, NOT TWO. This strategy holds to settlement, and Kalshi charges
    nothing at resolution — a winner held to expiry pays only the entry fee.
    Selling before settlement pays the fee a second time, which is what the
    3.6c figure was probably measuring.

    The fee falls as price rises, because P*(1-P) shrinks toward the extremes:

        58c -> 1.71c      69c -> 1.50c      79c -> 1.16c
        62c -> 1.65c      72c -> 1.41c      99c -> 0.07c

    ⚠️ ROUNDING BITES SMALL ORDERS. The round-up is on the TRADE, not the
    contract. An 8-contract order at 62c owes $0.135 and pays $0.14 — 1.74c per
    contract instead of 1.65c. At a $5 stake that is a few percent; at real
    size it disappears. Do not read the per-contract rate off a small ticket.

    ⚠️ THIS IS THE FEE, NOT THE COST. It does not include slippage. A displayed
    58c has filled at 61c on a manual trade. See the SLIPPAGE note at the top —
    it cannot be measured from this file because nothing here places an order.
    """
    try:
        p = float(price_cents) / 100.0
        if p <= 0 or p >= 1:
            return 0.0
        contracts = float(amount) / p
        raw = FEE_COEFFICIENT * contracts * p * (1.0 - p)
        return math.ceil(raw * 100.0) / 100.0
    except Exception:
        return 0.0


def break_even_pct(price_cents):
    """Win rate needed to break even at this entry price, fee included.

    Buying at P and holding to settlement: a win returns (1-P), a loss costs P,
    and the entry fee f is paid either way. Break-even w solves

        w*(1-P) - (1-w)*P - f = 0   ->   w = P + f

    with everything in dollars. So it is simply the price plus the per-contract
    fee — which is why a cheap bet in the band needs a LOWER win rate than an
    expensive one even though the fee itself is larger.

        58c -> 59.7%      69c -> 70.5%      79c -> 80.2%
        62c -> 63.7%      72c -> 73.4%
    """
    try:
        p = float(price_cents) / 100.0
        if p <= 0 or p >= 1:
            return None
        f = FEE_COEFFICIENT * p * (1.0 - p)
        return round((p + f) * 100.0, 1)
    except Exception:
        return None


# ── Logging pass ─────────────────────────────────────────────────────────────
def run_window(label, band_lo, band_hi, now_et, minutes_late):
    today = now_et.strftime("%Y-%m-%d")
    tag = f"{TAG_PREFIX}_{label}"
    print(f"\n=== {label} | band {band_lo}-{band_hi - 1}c | {today} "
          f"| {minutes_late} min after target ===")
    if minutes_late >= 10:
        print(f"  ⚠️ entry drift {minutes_late} min. Price at entry is the "
              f"strategy — check minutes_late across the sample.")

    obs = fetch_obs_today(today)
    if not obs:
        print("  (no obs_live rows — distance columns will be null, "
              "bets proceed unchanged)")

    logged, skipped_band, skipped_nomarket = [], 0, 0

    for city, series in SERIES.items():
        markets = ladder_for(city, series, now_et)

        top = top_three(markets)
        if top is None:
            skipped_nomarket += 1
            print(f"  {city:<15} no ladder")
            continue

        m1, ask, m2, runner_ask, m3, third_ask, sigma_p, n_priced, top3_share = top
        bracket = label_of(m1)
        runner_bracket = label_of(m2)
        third_bracket = label_of(m3) if m3 else None

        if not (band_lo <= ask < band_hi):
            skipped_band += 1
            print(f"  {city:<15} {bracket:<16} {ask:>3}c  — out of band")
            continue

        lo, hi = bracket_bounds(bracket)
        r_lo, r_hi = bracket_bounds(runner_bracket)

        # V1.5: two-sided quote and depth. Columns only.
        bid = bid_cents(m1)
        spread = (ask - bid) if bid is not None else None

        # which side is the runner-up on? this is the V1.2 question.
        side = ""
        if lo is not None and r_lo is not None:
            side = " ↑" if r_lo > lo else " ↓" if r_lo < lo else ""

        row = {
            "date": today,
            "city": city,
            "series": series,
            "event_ticker": m1.get("event_ticker"),
            "market_ticker": m1.get("ticker"),
            "bracket": bracket,
            "bracket_lo": lo,
            "bracket_hi": hi,
            "window_label": label,
            "yes_ask_cents": ask,
            "sigma_p": sigma_p,
            "n_brackets": n_priced,
            "strategy_tag": tag,
            "amount": STAKE,
            "result": "Pending",
            "placed_at": now_et.isoformat(),
            "minutes_late": minutes_late,
            "fee_dollars": fee_for(STAKE, ask),
            "runner_bracket": runner_bracket,
            "runner_ask": runner_ask,
            "runner_lo": r_lo,
            "runner_hi": r_hi,
            "gap_cents": ask - runner_ask,
            # ⚠️ V1.5 — columns only, nothing is filtered on these.
            "yes_bid_cents": bid,
            "spread_cents": spread,
            "volume": volume_of(m1),
            "third_bracket": third_bracket,
            "third_ask": third_ask,
            "top3_share": top3_share,
        }
        # V1.4: columns only. No bet is skipped for being far from its bracket.
        row.update(obs_fields(obs.get(city), lo, "entry"))

        if insert_bet(row):
            d2l = row.get("degrees_to_lo")
            logged.append(f"{city} {bracket} @ {ask}c")
            print(f"  {city:<15} {bracket:<16} {ask:>3}c  ✅ LOGGED  "
                  f"(BE {break_even_pct(ask):.1f}% · fee ${fee_for(STAKE, ask):.2f} · "
                  f"Σp {sigma_p:.2f} · runner {runner_bracket} @ {runner_ask}c"
                  f"{side} · gap {ask - runner_ask}c"
                  f"{f' · to-lo {d2l:+.1f}F' if d2l is not None else ''})")
        else:
            print(f"  {city:<15} {bracket:<16} {ask:>3}c  insert failed")

        time.sleep(0.25)

    print(f"\n  logged {len(logged)} | out of band {skipped_band} | "
          f"no ladder {skipped_nomarket}")
    if not logged:
        print("  (no rows written — a retry cron may fire this window again; "
              "any such entry carries a larger minutes_late)")
    return logged


# ── Snapshot pass ────────────────────────────────────────────────────────────
def run_snapshot(label, now_et, minutes_late):
    """Record the favorite for every city at this hour. NO BET, NO BAND FILTER.

    ⚠️ This does not stake anything and does not write to favorites_bets. It
    exists to answer two questions the bet record structurally cannot:
      - is there a better hour between the three betting windows
      - how often does the favorite win at prices OUTSIDE 58-79

    Never pool these rows with FAV V1 results.
    """
    today = now_et.strftime("%Y-%m-%d")
    print(f"\n=== SNAPSHOT {label} | {today} | {minutes_late} min after target "
          f"| no bets placed ===")

    obs = fetch_obs_today(today)
    written = in_band_n = nomarket = 0

    for city, series in SERIES.items():
        markets = ladder_for(city, series, now_et)
        top = top_three(markets)
        if top is None:
            nomarket += 1
            continue

        m1, ask, m2, runner_ask, m3, third_ask, sigma_p, n_priced, top3_share = top
        bracket = label_of(m1)
        runner_bracket = label_of(m2)
        third_bracket = label_of(m3) if m3 else None
        lo, hi = bracket_bounds(bracket)
        r_lo, r_hi = bracket_bounds(runner_bracket)
        bid = bid_cents(m1)
        spread = (ask - bid) if bid is not None else None
        in_band = bool(SNAP_BAND_LO <= ask < SNAP_BAND_HI)
        if in_band:
            in_band_n += 1

        row = {
            "date": today,
            "city": city,
            "snap_label": label,
            "series": series,
            "event_ticker": m1.get("event_ticker"),
            "market_ticker": m1.get("ticker"),
            "bracket": bracket,
            "bracket_lo": lo,
            "bracket_hi": hi,
            "yes_ask_cents": ask,
            "runner_bracket": runner_bracket,
            "runner_ask": runner_ask,
            "runner_lo": r_lo,
            "runner_hi": r_hi,
            "gap_cents": ask - runner_ask,
            "sigma_p": sigma_p,
            "n_brackets": n_priced,
            "in_band": in_band,
            # ⚠️ V1.5 — columns only.
            "yes_bid_cents": bid,
            "spread_cents": spread,
            "volume": volume_of(m1),
            "third_bracket": third_bracket,
            "third_ask": third_ask,
            "top3_share": top3_share,
            "result": "Pending",
            "captured_at": now_et.isoformat(),
            "minutes_late": minutes_late,
        }
        row.update(obs_fields(obs.get(city), lo, "snap"))

        if insert_snapshot(row):
            written += 1
        time.sleep(0.25)

    print(f"  captured {written}/{len(SERIES)} | in band {in_band_n} | "
          f"no ladder {nomarket}")
    return written


# ── Settlement pass ──────────────────────────────────────────────────────────
def settle_table(table, id_field="market_ticker"):
    """Score against Kalshi's own result field, not a temperature we computed.

    Shared by favorites_bets and favorites_snapshots. Snapshots carry no stake,
    so they get result only — no profit, no fee, no net.
    """
    is_bets = (table == "favorites_bets")
    print(f"\n=== Settlement: {table} ===")
    pending = fetch_pending(table)
    if not pending:
        print("  nothing pending")
        return

    today = dt.datetime.now(ET).strftime("%Y-%m-%d")
    by_event = {}
    for b in pending:
        if b.get("date", "") >= today:
            continue
        by_event.setdefault(b.get("event_ticker"), []).append(b)

    if not by_event:
        print(f"  {len(pending)} pending, none from a prior day yet")
        return

    won = lost = 0
    gross = net = 0.0
    for et_ticker, rows in by_event.items():
        if not et_ticker:
            continue
        markets = kalshi_markets({"event_ticker": et_ticker, "limit": 40})
        results = {m.get("ticker"): m.get("result") for m in markets}
        for b in rows:
            res = results.get(b.get(id_field))
            if res not in ("yes", "no"):
                continue

            if res == "yes":
                won += 1
            else:
                lost += 1

            updates = {
                "result": "Won" if res == "yes" else "Lost",
                "settled_at": dt.datetime.now(ET).isoformat(),
            }

            if is_bets:
                price = float(b.get("yes_ask_cents") or 0)
                amount = float(b.get("amount") or STAKE)
                if res == "yes" and price > 0:
                    profit = round(amount * (100.0 - price) / price, 2)
                else:
                    profit = round(-amount, 2)
                fee = b.get("fee_dollars")
                if fee is None:
                    fee = fee_for(amount, price)
                fee = float(fee)
                net_profit = round(profit - fee, 4)
                gross += profit
                net += net_profit
                updates.update({"profit": profit, "fee_dollars": fee,
                                "net_profit": net_profit})

            update_row(table, b["id"], updates)
        time.sleep(0.25)

    n = won + lost
    if n:
        print(f"  settled {n}: {won} won, {lost} lost ({100.0*won/n:.1f}%)")
        if is_bets:
            print(f"  gross ${gross:+.2f} | net ${net:+.2f} "
                  f"(Kalshi published formula, one entry fee, no settlement fee)")
        else:
            print("  (snapshots carry no stake — result only, never pooled "
                  "with FAV V1)")
    else:
        print("  no results available yet")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    now_et = dt.datetime.now(ET)
    print(f"FAV V1.5 | {now_et:%Y-%m-%d %H:%M} ET | {len(SERIES)} cities")
    print("no forecast — buying the market's own favorite, in band")
    print(f"latch: fires 0 to +{WINDOW_LATCH_MIN} min after target, never early")
    print("logging runner-up + obs-at-entry — columns only, no bet is filtered")
    print(f"snapshots at {', '.join(l for _, _, l in SNAPSHOT_HOURS)} "
          f"— no bets, no band filter\n")

    today = now_et.strftime("%Y-%m-%d")
    fired = False

    for hh, mm, label, lo, hi in WINDOWS:
        target = now_et.replace(hour=hh, minute=mm, second=0, microsecond=0)
        delta_min = (now_et - target).total_seconds() / 60.0

        # one-sided: early never fires
        if not (0 <= delta_min <= WINDOW_LATCH_MIN):
            continue

        if window_already_logged(today, label):
            print(f"({label} already has rows for {today} — skipping, "
                  f"this run is a retry after a successful one)")
            fired = True
            continue

        run_window(label, lo, hi, now_et, int(round(delta_min)))
        fired = True

    for hh, mm, label in SNAPSHOT_HOURS:
        target = now_et.replace(hour=hh, minute=mm, second=0, microsecond=0)
        delta_min = (now_et - target).total_seconds() / 60.0
        if not (0 <= delta_min <= WINDOW_LATCH_MIN):
            continue
        if snapshot_already_logged(today, label):
            print(f"(snapshot {label} already captured for {today} — skipping)")
            fired = True
            continue
        run_snapshot(label, now_et, int(round(delta_min)))
        fired = True

    if not fired:
        def mins_until(h, m):
            t = now_et.replace(hour=h, minute=m, second=0, microsecond=0)
            d = (t - now_et).total_seconds() / 60.0
            return d if d >= 0 else d + 1440.0
        allw = [(w[0], w[1], w[2]) for w in WINDOWS] + list(SNAPSHOT_HOURS)
        nxt = min(allw, key=lambda w: mins_until(w[0], w[1]))
        print(f"(nothing active — next is {nxt[2]} at "
              f"{nxt[0]:02d}:{nxt[1]:02d} ET, in {mins_until(nxt[0], nxt[1]):.0f} min)")

    settle_table("favorites_bets")
    settle_table("favorites_snapshots")

    print("\nThe tally (break-even is price-dependent — see break_even_pct):")
    print("  select coalesce(window_label,'TOTAL') win_label, count(*) n,")
    print("         sum((result='Won')::int) wins,")
    print("         round(100.0*avg((result='Won')::int),1) win_pct,")
    print("         round(avg(yes_ask_cents),1) avg_ask,")
    print("         round(avg(yes_ask_cents")
    print("               + 0.07*yes_ask_cents*(100-yes_ask_cents)/100.0),1) break_even_pct,")
    print("         round(sum(profit),2) gross,")
    print("         round(sum(net_profit),2) net")
    print("  from favorites_bets where result in ('Won','Lost')")
    print("  group by rollup (window_label) order by win_label nulls last;")
    print("\n  ⚠️ favorites_snapshots is a SEPARATE table. Never UNION it with")
    print("     favorites_bets — no stake, no fee, no band. See the V1.4 block.")


if __name__ == "__main__":
    main()
