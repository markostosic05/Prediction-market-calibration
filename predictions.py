#!/usr/bin/env python3
"""
Prediction-market research toolkit (Polymarket + Kalshi public data, no API keys).

  python predictions.py fetch    [--n 400] [--hours-before 24] [--min-volume 10000]
        Downloads resolved binary Polymarket markets and the YES price N hours before each one ended.
        -> calibration_data.csv  (resumable: rerun to add more)

  python predictions.py analyze  [--csv calibration_data.csv]
        Calibration table, Brier score, favourite-longshot test (logistic fit), and calibration.png.

  python predictions.py fairvalue [--top 10] [--kalshi TICKER ...]
        Live order books: mid price vs the price you'd ACTUALLY get for $100 / $1,000 after walking
        the book, plus fees (Kalshi's published taker-fee formula; Polymarket fee assumed 0 unless set).

Needs: pip install requests numpy matplotlib
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
HERE = Path(__file__).resolve().parent
S = requests.Session()
S.headers["User-Agent"] = "pm-research/0.1 (research script)"


def get(url, params=None, tries=3):
    """GET JSON with polite retries. Returns None on failure."""
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(5 * (i + 1))
                continue
            if r.status_code != 200:
                return None
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(2 * (i + 1))
    return None


def jlist(x):
    """Polymarket returns some list fields as JSON strings."""
    if isinstance(x, list):
        return x
    try:
        return json.loads(x or "[]")
    except (TypeError, ValueError):
        return []


def iso_ts(s):
    from datetime import datetime
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


# ============================================================ 1. calibration data
STATS = {}


def count(key, n=1):
    STATS[key] = STATS.get(key, 0) + n


def resolved_markets(min_volume, page=100, max_pages=200):
    """Yield resolved binary (Yes/No) Polymarket markets, newest first."""
    params = {"closed": "true", "limit": page, "order": "endDate", "ascending": "false", "volume_num_min": min_volume}
    offset = 0
    for p in range(max_pages):
        data = get(f"{GAMMA}/markets", dict(params, offset=offset))
        if not data and p == 0 and "order" in params:          # sorting not accepted -> try without it
            print("  (Polymarket refused the sort order - continuing unsorted)")
            params.pop("order"), params.pop("ascending")
            data = get(f"{GAMMA}/markets", dict(params, offset=0))
        if not data:
            count("pages_stopped_empty")
            return
        count("pages")
        count("markets_listed", len(data))
        offset += len(data)                      # page by what the API actually returned (it caps the size)
        for m in data:
            outs, prices, toks = jlist(m.get("outcomes")), jlist(m.get("outcomePrices")), jlist(m.get("clobTokenIds"))
            if [o.lower() for o in outs] != ["yes", "no"] or len(prices) != 2 or len(toks) != 2:
                count("skipped_not_yes_no")
                continue
            try:
                yes_final = float(prices[0])
            except ValueError:
                continue
            if yes_final not in (0.0, 1.0):          # unresolved, or a 50/50 resolution
                count("skipped_not_resolved_0_1")
                continue
            end = iso_ts(m.get("closedTime") or m.get("endDate"))
            if end:
                yield dict(id=m.get("id"), question=m.get("question"), end_ts=end, yes_token=toks[0],
                           outcome=int(yes_final), volume=float(m.get("volumeNum") or m.get("volume") or 0),
                           category=m.get("category") or "")


def price_before(token, end_ts, hours):
    """YES price at (end - hours), from the CLOB price history (last point at or before that time)."""
    t = end_ts - hours * 3600
    h = get(f"{CLOB}/prices-history", {"market": token, "startTs": int(t - 6 * 3600), "endTs": int(t),
                                       "fidelity": 60})
    pts = [x for x in (h or {}).get("history", []) if x.get("t", 0) <= t]
    if not pts:                                       # some closed markets only answer the full history
        h = get(f"{CLOB}/prices-history", {"market": token, "interval": "max", "fidelity": 60})
        pts = [x for x in (h or {}).get("history", []) if t - 12 * 3600 <= x.get("t", 0) <= t]
        if pts:
            count("price_from_full_history")
    return float(pts[-1]["p"]) if pts else None


def cmd_fetch(a):
    out = HERE / a.csv
    done = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            done = {r["id"] for r in csv.DictReader(f)}
    cols = ["id", "question", "category", "end_ts", "volume", "hours_before", "price", "outcome"]
    new = not out.exists()
    n_ok = n_seen = 0
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for m in resolved_markets(a.min_volume):
            if m["id"] in done:
                count("already_saved")
                continue
            n_seen += 1
            p = price_before(m["yes_token"], m["end_ts"], a.hours_before)
            if p is None:
                count("no_price_history")
            elif not 0 < p < 1:
                count("price_exactly_0_or_1")
            if p is not None and 0 < p < 1:
                w.writerow(dict(id=m["id"], question=m["question"], category=m["category"], end_ts=int(m["end_ts"]),
                                volume=round(m["volume"]), hours_before=a.hours_before, price=p, outcome=m["outcome"]))
                f.flush()
                n_ok += 1
                if n_ok % 25 == 0:
                    print(f"  {n_ok} markets saved ({n_seen} checked)", flush=True)
            if n_ok >= a.n:
                break
            time.sleep(0.15)                                 # be polite
    print(f"done: {n_ok} new markets with a price {a.hours_before}h before the end -> {out.name}")
    print("where markets went: " + ", ".join(f"{k} {v}" for k, v in STATS.items()))


# ============================================================ 2. analysis
def logistic_fit(x, y, iters=50):
    """Fit P(yes) = sigmoid(a + b*x) by Newton's method. Returns (a, b, standard errors)."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return beta, se


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cmd_analyze(a):
    path = HERE / a.csv
    if not path.exists():
        sys.exit(f"{path.name} not found - run: python predictions.py fetch")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    p = np.array([float(r["price"]) for r in rows])
    y = np.array([int(r["outcome"]) for r in rows])
    n = len(p)
    print(f"\n{n} resolved Polymarket markets, YES price {rows[0]['hours_before']}h before the end\n")
    print(f"  {'price bucket':>13} {'markets':>8} {'avg price':>10} {'YES won':>8} {'95% range':>15}")
    edges = np.linspace(0, 1, 11)
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        k, nn = int(y[m].sum()), int(m.sum())
        if nn == 0:
            continue
        ci = wilson(k, nn)
        table.append((p[m].mean(), k / nn, ci, nn))
        flag = "  <- off" if not (ci[0] <= p[m].mean() <= ci[1]) else ""
        print(f"  {lo:>5.0%} - {hi:>4.0%} {nn:>8} {p[m].mean():>10.1%} {k / nn:>8.1%}   {ci[0]:>5.0%} - {ci[1]:>4.0%}{flag}")
    brier = np.mean((p - y) ** 2)
    base = np.mean((y.mean() - y) ** 2)
    print(f"\n  Brier score {brier:.4f}  (always guessing the base rate: {base:.4f}; lower = better)")
    # favourite-longshot: regress outcome on logit(price). Slope 1 = perfectly calibrated;
    # slope > 1 = longshots overpriced / favourites underpriced; slope < 1 = the opposite.
    pc = np.clip(p, 0.01, 0.99)
    (b0, b1), (s0, s1) = logistic_fit(np.log(pc / (1 - pc)), y)
    z = (b1 - 1) / s1
    print(f"  logistic fit: outcome ~ logit(price): slope {b1:.2f} (+/- {1.96 * s1:.2f}), intercept {b0:+.2f}")
    verdict = ("no clear bias - prices look calibrated" if abs(z) < 1.96 else
               "longshots look OVERpriced, favourites UNDERpriced (classic favourite-longshot bias)" if b1 > 1 else
               "favourites look OVERpriced, longshots UNDERpriced")
    print(f"  -> {verdict} (z = {z:+.1f})")
    mid = (p >= 0.1) & (p <= 0.9)
    print(f"\n  {(~mid).mean():.0%} of markets were priced under 10% or over 90% (outcome nearly known) - they make")
    print(f"  the Brier score look good. Uncertain markets only (10-90%): n = {mid.sum()}", end="")
    if mid.sum() >= 30:
        pm_ = np.clip(p[mid], 0.01, 0.99)
        (c0, c1), (t0, t1) = logistic_fit(np.log(pm_ / (1 - pm_)), y[mid])
        print(f", Brier {np.mean((p[mid] - y[mid]) ** 2):.4f}, slope {c1:.2f} (+/- {1.96 * t1:.2f})")
    else:
        print(" - too few for a reliable fit (need 30+, ideally hundreds)")
    print("\n  Caveats: one price per market, 'end' is Polymarket's listed end/close time, no fees or spread,")
    print("  and markets are not independent (many share an event). Treat it as a first look, not a strategy.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", label="perfect calibration")
        xs = [t[0] for t in table]
        ys = [t[1] for t in table]
        err = np.array([[t[1] - t[2][0] for t in table], [t[2][1] - t[1] for t in table]])
        ax.errorbar(xs, ys, yerr=err, fmt="o", color="#2b6cb0", capsize=3, label="markets (95% range)")
        grid = np.linspace(0.01, 0.99, 200)
        ax.plot(grid, 1 / (1 + np.exp(-(b0 + b1 * np.log(grid / (1 - grid))))), color="#c05621", lw=1.5,
                label=f"logistic fit (slope {b1:.2f})")
        ax.set(xlabel=f"YES price {rows[0]['hours_before']}h before the end", ylabel="share that resolved YES",
               xlim=(0, 1), ylim=(0, 1), title=f"Polymarket calibration (n={n})")
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        fig.savefig(HERE / "calibration.png", dpi=150)
        print("\n  chart saved -> calibration.png")
    except ImportError:
        print("\n  (pip install matplotlib for the chart)")


# ============================================================ 3. fair value from the order book
def kalshi_fee(price, contracts, rate=0.07):
    """Kalshi taker fee: rate x C x P x (1-P), rounded UP to the cent (P in dollars)."""
    return math.ceil(rate * contracts * price * (1 - price) * 100 - 1e-9) / 100


def walk(levels, dollars):
    """Buy `dollars` worth from levels [(price, size), ...] sorted best first. -> (avg price, contracts, filled $)."""
    spent = qty = 0.0
    for px, sz in levels:
        take = min(sz, (dollars - spent) / px)
        if take <= 0:
            break
        spent += take * px
        qty += take
        if spent >= dollars - 1e-9:
            break
    return (spent / qty if qty else None), qty, spent


def walk_sell(levels, contracts):
    """Sell `contracts` into bid levels sorted best first. -> avg price (None if the book is too thin)."""
    got = left = 0.0
    left = contracts
    for px, sz in levels:
        take = min(sz, left)
        got += take * px
        left -= take
        if left <= 1e-9:
            return got / contracts
    return None


def fair_value(bids, asks, size, fee_fn=None):
    """Executable round trip for `size` dollars: buy-side and sell-side average prices, incl. fees."""
    buy_px, qty, filled = walk(asks, size)
    if buy_px is None or filled < size * 0.999:
        return None
    sell_px = walk_sell(bids, qty)
    if sell_px is None:
        return None
    fb = fee_fn(buy_px, qty) / qty if fee_fn else 0.0
    fs = fee_fn(sell_px, qty) / qty if fee_fn else 0.0
    eff_buy, eff_sell = buy_px + fb, sell_px - fs
    return dict(buy=eff_buy, sell=eff_sell, mid_exec=(eff_buy + eff_sell) / 2, cost=eff_buy - eff_sell)


def poly_book(token):
    b = get(f"{CLOB}/book", {"token_id": token})
    if not b:
        return None, None
    bids = sorted(((float(x["price"]), float(x["size"])) for x in b.get("bids", [])), reverse=True)
    asks = sorted((float(x["price"]), float(x["size"])) for x in b.get("asks", []))
    return bids, asks


def kalshi_book(ticker):
    """Kalshi lists BIDS for YES and for NO. A NO bid at x is a YES offer at 1-x."""
    b = (get(f"{KALSHI}/markets/{ticker}/orderbook") or {}).get("orderbook") or {}

    def lv(key):
        out = []
        for x in b.get(key) or []:
            px = float(x[0])
            out.append((px / 100 if px > 1 else px, float(x[1])))
        return out
    yes_bids = sorted(lv("yes"), reverse=True)
    yes_asks = sorted((round(1 - px, 4), sz) for px, sz in lv("no"))
    return yes_bids, yes_asks


def print_fv(name, bids, asks, fee_fn, sizes=(100, 1000)):
    if not bids or not asks:
        print(f"  {name[:48]:<48}  (empty book)")
        return
    mid = (bids[0][0] + asks[0][0]) / 2
    line = f"  {name[:48]:<48} mid {mid:6.3f}  spread {asks[0][0] - bids[0][0]:5.3f}"
    for s in sizes:
        fv = fair_value(bids, asks, s, fee_fn)
        line += (f" | ${s:>4}: buy {fv['buy']:.3f} sell {fv['sell']:.3f} cost {fv['cost'] * 100:4.1f}c"
                 f" = {fv['cost'] / fv['buy']:4.0%} of price" if fv else f" | ${s:>4}: book too thin")
    print(line)


def cmd_fairvalue(a):
    print("\nLive books. 'cost' = what a round trip of that size costs per $1 contract (spread + depth + fees).")
    print("A model's probability has to beat the BUY price (or be below the SELL price) by more than that.")
    print("'% of price' = the same cost relative to what you pay: tiny in cents can be huge for a 1-cent longshot.\n")
    data = get(f"{GAMMA}/markets", {"active": "true", "closed": "false", "limit": a.top * 3,
                                    "order": "volume24hr", "ascending": "false"}) or []
    shown = 0
    print("Polymarket (fee assumed 0 - check the market's fee settings):")
    for m in data:
        toks, outs = jlist(m.get("clobTokenIds")), jlist(m.get("outcomes"))
        if len(toks) != 2 or [o.lower() for o in outs] != ["yes", "no"]:
            continue
        bids, asks = poly_book(toks[0])
        print_fv(m.get("question") or "?", bids, asks, None)
        shown += 1
        if shown >= a.top:
            break
        time.sleep(0.2)
    if a.kalshi:
        print("\nKalshi (taker fee 0.07 x C x P x (1-P), rounded up to the cent):")
        for t in a.kalshi:
            bids, asks = kalshi_book(t)
            print_fv(t, bids, asks, kalshi_fee)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--n", type=int, default=400)
    f.add_argument("--hours-before", type=float, default=24)
    f.add_argument("--min-volume", type=float, default=10000)
    f.add_argument("--csv", default="calibration_data.csv")
    z = sub.add_parser("analyze")
    z.add_argument("--csv", default="calibration_data.csv")
    v = sub.add_parser("fairvalue")
    v.add_argument("--top", type=int, default=10)
    v.add_argument("--kalshi", nargs="*", default=[])
    if len(sys.argv) == 1:                       # started without a command (e.g. F5 in Visual Studio)
        print("What should I run?\n  1 = fetch      download resolved markets (do this first)\n"
              "  2 = analyze    calibration table + chart\n  3 = fairvalue  live order books")
        choice = input("Type 1, 2 or 3 and press Enter: ").strip()
        cmd = {"1": "fetch", "2": "analyze", "3": "fairvalue"}.get(choice)
        if not cmd:
            sys.exit("nothing chosen")
        sys.argv.append(cmd)
    a = ap.parse_args()
    {"fetch": cmd_fetch, "analyze": cmd_analyze, "fairvalue": cmd_fairvalue}[a.cmd](a)


if __name__ == "__main__":
    main()