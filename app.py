import random
import time
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# =============================
# CONFIG
# =============================
CARDS = [-10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]
AMOUNT_OF_PLAYERS = 5              # P0 = YOU, P1..P4 = bots
COMMUNAL_CARDS_NUMBER = 3
OPENING_AVG_CARD_VALUE = sum(CARDS) / len(CARDS)

GAME_SECONDS = 120                 # 2 minutes
TICK_SECONDS = 1.0                 # update cadence (bots quote + possibly trade)
P_OTHER_ACT_PER_TICK = 0.25        # each bot attempts a trade with this prob each tick

# User trade cooldown (your "Execute trades" button)
USER_TRADE_COOLDOWN_SECONDS = 2.0

# Always 4 bots are Smart Econ Students, except:
# One randomly chosen "special seat" has 10% chance to be Less-Stupid Econ Student;
# otherwise that seat is also Smart.
P_SPECIAL_LESS_STUPID = 0.10

ARCHETYPE_SMART = "smart_econ_student"
ARCHETYPE_LESS_STUPID = "less_stupid_econ_student"

ARCHETYPE_DISPLAY = {
    ARCHETYPE_SMART: "Smart Econ Student",
    ARCHETYPE_LESS_STUPID: "Less-Stupid Econ Student",
}

# =============================
# STREAMLIT PAGE CONFIG
# =============================
st.set_page_config(page_title="Trading Game (Continuous)", layout="wide")

# =============================
# INSTRUCTIONS (Start Screen)
# =============================
INSTRUCTIONS_MD = f"""
## How to play (quick)

### What the game is
- There are **5 players**: **you (P0)** and **4 AI players (P1–P4)**.
- Each round lasts **{GAME_SECONDS} seconds (2 minutes)**.
- At the start of the round, **8 unique cards** are drawn from a deck:
  **-10, 1–15, 20**
- The cards are split into:
  - **3 communal cards** (everyone can see these)
  - **5 private player cards** (each player gets 1, only they see it)

### Goal
You make money by trading at prices that are “wrong” relative to the **true total sum** (communal + all private cards).

**PnL logic**
- If you **buy** at price **P**, your profit is: **TrueSum − P**
- If you **sell** at price **P**, your profit is: **P − TrueSum**

So:
- Buying is good when price is **too low**
- Selling is good when price is **too high**

### What you can see during the round
- Your private card
- The 3 communal cards
- All market actions: quotes and trades

### What you cannot see during the round
- Other players’ private cards
- Player archetypes
- PnL (revealed at the end)

### What you can do
1) **Post a quote (Market Make)**
- Enter your **bid** and **ask**
- Other AI players can trade against you

2) **Trade instantly against players**
- For each AI player, choose:
  - **HB** = you **SELL** to them (you hit their **bid**)
  - **HA** = you **BUY** from them (you take their **ask**)

**Trade cooldown**
- You can press **Execute trades** at most **once every {USER_TRADE_COOLDOWN_SECONDS:.0f} seconds**.
- You can trade multiple AI players in that one action.
"""

# =============================
# SESSION STATE
# =============================
def ensure_state():
    ss = st.session_state

    # round lifecycle
    ss.setdefault("started", False)
    ss.setdefault("done", False)
    ss.setdefault("start_ts", None)
    ss.setdefault("end_ts", None)
    ss.setdefault("last_tick_ts", None)
    ss.setdefault("tick_id", 0)

    # round data
    ss.setdefault("opening_cards", None)
    ss.setdefault("communal_cards", None)
    ss.setdefault("player_cards", None)
    ss.setdefault("user_card", None)
    ss.setdefault("true_sum", None)
    ss.setdefault("player_types", None)

    # offers must always be list length AMOUNT_OF_PLAYERS
    ss.setdefault("offers", [None for _ in range(AMOUNT_OF_PLAYERS)])

    # logs
    ss.setdefault("trade_tape", [])
    ss.setdefault("trade_log", [])
    ss.setdefault("state_error", None)

    # smart beliefs
    ss.setdefault("beliefs", None)

    # user trade cooldown
    ss.setdefault("last_user_trade_ts", 0.0)


def listify(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return list(x)
    return [x]


def game_ready():
    ss = st.session_state
    if not ss.started:
        return False
    if ss.start_ts is None or ss.end_ts is None or ss.last_tick_ts is None:
        return False
    if ss.communal_cards is None or ss.player_cards is None or ss.player_types is None:
        return False
    if ss.beliefs is None:
        return False
    if not isinstance(ss.offers, list) or len(ss.offers) != AMOUNT_OF_PLAYERS:
        return False
    if not isinstance(ss.player_cards, list) or len(ss.player_cards) != AMOUNT_OF_PLAYERS:
        return False
    if not isinstance(ss.player_types, list) or len(ss.player_types) != AMOUNT_OF_PLAYERS:
        return False
    return True


def reset_to_start(reason: str):
    ss = st.session_state
    ss.started = False
    ss.done = False
    ss.start_ts = None
    ss.end_ts = None
    ss.last_tick_ts = None
    ss.tick_id = 0

    ss.opening_cards = None
    ss.communal_cards = None
    ss.player_cards = None
    ss.user_card = None
    ss.true_sum = None

    ss.player_types = None
    ss.offers = [None for _ in range(AMOUNT_OF_PLAYERS)]
    ss.trade_tape = []
    ss.trade_log = []
    ss.beliefs = None

    ss.last_user_trade_ts = 0.0
    ss.state_error = reason


# =============================
# MARKET HELPERS
# =============================
def make_quote(bid: int, ask: int):
    b, a = int(bid), int(ask)
    if b > a:
        b, a = a, b
    return {"bid": b, "ask": a}


def offers_df(offers):
    offers = offers if isinstance(offers, list) else []
    rows = []
    for i in range(AMOUNT_OF_PLAYERS):
        q = offers[i] if i < len(offers) else None
        if q is None:
            rows.append({"Player": f"P{i}", "Bid": "", "Ask": ""})
        else:
            rows.append({"Player": f"P{i}", "Bid": q["bid"], "Ask": q["ask"]})
    return pd.DataFrame(rows)


def get_best_bid_ask(offers):
    if offers is None or not isinstance(offers, list):
        return None, None, None, None

    best_bid, best_bid_player = None, None
    best_ask, best_ask_player = None, None

    for i, q in enumerate(offers):
        if q is None:
            continue
        if best_bid is None or q["bid"] > best_bid:
            best_bid, best_bid_player = q["bid"], i
        if best_ask is None or q["ask"] < best_ask:
            best_ask, best_ask_player = q["ask"], i

    return best_bid, best_bid_player, best_ask, best_ask_player


def _fmt_mmss_since_start(ts: float) -> str:
    ss = st.session_state
    if ss.start_ts is None:
        return "00:00"
    d = max(0, int(ts - ss.start_ts))
    m = d // 60
    s = d % 60
    return f"{m:02d}:{s:02d}"


def log_quote(who, bid, ask):
    t = _fmt_mmss_since_start(time.time())
    st.session_state.trade_tape.append(f"[{t}] QUOTE: P{who} now quotes {bid}/{ask}")


def log_trade(buyer, seller, price, maker_bid, maker_ask, aggress_code):
    t = _fmt_mmss_since_start(time.time())
    side = "BUY (took ask)" if aggress_code == "HA" else "SELL (hit bid)"
    st.session_state.trade_tape.append(
        f"[{t}] TRADE: P{buyer} {side} vs P{seller} @ {price} | maker quote was {maker_bid}/{maker_ask}"
    )


def record_trade(buyer, seller, price, maker_bid, maker_ask, aggress_code):
    st.session_state.trade_log.append({
        "ts": time.time(),
        "buyer": int(buyer),
        "seller": int(seller),
        "price": int(price),
        "maker_bid": int(maker_bid),
        "maker_ask": int(maker_ask),
        "aggress": aggress_code,
    })
    log_trade(buyer, seller, price, maker_bid, maker_ask, aggress_code)


# =============================
# LESS-STUPID ECON STUDENT (as requested)
# =============================
def less_stupid_econ_student_player(action, best_bid, best_ask, players_card, communal_cards):
    fair_value = OPENING_AVG_CARD_VALUE * (AMOUNT_OF_PLAYERS - 1) + sum(communal_cards) + players_card

    bid_spread = max(1, np.random.poisson(5))
    ask_spread = max(1, np.random.poisson(5))
    risk_buffer = np.random.poisson(2)

    if action == "setting":
        bid = int(round(fair_value - bid_spread, 0))
        ask = int(round(fair_value + ask_spread, 0))
        return bid, ask

    if best_bid is None or best_ask is None:
        return ""
    if fair_value - risk_buffer > best_ask:
        return "HA"
    if fair_value + risk_buffer < best_bid:
        return "HB"
    return ""


# =============================
# SMART ECON STUDENT (lightweight inference engine)
# =============================
def _ceil_int(x: float) -> int:
    return int(math.ceil(x))


def _nearest_card(value: float, card_set: set):
    if not card_set:
        return None
    return min(card_set, key=lambda c: (abs(c - value), c))


def _std_ceil(vals):
    vals = list(vals)
    if len(vals) <= 1:
        return 0
    return _ceil_int(float(np.std(vals)))


def init_beliefs(player_types):
    beliefs = {}
    for observer_i in range(1, AMOUNT_OF_PLAYERS):
        if player_types[observer_i] != ARCHETYPE_SMART:
            continue
        beliefs[observer_i] = {}
        for target_j in range(AMOUNT_OF_PLAYERS):
            if target_j == observer_i:
                continue
            beliefs[observer_i][target_j] = None
    return beliefs


def update_constraint_recent(beliefs, observer_i, target_j, lb, ub, tstep):
    if observer_i not in beliefs:
        return
    if target_j not in beliefs[observer_i]:
        return
    beliefs[observer_i][target_j] = {
        "lb": float(-1e18 if lb is None else lb),
        "ub": float(1e18 if ub is None else ub),
        "t": int(tstep),
    }


def compute_context_for_observer(observer_i, exclude_target, player_cards, communal_cards, beliefs):
    communal_sum = sum(communal_cards)
    own_card = player_cards[observer_i]

    remaining = set(CARDS) - set(communal_cards) - {own_card}

    other_players = [p for p in range(AMOUNT_OF_PLAYERS) if p != observer_i and (exclude_target is None or p != exclude_target)]

    constrained = []
    cons_map = beliefs.get(observer_i, {})
    for p in other_players:
        c = cons_map.get(p, None)
        if c is not None:
            constrained.append(p)

    poss = {}
    for p in constrained:
        c = cons_map[p]
        lb, ub = c["lb"], c["ub"]
        poss[p] = {x for x in remaining if (x > lb) and (x < ub)}

    assignments = {}
    for p in sorted(constrained, key=lambda k: len(poss[k])):
        cand = poss[p] & remaining
        if not cand:
            cand = set(remaining)
        m = float(np.mean(list(cand))) if cand else 0.0
        pick = _nearest_card(m, cand)
        if pick is None:
            pick = 0
        assignments[p] = pick
        if pick in remaining:
            remaining.remove(pick)

    assigned_sum = sum(assignments.values())
    assigned_count = len(assignments)
    unconstrained_count = len(other_players) - assigned_count

    avg_remaining = float(np.mean(list(remaining))) if remaining else 0.0
    other_sum = communal_sum + own_card + assigned_sum + unconstrained_count * avg_remaining

    return {
        "other_sum": float(other_sum),
        "remaining_cards": remaining,
        "avg_remaining": float(avg_remaining),
        "assignments": assignments,
        "unconstrained_count": unconstrained_count,
    }


def smart_fair_value_and_spread(observer_i, player_cards, communal_cards, beliefs, tstep):
    communal_sum = sum(communal_cards)
    deck_excl_communal = list(set(CARDS) - set(communal_cards))
    communal_fv = communal_sum + float(np.mean(deck_excl_communal)) * AMOUNT_OF_PLAYERS

    ctx = compute_context_for_observer(observer_i, exclude_target=None, player_cards=player_cards,
                                       communal_cards=communal_cards, beliefs=beliefs)
    total_est = ctx["other_sum"]

    if tstep <= 3:
        mid = int(round(0.85 * communal_fv + 0.15 * total_est))
        extra = 3
        widen_mult = 1.6
    else:
        mid = int(round(total_est))
        extra = 0
        widen_mult = 1.0

    remaining = list(ctx["remaining_cards"])
    avg_remaining = ctx["avg_remaining"]
    avg_round = max(_ceil_int(avg_remaining), 1)

    upper = [c for c in remaining if c > avg_round]
    lower = [c for c in remaining if c < avg_round]

    upper_std = _std_ceil(upper)
    lower_std = _std_ceil(lower)

    upper_std = _ceil_int(widen_mult * upper_std) + extra
    lower_std = _ceil_int(widen_mult * lower_std) + extra

    ask = mid + avg_round + upper_std
    bid = mid - avg_round - lower_std
    return bid, ask


def smart_take_intent(observer_i, best_bid, best_ask, player_cards, communal_cards, beliefs):
    if best_bid is None or best_ask is None:
        return ""

    ctx = compute_context_for_observer(observer_i, exclude_target=None, player_cards=player_cards,
                                       communal_cards=communal_cards, beliefs=beliefs)
    fv = ctx["other_sum"]
    remaining = list(ctx["remaining_cards"])
    rb = _ceil_int(float(np.std(remaining))) if len(remaining) > 1 else 1
    rb = max(rb, 1)

    if fv - rb > best_ask:
        return "HA"
    if fv + rb < best_bid:
        return "HB"
    return ""


def smart_observe_events(beliefs, player_cards, communal_cards, offers, trade_events, tstep):
    quote_events = []
    for q in range(AMOUNT_OF_PLAYERS):
        oq = offers[q]
        if oq is None:
            continue
        quote_events.append({"type": "quote", "q": q, "bid": int(oq["bid"]), "ask": int(oq["ask"])})

    for observer_i in list(beliefs.keys()):
        for ev in quote_events:
            q = ev["q"]
            if q == observer_i:
                continue

            ctx = compute_context_for_observer(observer_i, exclude_target=q, player_cards=player_cards,
                                               communal_cards=communal_cards, beliefs=beliefs)
            other_sum = ctx["other_sum"]

            lb = ev["bid"] - other_sum
            ub = ev["ask"] - other_sum
            update_constraint_recent(beliefs, observer_i, q, lb=lb, ub=ub, tstep=tstep)

        for tr in trade_events:
            actor = tr["actor"]
            if actor == observer_i:
                continue

            ctx = compute_context_for_observer(observer_i, exclude_target=actor, player_cards=player_cards,
                                               communal_cards=communal_cards, beliefs=beliefs)
            other_sum = ctx["other_sum"]
            price = tr["price"]

            if tr["code"] == "HA":
                lb = price - other_sum
                update_constraint_recent(beliefs, observer_i, actor, lb=lb, ub=None, tstep=tstep)
            elif tr["code"] == "HB":
                ub = price - other_sum
                update_constraint_recent(beliefs, observer_i, actor, lb=None, ub=ub, tstep=tstep)


# =============================
# GAME INIT / ENGINE
# =============================
def draw_round():
    total_cards = AMOUNT_OF_PLAYERS + COMMUNAL_CARDS_NUMBER
    opening_cards = random.sample(CARDS, total_cards)
    communal_cards = opening_cards[:COMMUNAL_CARDS_NUMBER]
    player_cards = opening_cards[COMMUNAL_CARDS_NUMBER:COMMUNAL_CARDS_NUMBER + AMOUNT_OF_PLAYERS]
    return list(opening_cards), list(communal_cards), list(player_cards), player_cards[0], sum(opening_cards)


def start_game():
    opening_cards, communal_cards, player_cards, user_card, true_sum = draw_round()

    # 4 bots = smart, with one special seat that has 10% chance to be less-stupid (else smart)
    player_types = ["YOU"] + [ARCHETYPE_SMART] * (AMOUNT_OF_PLAYERS - 1)
    special_seat = random.randint(1, AMOUNT_OF_PLAYERS - 1)
    if random.random() < P_SPECIAL_LESS_STUPID:
        player_types[special_seat] = ARCHETYPE_LESS_STUPID

    beliefs = init_beliefs(player_types)
    offers = [None] * AMOUNT_OF_PLAYERS

    now = time.time()
    ss = st.session_state
    ss.started = True
    ss.done = False
    ss.start_ts = now
    ss.end_ts = now + GAME_SECONDS
    ss.last_tick_ts = now
    ss.tick_id = 0

    ss.opening_cards = opening_cards
    ss.communal_cards = communal_cards
    ss.player_cards = player_cards
    ss.user_card = user_card
    ss.true_sum = true_sum

    ss.player_types = player_types
    ss.beliefs = beliefs
    ss.trade_tape = []
    ss.trade_log = []
    ss.offers = offers
    ss.last_user_trade_ts = 0.0
    ss.state_error = None

    # initial bot quotes at tstep=1
    tstep = 1
    for i in range(1, AMOUNT_OF_PLAYERS):
        if player_types[i] == ARCHETYPE_SMART:
            bid, ask = smart_fair_value_and_spread(i, player_cards, communal_cards, beliefs, tstep=tstep)
        else:
            bid, ask = less_stupid_econ_student_player("setting", None, None, player_cards[i], communal_cards)
        offers[i] = make_quote(bid, ask)


def update_bot_quotes_each_tick(tstep):
    ss = st.session_state
    for i in range(1, AMOUNT_OF_PLAYERS):
        if ss.player_types[i] == ARCHETYPE_SMART:
            bid, ask = smart_fair_value_and_spread(i, ss.player_cards, ss.communal_cards, ss.beliefs, tstep=tstep)
        else:
            bid, ask = less_stupid_econ_student_player("setting", None, None, ss.player_cards[i], ss.communal_cards)
        ss.offers[i] = make_quote(bid, ask)


def take_intent_for_bot(i, best_bid, best_ask):
    ss = st.session_state
    if ss.player_types[i] == ARCHETYPE_SMART:
        return smart_take_intent(i, best_bid, best_ask, ss.player_cards, ss.communal_cards, ss.beliefs)
    return less_stupid_econ_student_player("taking", best_bid, best_ask, ss.player_cards[i], ss.communal_cards)


def process_ticks_if_needed():
    ss = st.session_state
    if not game_ready() or ss.done:
        return

    now = time.time()
    if now >= ss.end_ts:
        ss.done = True
        return

    elapsed = now - ss.last_tick_ts
    if elapsed < TICK_SECONDS:
        return

    n_ticks = min(int(elapsed // TICK_SECONDS), 5)

    for _ in range(n_ticks):
        ss.tick_id += 1
        tstep = ss.tick_id

        # 1) refresh bot quotes
        update_bot_quotes_each_tick(tstep=tstep)

        # 2) bots may trade (collect trade events for smart inference)
        trade_events = []

        for actor in range(1, AMOUNT_OF_PLAYERS):
            if random.random() > P_OTHER_ACT_PER_TICK:
                continue

            best_bid, best_bid_player, best_ask, best_ask_player = get_best_bid_ask(ss.offers)
            if best_bid is None or best_ask is None:
                break

            intent = take_intent_for_bot(actor, best_bid, best_ask)

            if intent == "HA":
                maker = best_ask_player
                if maker is None or maker == actor:
                    continue
                maker_q = ss.offers[maker]
                price = maker_q["ask"]

                record_trade(
                    buyer=actor,
                    seller=maker,
                    price=price,
                    maker_bid=maker_q["bid"],
                    maker_ask=maker_q["ask"],
                    aggress_code="HA",
                )
                trade_events.append({"type": "trade", "actor": actor, "code": "HA", "price": price})

            elif intent == "HB":
                maker = best_bid_player
                if maker is None or maker == actor:
                    continue
                maker_q = ss.offers[maker]
                price = maker_q["bid"]

                record_trade(
                    buyer=maker,
                    seller=actor,
                    price=price,
                    maker_bid=maker_q["bid"],
                    maker_ask=maker_q["ask"],
                    aggress_code="HB",
                )
                trade_events.append({"type": "trade", "actor": actor, "code": "HB", "price": price})

        # 3) smart players observe quotes + trades
        if ss.beliefs:
            smart_observe_events(ss.beliefs, ss.player_cards, ss.communal_cards, ss.offers, trade_events, tstep=tstep)

        if len(ss.trade_tape) > 700:
            ss.trade_tape = ss.trade_tape[-450:]

    ss.last_tick_ts = time.time()
    if ss.last_tick_ts >= ss.end_ts:
        ss.done = True


# =============================
# END STATS
# =============================
def build_player_trade_summary():
    ss = st.session_state
    true_sum = ss.true_sum

    per_player = {i: [] for i in range(AMOUNT_OF_PLAYERS)}
    buys = {i: 0 for i in range(AMOUNT_OF_PLAYERS)}
    sells = {i: 0 for i in range(AMOUNT_OF_PLAYERS)}

    for tr in ss.trade_log:
        b, s, p = tr["buyer"], tr["seller"], tr["price"]
        buys[b] += 1
        sells[s] += 1

        buyer_profit = true_sum - p
        per_player[b].append(buyer_profit)
        per_player[s].append(-buyer_profit)

    rows = []
    for i in range(AMOUNT_OF_PLAYERS):
        profits = per_player[i]
        n = len(profits)
        wins = sum(1 for x in profits if x > 0)
        losses = sum(1 for x in profits if x < 0)
        win_rate = (wins / n) if n else 0.0
        avg_pnl = float(np.mean(profits)) if n else 0.0
        avg_win = float(np.mean([x for x in profits if x > 0])) if wins else 0.0
        avg_loss = float(np.mean([x for x in profits if x < 0])) if losses else 0.0

        rows.append({
            "Player": f"P{i}",
            "Trades": n,
            "Buys": buys[i],
            "Sells": sells[i],
            "Win rate (%)": round(100 * win_rate, 1),
            "Avg gain/loss": round(avg_pnl, 2),
            "Avg win": round(avg_win, 2),
            "Avg loss": round(avg_loss, 2),
            "Total PnL": round(sum(profits), 2),
        })

    return pd.DataFrame(rows)


# =============================
# APP
# =============================
def app():
    st.title("Trading Game (Continuous 2-Minute Round)")

    ensure_state()
    live = st.sidebar.checkbox("Live auto-refresh", value=True)

    # Guard against partial state after hot reloads
    if st.session_state.started and not game_ready():
        reset_to_start("State became incomplete after a rerun/hot-reload. Press Start again.")
        st.rerun()

    # -------------------------
    # START SCREEN
    # -------------------------
    if not st.session_state.started:
        if st.session_state.state_error:
            st.warning(st.session_state.state_error)

        st.markdown(f"**Current time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown(INSTRUCTIONS_MD)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ Start Game"):
                start_game()
                st.rerun()
        with c2:
            if st.button("🧹 Hard Reset"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
        st.stop()

    # -------------------------
    # TICK ENGINE
    # -------------------------
    process_ticks_if_needed()

    ss = st.session_state
    now = time.time()

    remaining = max(0, int(ss.end_ts - now)) if ss.end_ts is not None else 0
    progress = 0.0
    if ss.start_ts is not None and ss.end_ts is not None and ss.end_ts > ss.start_ts:
        progress = min(1.0, max(0.0, (now - ss.start_ts) / GAME_SECONDS))

    colA, colB = st.columns([1, 2])

    with colA:
        st.subheader("Round Info")
        st.write(f"Time remaining: **{remaining}s**")
        st.progress(progress)
        st.write(f"Your card: **{ss.user_card}**")
        st.write("Communal cards:")
        st.table(pd.DataFrame({"Card": listify(ss.communal_cards)}))

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Restart Round"):
                start_game()
                st.rerun()
        with c2:
            if st.button("⏹ End Now"):
                ss.done = True
                st.rerun()

    with colB:
        st.subheader("Live Market Quotes (no size)")
        best_bid, best_bid_player, best_ask, best_ask_player = get_best_bid_ask(ss.offers)
        st.dataframe(offers_df(ss.offers), use_container_width=True)

        if best_bid is None:
            st.warning("No quotes in market.")
        else:
            st.info(f"Best Bid = **{best_bid}** (P{best_bid_player}) | Best Ask = **{best_ask}** (P{best_ask_player})")

    st.divider()

    # -------------------------
    # END OF ROUND (REVEAL)
    # -------------------------
    if ss.done:
        st.subheader("✅ Round Complete — Reveal")
        st.write("All cards drawn (communal + players):")
        st.table(pd.DataFrame({"Cards": listify(ss.opening_cards)}))
        st.success(f"TRUE TOTAL SUM = **{ss.true_sum}**")

        st.subheader("Player Archetypes (revealed at end)")
        types_table = pd.DataFrame({
            "Player": [f"P{i}" for i in range(AMOUNT_OF_PLAYERS)],
            "Archetype": ["YOU"] + [ARCHETYPE_DISPLAY.get(t, t) for t in ss.player_types[1:]],
        })
        st.table(types_table)

        st.subheader("Per-Player Trade Summary")
        st.dataframe(build_player_trade_summary(), use_container_width=True)

        st.subheader("Trade Tape")
        st.text("\n".join(ss.trade_tape) if ss.trade_tape else "No trades logged.")
        st.stop()

    # -------------------------
    # USER CONTROLS (live)
    # -------------------------
    st.subheader("Your Controls (live)")

    left, right = st.columns(2)

    # ---- Trade UI: show all players (no dropdown) ----
    with left:
        st.markdown("### 1) Trade against AI players (all shown)")

        # show current cooldown state
        now_ts = time.time()
        seconds_since = now_ts - float(ss.last_user_trade_ts or 0.0)
        remaining_cd = max(0.0, USER_TRADE_COOLDOWN_SECONDS - seconds_since)

        if remaining_cd > 0:
            st.warning(f"Cooldown: you can trade again in **{remaining_cd:.1f}s**")

        # one radio per player (P1..P4)
        user_trades = []
        for opp in range(1, AMOUNT_OF_PLAYERS):
            oq = ss.offers[opp]
            qtxt = "(no quote)" if oq is None else f"quote {oq['bid']}/{oq['ask']}"
            st.write(f"**P{opp}** — {qtxt}")

            # ✅ CHANGE: swap order so HB is shown before HA
            dec = st.radio(
                f"Action vs P{opp}",
                ["", "HB", "HA"],
                horizontal=True,
                key=f"user_dec_live_{opp}",
            )
            if dec in ("HA", "HB"):
                user_trades.append((opp, dec))

        execute_disabled = (remaining_cd > 0)
        execute_trades = st.button("Execute selected trades NOW", disabled=execute_disabled)

    # ---- Post quote ----
    with right:
        st.markdown("### 2) Post / update your quote (live)")
        user_bid = st.number_input("Your bid", value=0, step=1, key="user_bid_live")
        user_ask = st.number_input("Your ask", value=0, step=1, key="user_ask_live")
        post_quote = st.button("Post/Update my quote NOW")

    # -------------------------
    # HANDLE USER ACTIONS
    # -------------------------
    if post_quote:
        if user_bid == 0 or user_ask == 0:
            st.warning("Enter non-zero bid and ask to quote.")
        else:
            q = make_quote(int(user_bid), int(user_ask))
            ss.offers[0] = q
            log_quote(0, q["bid"], q["ask"])
            st.rerun()

    if execute_trades:
        now_ts = time.time()
        seconds_since = now_ts - float(ss.last_user_trade_ts or 0.0)
        if seconds_since < USER_TRADE_COOLDOWN_SECONDS:
            st.warning(f"Too soon — wait **{USER_TRADE_COOLDOWN_SECONDS - seconds_since:.1f}s** then trade again.")
        elif not user_trades:
            st.warning("Select at least one HA/HB action.")
        else:
            # mark cooldown start (one action every 2 seconds)
            ss.last_user_trade_ts = now_ts

            # apply trades at the AI player's quoted bid/ask
            user_trade_events = []
            for opp, dec in user_trades:
                oq = ss.offers[opp]
                if oq is None:
                    ss.trade_tape.append(f"[{_fmt_mmss_since_start(time.time())}] NO TRADE: P0 tried vs P{opp} but no quote.")
                    continue

                if dec == "HA":
                    record_trade(
                        buyer=0,
                        seller=opp,
                        price=oq["ask"],
                        maker_bid=oq["bid"],
                        maker_ask=oq["ask"],
                        aggress_code="HA",
                    )
                    user_trade_events.append({"type": "trade", "actor": 0, "code": "HA", "price": int(oq["ask"])})

                elif dec == "HB":
                    record_trade(
                        buyer=opp,
                        seller=0,
                        price=oq["bid"],
                        maker_bid=oq["bid"],
                        maker_ask=oq["ask"],
                        aggress_code="HB",
                    )
                    user_trade_events.append({"type": "trade", "actor": 0, "code": "HB", "price": int(oq["bid"])})

            # smart players observe immediately
            if ss.beliefs and user_trade_events:
                ss.tick_id += 1
                smart_observe_events(ss.beliefs, ss.player_cards, ss.communal_cards, ss.offers, user_trade_events, tstep=ss.tick_id)

            st.rerun()

    st.divider()
    st.subheader("Live Trade Tape (most recent at bottom)")
    st.text("\n".join(ss.trade_tape[-90:]) if ss.trade_tape else "No events yet.")
    st.caption("PnL + archetypes stay hidden until the 2-minute round ends.")

    # Auto-refresh loop
    if live and not ss.done:
        time.sleep(TICK_SECONDS)
        st.rerun()


app()
