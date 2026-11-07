"""
The bot's brain: hand tracking, opponent models, preflop and postflop
decisions. Everything here is plain functions over plain data, so the Model
that holds the data pickles cleanly.

Units: chips inside the tracker, big blinds inside the preflop tree.
"""

import math
import time

import numpy as np

import pbcore as pc
import pbpre
from player_action import DEALER, PlayerAction

STACKS = (3, 5, 8, 12, 16, 20, 25, 30, 40, 60, 100)


# -- data built once, before pickling --------------------------------------------


def load_data(folder="lab/data"):
    """Everything precomputed, as plain numpy arrays and dicts. Called by
    Model.__init__ while pickling; never while playing."""
    eq = np.load(folder + "/preflop_eq.npy")
    nan = np.isnan(eq)
    eq16 = np.where(nan, 0, np.round(eq * 60000)).astype(np.uint16)
    strength = np.nanmean(eq, axis=1).astype(np.float32)  # vs random hand
    trees = {}
    for s in STACKS:
        try:
            z = np.load(f"{folder}/preflop_{s:g}.npz")
        except FileNotFoundError:
            continue
        nodes = pbpre.build_tree(float(s))
        strat = {}
        for n in nodes:
            if n["player"] >= 0:
                strat[n["id"]] = z[f"n{n['id']}"].astype(np.float16)
        trees[s] = {"nodes": nodes, "strat": strat,
                    "freq": _node_freqs(nodes, strat)}
    return {"eq": eq16, "valid": ~nan, "strength": strength, "trees": trees}


def _node_freqs(nodes, strat):
    """Equilibrium action frequencies at each player node (reach-weighted,
    card removal ignored): the opponent model's prior."""
    freq = {}
    reach = {0: (np.ones(1326), np.ones(1326))}
    for n in nodes:  # children always come after parents
        if n["player"] < 0 or n["id"] not in reach:
            continue
        rs, rb = reach[n["id"]]
        sig = strat[n["id"]].astype(np.float64)
        r = rs if n["player"] == 0 else rb
        tot = r.sum()
        freq[n["id"]] = [float((r * sig[:, a]).sum() / max(tot, 1e-12))
                         for a in range(sig.shape[1])]
        for a, kid in enumerate(n["kids"]):
            if n["player"] == 0:
                reach[kid] = (rs * sig[:, a], rb)
            else:
                reach[kid] = (rs, rb * sig[:, a])
    return freq


# -- opponent statistics --------------------------------------------------------

# Priors measured on AI-written bots (lab/measure.py): they rarely fold to
# small bets, rarely bet when checked to and almost never bluff.
POST_PRIOR = {
    "bet": 0.22,  # bets when checked to / first to act
    "fold_s": 0.06, "fold_m": 0.18, "fold_l": 0.33, "fold_a": 0.72,
    "fold_r": 0.20,  # folds when raised after betting (their range is strong)
    "raise": 0.08,  # raises when facing a bet
    "bluff": 0.08,  # share of aggressive actions made with weak hands
}
FOLD_MEAN = (0.06 + 0.18 + 0.33 + 0.72) / 4
K_POST = 6.0
K_PRE = 6.0
LAM_K = 12.0  # preflop: P(play the best response) = hands / (hands + LAM_K)
BET_SIZES = (0.33, 0.66, 1.0)  # pot fractions when betting
RAISE_SIZES = (0.75, 1.2)  # pot fractions when raising a bet


def new_opp():
    return {"hands": 0, "pre": {}, "post": {}, "net": 0.0,
            "bet_size": [0.0, 0.0]}


def post_rate(opp, key):
    n_hit, n_all = opp["post"].get(key, (0.0, 0.0))
    prior = POST_PRIOR[key]
    if key.startswith("fold_") and key != "fold_r":
        # Share evidence across sizes: a bot that never folds to small bets
        # rarely folds to big ones either.
        tot_hit = sum(opp["post"].get(k, (0, 0))[0]
                      for k in ("fold_s", "fold_m", "fold_l", "fold_a"))
        tot_all = sum(opp["post"].get(k, (0, 0))[1]
                      for k in ("fold_s", "fold_m", "fold_l", "fold_a"))
        g = (tot_hit + K_POST * FOLD_MEAN) / (tot_all + K_POST)
        if FIX_SCALE:
            # Bounded: folding to small bets says little about all-ins.
            scale = min(1.3, max(0.5, g / FOLD_MEAN))
            prior = min(0.8, max(0.01, prior * scale))
        else:
            prior = min(0.97, max(0.01, prior * g / FOLD_MEAN))
    return (n_hit + K_POST * prior) / (n_all + K_POST)


LEARN = True  # the lab sets this False for the no-learning baseline

# Fixes from the independent review (browser Claude), switchable for A/B.
FIX_SCALE = True  # bound cross-size fold sharing (cap 0.8)
FIX_SIZEKEY = True  # an all-in uses its real size bucket, not "fold_a"
FIX_OVERBET = True  # no all-in over 2x pot unless equity when called >= .45
FIX_RR_CONSIST = False  # raise_ev uses the observe_opp re-raise model
FIX_RAISE_BLUFF = True  # opponent raises: value from the top, bluffs from
#                         the bottom (the old model made raises pure value)
FIX_REALISE = True  # realised equity never above the equity itself
OVERBET_MIN_N = 8  # observations needed before overbet bluffs are allowed


ALPHA_VAL = 1.0  # bluff inference from bet frequency: off (A/B test: it lost chips)


def bluff_rate(opp):
    """Share of the opponent's aggression that is bluffs. Showdowns alone
    under-count bluffs (bluffs usually end in a fold, unseen), so also
    infer it from how often they bet: betting more than the ~30% of hands
    a value bettor has means the rest are bluffs."""
    shown = post_rate(opp, "bluff")
    n_bets = opp["post"].get("bet", (0.0, 0.0))[1]
    alpha = post_rate(opp, "bet")
    implied = max(0.0, (alpha - ALPHA_VAL) / max(alpha, 1e-9))
    trust = n_bets / (n_bets + 10.0)
    return max(shown, trust * implied)


def post_add(opp, key, hit):
    if not LEARN:
        return
    n_hit, n_all = opp["post"].get(key, (0.0, 0.0))
    opp["post"][key] = (n_hit + (1.0 if hit else 0.0), n_all + 1.0)


def size_key(frac):
    if frac >= 3.0:
        return "fold_a"
    if frac <= 0.4:
        return "fold_s"
    if frac <= 0.8:
        return "fold_m"
    return "fold_l"


# -- the hand tracker -----------------------------------------------------------


def new_hand(model, gs, start=None):
    me = model.player_id
    n = gs.num_players
    if start is None:
        start = [gs.stacks[p] + gs.bets[p] for p in range(n)]
    bb = gs.big_blind
    h = {
        "n": n, "me": me, "bb": bb, "sb": gs.small_blind,
        "button": gs.button, "start": start,
        "contrib": [0] * n, "bets": [0] * n, "folded": [False] * n,
        "board": [], "street": "preflop", "seen": 0,
        "hole": [pc.card_id(c) for c in model.hand],
        "log": [],  # (street, player, kind, total, facing, pot) per action
        "opp_agg": [],  # (street, board_len) of opponent aggression
    }
    # Blinds, as the engine posts them.
    small = gs.button if n == 2 else (gs.button + 1) % n
    big = (small + 1) % n
    for p, amt in ((small, gs.small_blind), (big, bb)):
        amt = min(amt, start[p])
        h["bets"][p] += amt
        h["contrib"][p] += amt
    if n == 2:
        opp = 1 - me
        h["opp"] = opp
        h["team"] = gs.teams[opp]
        eff = min(start) / bb
        solved = list(model.data["trees"]) or list(STACKS)
        h["S"] = min(solved, key=lambda s: abs(math.log(max(eff, 1) / s)))
        tree = model.data["trees"].get(h["S"])
        h["node"] = 0 if tree is not None and eff >= 2 else None
        w = np.ones(1326)
        w[pc.blocked(h["hole"])] = 0.0
        h["w"] = w
    return h


def apply_action(model, h, a):
    """Updates the tracker (and the opponent model) with one action."""
    p, kind = a.player_id, a.kind
    if p == DEALER:
        if kind == "reveal_card":
            h["board"].append(pc.card_id(a.card))
            nb = len(h["board"])
            if nb in (3, 4, 5):
                h["street"] = {3: "flop", 4: "turn", 5: "river"}[nb]
                h["bets"] = [0] * h["n"]
                if h["n"] == 2:
                    h["w"][pc.blocked(h["board"])] = 0.0
                    h["strength"] = None
                    h["node"] = None  # postflop: the tree is done
        return
    if kind not in ("fold", "call", "raise"):
        return
    bets = h["bets"]
    facing = max(bets) - bets[p]
    pot = sum(h["contrib"])
    left = h["start"][p] - h["contrib"][p]
    if h["n"] == 2 and p == h.get("opp"):
        observe_opp(model, h, kind, a.amount, facing, pot)
    if kind == "fold":
        h["folded"][p] = True
    elif kind == "call":
        amt = min(facing, left)
        bets[p] += amt
        h["contrib"][p] += amt
    else:
        add = a.amount - bets[p]
        bets[p] = a.amount
        h["contrib"][p] += add
    h["log"].append((h["street"], p, kind, a.amount, facing, pot))
    if h["n"] == 2 and h["street"] == "preflop" and h.get("node") is not None:
        advance_node(model, h, p, kind, a.amount)


OFF_TREE = 0.47  # log gap (about 1.6x) beyond which a raise is off the tree


def advance_node(model, h, p, kind, amount):
    tree = model.data["trees"][h["S"]]
    nodes = tree["nodes"]
    node = nodes[h["node"]]
    pos = 0 if p == h["button"] else 1
    if node["player"] != pos:
        h["node"] = None
        return
    label = {"fold": "F", "call": "C", "raise": "R"}[kind]
    total = amount / h["bb"]
    if kind == "raise" and amount >= h["start"][p]:
        total = float(h["S"])  # all-in
    i, dist = pbpre.child_for(node, nodes, label, total, h["S"],
                              STRICT_ALLIN)
    off = False
    if i is None and FIX_4BET_RANGE and label == "R" and STRICT_ALLIN:
        # A non-all-in raise where the tree only has all-in (a 3x 4-bet):
        # narrow their range with the all-in policy, then play real chips.
        i, dist = pbpre.child_for(node, nodes, label, total, h["S"], False)
        off = True
    if i is None:
        h["node"] = None
        return
    if p == h.get("opp"):
        # Range update and statistics at this node.
        pol = opp_pre_policy(model, h, node)
        h["w"] = h["w"] * pol[:, i]
        opp = model.opp[h["team"]]
        cnt = opp["pre"].setdefault(f"{h['S']}:{node['id']}",
                                    [0.0] * len(node["acts"]))
        if LEARN:
            cnt[i] += 1.0
    kid = node["kids"][i]
    h["node"] = kid if nodes[kid]["player"] >= 0 else None
    if off or (dist is not None and dist > OFF_TREE):
        # Far from every tree size: the range update above still uses the
        # nearest size, but decisions now use the real chips.
        h["node"] = None


# -- preflop -------------------------------------------------------------------


def _eq_row(model, combo):
    return model.data["eq"][combo].astype(np.float64) / 60000.0


def _percentile(score, w):
    """Weighted percentile (0..1) of every combo's score inside range w."""
    order = np.argsort(score, kind="stable")
    ws = w[order]
    cum = np.cumsum(ws)
    tot = cum[-1] if len(cum) else 0.0
    q = np.zeros(1326)
    if tot <= 0:
        return q
    q[order] = (cum - ws / 2) / tot
    return q


AGGR = {"F": 0, "C": 1, "R": 2, "A": 3}


def _slices(q, freqs, acts, temp=0.03):
    """A per-combo policy that plays each action with its frequency,
    weakest combos (low q) on the most passive action."""
    order = sorted(range(len(acts)),
                   key=lambda i: (AGGR[acts[i][0]], acts[i][1]))
    pol = np.zeros((1326, len(acts)))
    lo = 0.0
    for i in order:
        hi = lo + freqs[i]
        a = 1 / (1 + np.exp(-(q - lo) / temp)) if lo > 0 else 1.0
        b = 1 / (1 + np.exp(-(q - hi) / temp)) if hi < 1 else 0.0
        pol[:, i] = np.clip(a - b, 0, 1)
        lo = hi
    s = pol.sum(1, keepdims=True)
    return pol / np.where(s > 0, s, 1)


def opp_pre_policy(model, h, node, w=None):
    """The opponent's per-combo policy at a tree node: equilibrium,
    shifted to frequency slices as we see how they actually play."""
    tree = model.data["trees"][h["S"]]
    eqp = tree["strat"][node["id"]].astype(np.float64)
    opp = model.opp[h["team"]]
    cnt = opp["pre"].get(f"{h['S']}:{node['id']}")
    prior = tree["freq"][node["id"]]
    if not cnt or sum(cnt) == 0:
        return eqp
    n = sum(cnt)
    freqs = [(c + K_PRE * f) / (n + K_PRE) for c, f in zip(cnt, prior)]
    mu = n / (n + 8.0)
    w = h["w"] if w is None else w
    q = _percentile(model.data["strength"].astype(np.float64), w)
    return (1 - mu) * eqp + mu * _slices(q, freqs, node["acts"])


def _pre_ev(model, h, nid, w, erow, frow):
    nodes = model.data["trees"][h["S"]]["nodes"]
    n = nodes[nid]
    mepos = 0 if h["me"] == h["button"] else 1
    c_me, c_op = n["c"][mepos], n["c"][1 - mepos]
    tot = w.sum()
    if n["player"] < 0:
        if n["kind"] == "fold":
            return -c_me * tot if n["loser"] == mepos else c_op * tot
        pot = c_me + c_op
        row = erow if n["kind"] == "show" else frow
        return pot * float(row @ w) - c_me * tot
    if n["player"] == mepos:
        return max(_pre_ev(model, h, k, w, erow, frow) for k in n["kids"])
    pol = opp_pre_policy(model, h, n, w)
    return sum(_pre_ev(model, h, k, w * pol[:, a], erow, frow)
               for a, k in enumerate(n["kids"]))


def preflop_decide(model, h, gs, rng):
    tree = model.data["trees"][h["S"]]
    nodes = tree["nodes"]
    node = nodes[h["node"]]
    combo = pc.combo_of(*h["hole"])
    sig = tree["strat"][node["id"]][combo].astype(np.float64)
    sig = sig / max(sig.sum(), 1e-9)
    opp = model.opp[h["team"]]
    lam = opp["hands"] / (opp["hands"] + LAM_K)
    choice = int(rng.choice(len(sig), p=sig))
    if opp["hands"] >= 4:
        erow = _eq_row(model, combo)
        valid = model.data["valid"][combo]
        e_sb = erow if h["me"] == h["button"] else 1 - erow
        share_sb = pbpre.flop_share_sb(e_sb)
        frow = (share_sb if h["me"] == h["button"] else 1 - share_sb) * valid
        erow = erow * valid
        w = h["w"] * valid
        if w.sum() > 0:
            evs = np.array([_pre_ev(model, h, k, w, erow, frow)
                            for k in node["kids"]]) / w.sum()
            best = int(np.argmax(evs))
            mixed = float((sig * evs).sum())
            h["pre_evs"] = evs
            # Deviate from equilibrium only when the model says it pays.
            if evs[best] - mixed > 0.04 and rng.random() < lam:
                choice = best
    return tree_action(h, gs, node["acts"][choice])


def tree_action(h, gs, act):
    label, total = act
    me = h["me"]
    if label == "F":
        if gs.call_amount == 0:
            return PlayerAction.call(me)
        return PlayerAction.fold(me)
    if label == "C" or not gs.can_raise:
        return PlayerAction.call(me)
    lo, hi = gs.raise_range
    if label == "A":
        return PlayerAction.raise_to(me, hi)
    amt = int(round(total * h["bb"]))
    return PlayerAction.raise_to(me, min(hi, max(lo, amt)))


# -- postflop ------------------------------------------------------------------


def board_strength(h, rng):
    if h.get("strength") is None:
        if h["board"]:
            h["strength"] = pc.strength_vs_random(h["board"], 40, rng)
        else:
            h["strength"] = None
    return h["strength"]


def _sig(x):
    return 1 / (1 + np.exp(-x))


def agg_prob(q, alpha, beta, temp=0.05):
    """P(an aggressive action | range percentile q): value from the top,
    bluffs from the bottom, alpha of the range in total."""
    t_val = 1 - alpha * (1 - beta)
    t_bl = alpha * beta
    return np.clip(_sig((q - t_val) / temp) + _sig((t_bl - q) / temp), 0, 1)


def observe_opp(model, h, kind, amount, facing, pot):
    """Postflop: statistics and the range update for one opponent action
    (preflop is handled by advance_node)."""
    if h["street"] == "preflop":
        return
    opp = model.opp[h["team"]]
    rng = model._rng
    s = board_strength(h, rng)
    if s is None:
        return
    w = h["w"]
    q = _percentile(np.nan_to_num(s), w)
    beta = bluff_rate(opp)
    if facing == 0:
        alpha = post_rate(opp, "bet")
        bet = kind == "raise"
        post_add(opp, "bet", bet)
        p = agg_prob(q, alpha, beta)
        h["w"] = w * (p if bet else 1 - p) + 1e-9 * w
        if bet:
            frac = amount / max(pot, 1)
            opp["bet_size"][0] += frac
            opp["bet_size"][1] += 1
            h["opp_agg"].append(len(h["board"]))
    else:
        frac = facing / max(pot - facing, 1)
        # Raised after betting this street: a different, stronger range.
        key = "fold_r" if h["bets"][h["opp"]] > 0 else size_key(frac)
        f = post_rate(opp, key)
        r = post_rate(opp, "raise")
        can_rr = True
        if FIX_OBS_ALLIN:
            # They cannot re-raise when we are all-in or they can only call.
            me = h["me"]
            can_rr = (h["start"][me] > h["contrib"][me]
                      and h["start"][h["opp"]] - h["contrib"][h["opp"]] > facing)
        post_add(opp, key, kind == "fold")
        if kind != "fold" and can_rr:
            post_add(opp, "raise", kind == "raise")
        p_fold = _sig((f - q) / 0.05)
        if not can_rr:
            p_raise = np.zeros_like(q)
        elif FIX_RAISE_BLUFF:
            t_val = 1 - r * (1 - beta)
            p_raise = np.clip(_sig((q - t_val) / 0.05)
                              + _sig((r * beta - q) / 0.05), 0, 1)
            p_fold = p_fold * (1 - p_raise)
        else:
            p_raise = agg_prob(q, r, beta) * (1 - p_fold)
        p_call = np.clip(1 - p_fold - p_raise, 0, 1)
        like = {"fold": p_fold, "call": p_call, "raise": p_raise}[kind]
        h["w"] = w * like + 1e-9 * w
        if kind == "raise":
            h["opp_agg"].append(len(h["board"]))


def _samples(h, n, rng):
    """Opponent combos drawn from the range and our result against each:
    (combo index, result 0/0.5/1)."""
    hole, board = h["hole"], h["board"]
    w = h["w"].copy()
    w[pc.blocked(hole + board)] = 0
    tot = w.sum()
    if tot <= 0:
        w = (~pc.blocked(hole + board)).astype(float)
        tot = w.sum()
    idx = rng.choice(1326, n, p=w / tot)
    opp = pc.COMBOS[idx]
    missing = 5 - len(board)
    pool = np.array([c for c in range(52) if c not in hole and c not in board])
    if missing:
        cand = pc._draw(rng, pool, n, missing + 2)
        ok = (cand != opp[:, :1]) & (cand != opp[:, 1:2])
        order = np.argsort(~ok, axis=1, kind="stable")[:, :missing]
        run = np.take_along_axis(cand, order, axis=1)
    else:
        run = np.zeros((n, 0), np.int64)
    bt = np.broadcast_to(np.array(board, np.int64), (n, len(board)))
    full = np.concatenate([bt, run], axis=1)
    mine = pc.evaluate_batch(np.concatenate(
        [np.broadcast_to(np.array(hole, np.int64), (n, 2)), full], axis=1))
    theirs = pc.evaluate_batch(np.concatenate([opp, full], axis=1))
    nxt = run[:, 0] if missing else None  # the next street's card
    return idx, (mine > theirs) + 0.5 * (mine == theirs), nxt


FIX_NORAISE_ALLIN = True  # Codex round 2, bug 1
STRICT_ALLIN = True  # Codex round 2, bug 2: no non-all-in raise on the A node
FIX_4BET_RANGE = False  # review4 B1: range update for unmapped 4-bets
FIX_OVER_COUNT = False  # review4 B2: overbet guard counts fold_l+fold_a
BIG_VALUE = False  # review4 4.2: 1.6-pot value size vs proven non-folders
FIX_OBS_ALLIN = False  # codex4 1.4: no phantom re-raise when they can only call
N_SAMPLES = 2500  # sampled opponent hands + runouts per postflop decision
SOFT_BUDGET = 2.0  # seconds per move we allow ourselves (hard limit: 10)
LOOKAHEAD = False  # value check/call lines one street further ahead
LOOK_W = 1.0  # weight of the lookahead value against the one-street value
LOOK_SIZES = (0.5, 1.0)  # our bet sizes on the next street


def next_street_value(opp, q, r, nxt, w, pot_n, eff_n):
    """Expected chips from the next street on, if we reach it with pot
    pot_n and eff_n behind, over the sampled worlds weighted by w. Per next
    card (known when we act then, so no hindsight) we pick the best of
    checking it through or betting into the opponent's modelled folds and
    calls. Their own betting next street is not modelled."""
    n = len(r)
    if nxt is None or eff_n <= 0:
        return float((w * r).sum()) * pot_n / n

    def by_card(x):
        return np.bincount(nxt, weights=x, minlength=52)

    best = by_card(w * r * pot_n)
    for frac in LOOK_SIZES:
        bet = min(eff_n, frac * pot_n)
        f = post_rate(opp, size_key(bet / max(pot_n, 1)))
        p_fold = _sig((f - q) / 0.05)
        term = w * (p_fold * pot_n
                    + (1 - p_fold) * (r * (pot_n + 2 * bet) - bet))
        best = np.maximum(best, by_card(term))
    return float(best.sum()) / n


def postflop_decide(model, h, gs, rng, budget):
    """EV-maximising action against the opponent's modelled range and
    responses. Works preflop too (off-tree spots), with an empty board."""
    me = h["me"]
    opp = model.opp[h["team"]]
    n = N_SAMPLES if budget > 1.0 else 600
    look_ok = budget > 0.5
    idx, r, nxt = _samples(h, n, rng)
    s = board_strength(h, rng)
    if s is None:  # preflop off the tree: strength = equity vs random
        s = model.data["strength"].astype(np.float64)
    q_all = _percentile(np.nan_to_num(s), h["w"])
    q = q_all[idx]
    pot = gs.pot
    call = gs.call_amount
    my_bet = gs.bets[me]
    eff = min(gs.stacks[me], max(gs.stacks[p] + gs.bets[p] - my_bet
                                 for p in gs.players_in_hand if p != me))
    beta = bluff_rate(opp)
    river = len(h["board"]) == 5
    # Realisation: equity is worth less before the river, more so out of
    # position and with weak hands.
    ip = me == h["button"]
    def realise(eq):
        if river:
            return eq
        k = 0.97 if ip else 0.9
        out = eq * (k + 0.1 * (eq - 0.5))
        return min(eq, out) if FIX_REALISE else out

    eq = float(r.mean())
    options = []  # (ev, action)

    def raise_ev(total):
        """EV (chips, from now) of raising to street total."""
        add = total - my_bet
        opp_add = total - max(gs.bets[p] for p in gs.players_in_hand
                              if p != me)
        frac = opp_add / max(pot + call, 1)
        allin = total >= gs.max_raise_to
        if call > 0:  # raising their bet: they fold like a bettor does
            f = post_rate(opp, "fold_r")
        elif FIX_SIZEKEY:
            f = post_rate(opp, size_key(frac))
        else:
            f = post_rate(opp, "fold_a" if allin else size_key(frac))
        p_fold = _sig((f - q) / 0.05)
        rr = 0.0 if allin else post_rate(opp, "raise")
        if FIX_NORAISE_ALLIN and allin:
            p_raise = np.zeros_like(q)  # nobody can re-raise an all-in
        elif FIX_RR_CONSIST:
            # Same re-raise model as observe_opp: value from the top,
            # bluffs from the bottom.
            t_val = 1 - rr * (1 - beta)
            p_raise = np.clip(_sig((q - t_val) / 0.05)
                              + _sig((rr * beta - q) / 0.05), 0, 1)
            p_fold = p_fold * (1 - p_raise)
        else:
            p_raise = agg_prob(q, rr, beta) * (1 - p_fold)
        p_call = np.clip(1 - p_fold - p_raise, 0, 1)
        final = pot + add + opp_add
        m_call = float(p_call.mean())
        ev_call = 0.0
        if m_call > 1e-9:
            eq_c = float((p_call * r).sum() / p_call.sum())
            eq_c = eq_c if allin else realise(eq_c)
            if FIX_OVERBET and frac > 2.0 and eq_c < 0.45:
                # No overbet bluffs unless we have actually seen this
                # opponent fold to big bets often enough.
                key = "fold_r" if call > 0 else size_key(frac)
                keys = ((key,) if call > 0 or not FIX_OVER_COUNT
                        else ("fold_l", "fold_a"))
                n_seen = sum(opp["post"].get(k, (0.0, 0.0))[1] for k in keys)
                if n_seen < OVERBET_MIN_N:
                    return -1e9
            ev_call = m_call * (eq_c * final - add)
            if LOOKAHEAD and look_ok and not allin and not river and nxt is not None:
                # Called: the hand goes on to the next street.
                look = (next_street_value(opp, q, r, nxt, p_call, final,
                                          eff - add) - m_call * add)
                ev_call = (1 - LOOK_W) * ev_call + LOOK_W * look
        ev_fold = float(p_fold.mean()) * pot
        ev_raise = 0.0
        if p_raise.sum() > 0:
            # Their re-raise: treated as all-in; we call or fold.
            eq_r = float((p_raise * r).sum() / p_raise.sum())
            big = pot + eff + max(0, eff - call)
            v_r = max(-add, eq_r * big - eff)
            ev_raise = float(p_raise.mean()) * v_r
        return ev_fold + ev_call + ev_raise

    if call > 0:
        options.append((0.0, PlayerAction.fold(me)))
        allin_call = call >= gs.stacks[me]
        e_call = eq if (allin_call or river) else realise(eq)
        ev_c = e_call * (pot + call) - call
        if LOOKAHEAD and look_ok and not allin_call and not river and nxt is not None:
            ones = np.ones(len(r))
            look = next_street_value(opp, q, r, nxt, ones, pot + call,
                                     eff - call) - call
            ev_c = (1 - LOOK_W) * ev_c + LOOK_W * look
        options.append((ev_c, PlayerAction.call(me)))
    else:
        # Checking: if we close the street, we just realise equity; if
        # the opponent acts next, they may bet.
        closes = gs.acted[1 - me] if h["n"] == 2 else True
        if closes:
            ev_check = realise(eq) * pot
            if LOOKAHEAD and look_ok and not river and nxt is not None:
                look = next_street_value(opp, q, r, nxt, np.ones(len(r)),
                                         pot, eff)
                ev_check = (1 - LOOK_W) * ev_check + LOOK_W * look
        else:
            alpha = post_rate(opp, "bet")
            p_b = agg_prob(q, alpha, beta)
            mass = p_b.mean()
            if mass > 1e-6:
                eq_b = float((p_b * r).sum() / p_b.sum())
                bs = opp["bet_size"]
                frac = bs[0] / bs[1] if bs[1] >= 3 else 0.6
                bet = min(eff, frac * pot)
                v_b = max(0.0, realise(eq_b) * (pot + 2 * bet) - bet)
            else:
                v_b = 0.0
            eq_c = float(((1 - p_b) * r).sum() / max((1 - p_b).sum(), 1e-9))
            ev_check = mass * v_b + (1 - mass) * realise(eq_c) * pot
        options.append((ev_check, PlayerAction.call(me)))
    if gs.can_raise:
        lo, hi = gs.raise_range
        cur = gs.current_bet
        sizes = set()
        base = pot + call
        for frac in (BET_SIZES if call == 0 else RAISE_SIZES):
            sizes.add(int(min(hi, max(lo, cur + frac * base))))
        if BIG_VALUE and call == 0:
            n_l = opp["post"].get("fold_l", (0.0, 0.0))[1]
            if n_l >= 5 and post_rate(opp, "fold_l") < 0.15:
                # Seen not folding to pot bets: value-bet bigger.
                sizes.add(int(min(hi, max(lo, cur + 1.6 * base))))
        sizes.add(hi)
        for total in sorted(sizes):
            # A raise that leaves under ~a third of a pot behind is an
            # all-in in practice.
            if hi - total < 0.35 * (pot + total):
                total = hi
            options.append((raise_ev(total),
                            PlayerAction.raise_to(me, int(total))))
    best = max(options, key=lambda o: o[0])
    h["post_evs"] = [(round(e, 1), a.kind, a.amount) for e, a in options]
    return best[1]


# -- learning after the hand -------------------------------------------------------


def learn(model, h, actions, gs):
    if h is not None and h.get("broken"):
        return
    if h is None or h.get("hand_no") != model.hand_no:
        # We never acted: heads-up that means the small blind folded.
        h = _stale_hand(model, actions, gs)
        if h is None:
            return
    for a in actions:
        if a.kind in ("fold", "call", "raise") or a.player_id == DEALER:
            apply_action(model, h, a)
    if h["n"] != 2:
        return
    opp = model.opp[h["team"]]
    if LEARN:
        opp["hands"] += 1
    shown = [pc.card_id(a.card) for a in actions
             if a.kind == "reveal_card" and a.player_id == h["opp"]]
    if len(shown) == 2 and h["opp_agg"]:
        combo = pc.combo_of(shown[0], shown[1])
        rng = model._rng
        for nb in h["opp_agg"]:
            board = h["board"][:nb]
            if len(board) < 3:
                continue
            s = pc.strength_vs_random(board, 20, rng)
            post_add(opp, "bluff", bool(np.nan_to_num(s[combo]) < 0.5))
    for a in actions:
        if a.kind == "win" and a.player_id == h["me"]:
            pass
    net = gs.stacks[h["me"]] - h["start"][h["me"]]
    opp["net"] += net / h["bb"]


def _stale_hand(model, actions, gs):
    if gs.num_players != 2:
        return None
    won = {a.player_id: a.amount for a in actions if a.kind == "win"}
    start = []
    for p in range(2):
        posted = gs.small_blind if p == gs.button else gs.big_blind
        start.append(gs.stacks[p] - won.get(p, 0) + posted)
    h = new_hand(model, gs, start)
    h["hand_no"] = model.hand_no
    if h["team"] not in model.opp:
        model.opp[h["team"]] = new_opp()
    model.h = h
    return h


# -- the simple player (multiway, emergencies) -------------------------------------


def simple_decide(model, gs, rng):
    me = model.player_id
    hole = [pc.card_id(c) for c in model.hand]
    board = [pc.card_id(c) for c in gs.board]
    others = max(1, len(gs.players_in_hand) - 1)
    e = pc.equity_vs_random(hole, board, 300, rng) ** others
    if e > 0.7 and gs.can_raise:
        lo, hi = gs.raise_range
        return PlayerAction.raise_to(me, int(min(hi, max(lo, gs.current_bet + 0.66 * gs.pot))))
    if gs.call_amount == 0 or e > gs.pot_odds + 0.02:
        return PlayerAction.call(me)
    return PlayerAction.fold(me)


def decide(model, actions, gs, started):
    rng = model._rng
    h = model.h
    if h is None or h.get("hand_no") != model.hand_no:
        h = new_hand(model, gs)
        h["hand_no"] = model.hand_no
        model.h = h
        if h["n"] == 2 and h["team"] not in model.opp:
            model.opp[h["team"]] = new_opp()
    if h.get("broken"):
        return simple_decide(model, gs, rng)
    for a in actions:
        apply_action(model, h, a)
    if h["n"] != 2:
        return simple_decide(model, gs, rng)
    if h["street"] == "preflop" and h.get("node") is not None:
        return preflop_decide(model, h, gs, rng)
    # Anytime safety: the full computation takes ~0.05-0.3 s; if this move
    # has already used much of the soft budget (slow or busy machine), cut
    # the samples and skip the lookahead. Far below the 10 s hard limit.
    budget = SOFT_BUDGET - (time.perf_counter() - started)
    return postflop_decide(model, h, gs, rng, budget)
