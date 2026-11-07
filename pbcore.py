"""
Core helpers: card ids, the 1326 two-card combos, and a fast numpy hand
evaluator that scores many 5-7 card hands at once.

Card id = rank_index * 4 + suit_index, rank_index 0..12 for 2..A and suit
index 0..3 for C, D, H, S (the order of card_utils.full_deck()).

A score is a plain integer: a higher score wins, equal scores split. It is
category * 16**5 + kickers, with the categories of card_utils.CATEGORIES.
"""

import numpy as np

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "CDHS"

B = 16 ** 5
_POW2 = (1 << np.arange(13)).astype(np.int64)


def card_id(card) -> int:
    """A card_utils.Card (or anything with .rank and .suit) as an id."""
    return RANK_CHARS.index(card.rank) * 4 + SUIT_CHARS.index(card.suit)


def card_str(cid: int) -> str:
    return RANK_CHARS[cid >> 2] + SUIT_CHARS[cid & 3]


def _build_tables():
    n = 1 << 13
    hb = np.full(n, -1, np.int64)
    straight = np.full(n, -1, np.int64)
    top = np.zeros((6, n), np.int64)
    for m in range(1, n):
        hb[m] = m.bit_length() - 1
        m14 = (m << 1) | ((m >> 12) & 1)
        s = m14 & (m14 >> 1) & (m14 >> 2) & (m14 >> 3) & (m14 >> 4)
        if s:
            straight[m] = s.bit_length() - 1 + 3
        ranks = [r for r in range(12, -1, -1) if m >> r & 1]
        for k in range(1, 6):
            v = 0
            for i, r in enumerate(ranks[:k]):
                v += r * 16 ** (4 - i)
            top[k, m] = v
    return hb, straight, top


HB, STRAIGHT, TOP = _build_tables()


def _bit(x):
    return np.where(x >= 0, np.left_shift(1, np.maximum(x, 0)), 0)


def evaluate_batch(cards) -> np.ndarray:
    """Scores for an (N, K) int array of card ids, 5 <= K <= 7."""
    cards = np.asarray(cards, dtype=np.int64)
    n = cards.shape[0]
    if n == 0:
        return np.zeros(0, np.int64)
    r = cards >> 2
    s = cards & 3
    rows = np.arange(n, dtype=np.int64)[:, None]
    rc = np.bincount((r + rows * 13).ravel(), minlength=n * 13).reshape(n, 13)
    sc = np.bincount((s + rows * 4).ravel(), minlength=n * 4).reshape(n, 4)

    mask = (rc > 0).astype(np.int64) @ _POW2
    m4 = (rc == 4).astype(np.int64) @ _POW2
    m3 = (rc == 3).astype(np.int64) @ _POW2
    m2 = (rc == 2).astype(np.int64) @ _POW2

    fs = sc.argmax(1)
    has_flush = sc.max(1) >= 5
    fmask = np.where(s == fs[:, None], np.left_shift(1, r), 0).sum(1)
    fmask = np.where(has_flush, fmask, 0)

    q = HB[m4]
    t1 = HB[m3]
    t2 = HB[m3 & ~_bit(t1)]
    p1 = HB[m2]
    p2 = HB[m2 & ~_bit(p1)]
    st = STRAIGHT[mask]
    sf = STRAIGHT[fmask]
    fh_pair = np.maximum(t2, p1)

    D4, D3 = 16 ** 4, 16 ** 3
    sf_v = 8 * B + sf * D4
    quad_v = 7 * B + q * D4 + TOP[1][mask & ~_bit(q)] // 16
    fh_v = 6 * B + t1 * D4 + fh_pair * D3
    fl_v = 5 * B + TOP[5][fmask]
    st_v = 4 * B + st * D4
    tr_v = 3 * B + t1 * D4 + TOP[2][mask & ~_bit(t1)] // 16
    tp_v = (2 * B + p1 * D4 + p2 * D3
            + TOP[1][mask & ~_bit(p1) & ~_bit(p2)] // 256)
    pr_v = 1 * B + p1 * D4 + TOP[3][mask & ~_bit(p1)] // 16
    hi_v = TOP[5][mask]

    return np.select(
        [has_flush & (sf >= 0), q >= 0, (t1 >= 0) & (fh_pair >= 0),
         has_flush, st >= 0, t1 >= 0, (p1 >= 0) & (p2 >= 0), p1 >= 0],
        [sf_v, quad_v, fh_v, fl_v, st_v, tr_v, tp_v, pr_v],
        default=hi_v,
    )


# -- the 1326 combos ----------------------------------------------------------

COMBOS = np.array([(a, b) for a in range(52) for b in range(a + 1, 52)],
                  dtype=np.int64)  # (1326, 2), a < b
COMBO_INDEX = np.full((52, 52), -1, np.int64)
for _i, (_a, _b) in enumerate(COMBOS):
    COMBO_INDEX[_a, _b] = _i
    COMBO_INDEX[_b, _a] = _i
# CARD_MASK[i]: 52-bit mask of combo i's cards, as a bool (1326, 52) table
CARD_HIT = np.zeros((1326, 52), bool)
CARD_HIT[np.arange(1326), COMBOS[:, 0]] = True
CARD_HIT[np.arange(1326), COMBOS[:, 1]] = True


def combo_of(c1: int, c2: int) -> int:
    return int(COMBO_INDEX[c1, c2])


def blocked(dead) -> np.ndarray:
    """Bool (1326,) of combos that use any card in dead."""
    dead = list(dead)
    if not dead:
        return np.zeros(1326, bool)
    return CARD_HIT[:, dead].any(1)


def hand_class(c1: int, c2: int) -> int:
    """The 169 starting-hand classes: 0..168. Pairs are r*13+r, suited
    hi*13+lo, offsuit lo*13+hi (the usual 13x13 grid)."""
    r1, r2 = c1 >> 2, c2 >> 2
    hi, lo = max(r1, r2), min(r1, r2)
    if r1 == r2:
        return hi * 13 + hi
    if (c1 & 3) == (c2 & 3):
        return hi * 13 + lo
    return lo * 13 + hi


COMBO_CLASS = np.array([hand_class(a, b) for a, b in COMBOS], np.int64)


def class_name(k: int) -> str:
    a, b = divmod(k, 13)
    if a == b:
        return RANK_CHARS[a] * 2
    if a > b:
        return RANK_CHARS[a] + RANK_CHARS[b] + "s"
    return RANK_CHARS[b] + RANK_CHARS[a] + "o"


# -- Monte Carlo equity ---------------------------------------------------------


def _draw(rng, pool: np.ndarray, n: int, k: int) -> np.ndarray:
    """(n, k) cards drawn without replacement (within a row) from pool."""
    if k == 0:
        return np.zeros((n, 0), np.int64)
    keys = rng.random((n, len(pool)))
    pick = np.argpartition(keys, k - 1, axis=1)[:, :k]
    # argpartition gives a uniform subset but not a uniformly random order
    # (it depends on the numpy version); callers use positions, so sort the
    # k picks by their random keys.
    order = np.argsort(np.take_along_axis(keys, pick, axis=1), axis=1)
    return pool[np.take_along_axis(pick, order, axis=1)]


def equity_vs_random(hole, board, n: int, rng) -> float:
    """Share of the pot won by hole against one uniformly random hand."""
    hole = list(hole)
    board = list(board)
    missing = 5 - len(board)
    pool = np.array([c for c in range(52) if c not in hole and c not in board])
    draw = _draw(rng, pool, n, missing + 2)
    opp = draw[:, :2]
    run = draw[:, 2:]
    b = np.broadcast_to(np.array(board, np.int64), (n, len(board)))
    full = np.concatenate([b, run], axis=1)
    mine = evaluate_batch(np.concatenate(
        [np.broadcast_to(np.array(hole, np.int64), (n, 2)), full], axis=1))
    theirs = evaluate_batch(np.concatenate([opp, full], axis=1))
    return float(((mine > theirs) + 0.5 * (mine == theirs)).mean())


def equity_vs_range(hole, board, weights, n: int, rng) -> float:
    """Share of the pot won by hole against a range: weights (1326,) over
    combos (need not sum to 1; blocked combos are dropped)."""
    hole = list(hole)
    board = list(board)
    w = np.array(weights, np.float64)
    w[blocked(hole + board)] = 0.0
    total = w.sum()
    if total <= 0:
        return equity_vs_random(hole, board, n, rng)
    opp_idx = rng.choice(1326, n, p=w / total)
    opp = COMBOS[opp_idx]
    missing = 5 - len(board)
    pool = np.array([c for c in range(52) if c not in hole and c not in board])
    if missing:
        cand = _draw(rng, pool, n, missing + 2)
        ok = (cand != opp[:, :1]) & (cand != opp[:, 1:2])
        order = np.argsort(~ok, axis=1, kind="stable")[:, :missing]
        run = np.take_along_axis(cand, order, axis=1)
    else:
        run = np.zeros((n, 0), np.int64)
    b = np.broadcast_to(np.array(board, np.int64), (n, len(board)))
    full = np.concatenate([b, run], axis=1)
    mine = evaluate_batch(np.concatenate(
        [np.broadcast_to(np.array(hole, np.int64), (n, 2)), full], axis=1))
    theirs = evaluate_batch(np.concatenate([opp, full], axis=1))
    return float(((mine > theirs) + 0.5 * (mine == theirs)).mean())


def river_scores(board) -> np.ndarray:
    """Score of every combo on a full 5-card board (blocked combos: -1)."""
    board = list(board)
    live = ~blocked(board)
    out = np.full(1326, -1, np.int64)
    idx = np.nonzero(live)[0]
    hands = np.concatenate(
        [COMBOS[idx], np.broadcast_to(np.array(board, np.int64),
                                       (len(idx), 5))], axis=1)
    out[idx] = evaluate_batch(hands)
    return out


def strength_vs_random(board, n_runouts: int, rng) -> np.ndarray:
    """(1326,) equity of each combo against a random hand on this board,
    estimated with n_runouts shared runouts and random opponents (river:
    exact rank percentile). Blocked combos: NaN."""
    board = list(board)
    live = ~blocked(board)
    idx = np.nonzero(live)[0]
    out = np.full(1326, np.nan)
    if len(board) == 5:
        sc = river_scores(board)[idx]
        order = np.sort(sc)
        lo = np.searchsorted(order, sc, side="left")
        hi = np.searchsorted(order, sc, side="right")
        out[idx] = (lo + 0.5 * (hi - lo)) / len(sc)
        return out
    missing = 5 - len(board)
    pool = np.array([c for c in range(52) if c not in board])
    acc = np.zeros(len(idx))
    cnt = np.zeros(len(idx))
    bt = np.array(board, np.int64)
    for _ in range(n_runouts):
        run = rng.choice(pool, missing, replace=False)
        full = np.concatenate([bt, run])
        ok = ~CARD_HIT[idx][:, run].any(1)
        sc = np.full(len(idx), -1, np.int64)
        sel = idx[ok]
        sc[ok] = evaluate_batch(np.concatenate(
            [COMBOS[sel], np.broadcast_to(full, (len(sel), 5))], axis=1))
        # Percentile of each live combo among the others on this runout.
        vals = np.sort(sc[ok])
        lo = np.searchsorted(vals, sc[ok], side="left")
        hi = np.searchsorted(vals, sc[ok], side="right")
        acc[ok] += (lo + 0.5 * (hi - lo)) / max(1, len(vals))
        cnt[ok] += 1
    out[idx] = acc / np.maximum(cnt, 1)
    return out
