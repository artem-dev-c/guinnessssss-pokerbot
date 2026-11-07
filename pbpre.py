"""
The heads-up preflop game tree, in big blinds.

Shared by the offline solver (lab/solve_preflop.py) and the bot: the
solver computes a strategy for every node, the bot maps the real hand onto
the same nodes. SB is the button (posts 0.5, acts first preflop, in
position after the flop); BB posts 1.

A node is a plain dict:
    player: 0 (SB), 1 (BB), or -1 for a terminal
    acts:   list of (label, total) for a player node: label "F", "C", "R",
            "A"; total = the actor's total chips in after the action (bb)
    kids:   child node ids, parallel to acts
    c:      [sb, bb] chips in when the node is reached
    kind:   terminal only: "fold" (loser in "loser"), "show" (all-in, called)
            or "flop" (the hand goes to the flop, c[0] == c[1])
"""

import math


def build_tree(stack: float, opens=(2.0, 3.0), iso=4.0, three=3.0):
    nodes: list[dict] = []

    def add(node):
        node["id"] = len(nodes)
        nodes.append(node)
        return node["id"]

    def rec(c, actor, first, raises, last_raise, path):
        top = max(c)
        owed = top - c[actor]
        acts = []
        if owed > 0:
            acts.append(("F", c[actor]))
        acts.append(("C", min(top, stack)))
        if top < stack and raises < 4:
            if first and actor == 0:
                sizes = list(opens)
            elif raises == 0 and actor == 1:
                sizes = [iso]  # BB raising a limp
            elif raises == 1 and actor == 1:
                sizes = [three * top]  # BB 3-bet
            else:
                sizes = []
            for s in sizes:
                if s - top >= last_raise and s < 0.55 * stack:
                    acts.append(("R", s))
            acts.append(("A", stack))
        me = add({"player": actor, "acts": acts, "kids": [], "c": list(c),
                  "path": path})
        kids = []
        for label, total in acts:
            nc = list(c)
            p = path + label + ("%g" % total if label == "R" else "") + "."
            if label == "F":
                kid = add({"player": -1, "kind": "fold", "loser": actor,
                           "c": nc, "acts": [], "kids": [], "path": p})
            elif label == "C":
                nc[actor] = total
                closes = not (first and actor == 0)
                if not closes:
                    kid = rec(nc, 1, False, raises, last_raise, p)
                else:
                    kind = "show" if max(nc) >= stack else "flop"
                    kid = add({"player": -1, "kind": kind, "c": nc,
                               "acts": [], "kids": [], "path": p})
            else:
                nc[actor] = total
                kid = rec(nc, 1 - actor, False, raises + 1,
                          max(last_raise, total - top), p)
            kids.append(kid)
        nodes[me]["kids"] = kids
        return me

    rec([0.5, 1.0], 0, True, 0, 1.0, "")
    return nodes


def child_for(node, nodes, label, total, stack, strict=False):
    """(child index, distance) of the child of node matching a real
    action: label F/C/R (a raise to total bb). A raise maps to the
    nearest raise size in the tree in log space; only a raise close to
    the whole stack counts as all-in. distance is that log gap (0 for
    folds and calls); a large one means the action is off the tree."""
    acts = node["acts"]
    if label in ("F", "C"):
        for i, (lab, _) in enumerate(acts):
            if lab == label:
                return i, 0.0
        return None, None
    best, best_d = None, None
    for i, (lab, t) in enumerate(acts):
        if lab not in ("R", "A"):
            continue
        d = abs(math.log(max(total, 1e-6) / t))
        if lab == "A" and total >= 0.8 * stack:
            d = 0.0
        elif lab == "A" and strict:
            continue  # only a real (near) all-in maps to the all-in node
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best, best_d


def flop_share_sb(e_sb):
    """SB's (in position) realised share of a pot that goes to the flop,
    from its all-in equity e_sb (array or float). Weak hands realise less,
    the out-of-position BB less than the SB."""
    rs = 1.0 + 0.3 * (e_sb - 0.5)
    rb = 0.82 + 0.3 * ((1 - e_sb) - 0.5)
    den = e_sb * rs + (1 - e_sb) * rb
    return e_sb * rs / (den + 1e-12)
