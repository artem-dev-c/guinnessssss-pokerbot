import time

import numpy as np

import pbbrain
from card_utils import Card
from game_state import GameStateView
from player_action import PlayerAction


class Model:
    """
    Heads-up no-limit bot.

    Preflop: a CFR-solved equilibrium for the stack depth (precomputed,
    stored in this object), shifted towards a best response to each
    opponent as their frequencies become clear. Postflop: tracks the
    opponent's range action by action and picks the action with the best
    expected value against it, with fold, call and raise rates learned per
    opponent. Anything unexpected falls back to a simple, always-legal
    move: the bot never faults on purpose.
    """

    DEBUG = False  # tests set this to see errors instead of fallbacks

    def __init__(self):
        self.hand = []
        self.player_id = None
        self.hand_no = 0
        self.h = None
        self.opp = {}  # team name -> what we have learned about them
        self.data = pbbrain.load_data()
        self._rng = None
        self.fallbacks = {}  # reason -> count, for testing

    def _note(self, where, error):
        key = f"{where}: {repr(error)[:100]}"
        self.fallbacks[key] = self.fallbacks.get(key, 0) + 1

    def seed(self, hand: list[Card], player_id: int):
        self.hand = hand
        self.player_id = player_id
        self.hand_no += 1

    def _ready(self):
        if self._rng is None:
            self._rng = np.random.default_rng(1009 + 7919 * self.hand_no)

    def run(self, actions: list[PlayerAction],
            game_state: GameStateView) -> PlayerAction:
        started = time.perf_counter()
        self._ready()
        try:
            action = pbbrain.decide(self, actions, game_state, started)
            game_state.validate_action(action)
            return action
        except Exception as error:
            if Model.DEBUG:
                raise
            self._note("run", error)
            # Play the rest of this hand with the simple player.
            self.h = {"broken": True, "hand_no": self.hand_no}
        try:
            action = pbbrain.simple_decide(self, game_state, self._rng)
            game_state.validate_action(action)
            return action
        except Exception:
            if game_state.call_amount == 0:
                return PlayerAction.call(self.player_id)
            return PlayerAction.fold(self.player_id)

    def end_game(self, actions: list[PlayerAction],
                 game_state: GameStateView):
        self._ready()
        try:
            pbbrain.learn(self, self.h, actions, game_state)
        except Exception as error:
            if Model.DEBUG:
                raise
            self._note("end_game", error)
        self.h = None
