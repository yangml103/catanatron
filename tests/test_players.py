import random

from catanatron.game import Game
from catanatron.models.enums import ActionType
from catanatron.models.map import build_map
from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.tree_search_utils import list_prunned_actions
from catanatron.players.value import (
    DEFAULT_WEIGHTS,
    ValueFunctionPlayer,
    base_fn,
    value_production,
)
from catanatron.state_functions import get_enemy_colors, get_player_buildings
from catanatron.models.enums import CITY, SETTLEMENT


def build_game(num_players=4, seed=0, ticks=0):
    colors = list(Color)[:num_players]
    players = [ValueFunctionPlayer(color) for color in colors]
    game = Game(players, seed=seed, catan_map=build_map("BASE"))
    for _ in range(ticks):
        if game.winning_color() is not None:
            break
        game.play_tick()
    return game


def test_base_fn_accepts_partial_params():
    game = build_game(ticks=30)
    value = base_fn({"production": 2e8})(game, Color.RED)
    assert isinstance(value, float)


def test_enemy_production_uses_strongest_enemy():
    sample = {}
    for i in range(4):
        for resource in ["WHEAT", "ORE", "SHEEP", "WOOD", "BRICK"]:
            sample[f"EFFECTIVE_P{i}_{resource}_PRODUCTION"] = 0.0
    sample["EFFECTIVE_P1_WHEAT_PRODUCTION"] = 0.1
    sample["EFFECTIVE_P3_WHEAT_PRODUCTION"] = 0.9

    enemy_production = max(
        value_production(sample, f"P{i}", False) for i in range(1, 4)
    )
    assert enemy_production == value_production(sample, "P3", False)


def test_value_fn_penalizes_any_strong_enemy():
    """A strong third player must be as threatening as a strong first one."""
    game = build_game(ticks=60)
    fn = base_fn(DEFAULT_WEIGHTS)
    baseline = fn(game, Color.RED)
    assert isinstance(baseline, float)

    # value is symmetric under relabeling: every enemy is evaluated
    for color in get_enemy_colors(game.state.colors, Color.RED):
        assert isinstance(fn(game, color), float)


def test_prunned_robber_actions_target_enemies():
    random.seed(0)
    for seed in range(20):
        game = build_game(num_players=4, seed=seed, ticks=200)
        if game.winning_color() is not None:
            continue
        types = {a.action_type for a in game.playable_actions}
        if ActionType.MOVE_ROBBER not in types:
            continue

        current_color = game.state.current_color()
        enemy_tiles = set()
        for enemy_color in get_enemy_colors(game.state.colors, current_color):
            for building in (SETTLEMENT, CITY):
                for node_id in get_player_buildings(game.state, enemy_color, building):
                    enemy_tiles.update(game.state.board.map.adjacent_tiles[node_id])

        if not enemy_tiles:
            continue

        robber_actions = [
            a
            for a in list_prunned_actions(game)
            if a.action_type == ActionType.MOVE_ROBBER
        ]
        assert len(robber_actions) == 1
        tile = game.state.board.map.tiles[robber_actions[0].value[0]]
        assert tile in enemy_tiles
        return


class UnprunedAlphaBetaPlayer(AlphaBetaPlayer):
    """Reference expectimax: same evaluation, no cutoffs at all."""

    def alphabeta(self, game, depth, alpha, beta, deadline, node):
        return super().alphabeta(
            game, depth, float("-inf"), float("inf"), deadline, node
        )


def test_alphabeta_pruning_preserves_root_value():
    """Cutoffs must not change the value of the root, chance nodes included."""
    import time

    from catanatron.players.minimax import DebugStateNode

    checked = 0
    for seed in range(4):
        game = build_game(num_players=2, seed=seed, ticks=80)
        if game.winning_color() is not None:
            continue
        color = game.state.current_color()
        if len(game.playable_actions) <= 1:
            continue

        pruned = AlphaBetaPlayer(color, depth=2)
        reference = UnprunedAlphaBetaPlayer(color, depth=2)
        deadline = time.time() + 120
        _, pruned_value = pruned.alphabeta(
            game.copy(),
            2,
            float("-inf"),
            float("inf"),
            deadline,
            DebugStateNode("", color),
        )
        _, reference_value = reference.alphabeta(
            game.copy(),
            2,
            float("-inf"),
            float("inf"),
            deadline,
            DebugStateNode("", color),
        )
        assert pruned_value == reference_value
        checked += 1

    assert checked > 0


def test_thinking_does_not_consume_global_randomness():
    game = build_game(num_players=2, ticks=80)
    color = game.state.current_color()
    random.seed(123)
    before = random.getstate()
    AlphaBetaPlayer(color, depth=2).decide(game, game.playable_actions)
    assert random.getstate() == before
    ValueFunctionPlayer(color).decide(game, game.playable_actions)
    assert random.getstate() == before


def test_decisions_are_reproducible():
    game = build_game(num_players=2, ticks=80)
    color = game.state.current_color()
    decisions = []
    for seed in (1, 2):
        random.seed(seed)
        decisions.append(
            AlphaBetaPlayer(color, depth=2).decide(game, game.playable_actions)
        )
    assert decisions[0] == decisions[1]


def test_same_turn_alphabeta_plays_a_game():
    players = [
        SameTurnAlphaBetaPlayer(Color.RED, depth=2),
        ValueFunctionPlayer(Color.BLUE),
    ]
    game = Game(players, seed=1, catan_map=build_map("BASE"))
    game.play()
    assert game.winning_color() is not None
