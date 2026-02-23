"""Core game logic for Snake Battle Royale."""

from __future__ import annotations

import math
import random
from typing import Any

from config import GameConfig
from grid import Grid
from player import Player
from snake import Action, Direction, Pos, Snake

SNAKE_NAMES = ["Viper", "Cobra", "Python", "Mamba", "Rattler", "Sidewinder", "Adder", "Taipan"]

SNAKE_COLORS = [
    (0, 200, 80),     # green
    (220, 50, 50),    # red
    (50, 120, 220),   # blue
    (220, 180, 30),   # yellow
    (180, 60, 220),   # purple
    (220, 130, 30),   # orange
    (30, 200, 200),   # cyan
    (200, 100, 150),  # pink
]

_OPPOSITE = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST,
}


class Game:
    """Manages all game state and the main tick logic."""

    def __init__(self, config: GameConfig, players: list[Player]) -> None:
        if len(players) != config.num_snakes:
            raise ValueError(f"Expected {config.num_snakes} players, got {len(players)}")

        self.config = config
        self.grid = Grid(config)
        self.players = players
        self.snakes: list[Snake] = []
        self.tick = 0
        self.winners: list[Snake] = []
        self.game_over = False

        self._spawn_snakes()
        self.grid.spawn_apples(config.initial_apple_count, self.snakes)

    # -- spawning -------------------------------------------------------------

    def _spawn_snakes(self) -> None:
        """Spawn all snakes at random, non-overlapping, wall-safe positions."""
        min_dist = 5
        safety_margin = 3
        occupied: set[Pos] = set()
        heads: list[Pos] = []
        directions = list(Direction)

        for i in range(self.config.num_snakes):
            placed = False
            for _ in range(5000):
                d = random.choice(directions)
                dx, dy = d.value

                # Head must be far enough from walls so the snake body fits behind
                # and there's a safety margin ahead
                margin_ahead = safety_margin
                body_behind = self.config.initial_snake_length - 1

                # Compute valid head ranges
                if dx == 1:   # EAST
                    hx_min, hx_max = body_behind, self.config.grid_size - 1 - margin_ahead
                elif dx == -1:  # WEST
                    hx_min, hx_max = margin_ahead, self.config.grid_size - 1 - body_behind
                else:
                    hx_min, hx_max = body_behind, self.config.grid_size - 1 - body_behind

                if dy == 1:   # SOUTH
                    hy_min, hy_max = body_behind, self.config.grid_size - 1 - margin_ahead
                elif dy == -1:  # NORTH
                    hy_min, hy_max = margin_ahead, self.config.grid_size - 1 - body_behind
                else:
                    hy_min, hy_max = body_behind, self.config.grid_size - 1 - body_behind

                if hx_min > hx_max or hy_min > hy_max:
                    continue

                hx = random.randint(hx_min, hx_max)
                hy = random.randint(hy_min, hy_max)
                head = (hx, hy)

                # Build body behind the head (opposite of direction)
                odx, ody = _OPPOSITE[d].value
                positions = [
                    (hx + odx * j, hy + ody * j)
                    for j in range(self.config.initial_snake_length)
                ]

                # Check overlaps
                pos_set = set(positions)
                if pos_set & occupied:
                    continue

                # Check minimum distance from other heads
                if any(abs(head[0] - oh[0]) + abs(head[1] - oh[1]) < min_dist for oh in heads):
                    continue

                # Valid spawn
                name = SNAKE_NAMES[i % len(SNAKE_NAMES)]
                color = SNAKE_COLORS[i % len(SNAKE_COLORS)]
                snake = Snake(i, name, positions, d, color, self.config)
                self.snakes.append(snake)
                occupied.update(pos_set)
                heads.append(head)
                placed = True
                break

            if not placed:
                raise RuntimeError(
                    f"Could not find valid spawn for snake {i}. "
                    "Try a larger grid or fewer snakes."
                )

    # -- game state for players -----------------------------------------------

    def get_game_state(self) -> dict[str, Any]:
        """Build a read-only snapshot of the game state for player AI."""
        return {
            "grid_size": self.config.grid_size,
            "tick": self.tick,
            "apples": list(self.grid.apples),
            "snakes": [
                {
                    "id": s.snake_id,
                    "name": s.name,
                    "positions": list(s.positions),
                    "direction": s.direction,
                    "length": s.length,
                    "alive": s.alive,
                    "stamina": s.stamina,
                    "max_stamina": s.max_stamina,
                }
                for s in self.snakes
            ],
        }

    # -- main tick ------------------------------------------------------------

    def progress_game(self, player_decisions: dict[int, Action] | None = None) -> None:
        """Advance the game by one tick.

        Args:
            player_decisions: Optional mapping of snake_id -> Action.
                If not provided, each player's get_action() is called.
        """
        if self.game_over:
            return

        self.tick += 1
        state = self.get_game_state()

        # 1. Collect actions
        actions: dict[int, Action] = {}
        for snake in self.snakes:
            if not snake.alive:
                continue
            if player_decisions and snake.snake_id in player_decisions:
                actions[snake.snake_id] = player_decisions[snake.snake_id]
            else:
                actions[snake.snake_id] = self.players[snake.snake_id].get_action(state)

        # 2. Update stamina
        for snake in self.snakes:
            if not snake.alive:
                continue
            action = actions[snake.snake_id]
            if action.is_boost and snake.can_boost():
                snake.drain_stamina()
            else:
                snake.recover_stamina()
                # Downgrade boost action if no stamina
                if action.is_boost:
                    downgrade = {
                        Action.BOOST_STRAIGHT: Action.STRAIGHT,
                        Action.BOOST_LEFT: Action.TURN_LEFT,
                        Action.BOOST_RIGHT: Action.TURN_RIGHT,
                    }
                    actions[snake.snake_id] = downgrade[action]

        # 3. Compute new head positions for all snakes
        new_heads_map: dict[int, list[Pos]] = {}
        for snake in self.snakes:
            if not snake.alive:
                continue
            new_heads_map[snake.snake_id] = snake.compute_new_heads(actions[snake.snake_id])

        # 4. Move all snakes
        for snake in self.snakes:
            if not snake.alive:
                continue
            snake.apply_move(new_heads_map[snake.snake_id], actions[snake.snake_id])

        # 5. Collision detection
        dead_ids: set[int] = set()
        alive_snakes = [s for s in self.snakes if s.alive]

        # Build body lookup (exclude heads for body collision, include for wall/self)
        for snake in alive_snakes:
            head = snake.head

            # Wall collision
            if not self.grid.in_bounds(head):
                dead_ids.add(snake.snake_id)
                continue

            # Also check intermediate position for boosting snakes
            heads_list = new_heads_map[snake.snake_id]
            for h in heads_list:
                if not self.grid.in_bounds(h):
                    dead_ids.add(snake.snake_id)
                    break

        # Body collision: head hits any body segment (excluding own head)
        # Build full body set per snake (positions minus head)
        body_tiles: dict[int, set[Pos]] = {}
        for s in alive_snakes:
            body_tiles[s.snake_id] = set(s.positions[1:])

        for snake in alive_snakes:
            if snake.snake_id in dead_ids:
                continue
            head = snake.head
            # Check against all snakes' bodies (including own tail)
            for other in alive_snakes:
                if other.snake_id == snake.snake_id:
                    # Self collision: head in own body
                    if head in body_tiles[other.snake_id]:
                        dead_ids.add(snake.snake_id)
                        break
                else:
                    # Hit other snake's body (any segment including head-body overlap)
                    if head in set(other.positions):
                        dead_ids.add(snake.snake_id)
                        break

            # Also check intermediate tiles for boost moves
            if snake.snake_id not in dead_ids and len(new_heads_map[snake.snake_id]) > 1:
                intermediate = new_heads_map[snake.snake_id][0]  # first tile passed through
                for other in alive_snakes:
                    if other.snake_id == snake.snake_id:
                        if intermediate in body_tiles[other.snake_id]:
                            dead_ids.add(snake.snake_id)
                            break
                    else:
                        if intermediate in set(other.positions):
                            dead_ids.add(snake.snake_id)
                            break

        # Head-to-head collision: two heads at same tile
        head_positions: dict[Pos, list[int]] = {}
        for snake in alive_snakes:
            if snake.snake_id in dead_ids:
                continue
            pos = snake.head
            head_positions.setdefault(pos, []).append(snake.snake_id)
        for pos, ids in head_positions.items():
            if len(ids) > 1:
                dead_ids.update(ids)

        # 6. Remove dead snakes
        for snake in self.snakes:
            if snake.snake_id in dead_ids:
                snake.kill()

        # 7. Apple consumption
        apples_eaten: set[Pos] = set()
        # Check if multiple snakes eat the same apple (treat as head collision)
        apple_eaters: dict[Pos, list[int]] = {}
        for snake in self.snakes:
            if not snake.alive:
                continue
            if snake.head in self.grid.apples:
                apple_eaters.setdefault(snake.head, []).append(snake.snake_id)

        for pos, ids in apple_eaters.items():
            if len(ids) > 1:
                # Multiple snakes on same apple: all die
                for sid in ids:
                    self.snakes[sid].kill()
            else:
                # Single snake eats apple
                self.snakes[ids[0]].schedule_grow(1)
                apples_eaten.add(pos)

        self.grid.apples -= apples_eaten

        # 8. Maintain apple count
        self.grid.maintain_apple_count(self.config.initial_apple_count, self.snakes)

        # 9. Win condition
        tick_winners = [s for s in self.snakes if s.alive and s.length >= self.config.win_length]
        if tick_winners:
            self.winners = tick_winners
            self.game_over = True

        # Also end if no snakes alive
        if not any(s.alive for s in self.snakes):
            self.game_over = True
