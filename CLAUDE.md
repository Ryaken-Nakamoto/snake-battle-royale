# Snake Battle Royale — Project Overview

## What the Game Is

**Snake Battle Royale** is a multiplayer snake game built with Python + Pygame. One human player competes against AI-controlled snakes on a shared grid. The first snake to reach the configured `win_length` (default 50 segments) wins.

Run with: `python main.py`

---

## Code Structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point; pygame event loop, input handling |
| `game.py` | `Game` class — all game logic, tick progression |
| `snake.py` | `Snake` entity, `Direction`/`Action` enums, movement math |
| `grid.py` | `Grid` class — bounds checking, apple spawning |
| `player.py` | `Player` ABC, `HumanPlayer`, `RandomPlayer` |
| `renderer.py` | Pygame rendering — grid, HUD, overlays |
| `config.py` | `GameConfig` dataclass, JSON loading/validation |
| `config.json` | Tunable game parameters |

---

## Game Loop (one tick = `game.progress_game()`)

Each tick runs the following steps in order:

1. **Collect actions** — each `Player.get_action(state)` returns an `Action` enum value
2. **Update stamina** — boosting drains stamina; not boosting recovers it
3. **Compute new head positions** — 1 tile normally, 2 tiles when boosting (`boost_multiplier`)
4. **Apply moves** — prepend new head(s), remove tail(s) unless growing
5. **Collision detection** (all checked before any snake is removed):
   - **Wall** — head out of bounds → die
   - **Self** — head hits own body → die
   - **Body** — head hits any segment of another snake → die
   - **Head-to-head** — two heads land on same tile → both die
   - **Intermediate tile** (boost) — the tile passed *through* during a 2-tile move is also checked
6. **Remove dead snakes**
7. **Apple consumption** — snake grows by 1; if two snakes eat the same apple simultaneously, both die
8. **Maintain apple count** — respawn apples to keep count at `initial_apple_count`
9. **Win condition** — any snake with `length >= win_length` is declared a winner; game ends if all snakes die

---

## Snake Mechanics

### Positions
`snake.positions` is a list where `positions[0]` is the **head** and the last element is the **tail**.

### Movement
On each tick, new head position(s) are prepended. An equal number of tail segments are removed, unless the snake is scheduled to grow (`_grow_pending > 0`).

### Stamina
Stamina enables the **boost** mechanic:
- `max_stamina = floor(max_stamina_factor * sqrt(length))` — longer snakes have more stamina
- Boosts consume `stamina_drain_rate` per tick
- Recovery rate = `max_stamina / (stamina_recovery_factor * sqrt(length))`
- If a boost action is requested but `stamina < drain_rate`, the action is downgraded to a normal move

### Actions (enum `Action`)
| Action | Effect |
|--------|--------|
| `STRAIGHT` | Move forward 1 tile |
| `TURN_LEFT` | Turn left, move 1 tile |
| `TURN_RIGHT` | Turn right, move 1 tile |
| `BOOST_STRAIGHT` | Move forward 2 tiles (costs stamina) |
| `BOOST_LEFT` | Turn left, move 2 tiles (costs stamina) |
| `BOOST_RIGHT` | Turn right, move 2 tiles (costs stamina) |

Cannot reverse direction (no SOUTH from NORTH, etc.).

---

## Human Controls

Keys map to **absolute directions** — which two keys are active depends on the snake's current orientation:

| Situation | Key | Effect |
|-----------|-----|--------|
| Moving East or West | `W` | Turn North |
| Moving East or West | `S` | Turn South |
| Moving North or South | `A` | Turn West |
| Moving North or South | `D` | Turn East |
| Any | `Space` (hold) | Boost in current direction |
| Any | No key | Go straight |
| Any | `P` | Pause / unpause |
| Any | `ESC` | Quit |

Pressing the key for the opposite direction (reversing) is ignored — the snake continues straight.

---

## AI Players

Currently only `RandomPlayer` is implemented — it picks a random `Action` every tick with no awareness of the game state. The `Player` ABC in `player.py` defines the interface:

```python
def get_action(self, game_state: dict[str, Any]) -> Action:
```

`game_state` contains: `grid_size`, `tick`, `apples` (list of positions), and `snakes` (list of dicts with `id`, `name`, `positions`, `direction`, `length`, `alive`, `stamina`, `max_stamina`).

New AI players are added by subclassing `Player` and registering them in `main.py`'s `build_players()`.

---

## Configuration (`config.json`)

| Key | Default | Description |
|-----|---------|-------------|
| `grid_size` | 300 | Grid is `grid_size × grid_size` tiles |
| `tile_size` | 10 | Pixel size per tile (used for display scaling) |
| `num_snakes` | 5 | Total snakes (1 human + 4 AI) |
| `initial_snake_length` | 3 | Starting length of each snake |
| `apple_density` | 50 | Apples = `(grid_size²) / apple_density` |
| `win_length` | 50 | Segments needed to win |
| `boost_multiplier` | 2.0 | Tiles moved per tick when boosting |
| `max_stamina_factor` | 1.0 | Scales max stamina with snake length |
| `stamina_recovery_factor` | 1.0 | Scales recovery speed |
| `stamina_drain_rate` | 1.0 | Stamina cost per boost tick |
| `fps` | 10 | Game ticks per second |

---

## Rendering (HUD)

- Top bar shows each snake's name, current length, and dead status `[X]`
- Human snake is outlined in white with a gold triangle above the head
- Green stamina bar displayed for human player only
- Apples rendered as red tiles
- Snake head rendered with a brighter inner highlight

---

## Spawn Logic

Snakes are placed at random positions with:
- Minimum Manhattan distance of 5 between any two heads
- Body placed behind the head (opposite of facing direction)
- Safety margin of 3 tiles clear in front of the head
- Up to 5000 placement attempts per snake before raising an error
