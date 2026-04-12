# Parses grid data

from game import Game
from grid import Grid
from snake import Snake, Pos, Direction, turn_left, turn_right

# Number of features returned by get_features (must match neural_network.N_FEATURES).
# Layout:
#   0..15  : 8 rays × (distance, is_wall_flag)  in head-relative directions
#   16..17 : nearest apple direction (unit vector, head-relative)
#   18     : nearest apple distance (Manhattan, normalized)
#   19..20 : second-nearest apple direction (unit vector, head-relative)
#   21     : stamina / max_stamina
#   22     : length / win_length
#   23..24 : nearest enemy head direction (unit vector, head-relative)
N_FEATURES = 25


def get_grid_data(game: Game) -> list[list[int]]:
    grid = game.get_grid()
    snakes = game.get_snakes()
    snakeTiles: set[Pos] = grid.occupied_tiles(snakes)
    appleTiles: set[Pos] = grid.get_apple_positions()
    n: int = grid.get_grid_size()
    gridData: list[list[int]] = [[0 for cols in range(n)] for rows in range(n)]
    for tile in appleTiles:
        gridData[tile[1]][tile[0]] = 1

    for tile in snakeTiles:
        gridData[tile[1]][tile[0]] = 2

    return gridData


def _unit_vec(dx: float, dy: float) -> tuple[float, float]:
    """Return (dx, dy) normalized to unit length, or (0, 0) if zero."""
    mag = (dx * dx + dy * dy) ** 0.5
    if mag < 1e-9:
        return (0.0, 0.0)
    return (dx / mag, dy / mag)


def get_features(game_state: dict, player_id: int) -> list[float]:
    """Returns a list of N_FEATURES values, all roughly in [-1, 1].

    Designed for fast learning: every feature is directly actionable, head-relative,
    and bounded. See module-level comment for layout.
    """
    grid_size: int = game_state["grid_size"]

    snake_data = next((s for s in game_state["snakes"] if s["id"] == player_id), None)
    if snake_data is None or not snake_data["alive"]:
        return []

    hx, hy = snake_data["positions"][0]
    heading: Direction = snake_data["direction"]
    fx, fy = heading.value
    rx, ry = turn_right(heading).value

    def to_local(dx: int, dy: int) -> tuple[float, float]:
        """Project a world delta into (forward, right) head-relative coords."""
        return (dx * fx + dy * fy, dx * rx + dy * ry)

    # ---- Build occupied set (all snake bodies, including own; head excluded) ----
    occupied: set[Pos] = {
        pos
        for s in game_state["snakes"] if s["alive"]
        for pos in s["positions"]
    }
    occupied.discard((hx, hy))

    # ---- 8 rays in head-relative directions ----
    # Each ray: (forward_component, right_component) in WORLD coords by combining
    # the heading basis with the desired local direction.
    # Local directions (fwd, right):
    local_dirs = [
        ( 1,  0),  # forward
        ( 1,  1),  # fwd-right
        ( 0,  1),  # right
        (-1,  1),  # back-right
        (-1,  0),  # back
        (-1, -1),  # back-left
        ( 0, -1),  # left
        ( 1, -1),  # fwd-left
    ]

    # Maximum ray length we'll allow (used for normalization). Diagonals reach ~grid_size,
    # cardinals reach grid_size; using grid_size as the divisor keeps everything ≤ ~1.
    max_ray = float(grid_size)

    ray_features: list[float] = []
    for lf, lr in local_dirs:
        # Convert local direction to world direction via heading basis.
        wdx = lf * fx + lr * rx
        wdy = lf * fy + lr * ry
        # Step until we hit a wall or a body.
        x, y = hx, hy
        steps = 0
        hit_wall = True  # default: nothing in the way means we walk to the wall
        while True:
            x += wdx
            y += wdy
            steps += 1
            if not (0 <= x < grid_size and 0 <= y < grid_size):
                hit_wall = True
                break
            if (x, y) in occupied:
                hit_wall = False
                break
            if steps >= grid_size:  # safety cap
                break
        # Normalize distance so closer = bigger urgency. We encode raw distance
        # (smaller = more dangerous) so the network sees a clean linear signal.
        ray_features.append(steps / max_ray)
        ray_features.append(1.0 if hit_wall else 0.0)

    # ---- Apple features (nearest and second-nearest) ----
    apples: list[Pos] = game_state["apples"]
    closest_apples = sorted(apples, key=lambda ap: abs(ap[0] - hx) + abs(ap[1] - hy))[:2]

    if len(closest_apples) >= 1:
        ax, ay = closest_apples[0]
        dfwd, drgt = to_local(ax - hx, ay - hy)
        u_fwd, u_rgt = _unit_vec(dfwd, drgt)
        apple1_dir = [u_fwd, u_rgt]
        apple1_dist = (abs(ax - hx) + abs(ay - hy)) / (2.0 * grid_size)  # max Manhattan dist on grid is ~2*grid_size
    else:
        apple1_dir = [0.0, 0.0]
        apple1_dist = 1.0  # treat "no apple" as max distance

    if len(closest_apples) >= 2:
        ax2, ay2 = closest_apples[1]
        d2fwd, d2rgt = to_local(ax2 - hx, ay2 - hy)
        u2_fwd, u2_rgt = _unit_vec(d2fwd, d2rgt)
        apple2_dir = [u2_fwd, u2_rgt]
    else:
        apple2_dir = [0.0, 0.0]

    # ---- Stamina, length ----
    stamina_norm = (
        snake_data["stamina"] / snake_data["max_stamina"]
        if snake_data["max_stamina"] > 0 else 0.0
    )
    length_norm = snake_data["length"] / game_state["win_length"]

    # ---- Nearest enemy head direction (unit vector, head-relative) ----
    enemies = [s for s in game_state["snakes"] if s["alive"] and s["id"] != player_id]
    if enemies:
        closest_enemy = min(
            enemies,
            key=lambda s: abs(s["positions"][0][0] - hx) + abs(s["positions"][0][1] - hy),
        )
        ex, ey = closest_enemy["positions"][0]
        edfwd, edrgt = to_local(ex - hx, ey - hy)
        enemy_dir = list(_unit_vec(edfwd, edrgt))
    else:
        enemy_dir = [0.0, 0.0]

    return (
        ray_features                       # 16
        + apple1_dir                       # 2
        + [apple1_dist]                    # 1
        + apple2_dir                       # 2
        + [stamina_norm, length_norm]      # 2
        + enemy_dir                        # 2
    )                                       # = 25