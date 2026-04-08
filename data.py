# Parses grid data

from game import Game
from grid import Grid
from snake import Snake, Pos, Direction, turn_left, turn_right

def get_grid_data(game : Game) -> list[list[int]]:
    grid = game.get_grid()
    snakes = game.get_snakes()
    snakeTiles : set[Pos] = grid.occupied_tiles(snakes)
    appleTiles : set[Pos] = grid.get_apple_positions()
    n : int = grid.get_grid_size()
    gridData : list[list[int]] = [[0 for cols in range(n)] for rows in range(n)]
    for tile in appleTiles:
        gridData[tile[1]][tile[0]] = 1
    
    for tile in snakeTiles:
        gridData[tile[1]][tile[0]] = 2

    return gridData


# Gets features, returns a list of 17 normalized values:
#   0-3: 2 closest apples as (forward, right) in local frame, normalized
#   4-6: wall distance in straight/left/right, normalized
#   7-9: nearest body distance in straight/left/right, normalized
#   10: stamina / max_stamina
#   11: length / win_length
#   12-13: closest enemy head as (forward, right) in local frame, normalized
#   14-16: immediate danger flags (1=danger, 0=safe) for straight/left/right
def get_features(game_state: dict, player_id: int) -> list[float]:
    grid_size: int = game_state["grid_size"]

    snake_data = next((s for s in game_state["snakes"] if s["id"] == player_id), None)
    if snake_data is None or not snake_data["alive"]:
        return []

    hx, hy = snake_data["positions"][0]
    heading: Direction = snake_data["direction"]

    def to_local(dx: int, dy: int) -> tuple[float, float]:
        fx, fy = heading.value
        rx, ry = turn_right(heading).value
        return ((dx * fx + dy * fy) / grid_size, (dx * rx + dy * ry) / grid_size)

    # 2 closest apples in local frame
    apples: list[Pos] = game_state["apples"]
    closest_apples = sorted(apples, key=lambda ap: abs(ap[0] - hx) + abs(ap[1] - hy))[:2]
    apple_features: list[float] = []
    for ap in closest_apples:
        fwd, right = to_local(ap[0] - hx, ap[1] - hy)
        apple_features += [fwd, right]
    while len(apple_features) < 4:
        apple_features.append(0.0)

    # occupied tiles built from snake positions in game_state
    occupied: set[Pos] = {
        pos
        for s in game_state["snakes"] if s["alive"]
        for pos in s["positions"]
    }
    occupied.discard((hx, hy))

    # wall and body distances in straight/left/right
    wall_dists: list[float] = []
    body_dists: list[float] = []
    for d in [heading, turn_left(heading), turn_right(heading)]:
        dx, dy = d.value
        x, y = hx, hy
        wall_d = 0
        body_d = None
        while True:
            x += dx
            y += dy
            wall_d += 1
            if not (0 <= x < grid_size and 0 <= y < grid_size):
                break
            if body_d is None and (x, y) in occupied:
                body_d = wall_d
        wall_dists.append(wall_d / grid_size)
        body_dists.append(body_d / grid_size if body_d is not None else 1.0)

    # stamina and length
    stamina_norm = snake_data["stamina"] / snake_data["max_stamina"] if snake_data["max_stamina"] > 0 else 0.0
    length_norm = snake_data["length"] / game_state["win_length"]

    # closest enemy head in local frame
    enemies = [s for s in game_state["snakes"] if s["alive"] and s["id"] != player_id]
    enemy_features: list[float] = [0.0, 0.0]
    if enemies:
        closest = min(enemies, key=lambda s: abs(s["positions"][0][0] - hx) + abs(s["positions"][0][1] - hy))
        ex, ey = closest["positions"][0]
        enemy_features = list(to_local(ex - hx, ey - hy))

    # immediate danger: is the very next tile in each direction a wall or body?
    danger_flags: list[float] = []
    for d in [heading, turn_left(heading), turn_right(heading)]:
        dx, dy = d.value
        nx, ny = hx + dx, hy + dy
        is_danger = not (0 <= nx < grid_size and 0 <= ny < grid_size) or (nx, ny) in occupied
        danger_flags.append(1.0 if is_danger else 0.0)

    return apple_features + wall_dists + body_dists + [stamina_norm, length_norm] + enemy_features + danger_flags
