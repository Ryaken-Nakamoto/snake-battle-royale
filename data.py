# Parses grid data

from game import Game
from grid import Grid
from snake import Snake, Pos, turn_left, turn_right
from player import HumanPlayer, SafeRandomPlayer, RandomPlayer

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

# Gets features, retuns a list of 14 normalized values:
#   0-3: 2 closest apples as (forward, right) in local frame, normalized
#   4-6: wall distance in straight/left/right, normalized
#   7-9: nearest body distance in straight/left/right, normalized
#   10: stamina / max_stamina
#   11: length / win_length
#   12-13: closest enemy head as (forward, right) in local frame, normalized
def get_features(game: Game, player_id: int) -> list[float]:
    if isinstance(game.players[player_id], HumanPlayer):
        return []

    snake: Snake = game.snakes[player_id]
    hx, hy = snake.positions[0]
    heading = snake.direction
    grid_size = game.grid.size

    def to_local(dx: int, dy: int) -> tuple[float, float]:
        fx, fy = heading.value
        rx, ry = turn_right(heading).value
        return ((dx * fx + dy * fy) / grid_size, (dx * rx + dy * ry) / grid_size)

    # gets 2 closest apples in local frame
    apple_positions: list[Pos] = list(game.grid.apples)
    closest_apples = sorted(
        apple_positions,
        key=lambda ap: abs(ap[0] - hx) + abs(ap[1] - hy)
    )[:2]
    apple_features: list[float] = []
    for ap in closest_apples:
        fwd, right = to_local(ap[0] - hx, ap[1] - hy)
        apple_features += [fwd, right]
    while len(apple_features) < 4:
        apple_features.append(0.0)

    # gets distances from both walls and snake bodies in the 3 relative directions
    occupied: set[Pos] = game.grid.occupied_tiles(game.get_snakes())
    occupied.discard((hx, hy))

    dirs = [heading, turn_left(heading), turn_right(heading)]
    wall_dists: list[float] = []
    body_dists: list[float] = []
    for d in dirs:
        dx, dy = d.value
        x, y = hx, hy
        wall_d = 0
        body_d = None
        while True:
            x += dx
            y += dy
            wall_d += 1
            if not game.grid.in_bounds((x, y)):
                break
            if body_d is None and (x, y) in occupied:
                body_d = wall_d
        wall_dists.append(wall_d / grid_size)
        body_dists.append((body_d if body_d is not None else wall_d) / grid_size)

    # gets own features, stamina and length
    stamina_norm = snake.stamina / snake.max_stamina if snake.max_stamina > 0 else 0.0
    length_norm = snake.length / game.config.win_length

    # Gets closest enemy head in local fram
    enemies = [s for s in game.get_snakes() if s.alive and s.snake_id != player_id]
    enemy_features: list[float] = [0.0, 0.0]
    if enemies:
        closest = min(enemies, key=lambda s: abs(s.head[0] - hx) + abs(s.head[1] - hy))
        enemy_features = list(to_local(closest.head[0] - hx, closest.head[1] - hy))

    return apple_features + wall_dists + body_dists + [stamina_norm, length_norm] + enemy_features


    
    
