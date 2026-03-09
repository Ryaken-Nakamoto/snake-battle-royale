# Parses grid data

from game import Game
from grid import Grid
from snake import Snake, Pos

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
    
    
