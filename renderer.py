"""Pygame rendering for Snake Battle Royale."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import pygame

if TYPE_CHECKING:
    from config import GameConfig
    from game import Game


class _Viewport(NamedTuple):
    tile: int
    cam_x: int
    cam_y: int
    view_w: int
    view_h: int


# Maximum window dimensions (the grid is scaled to fit within this)
MAX_WINDOW_W = 1200
MAX_WINDOW_H = 900


class Renderer:
    """Handles all pygame drawing, scaling the grid to fit the screen."""

    BACKGROUND = (15, 15, 15)
    APPLE_COLOR = (255, 40, 40)
    TEXT_COLOR = (240, 240, 240)
    STAMINA_BG = (60, 60, 60)
    STAMINA_FG = (50, 220, 100)

    def __init__(self, config: GameConfig, human_snake_id: int) -> None:
        self.config = config
        self.human_snake_id = human_snake_id
        self.hud_height = 44

        # Compute a tile size that fits the grid into the max window
        max_grid_h = MAX_WINDOW_H - self.hud_height
        tile_from_w = MAX_WINDOW_W / config.grid_size
        tile_from_h = max_grid_h / config.grid_size
        self.tile = max(1, int(min(tile_from_w, tile_from_h)))

        self.grid_pixel = config.grid_size * self.tile
        self.win_w = self.grid_pixel
        self.win_h = self.grid_pixel + self.hud_height

        pygame.init()
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Snake Battle Royale")
        self.font = pygame.font.SysFont("monospace", 13)
        self.big_font = pygame.font.SysFont("monospace", 32, bold=True)
        self.med_font = pygame.font.SysFont("monospace", 18)

    def _compute_viewport(self, game: Game) -> _Viewport:
        """Compute viewport for rendering, with optional scoped view."""
        if not self.config.limit_scope:
            return _Viewport(self.tile, 0, 0, self.config.grid_size, self.config.grid_size)

        human = next((s for s in game.snakes if s.snake_id == self.human_snake_id and s.alive), None)
        if human is None:
            return _Viewport(self.tile, 0, 0, self.config.grid_size, self.config.grid_size)

        radius = 7 + math.floor(math.sqrt(human.length))
        diameter = 2 * radius + 1
        side = min(self.win_w, self.win_h - self.hud_height)
        dyn_tile = max(1, side // diameter)

        hx, hy = human.head
        cam_x = max(0, min(hx - radius, self.config.grid_size - diameter))
        cam_y = max(0, min(hy - radius, self.config.grid_size - diameter))

        return _Viewport(dyn_tile, cam_x, cam_y, diameter, diameter)

    def _grid_rect(self, x: int, y: int) -> pygame.Rect:
        return pygame.Rect(
            x * self.tile,
            y * self.tile + self.hud_height,
            self.tile,
            self.tile,
        )

    # -- main draw ------------------------------------------------------------

    def draw(self, game: Game, paused: bool = False) -> None:
        vp = self._compute_viewport(game)
        self.screen.fill(self.BACKGROUND)
        self._draw_apples(game, vp)
        self._draw_snakes(game, vp)
        self._draw_hud(game)

        if paused:
            self._draw_overlay("PAUSED", "(P to resume)")
        if game.game_over:
            self._draw_win_screen(game)

        pygame.display.flip()

    # -- components -----------------------------------------------------------

    def _draw_apples(self, game: Game, vp: _Viewport) -> None:
        for ax, ay in game.grid.apples:
            if not (vp.cam_x <= ax < vp.cam_x + vp.view_w and vp.cam_y <= ay < vp.cam_y + vp.view_h):
                continue
            px = (ax - vp.cam_x) * vp.tile
            py = (ay - vp.cam_y) * vp.tile + self.hud_height
            self.screen.fill(self.APPLE_COLOR, (px, py, vp.tile, vp.tile))

    def _draw_snakes(self, game: Game, vp: _Viewport) -> None:
        for snake in game.snakes:
            if not snake.alive:
                continue
            is_human = snake.snake_id == self.human_snake_id
            col = snake.color

            for i, (sx, sy) in enumerate(snake.positions):
                if not (vp.cam_x <= sx < vp.cam_x + vp.view_w and vp.cam_y <= sy < vp.cam_y + vp.view_h):
                    continue
                px = (sx - vp.cam_x) * vp.tile
                py = (sy - vp.cam_y) * vp.tile + self.hud_height
                rect = pygame.Rect(px, py, vp.tile, vp.tile)
                pygame.draw.rect(self.screen, col, rect)

                if i == 0:
                    # Brighter head
                    highlight = tuple(min(255, c + 70) for c in col)
                    inner = rect.inflate(-max(2, vp.tile // 3), -max(2, vp.tile // 3))
                    pygame.draw.rect(self.screen, highlight, inner)
                    if is_human:
                        # Gold triangle marker above head
                        cx = rect.centerx
                        cy = rect.top - 1
                        size = max(3, vp.tile // 2)
                        pygame.draw.polygon(
                            self.screen,
                            (255, 215, 0),
                            [(cx - size, cy), (cx, cy - size), (cx + size, cy)],
                        )

                if is_human:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

    def _draw_hud(self, game: Game) -> None:
        pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, self.win_w, self.hud_height))

        x = 8
        for snake in game.snakes:
            # Color swatch
            pygame.draw.rect(self.screen, snake.color, (x, 6, 10, 10))
            if snake.snake_id == self.human_snake_id:
                pygame.draw.rect(self.screen, (255, 255, 255), (x - 1, 5, 12, 12), 1)

            # Name + length
            status = "" if snake.alive else " [X]"
            label = f"{snake.name}:{snake.length}{status}"
            fg = self.TEXT_COLOR if snake.alive else (100, 100, 100)
            txt = self.font.render(label, True, fg)
            self.screen.blit(txt, (x + 14, 4))

            # Stamina bar (human only)
            if snake.snake_id == self.human_snake_id and snake.alive:
                bw = 60
                bx, by = x + 14, 22
                pygame.draw.rect(self.screen, self.STAMINA_BG, (bx, by, bw, 7))
                fill = (snake.stamina / snake.max_stamina * bw) if snake.max_stamina > 0 else 0
                pygame.draw.rect(self.screen, self.STAMINA_FG, (bx, by, int(fill), 7))
                st = self.font.render("STA", True, (130, 130, 130))
                self.screen.blit(st, (bx + bw + 3, by - 3))

            x += max(140, self.win_w // (len(game.snakes) + 1))

    def _draw_overlay(self, title: str, subtitle: str = "") -> None:
        overlay = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        txt = self.big_font.render(title, True, (255, 255, 255))
        self.screen.blit(txt, txt.get_rect(center=(self.win_w // 2, self.win_h // 2 - 20)))
        if subtitle:
            sub = self.med_font.render(subtitle, True, (200, 200, 200))
            self.screen.blit(sub, sub.get_rect(center=(self.win_w // 2, self.win_h // 2 + 20)))

    def _draw_win_screen(self, game: Game) -> None:
        if game.winners:
            if len(game.winners) == 1:
                title = f"{game.winners[0].name} WINS!"
            else:
                names = ", ".join(w.name for w in game.winners)
                title = f"TIE: {names}"
        else:
            title = "ALL SNAKES DEAD"
        self._draw_overlay(title, "ESC to quit")

    def cleanup(self) -> None:
        pygame.quit()
