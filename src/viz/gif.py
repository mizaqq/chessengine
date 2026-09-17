"""Render a played game as an animated GIF with Pillow only (no cairo needed).

Each frame: the board after a move (last move highlighted), a caption with the
move number, side, SAN, the mover's probability for the move and the value head's
number. Pieces are drawn with Unicode chess glyphs when a system font has them
(macOS: Apple Symbols), otherwise as letters.
"""
from pathlib import Path
from typing import Optional

import chess
from PIL import Image, ImageDraw, ImageFont

LIGHT, DARK, LAST = (240, 217, 181), (181, 136, 99), (205, 210, 106)
GLYPHS = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
          "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}
FONT_CANDIDATES = ["/System/Library/Fonts/Apple Symbols.ttf", "/Library/Fonts/Arial Unicode.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"]


def _font(size: int, glyphs: bool):
    if glyphs:
        for path in FONT_CANDIDATES:
            try:
                return ImageFont.truetype(path, size), True
            except OSError:
                continue
    try:
        return ImageFont.truetype("Arial.ttf", size), False
    except OSError:
        return ImageFont.load_default(), False


def render_board(board: chess.Board, size: int = 360, lastmove: Optional[chess.Move] = None,
                 caption: str = "", flip: bool = False) -> Image.Image:
    sq = size // 8
    cap_h = 44
    img = Image.new("RGB", (sq * 8, sq * 8 + cap_h), (250, 250, 250))
    d = ImageDraw.Draw(img)
    piece_font, glyphs = _font(int(sq * 0.8), True)
    text_font, _ = _font(13, False)
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, 7 - rank) if not flip else chess.square(7 - file, rank)
            color = LIGHT if (file + rank) % 2 == 0 else DARK
            if lastmove is not None and square in (lastmove.from_square, lastmove.to_square):
                color = LAST
            x0, y0 = file * sq, rank * sq
            d.rectangle([x0, y0, x0 + sq, y0 + sq], fill=color)
            piece = board.piece_at(square)
            if piece is not None:
                sym = piece.symbol()
                txt = GLYPHS[sym] if glyphs else sym
                fill = (20, 20, 20) if piece.color == chess.BLACK else (250, 250, 250)
                outline = (20, 20, 20)
                bbox = d.textbbox((0, 0), txt, font=piece_font)
                tx = x0 + (sq - (bbox[2] - bbox[0])) / 2 - bbox[0]
                ty = y0 + (sq - (bbox[3] - bbox[1])) / 2 - bbox[1]
                if glyphs:
                    # white glyphs are drawn as the filled black glyph in white with a dark outline
                    d.text((tx, ty), GLYPHS[sym.lower()], font=piece_font, fill=fill,
                           stroke_width=1, stroke_fill=outline)
                else:
                    d.text((tx, ty), txt, font=piece_font, fill=fill, stroke_width=1, stroke_fill=outline)
    d.text((6, sq * 8 + 4), caption, font=text_font, fill=(20, 20, 20))
    return img


def game_to_gif(game, path, size: int = 360, ms: int = 700, hold_last_ms: int = 2500, flip: bool = False) -> str:
    """Write `game` (a GameRecord from src.viz.play.play_game) to an animated GIF and
    return the path. `ms` per frame, the final frame held `hold_last_ms`."""
    frames = [render_board(chess.Board(), size, None, "Start position", flip)]
    for rec in game.moves:
        board = chess.Board(rec.fen_after)
        last = chess.Board(rec.fen_before).parse_san(rec.san)
        caption = (f"{rec.move_number}. {rec.side} {rec.san}  p={rec.prob_played:.2f}  "
                   f"value={rec.value:+.2f}  material={rec.material:+.0f}")
        frames.append(render_board(board, size, last, caption, flip))
    frames[-1] = frames[-1].copy()
    d = ImageDraw.Draw(frames[-1])
    d.text((6, size // 8 * 8 + 24), f"result: {game.result}", font=_font(13, False)[0], fill=(160, 20, 20))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    durations = [ms] * (len(frames) - 1) + [hold_last_ms]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
    return str(path)
