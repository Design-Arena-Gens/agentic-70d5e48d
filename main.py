from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pygame

# Board configuration
ROWS = 8
COLS = 8
SQUARE_SIZE = 90
BOARD_SIZE = ROWS * SQUARE_SIZE
INFO_PANEL_HEIGHT = 80
WIDTH = BOARD_SIZE
HEIGHT = BOARD_SIZE + INFO_PANEL_HEIGHT

# Colors
BOARD_LIGHT = (240, 217, 181)
BOARD_DARK = (181, 136, 99)
WHITE_COLOR = (240, 240, 240)
BLACK_COLOR = (30, 30, 30)
HIGHLIGHT_COLOR = (66, 135, 245)
MOVE_HINT_COLOR = (255, 215, 0)
STATUS_BG = (32, 34, 37)
STATUS_TEXT = (235, 235, 235)

DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def opponent(color: str) -> str:
    return "black" if color == "white" else "white"


def build_state(board_matrix: List[List[Optional["Piece"]]]) -> List[List[Optional[Tuple[str, bool]]]]:
    """Create a lightweight snapshot of the board suitable for move simulations."""
    snapshot: List[List[Optional[Tuple[str, bool]]]] = []
    for row in board_matrix:
        snapshot.append([None if piece is None else (piece.color, piece.king) for piece in row])
    return snapshot


def copy_state(state: List[List[Optional[Tuple[str, bool]]]]) -> List[List[Optional[Tuple[str, bool]]]]:
    return [row.copy() for row in state]


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < ROWS and 0 <= col < COLS


def is_promotion_row(row: int, color: str) -> bool:
    return (color == "white" and row == 0) or (color == "black" and row == ROWS - 1)


class Piece:
    PADDING = 12
    OUTLINE = 2

    COLOR_MAP = {
        "white": WHITE_COLOR,
        "black": BLACK_COLOR,
    }

    def __init__(self, row: int, col: int, color: str):
        self.row = row
        self.col = col
        self.color = color
        self.king = False

    def move(self, row: int, col: int) -> None:
        self.row = row
        self.col = col

    def draw(self, win: pygame.Surface) -> None:
        radius = SQUARE_SIZE // 2 - self.PADDING
        x = self.col * SQUARE_SIZE + SQUARE_SIZE // 2
        y = self.row * SQUARE_SIZE + SQUARE_SIZE // 2
        pygame.draw.circle(win, (10, 10, 10), (x, y), radius + self.OUTLINE)
        pygame.draw.circle(win, self.COLOR_MAP[self.color], (x, y), radius)
        if self.king:
            crown_radius = radius // 2
            pygame.draw.circle(win, MOVE_HINT_COLOR, (x, y), crown_radius, width=3)
            pygame.draw.circle(win, (255, 255, 255), (x, y), crown_radius - 6, width=2)


class Board:
    def __init__(self, setup: bool = True):
        self.board: List[List[Optional[Piece]]] = []
        self.white_left = 0
        self.black_left = 0
        self.white_kings = 0
        self.black_kings = 0
        if setup:
            self._create_board()

    def _create_board(self) -> None:
        self.board = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.white_left = 0
        self.black_left = 0
        self.white_kings = 0
        self.black_kings = 0
        for row in range(ROWS):
            for col in range(COLS):
                if (row + col) % 2 == 1:
                    if row < 3:
                        piece = Piece(row, col, "black")
                        self.board[row][col] = piece
                        self.black_left += 1
                    elif row > 4:
                        piece = Piece(row, col, "white")
                        self.board[row][col] = piece
                        self.white_left += 1

    def draw_squares(self, win: pygame.Surface) -> None:
        win.fill(BOARD_LIGHT)
        for row in range(ROWS):
            for col in range((row + 1) % 2, COLS, 2):
                pygame.draw.rect(
                    win,
                    BOARD_DARK,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                )

    def draw(self, win: pygame.Surface) -> None:
        self.draw_squares(win)
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece:
                    piece.draw(win)

    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        if not in_bounds(row, col):
            return None
        return self.board[row][col]

    def move(self, piece: Piece, row: int, col: int, captured: Optional[List[Tuple[int, int]]] = None) -> None:
        self.board[piece.row][piece.col] = None
        self.board[row][col] = piece
        piece.move(row, col)

        if captured:
            for r, c in captured:
                captured_piece = self.board[r][c]
                if captured_piece:
                    self.board[r][c] = None
                    if captured_piece.color == "white":
                        self.white_left -= 1
                        if captured_piece.king:
                            self.white_kings -= 1
                    else:
                        self.black_left -= 1
                        if captured_piece.king:
                            self.black_kings -= 1

        if not piece.king and is_promotion_row(row, piece.color):
            piece.king = True
            if piece.color == "white":
                self.white_kings += 1
            else:
                self.black_kings += 1

    def get_pieces(self, color: str) -> List[Piece]:
        pieces: List[Piece] = []
        for row in self.board:
            for piece in row:
                if piece and piece.color == color:
                    pieces.append(piece)
        return pieces

    def _capture_dfs(
        self,
        row: int,
        col: int,
        color: str,
        is_king: bool,
        state: List[List[Optional[Tuple[str, bool]]]],
        captured: List[Tuple[int, int]],
        sequences: List[Tuple[int, int, List[Tuple[int, int]]]],
    ) -> None:
        found_capture = False
        if is_king:
            for dr, dc in DIAGONALS:
                step = 1
                enemy_pos: Optional[Tuple[int, int]] = None
                while True:
                    nr = row + dr * step
                    nc = col + dc * step
                    if not in_bounds(nr, nc):
                        break
                    cell = state[nr][nc]
                    if cell is None:
                        if enemy_pos:
                            found_capture = True
                            new_state = copy_state(state)
                            new_state[row][col] = None
                            er, ec = enemy_pos
                            new_state[er][ec] = None
                            new_state[nr][nc] = (color, True)
                            self._capture_dfs(
                                nr,
                                nc,
                                color,
                                True,
                                new_state,
                                captured + [enemy_pos],
                                sequences,
                            )
                        step += 1
                        continue
                    cell_color, cell_king = cell
                    if cell_color == color:
                        break
                    if enemy_pos:
                        break
                    enemy_pos = (nr, nc)
                    step += 1
        else:
            for dr, dc in DIAGONALS:
                mid_r = row + dr
                mid_c = col + dc
                land_r = row + 2 * dr
                land_c = col + 2 * dc
                if not (in_bounds(land_r, land_c) and in_bounds(mid_r, mid_c)):
                    continue
                middle = state[mid_r][mid_c]
                landing = state[land_r][land_c]
                if middle and middle[0] != color and landing is None:
                    found_capture = True
                    new_state = copy_state(state)
                    new_state[row][col] = None
                    new_state[mid_r][mid_c] = None
                    promoted = is_king or is_promotion_row(land_r, color)
                    new_state[land_r][land_c] = (color, promoted)
                    self._capture_dfs(
                        land_r,
                        land_c,
                        color,
                        promoted,
                        new_state,
                        captured + [(mid_r, mid_c)],
                        sequences,
                    )
        if not found_capture and captured:
            sequences.append((row, col, captured))

    def _find_captures(self, piece: Piece) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        state = build_state(self.board)
        sequences: List[Tuple[int, int, List[Tuple[int, int]]]] = []
        self._capture_dfs(piece.row, piece.col, piece.color, piece.king, state, [], sequences)
        if not sequences:
            return {}
        max_capture = max(len(seq[2]) for seq in sequences)
        moves: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for dest_row, dest_col, captured in sequences:
            if len(captured) == max_capture:
                moves[(dest_row, dest_col)] = captured
        return moves

    def _find_simple_moves(self, piece: Piece) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        moves: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        if piece.king:
            for dr, dc in DIAGONALS:
                step = 1
                while True:
                    nr = piece.row + dr * step
                    nc = piece.col + dc * step
                    if not in_bounds(nr, nc):
                        break
                    if self.board[nr][nc] is None:
                        moves[(nr, nc)] = []
                    else:
                        break
                    step += 1
        else:
            dir_step = -1 if piece.color == "white" else 1
            for dc in (-1, 1):
                nr = piece.row + dir_step
                nc = piece.col + dc
                if in_bounds(nr, nc) and self.board[nr][nc] is None:
                    moves[(nr, nc)] = []
        return moves

    def _generate_piece_moves(self, piece: Piece) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        captures = self._find_captures(piece)
        if captures:
            return captures
        return self._find_simple_moves(piece)

    def get_player_moves(self, color: str) -> Dict[Piece, Dict[Tuple[int, int], List[Tuple[int, int]]]]:
        moves_map: Dict[Piece, Dict[Tuple[int, int], List[Tuple[int, int]]]] = {}
        capture_available = False
        for piece in self.get_pieces(color):
            moves = self._generate_piece_moves(piece)
            if moves:
                moves_map[piece] = moves
                if any(captures for captures in moves.values()):
                    capture_available = True
            else:
                moves_map[piece] = {}
        if capture_available:
            for piece in list(moves_map.keys()):
                moves = moves_map[piece]
                capture_moves = {dest: cap for dest, cap in moves.items() if cap}
                moves_map[piece] = capture_moves
        return moves_map

    def has_any_moves(self, color: str) -> bool:
        moves_map = self.get_player_moves(color)
        return any(moves_map[piece] for piece in moves_map)

    def evaluate(self, perspective: str) -> float:
        white_score = self.white_left + 0.7 * self.white_kings
        black_score = self.black_left + 0.7 * self.black_kings
        score = white_score - black_score
        return score if perspective == "white" else -score

    def clone(self) -> "Board":
        clone_board = Board(setup=False)
        clone_board.board = [[None for _ in range(COLS)] for _ in range(ROWS)]
        clone_board.white_left = 0
        clone_board.black_left = 0
        clone_board.white_kings = 0
        clone_board.black_kings = 0
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece:
                    new_piece = Piece(r, c, piece.color)
                    new_piece.king = piece.king
                    clone_board.board[r][c] = new_piece
                    if new_piece.color == "white":
                        clone_board.white_left += 1
                        if new_piece.king:
                            clone_board.white_kings += 1
                    else:
                        clone_board.black_left += 1
                        if new_piece.king:
                            clone_board.black_kings += 1
        return clone_board

    def generate_moves(self, color: str) -> List["MoveOption"]:
        options: List[MoveOption] = []
        moves_map = self.get_player_moves(color)
        for piece, moves in moves_map.items():
            if not moves:
                continue
            for dest, captures in moves.items():
                board_clone = self.clone()
                clone_piece = board_clone.get_piece(piece.row, piece.col)
                if clone_piece is None:
                    continue
                board_clone.move(clone_piece, dest[0], dest[1], captures.copy())
                options.append(
                    MoveOption(
                        start=(piece.row, piece.col),
                        end=dest,
                        captures=captures.copy(),
                        board=board_clone,
                    )
                )
        return options

    def check_winner(self) -> Optional[str]:
        if self.white_left == 0 or not self.has_any_moves("white"):
            return "black"
        if self.black_left == 0 or not self.has_any_moves("black"):
            return "white"
        return None


@dataclass
class MoveOption:
    start: Tuple[int, int]
    end: Tuple[int, int]
    captures: List[Tuple[int, int]]
    board: Board


class AI:
    def __init__(self, depth: int, color: str):
        self.depth = depth
        self.color = color

    def get_move(self, board: Board) -> Optional[MoveOption]:
        _, move = self._minimax(board, self.depth, True, float("-inf"), float("inf"))
        return move

    def _minimax(
        self,
        board: Board,
        depth: int,
        maximizing: bool,
        alpha: float,
        beta: float,
    ) -> Tuple[float, Optional[MoveOption]]:
        winner = board.check_winner()
        if depth == 0 or winner:
            score = board.evaluate(self.color)
            if winner == self.color:
                score += 100
            elif winner and winner != self.color:
                score -= 100
            return score, None

        current_color = self.color if maximizing else opponent(self.color)
        moves = board.generate_moves(current_color)
        if not moves:
            return board.evaluate(self.color), None

        best_move: Optional[MoveOption] = None

        if maximizing:
            max_eval = float("-inf")
            for move in moves:
                evaluation, _ = self._minimax(move.board, depth - 1, False, alpha, beta)
                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = move
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return max_eval, best_move

        min_eval = float("inf")
        for move in moves:
            evaluation, _ = self._minimax(move.board, depth - 1, True, alpha, beta)
            if evaluation < min_eval:
                min_eval = evaluation
                best_move = move
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return min_eval, best_move


class Game:
    def __init__(self, win: pygame.Surface):
        self.win = win
        self.font = pygame.font.SysFont("arial", 24)
        self.large_font = pygame.font.SysFont("arial", 32, bold=True)
        self._init()

    def _init(self) -> None:
        self.board = Board()
        self.turn = "white"
        self.selected: Optional[Piece] = None
        self.valid_moves: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.winner: Optional[str] = None
        self.ai = AI(depth=4, color="black")

    def update(self) -> None:
        self.board.draw(self.win)
        if self.selected:
            self._highlight_selected(self.selected)
        self._draw_valid_moves(self.valid_moves)
        self._draw_status_panel()
        pygame.display.flip()

    def reset(self) -> None:
        self._init()

    def _highlight_selected(self, piece: Piece) -> None:
        pygame.draw.rect(
            self.win,
            HIGHLIGHT_COLOR,
            (
                piece.col * SQUARE_SIZE,
                piece.row * SQUARE_SIZE,
                SQUARE_SIZE,
                SQUARE_SIZE,
            ),
            width=4,
        )

    def _draw_valid_moves(self, moves: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> None:
        for (row, col) in moves:
            pygame.draw.circle(
                self.win,
                MOVE_HINT_COLOR,
                (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                10,
            )

    def _draw_status_panel(self) -> None:
        panel_rect = pygame.Rect(0, BOARD_SIZE, WIDTH, INFO_PANEL_HEIGHT)
        pygame.draw.rect(self.win, STATUS_BG, panel_rect)

        if self.winner:
            text = f"Победитель: {'Вы' if self.winner == 'white' else 'AI'}"
            rendered = self.large_font.render(text, True, STATUS_TEXT)
            self.win.blit(rendered, (20, BOARD_SIZE + 20))
            prompt = self.font.render("Нажмите R чтобы начать заново или Esc для выхода", True, STATUS_TEXT)
            self.win.blit(prompt, (20, BOARD_SIZE + 50))
            return

        turn_text = "Ваш ход" if self.turn == "white" else "Ходит AI"
        rendered = self.large_font.render(turn_text, True, STATUS_TEXT)
        self.win.blit(rendered, (20, BOARD_SIZE + 20))

        moves_map = self.board.get_player_moves(self.turn)
        capture_available = any(
            captures for move_dict in moves_map.values() for captures in move_dict.values()
        )
        hint_text = "Взятие обязательно" if capture_available else "Выберите шашку и сделайте ход"
        hint_rendered = self.font.render(hint_text, True, STATUS_TEXT)
        self.win.blit(hint_rendered, (20, BOARD_SIZE + 50))

    def select(self, row: int, col: int) -> bool:
        piece = self.board.get_piece(row, col)
        if piece and piece.color == self.turn:
            self.selected = piece
            moves_map = self.board.get_player_moves(self.turn)
            self.valid_moves = moves_map.get(piece, {})
            return True
        return False

    def _move(self, row: int, col: int) -> bool:
        if not self.selected:
            return False
        if (row, col) not in self.valid_moves:
            return False
        captures = self.valid_moves[(row, col)]
        self.board.move(self.selected, row, col, captures)
        self._post_move()
        return True

    def _post_move(self) -> None:
        self.selected = None
        self.valid_moves = {}
        self._check_winner()
        if not self.winner:
            self.turn = opponent(self.turn)

    def _check_winner(self) -> None:
        self.winner = self.board.check_winner()

    def handle_click(self, pos: Tuple[int, int]) -> None:
        x, y = pos
        if y >= BOARD_SIZE or self.winner:
            return
        row = y // SQUARE_SIZE
        col = x // SQUARE_SIZE
        if self.selected:
            if not self._move(row, col):
                self.select(row, col)
        else:
            self.select(row, col)

    def ai_turn(self) -> None:
        if self.turn != self.ai.color or self.winner:
            return
        move = self.ai.get_move(self.board)
        if move is None:
            self.winner = opponent(self.ai.color)
            return
        piece = self.board.get_piece(*move.start)
        if not piece:
            return
        self.board.move(piece, move.end[0], move.end[1], move.captures)
        self._post_move()


def main() -> None:
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Русские шашки с AI")
    clock = pygame.time.Clock()
    game = Game(win)

    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
            elif event.type == pygame.MOUSEBUTTONDOWN and game.turn == "white":
                game.handle_click(pygame.mouse.get_pos())

        if game.turn == "black" and not game.winner:
            pygame.time.delay(250)
            game.ai_turn()

        game.update()

    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        pygame.quit()
        raise exc
