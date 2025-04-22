import chess
import chess.engine
import pygame
import random
import json
import os
from pathlib import Path
import csv
from datetime import datetime

pygame.init()
WIDTH, HEIGHT = 480, 480
SQUARE_SIZE = WIDTH // 8
WHITE = (238, 238, 210)
BLACK = (118, 150, 86)

piece_images = {}
piece_types = {'K': 'king', 'Q': 'queen', 'R': 'rook', 'B': 'bishop', 'N': 'knight', 'P': 'pawn'}

for piece_symbol, piece_name in piece_types.items():
    white_img = pygame.image.load(f'/Users/johndolye/ChessProject/Images/white-{piece_name}.png')
    black_img = pygame.image.load(f'/Users/johndolye/ChessProject/Images/black-{piece_name}.png')

    white_img = pygame.transform.scale(white_img, (SQUARE_SIZE, SQUARE_SIZE))
    black_img = pygame.transform.scale(black_img, (SQUARE_SIZE, SQUARE_SIZE))

    piece_images[f'w{piece_symbol}'] = white_img
    piece_images[f'b{piece_symbol}'] = black_img

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess AI Endgame")

STOCKFISH_PATH = "/Users/johndolye/ChessProject/stockfish/stockfish-macos-m1-apple-silicon"
StockFishEngine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 350,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0  
}

WEIGHTS_FILE = Path("endgame_weights.json")
DEFAULT_WEIGHTS = {
    'material': 3,         
    'king_position': 3,    
    'pawn_structure': 3,   
    'piece_activity': 3,   
    'endgame_tactics': 3,
	'piece_safety': 3,
}  

# Draws chessboard
def draw_board():
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

# Addes pieces to board
def draw_pieces(board):
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        color_prefix = 'w' if piece.color == chess.WHITE else 'b'
        piece_key = f'{color_prefix}{piece.symbol().upper()}'
        screen.blit(piece_images[piece_key], (col * SQUARE_SIZE, (7-row) * SQUARE_SIZE))

# Generates endgame positions makes sure to only accept valid ones
def generate_endgame_positions(num_positions=1):
    """Generate endgame positions meeting criteria."""
    positions = []
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    
    while len(positions) < num_positions:
        board = chess.Board()
        board.clear()
        
        wk_sq, bk_sq = place_kings_safely()
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        
        for color in [chess.WHITE, chess.BLACK]:
            num_pieces = random.randint(1, 3)
            added = 0
            while added < num_pieces:
                piece = random.choice(piece_types)
                sq = random_empty_square(board)
                if sq is None:
                    continue 
                    
                if piece == chess.PAWN:
                    rank = chess.square_rank(sq)
                    if (color == chess.WHITE and rank == 7) or (color == chess.BLACK and rank == 0):
                        continue
                board.set_piece_at(sq, chess.Piece(piece, color))
                added += 1
        
        if validate_endgame(board):
            board.turn = random.choice([chess.WHITE, chess.BLACK])
            positions.append(board)
        else:
            continue
    
    return positions[0] if positions else chess.Board()

# Helper function to make sure that the generated position is valid
def validate_endgame(board):
    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        return False
    
    for color in [chess.WHITE, chess.BLACK]:
        non_king_pieces = sum(
            1 for sq in board.piece_map().keys()
            if board.color_at(sq) == color 
            and board.piece_type_at(sq) != chess.KING
        )
        if not 1 <= non_king_pieces <= 3:
            return False
    
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if chess.square_distance(wk, bk) < 2:
        return False
    
    return board.is_valid() and not board.is_check()

# Locates empty square to place piece
def random_empty_square(board):
    for _ in range(1000):
        sq = chess.Square(random.randint(0, 63))
        if not board.piece_at(sq):
            return sq
    return None 

# Places king safely
def place_kings_safely(max_attempts=1000):
    for _ in range(max_attempts):
        wk = random.randint(0, 63)
        bk = random.randint(0, 63)
        if chess.square_distance(wk, bk) >= 2:
            return wk, bk
    return chess.E1, chess.E8 

# Evaluation function for piece safety heuristic
def piece_safety_evaluation(board):
    score = 0
    
    # Evaluate each piece's safety
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue 
            
        attackers = list(board.attackers(not piece.color, square))
        defenders = list(board.attackers(piece.color, square))
        
        # Skip if no attackers
        if not attackers:
            continue
            
        piece_value = PIECE_VALUES[piece.piece_type]
        multiplier = 1 if piece.color == chess.WHITE else -1
        
        # Case 1: Hanging piece (no defenders)
        if not defenders:
            score -= piece_value * multiplier * 0.8  # 80% penalty - piece is likely to be captured
            continue
            
        # Case 2: Piece is defended but might be under threat
        # Calculate the smallest attacker value
        min_attacker_value = min(PIECE_VALUES[board.piece_at(sq).piece_type] for sq in attackers)
        
        # If the piece is attacked by a lower value piece
        if min_attacker_value < piece_value:
            # Calculate exchange value
            exchange_value = piece_value - min_attacker_value
            
            # Check if defenders can equalize
            can_recapture = False
            for defender in defenders:
                defender_value = PIECE_VALUES[board.piece_at(defender).piece_type]
                if defender_value <= min_attacker_value:
                    can_recapture = True
                    break
                    
            # If no good recapture, apply penalty
            if not can_recapture:
                score -= exchange_value * multiplier * 0.5  # 50% penalty - might not be captured
    
    return score


# Evaluation function for minimax
def evaluate_endgame(board: chess.Board, weights: dict):
    if board.is_checkmate():
        return float('inf') if board.turn == chess.BLACK else float('-inf')
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    material_score = evaluate_material(board)
    king_score = evaluate_king_position(board)
    pawn_score = evaluate_pawn_structure(board)
    piece_score = evaluate_piece_activity(board)
    tactics_score = evaluate_endgame_tactics(board)
    safety_score = piece_safety_evaluation(board)

    score = (
        weights.get('material', DEFAULT_WEIGHTS['material']) * material_score +
        weights.get('king_position', DEFAULT_WEIGHTS['king_position']) * king_score +
        weights.get('pawn_structure', DEFAULT_WEIGHTS['pawn_structure']) * pawn_score +
        weights.get('piece_activity', DEFAULT_WEIGHTS['piece_activity']) * piece_score +
        weights.get('endgame_tactics', DEFAULT_WEIGHTS['endgame_tactics']) * tactics_score +
        weights.get('piece_safety', DEFAULT_WEIGHTS['piece_safety']) * safety_score
    )
    
    return score if board.turn == chess.WHITE else -score

# Evaluation function for material heuristic
def evaluate_material(board):
    
    white_material = sum(PIECE_VALUES[piece.piece_type] 
                        for piece in board.piece_map().values() 
                        if piece.color == chess.WHITE and piece.piece_type != chess.KING)
    
    black_material = sum(PIECE_VALUES[piece.piece_type] 
                        for piece in board.piece_map().values() 
                        if piece.color == chess.BLACK and piece.piece_type != chess.KING)
    
    material_score = white_material - black_material
    total_material = white_material + black_material
    
    if total_material < 1500:
        white_knights = len(list(board.pieces(chess.KNIGHT, chess.WHITE)))
        white_bishops = len(list(board.pieces(chess.BISHOP, chess.WHITE)))
        black_knights = len(list(board.pieces(chess.KNIGHT, chess.BLACK)))
        black_bishops = len(list(board.pieces(chess.BISHOP, chess.BLACK)))
        
        if white_bishops >= 2:
            material_score += 75 if total_material < 1000 else 50
        if black_bishops >= 2:
            material_score -= 75 if total_material < 1000 else 50
            
        white_pawns = len(list(board.pieces(chess.PAWN, chess.WHITE)))
        black_pawns = len(list(board.pieces(chess.PAWN, chess.BLACK)))
        
        if white_pawns < 4:
            material_score += white_bishops * 20 - white_knights * 10
        if black_pawns < 4:
            material_score -= black_bishops * 20 - black_knights * 10
    
    return material_score

# Evaluation function for king position heuristic
def evaluate_king_position(board):
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    
    wk_file = chess.square_file(white_king_square)
    wk_rank = chess.square_rank(white_king_square)
    bk_file = chess.square_file(black_king_square)
    bk_rank = chess.square_rank(black_king_square)
    
    king_distance = chess.square_distance(white_king_square, black_king_square)
    white_center_distance = distance_from_center(white_king_square)
    black_center_distance = distance_from_center(black_king_square)
    
    score = 0
    
    # Material-based phase detection for adaptive king evaluation
    total_pieces = len(board.piece_map()) - 2  # Excluding kings
    pawn_count = len(list(board.pieces(chess.PAWN, chess.WHITE))) + len(list(board.pieces(chess.PAWN, chess.BLACK)))
    
    # Deep endgame (few pieces, centralize king)
    if total_pieces <= 5:
        # King centrality is important in deep endgames
        score += (7 - white_center_distance) * 3.0
        score -= (7 - black_center_distance) * 3.0
        
        # King opposition
        file_diff = abs(wk_file - bk_file)
        rank_diff = abs(wk_rank - bk_rank)
        
        # Direct opposition
        if (file_diff == 0 and rank_diff == 2) or (file_diff == 2 and rank_diff == 0):
            score += 1.0 if board.turn == chess.BLACK else -1.0
        
        # Diagonal opposition
        elif file_diff == 2 and rank_diff == 2:
            score += 0.6 if board.turn == chess.BLACK else -0.6
            
        # Outflanking bonus (getting around the opponent's king)
        if king_distance <= 3 and pawn_count <= 2:
            # Bonus for pushing opponent's king to the edge
            edge_distance_white = min(wk_file, 7-wk_file, wk_rank, 7-wk_rank)
            edge_distance_black = min(bk_file, 7-bk_file, bk_rank, 7-bk_rank)
            score += 0.3 * (edge_distance_black - edge_distance_white)
    
    # Middlegame/early endgame (king safety and pawn proximity)
    else:
        # Kings should approach important pawns or stay back for safety
        score += evaluate_king_activity(board, white_king_square, black_king_square)
        
        # King safety from checks
        score += evaluate_king_safety(board, white_king_square, black_king_square)
    
    return score

# King activity helper function
def evaluate_king_activity(board, white_king_square, black_king_square):
    score = 0
    
    white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
    
    white_passed = [sq for sq in white_pawns if is_passed_pawn(board, sq, chess.WHITE)]
    black_passed = [sq for sq in black_pawns if is_passed_pawn(board, sq, chess.BLACK)]
    
    if white_passed:
        min_dist_to_passed = min(chess.square_distance(white_king_square, sq) for sq in white_passed)
        score += (8 - min_dist_to_passed) * 0.3
    
    if black_passed:
        min_dist_to_passed = min(chess.square_distance(black_king_square, sq) for sq in black_passed)
        score -= (8 - min_dist_to_passed) * 0.3
    
    if not white_passed and not black_passed:
        score += (7 - distance_from_center(white_king_square)) * 0.2
        score -= (7 - distance_from_center(black_king_square)) * 0.2
    
    return score

# Helper function makes sure king is safe
def evaluate_king_safety(board, white_king_square, black_king_square):
    score = 0
    
    white_checkers = 0
    black_checkers = 0
    
    for square, piece in board.piece_map().items():
        if piece.color == chess.BLACK and piece.piece_type != chess.KING:
            if white_king_square in board.attacks(square):
                white_checkers += 1
        elif piece.color == chess.WHITE and piece.piece_type != chess.KING:
            if black_king_square in board.attacks(square):
                black_checkers += 1
    
    score -= white_checkers
    score += black_checkers
    
    # Check for kings on edge (generally worse in endgames)
    white_edge = min(chess.square_file(white_king_square), 7-chess.square_file(white_king_square), 
                    chess.square_rank(white_king_square), 7-chess.square_rank(white_king_square))
    black_edge = min(chess.square_file(black_king_square), 7-chess.square_file(black_king_square), 
                    chess.square_rank(black_king_square), 7-chess.square_rank(black_king_square))
    
    if white_edge == 0:
        score -= 1
    if black_edge == 0:
        score += 1
    
    return score

# Evaluation function for pawn structure heuristic
def evaluate_pawn_structure(board):
    score = 0
    
    white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
    
    if not white_pawns and not black_pawns:
        return 0
    
    # 1. Pawn advancement with progressive weighting
    for square in white_pawns:
        rank = chess.square_rank(square)
        score += (2 ** rank) / 10
        
        if is_passed_pawn(board, square, chess.WHITE):
            passed_bonus = (rank + 1) ** 2
            score += passed_bonus
            
            if is_protected(board, square):
                score += rank * 0.3
                
            file = chess.square_file(square)
            path_clear = True
            for r in range(rank + 1, 8):
                if board.piece_at(chess.square(file, r)) is not None:
                    path_clear = False
                    break
            if path_clear:
                score += 2
    
    for square in black_pawns:
        rank = 7 - chess.square_rank(square) 
        score -= (2 ** rank) / 10
        
        if is_passed_pawn(board, square, chess.BLACK):
            passed_bonus = (rank + 1) ** 2 / 5
            score -= passed_bonus
            
            if is_protected(board, square):
                score -= rank * 0.3
                
            file = chess.square_file(square)
            path_clear = True
            for r in range(chess.square_rank(square) - 1, -1, -1):
                if board.piece_at(chess.square(file, r)) is not None:
                    path_clear = False
                    break
            if path_clear:
                score -= 0.5
    
    # 2. Pawn structure evaluation (doubled, isolated pawns)
    for file in range(8):
        white_in_file = sum(1 for sq in white_pawns if chess.square_file(sq) == file)
        black_in_file = sum(1 for sq in black_pawns if chess.square_file(sq) == file)
        
        # Penalty for doubled pawns
        if white_in_file > 1:
            score -= (white_in_file - 1) * 0.3
        if black_in_file > 1:
            score += (black_in_file - 1) * 0.3
        
        # Check for isolated pawns (no friendly pawns on adjacent files)
        if white_in_file > 0:
            isolated = True
            for adj_file in [file - 1, file + 1]:
                if 0 <= adj_file < 8:
                    if any(chess.square_file(sq) == adj_file for sq in white_pawns):
                        isolated = False
                        break
            if isolated:
                score -= 0.3 * white_in_file
        
        if black_in_file > 0:
            isolated = True
            for adj_file in [file - 1, file + 1]:
                if 0 <= adj_file < 8:
                    if any(chess.square_file(sq) == adj_file for sq in black_pawns):
                        isolated = False
                        break
            if isolated:
                score += 0.3 * black_in_file
    
    # 3. Connected pawns bonus
    for i, sq1 in enumerate(white_pawns):
        for sq2 in white_pawns[i+1:]:
            if abs(chess.square_file(sq1) - chess.square_file(sq2)) == 1:
                if abs(chess.square_rank(sq1) - chess.square_rank(sq2)) <= 1:
                    score += 0.2
    
    for i, sq1 in enumerate(black_pawns):
        for sq2 in black_pawns[i+1:]:
            if abs(chess.square_file(sq1) - chess.square_file(sq2)) == 1:
                if abs(chess.square_rank(sq1) - chess.square_rank(sq2)) <= 1:
                    score -= 0.2
    
    # 4. Promotion potential (which side is winning the pawn race)
    if white_pawns and black_pawns:
        most_advanced_white = max(chess.square_rank(sq) for sq in white_pawns)
        most_advanced_black = min(chess.square_rank(sq) for sq in black_pawns)
        
        white_to_promotion = 7 - most_advanced_white
        black_to_promotion = most_advanced_black
        
        if white_to_promotion < black_to_promotion - 1:
            score += (black_to_promotion - white_to_promotion) * 0.5
        elif black_to_promotion < white_to_promotion - 1:
            score -= (white_to_promotion - black_to_promotion) * 0.5
    
    return score

# Evaluation function for piece activity heuristic
def evaluate_piece_activity(board):
    score = 0
    
    # 1. Mobility (number of legal moves for each piece)
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue  # Kings evaluated separately
            
        attacks = len(list(board.attacks(square)))
        
        mobility_weight = {
            chess.PAWN: 0.02,
            chess.KNIGHT: 0.1,
            chess.BISHOP: 0.15,
            chess.ROOK: 0.06,
            chess.QUEEN: 0.05
        }.get(piece.piece_type, 0)
        
        if piece.color == chess.WHITE:
            score += attacks * mobility_weight
        else:
            score -= attacks * mobility_weight
    
    # 2. Piece safety (hanging pieces, threats)
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
            
        attackers = list(board.attackers(not piece.color, square))
        defenders = list(board.attackers(piece.color, square))
        
        piece_value = PIECE_VALUES[piece.piece_type]
        
        if attackers and not defenders:
            if piece.color == chess.WHITE:
                score -= piece_value
            else:
                score += piece_value * 0.5
        
        # Piece under attack but defended
        elif attackers:
            min_attacker_value = min(PIECE_VALUES[board.piece_at(sq).piece_type] for sq in attackers)
            if min_attacker_value < piece_value:
                if piece.color == chess.WHITE:
                    score -= 0.05 * (piece_value - min_attacker_value)
                else:
                    score += 0.05 * (piece_value - min_attacker_value)
    
    # 3. Piece positioning
    # Rooks
    for square in board.pieces(chess.ROOK, chess.WHITE):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        if not any(chess.square_file(sq) == file for sq in board.pieces(chess.PAWN, chess.WHITE)) and \
           not any(chess.square_file(sq) == file for sq in board.pieces(chess.PAWN, chess.BLACK)):
            score += 0.3
        
        if rank == 6:
            score += 0.5
            
        for pawn_sq in board.pieces(chess.PAWN, chess.WHITE):
            if is_passed_pawn(board, pawn_sq, chess.WHITE) and chess.square_file(pawn_sq) == file:
                if rank < chess.square_rank(pawn_sq):
                    score += 0.4
    
    for square in board.pieces(chess.ROOK, chess.BLACK):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        if not any(chess.square_file(sq) == file for sq in board.pieces(chess.PAWN, chess.WHITE)) and \
           not any(chess.square_file(sq) == file for sq in board.pieces(chess.PAWN, chess.BLACK)):
            score -= 0.3
        
        if rank == 1:
            score -= 0.5
            
        for pawn_sq in board.pieces(chess.PAWN, chess.BLACK):
            if is_passed_pawn(board, pawn_sq, chess.BLACK) and chess.square_file(pawn_sq) == file:
                if rank > chess.square_rank(pawn_sq):
                    score -= 0.4
    
    # Knights (prefer central positions)
    for square in board.pieces(chess.KNIGHT, chess.WHITE):
        score += (4 - distance_from_center(square)) * 0.05
    
    for square in board.pieces(chess.KNIGHT, chess.BLACK):
        score -= (4 - distance_from_center(square)) * 0.05
    
    # Bishops (penalty for blocked bishops)
    for square in board.pieces(chess.BISHOP, chess.WHITE):
        if len(list(board.attacks(square))) <= 5: 
            score -= 0.2
    
    for square in board.pieces(chess.BISHOP, chess.BLACK):
        if len(list(board.attacks(square))) <= 5:
            score += 0.2
    
    return score

# Helper function to determine passed pawns
def is_passed_pawn(board, square, color):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    enemy_color = not color
    
    direction = 1 if color == chess.WHITE else -1
    
    start_rank = rank + direction
    end_rank = 8 if direction > 0 else -1
    
    for r in range(start_rank, end_rank, direction):
        for f in range(max(0, file - 1), min(8, file + 2)):
            check_square = chess.square(f, r)
            piece = board.piece_at(check_square)
            if piece and piece.piece_type == chess.PAWN and piece.color == enemy_color:
                return False
    return True

# Finds disance from board center
def distance_from_center(square):
    file = chess.square_file(square) - 3.5
    rank = chess.square_rank(square) - 3.5
    return file + rank

# Check if piece is protected
def is_protected(board, square):
    color = board.color_at(square)
    for attack_square in board.attackers(color, square):
        return True
    return False

# Quiescence search to enhance minimax function
def quiescence_search(board, alpha, beta, depth, weights):
    stand_pat = evaluate_endgame(board, weights)
    
    if depth == 0:
        return stand_pat
    
    if stand_pat >= beta:
        return beta
    
    if alpha < stand_pat:
        alpha = stand_pat
    
    # Only look at capture moves, ordered by MVV-LVA
    captures = []
    for move in board.legal_moves:
        if board.is_capture(move):
            victim_value = 0
            from_piece = board.piece_at(move.from_square)
            
            if board.is_en_passant(move):
                victim_value = PIECE_VALUES[chess.PAWN]
            else:
                to_piece = board.piece_at(move.to_square)
                if to_piece:
                    victim_value = PIECE_VALUES.get(to_piece.piece_type, 0)
            
            aggressor_value = PIECE_VALUES.get(from_piece.piece_type, 0)
            move_score = victim_value * 100 - aggressor_value
            captures.append((move, move_score))
    
    captures.sort(key=lambda x: x[1], reverse=True)
    
    for move, _ in captures:
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, depth - 1, weights)
        board.pop()
        
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    
    return alpha

# Evaluatuion function for engame tactics heuristic
def evaluate_endgame_tactics(board):
    score = 0
    legal_moves = list(board.legal_moves)
    
    # 1. Check for stalemate risk/opportunity
    if not board.is_check() and len(legal_moves) < 3:
        if len(legal_moves) < 3:
            score += -2.0 if board.turn == chess.WHITE else 2.0
    
    # 2. Control of critical squares
    critical_squares = [
        chess.E4, chess.D4, chess.E5, chess.D5,  
        chess.C3, chess.F3, chess.C6, chess.F6   
    ]
    
    for square in critical_squares:
        white_control = len(list(board.attackers(chess.WHITE, square)))
        black_control = len(list(board.attackers(chess.BLACK, square)))
        score += 0.05 * (white_control - black_control)
    
    # 3. Check for potential checkmate patterns
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            score += 10 if board.turn != chess.WHITE else -10
        board.pop()
    
    # 4. Distance of pieces from enemy king (coordination)
    black_king = board.king(chess.BLACK)
    white_king = board.king(chess.WHITE)
    
    for square, piece in board.piece_map().items():
        if piece.piece_type != chess.KING and piece.piece_type != chess.PAWN:
            if piece.color == chess.WHITE:
                score += (8 - chess.square_distance(square, black_king)) * 0.1
            else:
                score -= (8 - chess.square_distance(square, white_king)) * 0.1
    
    return score

# Prelimnary move filter to improve minimax search efficiency
def order_moves(board, moves):
    scored_moves = []
    
    for move in moves:
        score = 0
        is_unsafe = False
        
        moving_piece = board.piece_at(move.from_square)
        if not moving_piece:
            continue
            
        piece_value = PIECE_VALUES.get(moving_piece.piece_type, 0)
        
        board.push(move)
        
        # 1. Checkmate detection (highest priority)
        if board.is_checkmate():
            score += 1000000
            board.pop()
            return [(move, score)]
        
        # 2. Check if the moved piece is now hanging
        to_square = move.to_square
        attackers = list(board.attackers(not board.turn, to_square))
        defenders = list(board.attackers(board.turn, to_square))

        if attackers and moving_piece.piece_type != chess.KING:
            min_attacker_value = min(PIECE_VALUES[board.piece_at(sq).piece_type] for sq in attackers)
            
            if not defenders:
                is_unsafe = True
                score -= piece_value * 10
            elif min_attacker_value < piece_value:
                attack_value = min_attacker_value  
                defense_value = 0
                if defenders:
                    defense_value = min(PIECE_VALUES[board.piece_at(sq).piece_type] for sq in defenders)
                
                if attack_value < piece_value and (not defenders or defense_value > attack_value):
                    is_unsafe = True
                    score -= (piece_value - attack_value) * 5
        
        # 3. Check detection (high priority)
        if board.is_check():
            checking_piece_square = to_square
            check_attackers = list(board.attackers(not board.turn, checking_piece_square))
            if not check_attackers:
                score += 3000  
                is_unsafe = False  
            else:
                score += 500  
        
        board.pop()
        
        if is_unsafe and len(list(board.legal_moves)) > 3:
            continue
        
        # 4. Captures prioritization (MVV-LVA)
        if board.is_capture(move):
            victim_value = 0
            
            # en passant
            if board.is_en_passant(move):
                victim_value = PIECE_VALUES[chess.PAWN]
            else:
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    victim_value = PIECE_VALUES.get(captured_piece.piece_type, 0)
            
            # MVV-LVA
            aggressor_value = PIECE_VALUES.get(moving_piece.piece_type, 0)
            score += victim_value * 100 - aggressor_value * 10
            
            if victim_value >= piece_value:
                is_unsafe = False
            
        # 5. Pawn promotion logic (very high priority)
        if moving_piece.piece_type == chess.PAWN:
            from_rank = chess.square_rank(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            
            if moving_piece.color == chess.WHITE:
                if to_rank > from_rank:
                    score += (to_rank - from_rank) * 50
                    
                    if to_rank == 7:
                        promotion_value = PIECE_VALUES.get(move.promotion or chess.QUEEN, 900)
                        score += promotion_value
                        is_unsafe = False 
            else:
                if to_rank < from_rank:
                    score += (from_rank - to_rank) * 50
                    
                    if to_rank == 0:
                        promotion_value = PIECE_VALUES.get(move.promotion or chess.QUEEN, 900)
                        score += promotion_value
                        is_unsafe = False
                            
        # 6. Piece centralization
        if moving_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            from_center_dist = distance_from_center(move.from_square)
            to_center_dist = distance_from_center(move.to_square)
            if to_center_dist < from_center_dist:
                score += 10
        
        # 7. Rook positioning bonuses
        if moving_piece.piece_type == chess.ROOK:
            to_file = chess.square_file(move.to_square)
            to_rank = chess.square_rank(move.to_square)
            
            open_file = True
            for r in range(8):
                check_sq = chess.square(to_file, r)
                piece_at_sq = board.piece_at(check_sq)
                if piece_at_sq and piece_at_sq.piece_type == chess.PAWN:
                    open_file = False
                    break
            
            if open_file:
                score += 30
            
            # Bonus for 7th rank (white) or 2nd rank (black)
            if (moving_piece.color == chess.WHITE and to_rank == 6) or \
               (moving_piece.color == chess.BLACK and to_rank == 1):
                score += 50
                
        scored_moves.append((move, score))
    
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    
    if not scored_moves:
        return [(move, 0) for move in moves]
    
    return scored_moves

# Get stockfish bot move
def get_stockfish_move(board, time_limit, difficulty):
    StockFishEngine.configure({"Skill Level": difficulty}) 
    result = StockFishEngine.play(board, chess.engine.Limit(time=time_limit)) 
    return result.move

# Main minimax function to decide which move to make. Utilizes alpha beta pruning
def minimax(board, depth, alpha, beta, maximizing_player, weights):

    if board.is_game_over():
        if board.is_checkmate():
            return None, float('inf') if board.turn != chess.WHITE else float('-inf')
        return None, 0  

    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return None, evaluate_endgame(board, weights)
    
    for move in legal_moves:
        board.push(move)
        is_checkmate = board.is_checkmate()
        board.pop()
        if is_checkmate:
            return move, float('inf') if maximizing_player else float('-inf')
    
    if depth == 0:
        eval_score = quiescence_search(board, alpha, beta, 3, weights)
        return None, eval_score
    
    scored_moves = order_moves(board, legal_moves)
    ordered_moves = [move for move, _ in scored_moves]
    
    best_move = ordered_moves[0] if ordered_moves else None
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in ordered_moves:
            board.push(move)
            _, eval_score = minimax(board, depth - 1, alpha, beta, False, weights)
            board.pop()
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return best_move, max_eval
    else:
        min_eval = float('inf')
        for move in ordered_moves:
            board.push(move)
            _, eval_score = minimax(board, depth - 1, alpha, beta, True, weights)
            board.pop()
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
                
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return best_move, min_eval


# Actual function to simulate games and than save the results
def simulate_games(num_games, stockfish_difficulty, position_pool=None, results_file=None):
    """Run simulation with fixed positions and save results to file."""
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    skipped = 0
    game_data = []  
    
    if position_pool is None:
        position_pool = load_position_pool()
    
    if len(position_pool) < num_games:
        print(f"Warning: Not enough positions in pool ({len(position_pool)}). Generating more.")
        additional_positions = generate_fixed_position_pool(num_games - len(position_pool))
        position_pool.extend(additional_positions)
        save_position_pool(position_pool)
    
    game_positions = position_pool[:num_games]

    for i in range(num_games):
        print(f"\n=== Game {i+1} ===")
        try:
            board = chess.Board(game_positions[i])
            if not board.is_valid() or board.is_game_over():
                print(f"Skipping invalid position: {game_positions[i]}")
                skipped += 1
                continue
                
            white_material = sum(PIECE_VALUES[piece.piece_type] 
                               for piece in board.piece_map().values() 
                               if piece.color == chess.WHITE and piece.piece_type != chess.KING)
            black_material = sum(PIECE_VALUES[piece.piece_type] 
                               for piece in board.piece_map().values() 
                               if piece.color == chess.BLACK and piece.piece_type != chess.KING)
            initial_material_diff = white_material - black_material
            
            starting_fen = board.fen()
            
            result = play_game_with_board(board, stockfish_difficulty, display_gui=True, verbose=True)

            if result in results:
                results[result] += 1
                game_data.append({
                    'game_num': i+1,
                    'starting_fen': starting_fen,
                    'initial_material_diff': initial_material_diff,
                    'result': result
                })
            elif result == "quit":
                print("Simulation manually exited.")
                break
            else:
                print(f"Skipped game {i+1} due to unexpected result: {result}")
                skipped += 1
        except Exception as e:
            print(f"Error in game {i+1}: {str(e)}")
            skipped += 1

    print("\n=== Simulation Complete ===")
    print(f"Games attempted: {num_games}")
    print(f"Games played: {sum(results.values())}")
    print(f"AI Wins (White): {results['1-0']}")
    print(f"Stockfish Wins (Black): {results['0-1']}")
    print(f"Draws: {results['1/2-1/2']}")
    print(f"Games skipped: {skipped}")
    
    if results_file is None:
        results_file = f"chess_results_SF{stockfish_difficulty}.csv"
    
    save_results_to_csv(
        results, 
        stockfish_difficulty, 
        sum(results.values()), 
        skipped, 
        filename=results_file
    )
    
    save_game_details(game_data, f"chess_game_details_SF{stockfish_difficulty}.csv")
    
    return results

# Save game results
def save_results_to_csv(results, stockfish_difficulty, games_played, skipped, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chess_results_{timestamp}.csv"
    
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'stockfish_difficulty', 'games_played', 
                     'ai_wins', 'stockfish_wins', 'draws', 'skipped',
                     'ai_win_rate', 'draw_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        ai_win_rate = results["1-0"] / games_played if games_played > 0 else 0
        draw_rate = results["1/2-1/2"] / games_played if games_played > 0 else 0
        
        writer.writerow({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'stockfish_difficulty': stockfish_difficulty,
            'games_played': games_played,
            'ai_wins': results["1-0"],
            'stockfish_wins': results["0-1"],
            'draws': results["1/2-1/2"],
            'skipped': skipped,
            'ai_win_rate': f"{ai_win_rate:.2f}",
            'draw_rate': f"{draw_rate:.2f}"
        })
    
    print(f"Results saved to {filename}")
    return filename

# Generates list of positions to test
def generate_fixed_position_pool(num_positions=100, max_material_diff=900):
    positions = []
    attempts = 0
    max_attempts = num_positions * 10  
    
    print(f"Generating {num_positions} positions with max material difference of {max_material_diff}...")
    
    while len(positions) < num_positions and attempts < max_attempts:
        attempts += 1
        board = generate_endgame_positions(num_positions=1)
        
        white_material = sum(PIECE_VALUES[piece.piece_type] 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.WHITE and piece.piece_type != chess.KING)
        black_material = sum(PIECE_VALUES[piece.piece_type] 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.BLACK and piece.piece_type != chess.KING)
        material_diff = abs(white_material - black_material)
        
        if material_diff <= max_material_diff and validate_endgame(board):
            board.turn = random.choice([chess.WHITE, chess.BLACK])
            if board.is_valid() and not board.is_game_over():
                positions.append(board.fen())
                if len(positions) % 10 == 0:
                    print(f"Generated {len(positions)} positions so far...")
    
    print(f"Successfully generated {len(positions)} positions after {attempts} attempts.")
    return positions

def save_position_pool(positions, filename="endgame_positions.json"):
    """Save the generated positions to a file."""
    with open(filename, 'w') as f:
        json.dump(positions, f)
    print(f"Saved {len(positions)} positions to {filename}")

def load_position_pool(filename="endgame_positions.json"):
    """Load positions from a file."""
    try:
        with open(filename, 'r') as f:
            positions = json.load(f)
        print(f"Loaded {len(positions)} positions from {filename}")
        return positions
    except FileNotFoundError:
        print(f"Position file {filename} not found. Generating new positions.")
        positions = generate_fixed_position_pool()
        save_position_pool(positions, filename)
        return positions

def play_game_with_board(board, stockfish_difficulty, display_gui=True, verbose=True):
    """
    Runs a single game between AI and Stockfish using a pre-generated board.
    Returns the result: "1-0", "0-1", or "1/2-1/2"
    """
    weights = DEFAULT_WEIGHTS

    if display_gui:
        screen.fill(WHITE)
        draw_board()
        draw_pieces(board)
        pygame.display.flip()

    ai_thinking = False
    stockfish_thinking = False
    
    white_material = sum(PIECE_VALUES[piece.piece_type] for piece in board.piece_map().values() if piece.color == chess.WHITE and piece.piece_type != chess.KING)
    black_material = sum(PIECE_VALUES[piece.piece_type] for piece in board.piece_map().values() if piece.color == chess.BLACK and piece.piece_type != chess.KING)
    material_advantage = white_material - black_material

    if verbose:
        print(f"\nStarting game with endgame position...")
        print(f"FEN: {board.fen()}")
        print(f"White material: {white_material}")
        print(f"Black material: {black_material}")
        print(f"Initial material advantage: {'White' if material_advantage > 0 else 'Black' if material_advantage < 0 else 'Equal'} ({abs(material_advantage)})")
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")

    while not board.is_game_over():
        if display_gui:
            draw_board()
            draw_pieces(board)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return "quit"

        if board.turn == chess.WHITE and not ai_thinking:
            ai_thinking = True
            best_move, eval_score = minimax(board, depth=5, alpha=float('-inf'), beta=float('inf'),
                                            maximizing_player=True, weights=weights)
            if best_move:
                board.push(best_move)
            else:
                if verbose:
                    print("AI has no legal moves.")
                if board.is_game_over():
                    return board.result()
                else:
                    print("Unexpected crash")
                    return "unknown"
            ai_thinking = False

        elif board.turn == chess.BLACK and not stockfish_thinking:
            stockfish_thinking = True
            move = get_stockfish_move(board, time_limit=0.1, difficulty=stockfish_difficulty)
            if move:
                board.push(move)
            else:
                if verbose:
                    print("Stockfish has no legal moves.")
                if board.is_game_over():
                    return board.result()
                else:
                    print("Unexpected crash")
                    return "unknown"
            stockfish_thinking = False

        if display_gui:
            pygame.time.wait(500)

    result = board.result()
    if verbose:
        print("Game Over!")
        print("Result:", result)

    return result

def save_game_details(game_data, filename):
    if not game_data:
        return
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['game_num', 'starting_fen', 'initial_material_diff', 'result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for game in game_data:
            writer.writerow(game)
    
    print(f"Detailed game data saved to {filename}")

if __name__ == "__main__":
    try:
        position_pool = load_position_pool()
    except:
        print("Error loading position pool. Generating new one...")
        position_pool = generate_fixed_position_pool(num_positions=100, max_material_diff=900)
        save_position_pool(position_pool)
    
    difficulties = [1, 3, 5, 10]
    games_per_level = 50  
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"chess_results_summary_{timestamp}.csv"
    
    for difficulty in difficulties:
        print(f"\n===== Running {games_per_level} games against Stockfish level {difficulty} =====")
        simulate_games(
            num_games=games_per_level, 
            stockfish_difficulty=difficulty, 
            position_pool=position_pool,
            results_file=summary_file
        )