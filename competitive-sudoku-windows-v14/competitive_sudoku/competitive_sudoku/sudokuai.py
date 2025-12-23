#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from collections import defaultdict
from typing import List, Tuple, Dict
import copy

def createLookups(board: SudokuBoard) -> Tuple[
    Dict[Tuple[int, int], List[Tuple[int, int]]],
    Dict[Tuple[int, int], Tuple[int, int]]]:
    """
    Creates two lookup dictionaries, that keep track of regions on a board.
    @param board: A sudoku board. It contains the current position of a game.
    @return regionsToIndexes:  dict mapping a region coordinate to a list of indexes 
                        that belong to that region.
    @return indexesToRegion:  dict mapping a square index to the coordinate of the 
                       region it belongs to.
    """
    rows = board.region_height()
    cols = board.region_width()
    N = board.N
    values = N*N
    regionsToIndexes = defaultdict(list)
    indexesToRegion = dict()

    for index in range(0,values):
        regionRow = index // N // rows
        regionCol = index % N // cols
        region = (regionRow, regionCol)
        regionsToIndexes[region].append(index)
        indexesToRegion[index] = region

    return regionsToIndexes, indexesToRegion


def arithmeticSum(value : int) -> int:
    """
    Calculates arithmetic sum, where the difference between values is equal to one.
    @param value: the last value for this arithmetic sequence.
    @return: calculated arithmetic sum.
    """
    return (value/2) * (2 + (value - 1))


def calculateScore(game_state: GameState, move : Move, 
                   regionsToIndexes : Dict[Tuple[int, int], List[Tuple[int, int]]], 
                   indexesToRegions : Dict[Tuple[int, int], Tuple[int, int]]) -> Tuple[int, int]: 
    """
    Calculates how the score changes after making a move.
    @param game_state: Current game state 
    @param move: Tested move.
    @param regionsToIndexes: dict mapping a region coordinate to a list of indexes 
                             that belong to that region.
    @param indexesToRegions: dict mapping a square index to the coordinate of the 
                             region it belongs to.
    """
    board = game_state.board # accessing the board
    score = game_state.scores # current score, before the move
    currentPlayer = game_state.current_player # used to evaluete who should be given a point
    N = board.N
    goal = arithmeticSum(N) # value used to check if all squares are filled, we assume that all previous moves are legal

    # flags to evaluate how many points to score
    flag1 = 0 # row wise
    flag2 = 0 # column wise
    flag3 = 0 # square wise

    # making the move 
    board.put(move.square, move.value)
    index = board.square2index(move.square)

    # checking each region
    sum1 = 0
    sum2 = 0
    sum3 = 0 

    # row and column check
    row_idx, col_idx = move.square
    for i in range(0, len(board.squares)):
        if i // N == row_idx:
            sum1 += board.squares[i]
        if i % N == col_idx:
            sum2 += board.squares[i] 

    if sum1 == goal:
        flag1 = 1

    if sum2 == goal:
        flag2 = 1   

    # checking to which squre the moive belongs
    region_specific = indexesToRegions[index]
    # getting all indexes of memebers of a region
    toCheckSquares = regionsToIndexes[region_specific]
  
    for i in toCheckSquares:
        if board.squares[i] == 0: # we find a square that os empty
            sum3 = 0 # goal needs a reset
            break
        else: 
            sum3 += board.squares[i]

    # evaluating points
    if sum3 == goal: 
        flag3 = 1
    result = flag1 + flag2 + flag3
    # one region filled in
    if result == 1:
        score[currentPlayer-1] = score[currentPlayer-1] + 1 #we substract 1 from currnet player to use them as indexes
        return score
    # two regions filled in
    elif result == 2:
        score[currentPlayer-1] = score[currentPlayer-1] + 3
        return score
    # 3 regions filled in
    elif result == 3:
        score[currentPlayer-1] = score[currentPlayer-1] + 7
        return score
    else:
        return score
    

def scoreDifference(score: Tuple[int,int], player: int):
    """
    Calculates score differnce for given player
    
    @param score: score at the current gamestate
    @param player: which player checks for best move
    @retunr difference: the differnece between players
    """ 
    if player == 1:
        return score[0] - score[1]
    else:
        return score[1] - score[0]


def generate_legal_moves(game_state: GameState) -> List[Move]:
    """
    All non-taboo, Sudoku-legal moves for the current player in the given game state.
    """
    board = game_state.board
    N = board.N
    allowed = game_state.player_squares()
    box_h, box_w = board.region_height(), board.region_width()

    def square_allowed(square):
        return allowed is None or square in allowed

    def is_legal(square, value):
        r, c = square
        # row/col check
        for j in range(N):
            if board.get((r, j)) == value:
                return False
        for i in range(N):
            if board.get((i, c)) == value:
                return False
        # box check
        r0 = (r // box_h) * box_h
        c0 = (c // box_w) * box_w
        for rr in range(r0, r0 + box_h):
            for cc in range(c0, c0 + box_w):
                if board.get((rr, cc)) == value:
                    return False
        return True

    moves = []
    for i in range(N):
        for j in range(N):
            square = (i, j)
            if board.get(square) != SudokuBoard.empty:
                continue
            if not square_allowed(square):
                continue
            for value in range(1, N + 1):
                if TabooMove(square, value) in game_state.taboo_moves:
                    continue
                if not is_legal(square, value):
                    continue
                moves.append(Move(square, value))
    return moves


def apply_move(game_state: GameState, move: Move) -> GameState:
    child = copy.deepcopy(game_state)  # work on a fresh copy so caller stays unchanged

    # calculateScore writes the value to the board and updates the score
    look1, look2 = createLookups(child.board)  # reuse one set per board size if you cache it
    child.scores = calculateScore(child, move, look1, look2)

    # keep history/occupancy up to date
    child.moves.append(move)
    if child.occupied_squares1 is not None:
        (child.occupied_squares1 if game_state.current_player == 1 else child.occupied_squares2).append(move.square)

    # pass the turn to the other player
    child.current_player = 2 if game_state.current_player == 1 else 1
    return child


def alphabeta(game_state: GameState,
              depth: int,
              alpha: float,
              beta: float,
              maximizing: bool,
              root_player: int
             ) -> Tuple[float, Move | None]:
    """
    Alpha-beta search that evaluates with scoreDifference for the root player.
    """
    legal_moves = generate_legal_moves(game_state)
    if depth == 0 or not legal_moves:
        return scoreDifference(game_state.scores, root_player), None

    best_move = None
    if maximizing:
        value = float('-inf')
        for move in legal_moves:
            child = apply_move(game_state, move)
            child_value, _ = alphabeta(child, depth - 1, alpha, beta, False, root_player)
            if child_value > value:
                value, best_move = child_value, move
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = float('inf')
        for move in legal_moves:
            child = apply_move(game_state, move)
            child_value, _ = alphabeta(child, depth - 1, alpha, beta, True, root_player)
            if child_value < value:
                value, best_move = child_value, move
            beta = min(beta, value)
            if alpha >= beta:
                break
    return value, best_move


import competitive_sudoku.sudokuai
from competitive_sudoku.sudoku import GameState 

class SudokuAI:
    """Sudoku AI that computes a move for a given sudoku configuration."""
    
    def __init__(self):
        self.best_move: List[int] = [0, 0, 0]
        self.lock = None
        self.player_number = -1

    def compute_best_move(self, game_state: GameState) -> None:
        raise NotImplementedError

    def propose_move(self, move: Move) -> None:
        if self.lock:
            self.lock.acquire()
        i, j = move.square
        self.best_move[0] = i
        self.best_move[1] = j
        self.best_move[2] = move.value
        if self.lock:
            self.lock.release()


