#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
from collections import defaultdict
from typing import List, Tuple, Dict

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

    # row check
    for i in range(0, len(board.squares)):
        if i // N == move.square[0] // N:
            sum1 += board.squares[i]
        if i % N == move.square[1] % N:
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




class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N

        # Check whether a cell is empty, a value in that cell is not taboo, and that cell is allowed
        def possible(i, j, value):
            return game_state.board.get((i, j)) == SudokuBoard.empty \
                   and not TabooMove((i, j), value) in game_state.taboo_moves \
                       and (i, j) in game_state.player_squares()

        all_moves = [Move((i, j), value) for i in range(N) for j in range(N)
                     for value in range(1, N+1) if possible(i, j, value)]
        

        look1, look2 = createLookups(game_state.board)
        
        # initialization, we just take fisrst move
        best_move = (all_moves[0], 0)

        for move in all_moves:
            score_move = calculateScore(game_state, move, look1, look2)
            score_diff = scoreDifference(score_move, game_state.current_player)
            if score_diff > best_move[1]:
                best_move = (move, score_diff)
        
        toTryMove = best_move[0]
        self.propose_move(toTryMove)





