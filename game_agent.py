"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    """
    This heuristic function is based on the number of legal moves available
    and each player's distance to the center of the board.
    
    10* (own_moves - opp_moves) + (own_distance_x + own_distance_y) -
    (opp_distance_x + opp_distance_y)
    
    This calculates the sum of the absolute number of squares from the player's
    position to the center of the board along x and y axes.
    This puts a positive coefficient to the player's distance to the center,
    effectively pushing it towards the edges of the board.
    """
    
    if game.is_loser(player):
        return float("-inf")
    
    if game.is_winner(player):
        return float("inf")
    
    center = game.width/2
    
    own_position = game.get_player_location(player)
    opp_position = game.get_player_location(game.get_opponent(player))
    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    own_distance_x = abs(center - own_position[0])
    own_distance_y = abs(center - own_position[1])
    
    opp_distance_x = abs(center - opp_position[0])
    opp_distance_y = abs(center - opp_position[1])
    
    return float(10* (opp_moves - own_moves) + (own_distance_x + own_distance_y) -
                 (opp_distance_x + opp_distance_y))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    """
    This function returns the difference between the number of legal moves the player has available
    and two times the number of legal moves available to the opponent.
    If the returned value is positive, then the player has more than three times as many available legal moves
    than the opponent.
    If the returned value is negative, then the player has less than one-third as many available legal moves
    than the opponent.
    """
    
    if game.is_loser(player):
        return float("-inf")
    
    if game.is_winner(player):
        return float("inf")
    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    return float(opp_moves - 3 * own_moves)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    """
    Changes the factors applied to the player and opponent based upon the state of the game.
    In the beginning of the game, where there are 25 or more blank spaces, more aggresiveness:
        own_moves - 3 * opp_moves
    Near the end of the game, be more defensive:
        3 * own_moves - opp_moves
    
    """
    
    if game.is_loser(player):
        return float("-inf")
    
    if game.is_winner(player):
        return float("inf")
    
    blank_spaces = len(game.get_blank_spaces())
    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    if blank_spaces > 25:
        return float(own_moves - 3 * opp_moves)
    else:
        return float(3 * own_moves - opp_moves)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        
        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return (-1, -1)
        

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            
            best_move = self.minimax(game, self.search_depth)
                            
        except SearchTimeout:
            #pass  # Handle any actions required after timeout as needed
            
        # Return the best move from the last completed depth search iteration 
            if (best_move == (-1, -1)):
                best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]
        return best_move


    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        
        best_move = None
        best_score = float("-inf")
        
        for move in legal_moves:
            score = self.minValue(game.forecast_move(move), depth - 1)
            if score > best_score:
                best_score, best_move = score, move
        
        return best_move
    
        
    def maxValue(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        if depth == 0:
            return self.score(game, self)
        
        best_score = float("-inf")
        
        legal_moves = game.get_legal_moves()
        
        for move in legal_moves:
            score = self.minValue(game.forecast_move(move), depth - 1)
            if score > best_score:
                best_score = score
                
        return best_score
    
    def minValue(self, game, depth):
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
                
        if depth == 0:
            return self.score(game, self)
        
        best_score = float("inf")
        
        legal_moves = game.get_legal_moves()
        
        for move in legal_moves:
            score = self.maxValue(game.forecast_move(move), depth - 1)
            if score <  best_score:
                best_score = score
        
        return best_score


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        
        best_move = (-1, -1)
    
        
        legal_moves = game.get_legal_moves()
        
        if len(legal_moves) == 0:
            return (-1, -1)
        
        
        try:
            if self.time_left() > self.TIMER_THRESHOLD: 
                depth = 1
                while True:
                    best_move = self.alphabeta(game, depth)
                    depth += 1
                
        except SearchTimeout:
            # Handle any actions required at timeout, if necessary
            #pass
        
            if (best_move == (-1, -1)):
                best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]
        return best_move
        
        # Alpha is the maximum lower bound of possible solutions
        # Beta is the minimum upper bound of possible solutions

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if depth == 0:
            return self.score(game, self)
            

        legal_moves = game.get_legal_moves()
        
        if len(legal_moves) == 0:
            return (-1, -1)
        
        best_move = (-1, -1)
        best_val = float("-inf")
                
        for move in legal_moves:
            val = self.minVal(game.forecast_move(move), depth - 1, alpha, beta)

            if val >= best_val:
                best_val = val
                best_move = move
            
            if best_val >= beta:
                return best_move
                
            alpha = max(alpha, best_val)
        return best_move
            
        
    
    def minVal(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
         
        legal_moves = game.get_legal_moves()
        
        if len(legal_moves) == 0 or depth == 0:
            return self.score(game, self)
        
        best_val = float("inf")
        
        for move in legal_moves:
            best_val = min(best_val, self.maxVal(game.forecast_move(move), depth - 1, alpha, beta))
            
            if best_val <= alpha:
                return best_val
            beta = min(beta, best_val)
        return best_val
         
    
    
    
    def maxVal(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        legal_moves = game.get_legal_moves()
        
        if len(legal_moves) == 0 or depth == 0:
            return self.score(game, self)
            
        best_val = float("-inf")
        
        for move in legal_moves:
            best_val = max(best_val, self.minVal(game.forecast_move(move), depth - 1, alpha, beta))
            if best_val >= beta:
                return best_val
            alpha = max(alpha, best_val)
        return best_val
            
        
