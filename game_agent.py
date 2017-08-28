"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def throw_timeout_if_player_outoftime(player):
    # throw a timeout exception if computer player has run out of time
    if player.time_left() < player.TIMER_THRESHOLD:
        raise SearchTimeout()
    return


def custom_score(game, player):
    """Weighted Improved Average heuristic

    This evaluation function effectively combines an aggressive version of the Improved score and the open score and center score
    to essentially "listen" to improved score when it gives a high enough value, but listen for a 2nd opinion when it gives a
    lower value than either open or center score.

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

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # calculate number of available moves for both players
    own_moves_left = game.get_legal_moves(player)
    own_moves = len(own_moves_left)
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # get the open score value
    openscore = float(len(own_moves_left))

    # get the weighted improved score value
    weightedimprovedscore = float(own_moves - (2 * opp_moves))
    # set the bias towards weighted improved score to listen to it more often
    bias = 2

    # calculate center score
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    centerscore = float((h - y) ** 2 + (w - x) ** 2)

    # get the weighted average of the 3 sample heuristics biased towards the weightedimprovedscore
    return ((bias*weightedimprovedscore) + centerscore + openscore)/3


def custom_score_2(game, player):
    """Opposite Improved Score

    This heuristic causes the agent to avoid improved score moves in order to run away from opponents. This was conceptualized
    as a response to knowledge that AB_Improved would try to limit the player's moves

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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # calculate number of available moves for both players
    own_moves_left = game.get_legal_moves(player)
    own_moves = len(own_moves_left)
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # calculate opposite value of running after the opponent
    return float(own_moves - opp_moves) * -1


def custom_score_3(game, player):
    """ImprovedLessOpponentsScore

    was designed to choose moves where the opponents possible moves were less while still
    attempting to block the opponents moves and keep your own moves open at the same time

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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # calculate number of available moves for both players
    own_moves_left = game.get_legal_moves(player)
    own_moves = len(own_moves_left)
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # get the improved score
    improvedscore = float(own_moves - opp_moves)
    # negation of the opponents available moves
    lessenopponentmoves = float(len(game.get_legal_moves(game.get_opponent(player)))) * -1

    return (lessenopponentmoves + improvedscore) / 2


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

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def min_val(self, board, depth):
        """
        Performs the min portion of the minimax algorithm


        :param board: isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        :param depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        :return:  float
            Returns the minimum score portion of the minima algorithm from among the child nodes of the current board state
        """

        throw_timeout_if_player_outoftime(self)

        # get allowable moves for player
        legal_moves = board.get_legal_moves()

        # return the score if we've reached a leaf node
        if depth <= 0 or not legal_moves:
            return self.score(board, self)

        # init minval
        minval = float('inf')

        # for each legal move available, keep track of the smallest max value of the child nodes
        for legal_move in legal_moves:
            nextboard = board.forecast_move(legal_move)
            nextmove_val = self.max_val(nextboard, depth - 1)

            if nextmove_val <= minval:
                minval = nextmove_val

        return minval
    
    def max_val(self, board, depth):

        """
           Performs the max portion of the minimax algorithm


           :param board: isolation.Board
               An instance of the Isolation game `Board` class representing the
               current game state

           :param depth : int
               Depth is an integer representing the maximum number of plies to
               search in the game tree before aborting

           :return:  float
               Returns the maximum score portion of the minimax algorithm from among the child nodes of the current board state
           """

        throw_timeout_if_player_outoftime(self)

        # get allowable moves for player
        legal_moves = board.get_legal_moves()

        # return the score if we've reached a leaf node
        if depth <= 0 or not legal_moves:
            return self.score(board, self)

        # init maxval
        maxval = float('-inf')

        # for each legal move available, keep track of the largest min value of the child nodes
        for legal_move in legal_moves:
            nextboard = board.forecast_move(legal_move)
            nextmove_val = self.min_val(nextboard, depth - 1)

            if nextmove_val >= maxval:
                maxval = nextmove_val

        return maxval

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
        throw_timeout_if_player_outoftime(self)

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return -1, -1

        # init vars to keep track of largest value
        maxmove = legal_moves[0]
        maxval = float('-inf')

        # for each legal move available, keep track of the largest min value of the child nodes
        # then return the corresponding best move once we reach the end of the list
        for nextmove in legal_moves:
            nextboard = game.forecast_move(nextmove)
            nextmove_val = self.min_val(nextboard, depth - 1)

            if nextmove_val >= maxval:
                maxmove = nextmove
                maxval = nextmove_val

        return maxmove

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

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        currentdepth = 1

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.

            # until we run out of time, find the best move by running alphabeta one depth lower than last time
            while True:
                best_move = self.alphabeta(game, currentdepth)
                currentdepth += 1

        except SearchTimeout:
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def min_val(self, board, depth, alpha, beta):
        """
       Performs the min portion of the minimax algorithm while keeping track of the lower and upper bounds that dictate
       Alpha Beta Pruning


       :param board: isolation.Board
           An instance of the Isolation game `Board` class representing the
           current game state

       :param depth : int
           Depth is an integer representing the maximum number of plies to
           search in the game tree before aborting

        :param alpha : float
            alpha is a float representing the maximum lower bound of possible scores

       :return:  float
           Returns the minimum score portion of the minimax algorithm from among the child nodes of the current board state
       """

        throw_timeout_if_player_outoftime(self)

        # get available moves
        legal_moves = board.get_legal_moves()

        # if at leaf node, then return the score
        if depth <= 0 or not legal_moves:
            return self.score(board, self)

        # init minvval
        minval = float('inf')

        # for each available move at current state, keep track of the smallest max val of the child nodes
        for legal_move in legal_moves:
            nextboard = board.forecast_move(legal_move)
            nextmove_val = self.max_val(nextboard, depth - 1, alpha, beta)

            minval = min(minval, nextmove_val)

            if minval <= alpha:
                return minval

            # then update beta in order to pass it to the next child
            beta = min(beta, minval)

        return minval

    def max_val(self, board, depth, alpha, beta):
        """
           Performs the max portion of the minimax algorithm while keeping track of the lower and upper bounds that dictate
           Alpha Beta Pruning


           :param board: isolation.Board
               An instance of the Isolation game `Board` class representing the
               current game state

           :param depth : int
               Depth is an integer representing the maximum number of plies to
               search in the game tree before aborting

            :param beta : float
                beta is a float representing the minimum upper bound of possible scores

           :return:  float
               Returns the minimum score portion of the minimax algorithm from among the child nodes of the current board state
        """

        throw_timeout_if_player_outoftime(self)

        # get available moves at current state
        legal_moves = board.get_legal_moves()

        # if at a leaf node, return the score
        if depth <= 0 or not legal_moves:
            return self.score(board, self)

        # init maxval
        maxval = float('-inf')

        # for each available move at current state, keep track of the largest min val of the child nodes
        for legal_move in legal_moves:
            nextboard = board.forecast_move(legal_move)
            nextmove_val = self.min_val(nextboard, depth - 1, alpha, beta)

            maxval = max(maxval, nextmove_val)

            if maxval >= beta:
                return maxval

            # update alpha in order to pass to next child
            alpha = max(alpha, maxval)

        return maxval

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
        throw_timeout_if_player_outoftime(self)

        # get available moves at current state
        legal_moves = game.get_legal_moves()
        # no moves available
        if not legal_moves:
            return -1, -1

        # init vars to keep track of best move
        maxmove = legal_moves[0]
        maxval = float("-inf")

        # for each available move, keep track of alpha and the corresponding best move so far
        for nextmove in legal_moves:
            nextboard = game.forecast_move(nextmove)
            nextmove_val = self.min_val(nextboard, depth - 1, alpha, beta)

            if nextmove_val >= maxval:
                maxmove = nextmove
                maxval = nextmove_val
                alpha = nextmove_val

        return maxmove
