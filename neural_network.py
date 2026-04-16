from abc import ABC, abstractmethod
from snake import Action

# Must match the length returned by get_features() in data.py.
# New layout (25 values):
#    0..15 : 8 rays × (distance, is_wall_flag) in head-relative directions
#            ray order: fwd, fwd-right, right, back-right, back, back-left, left, fwd-left
#   16..17 : nearest apple unit-direction (forward, right) in local frame
#      18  : nearest apple Manhattan distance, normalized
#   19..20 : second-nearest apple unit-direction
#      21  : stamina / max_stamina
#      22  : length / win_length
#   23..24 : nearest enemy head unit-direction
N_FEATURES = 25

# Feature-layout constants. Update these together with the layout above if you
# change get_features.
_RAY_FWD_DIST    = 0
_RAY_FWD_ISWALL  = 1
_RAY_RIGHT_DIST  = 4
_RAY_RIGHT_ISWALL = 5
_RAY_LEFT_DIST   = 12
_RAY_LEFT_ISWALL = 13
_APPLE1_FWD  = 16
_APPLE1_RIGHT = 17
_APPLE1_DIST  = 18


class Neural_Network(ABC):
    def __init__(self, genome: list[float]):
        self.genome = genome

    @staticmethod
    @abstractmethod
    def genome_length(n_features: int = N_FEATURES) -> int:
        """Return the number of genome weights needed for n_features inputs."""
        ...

    @abstractmethod
    def get_action(self, features: list[float]) -> Action:
        ...


class Basic_Neural_Network(Neural_Network):

    @staticmethod
    def genome_length(n_features: int = N_FEATURES) -> int:
        return 2 * n_features + 2

    def __init__(self, genome: list[float]):
        super().__init__(genome)

    def get_action(self, features: list[float]) -> Action:
        n = len(features)
        turn_score  = sum(self.genome[i]     * features[i] for i in range(n))
        boost_score = sum(self.genome[n + i] * features[i] for i in range(n))
        turn_score  += self.genome[2 * n]
        boost_score += self.genome[2 * n + 1]

        if turn_score < -0.33:
            action = Action.TURN_LEFT
        elif turn_score > 0.33:
            action = Action.TURN_RIGHT
        else:
            action = Action.STRAIGHT

        if boost_score > 0:
            action = {
                Action.STRAIGHT:   Action.BOOST_STRAIGHT,
                Action.TURN_LEFT:  Action.BOOST_LEFT,
                Action.TURN_RIGHT: Action.BOOST_RIGHT,
            }[action]

        return action


class Two_Layer_Neural_Network(Neural_Network):

    HIDDEN_SIZE = 8

    @staticmethod
    def genome_length(n_features: int = N_FEATURES) -> int:
        h = Two_Layer_Neural_Network.HIDDEN_SIZE
        return n_features * h + h + h * 2 + 2

    def __init__(self, genome: list[float]):
        super().__init__(genome)

    def get_action(self, features: list[float]) -> Action:
        n = len(features)
        h = self.HIDDEN_SIZE

        idx = 0
        W1 = self.genome[idx : idx + n * h]; idx += n * h
        b1 = self.genome[idx : idx + h];     idx += h
        W2 = self.genome[idx : idx + h * 2]; idx += h * 2
        b2 = self.genome[idx : idx + 2]

        # Hidden layer with ReLU
        hidden = [
            max(0.0, sum(W1[i * n + j] * features[j] for j in range(n)) + b1[i])
            for i in range(h)
        ]

        # Output layer
        turn_score  = sum(W2[i]     * hidden[i] for i in range(h)) + b2[0]
        boost_score = sum(W2[h + i] * hidden[i] for i in range(h)) + b2[1]

        if turn_score < -0.33:
            action = Action.TURN_LEFT
        elif turn_score > 0.33:
            action = Action.TURN_RIGHT
        else:
            action = Action.STRAIGHT

        if boost_score > 0:
            action = {
                Action.STRAIGHT:   Action.BOOST_STRAIGHT,
                Action.TURN_LEFT:  Action.BOOST_LEFT,
                Action.TURN_RIGHT: Action.BOOST_RIGHT,
            }[action]

        return action


class Base_algorithm(Neural_Network):
    """Simple greedy-to-apple policy with obstacle avoidance.

    Uses only a few of the new features:
      - apple1 unit direction (indices 16, 17) to decide "which way to aim"
      - ray distances forward/left/right to check for obstacles
    A ray is "safe" if its distance is above DANGER (raw distance is normalized
    by grid_size, so DANGER is a fraction of the board).
    """

    DANGER = 0.05  # tunable; higher = more risk-averse

    @staticmethod
    def genome_length(_n_features: int = N_FEATURES) -> int:
        return 2  # genome unused; genetic pipeline still needs a non-empty genome

    def __init__(self, genome: list[float]):
        super().__init__(genome)

    def get_action(self, features: list[float]) -> Action:
        # Apple direction (unit vector, head-relative)
        fwd   = features[_APPLE1_FWD]
        right = features[_APPLE1_RIGHT]

        # Forward / left / right ray distances. The old code used separate
        # wall and body distances; the new rays combine them into a single
        # "distance to the nearest obstacle" value, which is what we want.
        dist_fwd   = features[_RAY_FWD_DIST]
        dist_left  = features[_RAY_LEFT_DIST]
        dist_right = features[_RAY_RIGHT_DIST]

        def safe_straight() -> bool:
            return dist_fwd > self.DANGER

        def safe_right() -> bool:
            return dist_right > self.DANGER

        def safe_left() -> bool:
            return dist_left > self.DANGER

        if right > 0:
            if safe_right():    return Action.TURN_RIGHT
            if safe_straight(): return Action.STRAIGHT
            return Action.TURN_LEFT
        elif right < 0:
            if safe_left():     return Action.TURN_LEFT
            if safe_straight(): return Action.STRAIGHT
            return Action.TURN_RIGHT
        else:
            if fwd >= 0 and safe_straight(): return Action.STRAIGHT
            if safe_right():                 return Action.TURN_RIGHT
            return Action.TURN_LEFT