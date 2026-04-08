from abc import ABC, abstractmethod
from snake import Action

N_FEATURES = 14  # output length of get_features() in data.py


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
    """Single linear layer: genome encodes turn weights, boost weights, and 2 biases.

    Genome layout (length = 2*n + 2):
        genome[0:n]      — turn weights
        genome[n:2n]     — boost weights
        genome[2n]       — turn bias
        genome[2n + 1]   — boost bias
    """

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
    """Two-layer feedforward network: input → hidden (ReLU) → output.

    Genome layout (length = n*h + h + h*2 + 2):
        W1: n*h weights  (input → hidden)
        b1: h biases     (hidden layer)
        W2: h*2 weights  (hidden → output: turn, boost)
        b2: 2 biases     (output layer)
    """

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
    """Genome-independent baseline: steers toward the nearest apple, avoids walls/bodies.

    The genome is unused — this exists so it conforms to the Neural_Network interface
    and can be plugged into GeneticPlayer as a fixed baseline for comparison.

    Feature indices used (from data.py get_features):
        0,1   — closest apple (forward, right) in local frame, normalized
        4,5,6 — wall distance (straight, left, right), normalized
        7,8,9 — nearest body distance (straight, left, right), normalized
    """

    DANGER = 0.05  # normalized distance considered unsafe (~15 tiles on a 300-grid)

    @staticmethod
    def genome_length(_n_features: int = N_FEATURES) -> int:
        return 2  # genome unused; genetic pipeline still needs a non-empty genome

    def __init__(self, genome: list[float]):
        super().__init__(genome)

    def get_action(self, features: list[float]) -> Action:
        fwd, right = features[0], features[1]
        wall_fwd,  wall_left,  wall_right  = features[4], features[5], features[6]
        body_fwd,  body_left,  body_right  = features[7], features[8], features[9]

        def safe_straight() -> bool:
            return wall_fwd > self.DANGER and body_fwd > self.DANGER

        def safe_right() -> bool:
            return wall_right > self.DANGER and body_right > self.DANGER

        def safe_left() -> bool:
            return wall_left > self.DANGER and body_left > self.DANGER

        if right > 0:
            if safe_right():   return Action.TURN_RIGHT
            if safe_straight(): return Action.STRAIGHT
            return Action.TURN_LEFT
        elif right < 0:
            if safe_left():    return Action.TURN_LEFT
            if safe_straight(): return Action.STRAIGHT
            return Action.TURN_RIGHT
        else:
            if fwd >= 0 and safe_straight(): return Action.STRAIGHT
            if safe_right(): return Action.TURN_RIGHT
            return Action.TURN_LEFT
