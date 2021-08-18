
from numpy import (
    ndarray,
    array,
    sum as np_sum,
    power,
    append as np_append,
    delete,
    max as np_max,
    mean as np_mean,
    var as np_var
)
from numpy.random import (
    default_rng
)


class ExperienceBuffer:
    max_experiences: int
    buffer: list
    priorities: ndarray
    probabilities: ndarray

    alpha: float
    min_priority: float
    max_priority: float

    def __init__(
            self,
            num_max_experiences: int,
            alpha: float
    ) -> None:
        self.max_experiences = num_max_experiences  # Max buffer size

        self.buffer = []
        self.priorities = array([], dtype='float32')
        self.probabilities = array([], dtype='float32')  # prob_i = priority_i / sum_i(priority_i)

        self.alpha = alpha  # Priority adjustment factor, 0..1+
        self.min_priority = 1e-7  # Minimum priority, arbitrary small value
        self.max_priority = self.min_priority  # Maximum encountered priority so far, assigned to new exp.

        self.rng = default_rng()

    def get_len(
            self
    ) -> int:

        return len(self.buffer)

    def update_probabilities(
            self
    ) -> None:
        priority_sum = np_sum(self.priorities)
        self.probabilities = self.priorities / priority_sum

    def adjust_priorities(
            self,
            experience_ids: ndarray,
            new_priorities: ndarray
    ) -> None:
        for list_id in range(len(experience_ids)):
            self.priorities[experience_ids[list_id]] = max(self.min_priority,
                                                           power(abs(new_priorities[list_id]), self.alpha))

            if abs(new_priorities[list_id]) > self.max_priority:
                self.max_priority = abs(new_priorities[list_id])

    def add_experience(
            self,
            state,
            action,
            reward,
            next_state
    ) -> None:
        experience = {'state': state,
                      'action': action,
                      'reward': reward,
                      'next_state': next_state}
        self.buffer.append(experience)
        self.priorities = np_append(self.priorities, power(self.max_priority, self.alpha))

        if len(self.buffer) > self.max_experiences:
            self.buffer.pop(0)
            self.priorities = delete(self.priorities, 0)

    def sample(
            self,
            batch_size: int,
            beta: float
    ) -> tuple[list, ndarray, ndarray]:
        self.update_probabilities()

        # Weighted Sampling
        experience_ids = self.rng.choice(len(self.buffer), size=batch_size, replace=True, p=self.probabilities)

        sample_experiences = [self.buffer[ii] for ii in experience_ids]

        # IS weights correct for bias caused by sampling from a distribution different than the original one
        # Due to batch scaling we can remove N from the original IS weight formula
        sample_probabilities = self.probabilities[experience_ids]
        sample_importance_sampling_weights = power(sample_probabilities, -beta, dtype='float32')
        # Scale down weights per batch
        importance_sampling_weights = sample_importance_sampling_weights / np_max(sample_importance_sampling_weights)

        return sample_experiences, experience_ids, importance_sampling_weights
