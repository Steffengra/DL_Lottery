
from numpy import (
    ndarray,
    log2,
    zeros,
    concatenate,
)

from DL_Lottery_imports.basic_resource_grid import ResourceGrid
from DL_Lottery_imports.ue import UserEquipment


class Simulation:
    res_grid: ResourceGrid
    users: dict

    def __init__(
            self,
            config,
            rng,
    ) -> None:
        def initialize_users() -> None:
            """
            Initialize users according to config
            """
            for user_id in range(self.config.num_users):
                self.users[user_id] = UserEquipment(
                    user_id=user_id,
                    pos_initial={'x': self.rng.integers(low=self.config.ue_position_range['low'],
                                                        high=self.config.ue_position_range['high']),
                                 'y': self.rng.integers(low=self.config.ue_position_range['low'],
                                                        high=self.config.ue_position_range['high'])},
                    max_job_size_rb=self.config.max_job_size_rb,
                    rayleigh_fading_scale=self.config.ue_rayleigh_fading_scale,
                    rng=rng,
                )

        self.config = config
        self.rng = rng

        self.res_grid = ResourceGrid(total_resource_blocks=config.available_rb_ofdm)

        self.users = {}
        initialize_users()
        self.generate_new_jobs()

    def generate_new_jobs(
            self,
    ) -> None:
        for ue in self.users.values():
            if self.rng.random() < self.config.job_creation_probability:
                ue.generate_job()

    def insert_priority_job(
            self
    ) -> None:
        attempts = 0
        while attempts < 8:
            random_ue_id = self.rng.choice(list(self.users.keys()))
            ue_jobs = self.users[random_ue_id].jobs
            if ue_jobs:
                random_job_id = self.rng.choice(range(len(ue_jobs)))
                random_job = self.users[random_ue_id].jobs[random_job_id]
                if random_job.priority == 0:
                    random_job.set_priority()
                    return
            attempts += 1
        # if self.config.verbosity == 1:
        #     print('Could not assign a priority job')

    def update_user_channel_fading(
            self,
    ) -> None:
        for ue in self.users.values():
            ue.update_channel_fading()

    def get_expected_load(
            self,
    ) -> float:
        """
        Calculate expected load according to config
        """
        expected_rb = 0
        for ue in self.users.values():
            mean_job_size = (1 + ue.max_job_size_rb) / 2
            expected_rb += self.config.job_creation_probability * mean_job_size
        expected_load = expected_rb / self.res_grid.total_resource_blocks

        return expected_load

    def gather_state(
            self
    ) -> ndarray:
        """
        Gather a vector-shape feature representation of the current simulation state
        """
        # rb requested per user, normalized to rb available
        rb_requested = zeros(self.config.num_users)
        for user in self.users.values():
            for job in user.jobs:
                rb_requested[user.user_id] += job.size_rb
        rb_requested = rb_requested / self.res_grid.total_resource_blocks

        # rb requested for priority jobs per user, normalized to rb available
        rb_requested_priority = zeros(self.config.num_users)
        for user in self.users.values():
            for job in user.jobs:
                if job.priority == 1:
                    rb_requested_priority[user.user_id] += job.size_rb
        rb_requested_priority = rb_requested_priority / self.res_grid.total_resource_blocks

        # channel quality per user
        channel_quality = zeros(self.config.num_users)
        for user in self.users.values():
            channel_quality[user.user_id] = user.channel_fading

        # maximum steps delayed per user normalized by longest possible delay
        maximum_steps_delayed = zeros(self.config.num_users)
        for user in self.users.values():
            greatest_delay = 0
            for job in user.jobs:
                if job.delay_steps > greatest_delay:
                    greatest_delay = job.delay_steps
            maximum_steps_delayed[user.user_id] = greatest_delay
        maximum_steps_delayed = maximum_steps_delayed / max(self.config.timeout_step.values())

        return concatenate([
            rb_requested,
            rb_requested_priority,
            channel_quality,
            maximum_steps_delayed,
        ], dtype='float32')

    def allocate(
            self,
            percentage_allocation_solution: ndarray,
    ) -> dict:
        def serve_jobs(
                jobs: list,
                available_rb: int,
        ) -> (int, list):
            deletion_list = []
            allocated_rb = 0
            for job in jobs:
                rb_allocation = min(available_rb, job.size_rb)

                allocated_rb += rb_allocation
                job.size_rb -= rb_allocation
                available_rb -= rb_allocation

                if job.size_rb == 0:
                    deletion_list.append(job)
                if available_rb == 0:
                    break

            return allocated_rb, deletion_list

        allocated_rb_per_ue: dict = {ue_id: 0 for ue_id in range(self.config.num_users)}
        available_rb_per_ue: dict = {
            ue.user_id: int(percentage_allocation_solution[ue.user_id] * self.res_grid.total_resource_blocks)
            for ue in self.users.values()}

        # first, serve priority jobs
        for ue in self.users.values():
            ue_id = ue.user_id
            if available_rb_per_ue[ue_id] == 0:
                continue

            priority_jobs = []
            for job in ue.jobs:
                if job.priority == 1:
                    priority_jobs.append(job)
            if not priority_jobs:
                continue

            (
                allocated_rb,
                deletion_list,
            ) = serve_jobs(jobs=priority_jobs, available_rb=available_rb_per_ue[ue_id])
            allocated_rb_per_ue[ue_id] += allocated_rb
            available_rb_per_ue[ue_id] -= allocated_rb
            for job in deletion_list:
                ue.jobs.remove(job)

        # If resources are left, serve remaining jobs
        for ue in self.users.values():
            ue_id = ue.user_id
            if available_rb_per_ue[ue_id] == 0:
                continue

            (
                allocated_rb,
                deletion_list,
            ) = serve_jobs(jobs=ue.jobs, available_rb=available_rb_per_ue[ue_id])
            allocated_rb_per_ue[ue_id] += allocated_rb
            available_rb_per_ue[ue_id] -= allocated_rb
            for job in deletion_list:
                ue.jobs.remove(job)

        # TODO: Unallocated rb
        # unallocated_rb = self.res_grid.total_resource_blocks - sum(allocated_rb_per_ue.values())
        # if unallocated_rb > 0:
        #     if self.config.verbosity == 1:
        #         print(f'{unallocated_rb} rb left unallocated')

        return allocated_rb_per_ue

    def step(
            self,
            percentage_allocation_solution: ndarray,
            critical_events_chance: float,  # 1 in x
    ) -> (float, dict):
        # allocate rb according to allocation action
        allocated_rb_per_ue = self.allocate(percentage_allocation_solution=percentage_allocation_solution)

        # calculate capacity
        sum_capacity_kbit_per_second = 0.0
        for ue_id, rb in allocated_rb_per_ue.items():
            sum_capacity_kbit_per_second += (
                    rb * self.config.bandwidth_per_rb_khz
                    * log2(1 + self.users[ue_id].channel_fading * self.config.ue_snr_linear)) / 1_000

        # increase all jobs delay
        for ue in self.users.values():
            for job in ue.jobs:
                job.delay_steps += 1

        # remove timed out jobs, calculate timeouts
        sum_normal_timeouts = 0
        sum_priority_timeouts = 0
        for ue in self.users.values():
            deletion_list = []
            for job in ue.jobs:
                if job.priority == 1:
                    if job.delay_steps >= self.config.timeout_step['priority']:
                        deletion_list.append(job)
                        sum_priority_timeouts += 1
                else:
                    if job.delay_steps >= self.config.timeout_step['normal']:
                        deletion_list.append(job)
                        sum_normal_timeouts += 1

            for job in deletion_list:
                ue.jobs.remove(job)

        # if self.config.verbosity == 1:
            # print(f'r1 {sum_capacity_kbit_per_second}')
            # print(f'r2 {sum_timeouts}')
            # print(f'r3 {sum_priority_timeouts}')

        # calculate reward sum
        reward = (
                + self.config.reward_sum_weightings['sum_capacity_kbit_per_second'] * sum_capacity_kbit_per_second
                + self.config.reward_sum_weightings['sum_normal_timeouts'] * sum_normal_timeouts
                + self.config.reward_sum_weightings['sum_priority_timeouts'] * sum_priority_timeouts
        )

        unweighted_reward_components = {
            'sum_capacity_kbit_per_second': sum_capacity_kbit_per_second,
            'sum_normal_timeouts': sum_normal_timeouts,
            'sum_priority_timeouts': sum_priority_timeouts,
        }

        # move simulation to new state
        self.update_user_channel_fading()
        self.generate_new_jobs()
        if self.rng.random() < critical_events_chance:
            self.insert_priority_job()  # turn 1 job into priority job at a chance

        return reward, unweighted_reward_components

    def reset(
            self
    ) -> None:
        self.__init__(config=self.config, rng=self.rng)
