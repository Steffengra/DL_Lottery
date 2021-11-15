
from numpy import (
    ndarray,
    infty,
    ones,
    isnan,
    mean,
    concatenate,
    newaxis,
)
from numpy.random import (
    default_rng,
)
from tensorflow.keras.models import (
    load_model,
)
from os.path import (
    dirname,
    join,
)
from shutil import (
    copy2,
)
from datetime import (
    datetime,
)
from gzip import (
    open as gzip_open,
)
from pickle import (
    dump as pickle_dump,
    load as pickle_load,
)

from config import Config
from DL_Lottery_imports.td3 import TD3ActorCritic
from DL_Lottery_imports.simulation import Simulation
from DL_Lottery_imports.prune_network import prune_network
from DL_Lottery_imports.plotting_functions import (
    plot_scatter_plot,
)

# from tensorflow.keras import mixed_precision  # TODO: This may be useful on newer gpus
# mixed_precision.set_global_policy('mixed_float16')  # https://www.tensorflow.org/guide/mixed_precision


class Runner:
    config: Config
    time: float
    rng: default_rng
    profiler: object

    def __init__(
            self,
    ) -> None:
        self.config = Config()
        self.rng = self.config.rng

        if self.config.toggle_profiling:
            import cProfile
            self.profiler: cProfile.Profile = cProfile.Profile()

    def policy_training_criterion(
            self,
            simulation_step: int,
    ) -> bool:
        """Train policy networks only every k steps and/or only after j total steps to ensure a good value function"""
        if (
            simulation_step > self.config.train_policy_after_j_steps
            and
            (simulation_step % self.config.train_policy_every_k) == 0
        ):
            return True
        return False

    def add_random_distribution(
            self,
            action: ndarray,  # turns out its much faster to numpy the tensor and then do operations on ndarray
            tau_momentum: float,  # tau * random_distribution + (1 - tau) * action
    ) -> ndarray:
        """
        Mix an action vector with a random_uniform vector of same length
        by tau * random_distribution + (1 - tau) * action
        """
        if tau_momentum == 0.0:
            return action

        # create random action
        random_distribution = self.rng.random(size=self.config.num_actions_policy, dtype='float32')
        random_distribution = random_distribution / sum(random_distribution)

        # combine
        noisy_action = tau_momentum * random_distribution + (1 - tau_momentum) * action

        # normalize
        sum_noisy_action = sum(noisy_action)
        if sum_noisy_action != 0:
            noisy_action = noisy_action / sum_noisy_action

        return noisy_action

    def train(
            self,
            training_name: str,
            probability_critical_events: float,
            value_network_path: None or str = None,  # for loading from pretrained
            policy_network_path: None or str = None,  # for loading from pretrained
            anchoring_parameters_path: None or str = None,
    ) -> None:
        def progress_print() -> None:
            progress = (episode_id * self.config.num_steps_per_episode + step_id + 1) / self.config.steps_total
            timedelta = datetime.now() - real_time_start
            finish_time = real_time_start + timedelta / progress

            print('\rSimulation completed: {:.2%}, est. finish {:02d}:{:02d}:{:02d}'.format(
                progress, finish_time.hour, finish_time.minute, finish_time.second), end='')

        def save_networks() -> None:
            state = sim.gather_state()
            action = allocator.get_action(state)

            allocator.networks['policy']['target'](state[newaxis])
            allocator.networks['policy']['target'].save(
                join(self.config.model_path, 'actor_allocation_' + training_name))

            allocator.networks['value1']['target'](concatenate([state, action])[newaxis])
            allocator.networks['value1']['target'].save(
                join(self.config.model_path, 'critic_allocation_' + training_name))

            copy2(join(dirname(__file__), 'config.py'), self.config.model_path)

        def anneal_parameters() -> tuple:
            if simulation_step > self.config.exploration_noise_step_start_decay:
                exploration_noise_momentum_new = max(
                    0.0,
                    exploration_noise_momentum - self.config.exploration_noise_linear_decay_per_step
                )
            else:
                exploration_noise_momentum_new = exploration_noise_momentum

            return (
                exploration_noise_momentum_new,
            )

        # setup---------------------------------------------------------------------------------------------------------
        training_name = training_name
        if self.config.verbosity == 1:
            print('\n' + training_name)
        real_time_start = datetime.now()

        if self.config.toggle_profiling:
            self.profiler.enable()

        sim = Simulation(
            config=self.config,
            rng=self.rng,
        )
        if self.config.verbosity == 1:
            print('Expected load: {:.2f}'.format(sim.get_expected_load()))

        allocator = TD3ActorCritic(
            name='allocator',
            rng=self.rng,
            **self.config.td3_args
        )

        # load pretrained networks
        if value_network_path:
            allocator.load_pretrained_networks(value_path=value_network_path,
                                               policy_path=policy_network_path)
        # load anchoring parameters
        policy_parameters_anchor = None
        policy_parameters_fisher = None
        if anchoring_parameters_path:
            with gzip_open(anchoring_parameters_path, 'rb') as file:
                policy_parameters = pickle_load(file)
            policy_parameters_anchor = policy_parameters['final']
            policy_parameters_fisher = policy_parameters['fisher']

        exploration_noise_momentum = self.config.exploration_noise_momentum_initial

        # main loop-----------------------------------------------------------------------------------------------------
        per_episode_metrics: dict = {
            'reward_per_step': -infty * ones(self.config.num_episodes),
            'value_loss_mean': +infty * ones(self.config.num_episodes),
            'priority_timeouts_per_occurrence': +infty * ones(self.config.num_episodes),
        }
        policy_parameters: dict = {
            'initial': [],
            'final': [],
            'fisher': [],
        }

        for episode_id in range(self.config.num_episodes):
            episode_metrics: dict = {
                'rewards': -infty * ones(self.config.num_steps_per_episode),
                'value_losses': +infty * ones(self.config.num_steps_per_episode),
                'priority_timeouts': +infty * ones(self.config.num_steps_per_episode),
            }

            step_experience: dict = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}
            state_next: ndarray = sim.gather_state()

            # initialize network & log initial params
            if episode_id == 0:
                if not policy_network_path:  # not pretrained
                    allocator.networks['policy']['primary'].initialize_inputs(state_next[newaxis])
                policy_parameters['initial'] = allocator.get_policy_params('primary')

            for step_id in range(self.config.num_steps_per_episode):
                simulation_step = episode_id * self.config.num_steps_per_episode + step_id

                # determine state
                state_current = state_next
                step_experience['state'] = state_current

                # find allocation action based on state
                bandwidth_allocation_solution = allocator.get_action(state_current).numpy()
                noisy_bandwidth_allocation_solution = self.add_random_distribution(
                    action=bandwidth_allocation_solution,
                    tau_momentum=exploration_noise_momentum)
                step_experience['action'] = noisy_bandwidth_allocation_solution

                # step simulation based on action
                (
                    step_reward,
                    unweighted_step_reward_components,
                ) = sim.step(
                    percentage_allocation_solution=noisy_bandwidth_allocation_solution,
                    critical_events_chance=probability_critical_events,
                )
                step_experience['reward'] = step_reward

                # determine new state
                state_next = sim.gather_state()
                step_experience['next_state'] = state_next

                # save tuple (S, A, r, S_{new})
                allocator.add_experience(**step_experience)

                # train allocator off-policy
                train_policy = False
                if self.policy_training_criterion(simulation_step=simulation_step):
                    train_policy = True
                (
                    step_allocation_value1_loss,
                    weight_anchoring_lambda_updated,
                ) = allocator.train(
                    train_policy=train_policy,
                    policy_parameters_anchor=policy_parameters_anchor,
                    policy_parameters_fisher=policy_parameters_fisher,
                    **self.config.training_args,
                )
                if self.config.adaptive_anchoring_weight_lambda and weight_anchoring_lambda_updated is not None:
                    self.config.training_args['weight_anchoring_lambda'] = weight_anchoring_lambda_updated

                # anneal parameters
                (
                    exploration_noise_momentum,
                ) = anneal_parameters()

                # log step results
                episode_metrics['rewards'][step_id] = step_experience['reward']
                episode_metrics['value_losses'][step_id] = step_allocation_value1_loss
                episode_metrics['priority_timeouts'][step_id] = unweighted_step_reward_components['sum_priority_timeouts']

                # progress print
                if self.config.verbosity == 1:
                    if step_id % 50 == 0:
                        progress_print()

            # log episode results
            # Remove NaN entries, e.g. steps in which there was no training -> None loss
            episode_metrics['value_losses'] = episode_metrics['value_losses'][~isnan(episode_metrics['value_losses'])]

            per_episode_metrics['reward_per_step'][episode_id] = (
                    sum(episode_metrics['rewards']) / self.config.num_steps_per_episode
            )
            per_episode_metrics['value_loss_mean'][episode_id] = (
                mean(episode_metrics['value_losses'])
            )
            per_episode_metrics['priority_timeouts_per_occurrence'][episode_id] = (
                sum(episode_metrics['priority_timeouts']) / self.config.num_steps_per_episode / (
                    probability_critical_events + self.config.tiny_numerical_value)
            )

            # print episode results
            if self.config.verbosity == 1:
                print('\n', end='')
                print('episode per step reward: {:.2f}'.format(
                    sum(episode_metrics['rewards']) / self.config.num_steps_per_episode)
                )
                print('episode mean value loss: {:.2f}'.format(
                    mean(episode_metrics['value_losses']))
                )
                print('episode per occurrence priority timeouts: {:.2f}'.format(
                    sum(episode_metrics['priority_timeouts']) / self.config.num_steps_per_episode / (
                        probability_critical_events + self.config.tiny_numerical_value)
                ))

            # reset simulation for next episode
            sim.reset()

        # teardown------------------------------------------------------------------------------------------------------
        policy_parameters['final'] = allocator.get_policy_params('target')
        policy_parameters['fisher'] = allocator.get_policy_fisher_diagonal_estimate('primary')

        # save trained networks locally
        save_networks()

        # save logged results
        with gzip_open(join(self.config.log_path, training_name + '_per_episode_metrics.gzip'), 'wb') as file:
            pickle_dump(per_episode_metrics, file=file)
        with gzip_open(join(self.config.model_path, 'policy_parameters_' + training_name + '.gzip'), 'wb') as file:
            pickle_dump(policy_parameters, file=file)

        # end compute performance profiling
        if self.config.toggle_profiling:
            self.profiler.disable()
            if self.config.verbosity == 1:
                self.profiler.print_stats(sort='cumulative')
            self.profiler.dump_stats(join(self.config.performance_profile_path, training_name + '.profile'))

        # plots
        plot_scatter_plot(per_episode_metrics['reward_per_step'],
                          title='Per Step Reward')
        plot_scatter_plot(per_episode_metrics['value_loss_mean'],
                          title='Mean Value Loss')
        plot_scatter_plot(per_episode_metrics['priority_timeouts_per_occurrence'],
                          title='Per Occurrence Priority Timeouts')

    def train_critical_events(
            self,
            name: str = '',
            value_network_path: None or str = None,
            policy_network_path: None or str = None,
            anchoring_parameters_path: None or str = None,
    ) -> None:
        """Wrapper for 100% critical event chance"""
        self.train(
            training_name='training_critical_events_' + name,
            probability_critical_events=1,
            value_network_path=value_network_path,
            policy_network_path=policy_network_path,
            anchoring_parameters_path=anchoring_parameters_path,
        )

    def train_no_critical_events(
            self,
            name: str = '',
            value_network_path: None or str = None,
            policy_network_path: None or str = None,
            anchoring_parameters_path: None or str = None,
    ) -> None:
        """Wrapper for 0% critical event chance"""
        self.train(
            training_name='training_no_critical_events_' + name,
            probability_critical_events=0,
            value_network_path=value_network_path,
            policy_network_path=policy_network_path,
            anchoring_parameters_path=anchoring_parameters_path,
        )

    def train_normal(
            self,
            name: str = '',
            value_network_path: None or str = None,
            policy_network_path: None or str = None,
            anchoring_parameters_path: None or str = None,
    ) -> None:
        """Wrapper for normal amount of critical events"""
        self.train(
            training_name='training_normal_' + name,
            probability_critical_events=self.config.normal_priority_job_probability,
            value_network_path=value_network_path,
            policy_network_path=policy_network_path,
            anchoring_parameters_path=anchoring_parameters_path,
        )

    def train_fifty_percent_critical(
            self,
            name: str = '',
            value_network_path: None or str = None,
            policy_network_path: None or str = None,
            anchoring_parameters_path: None or str = None,
    ):
        """Wrapper for 50% critical rate"""
        self.train(
            training_name='training_fifty_percent_critical_events_' + name,
            probability_critical_events=0.5,
            value_network_path=value_network_path,
            policy_network_path=policy_network_path,
            anchoring_parameters_path=anchoring_parameters_path,
        )

    def train_on_random_data(
            self,
            value_network_path: None or str = None,
            policy_network_path: None or str = None,
            anchoring_parameters_path: None or str = None,
            name: str = '',
    ) -> None:
        def progress_print() -> None:
            progress = (episode_id * self.config.num_steps_per_episode + step_id + 1) / self.config.steps_total
            timedelta = datetime.now() - real_time_start
            finish_time = real_time_start + timedelta / progress

            print('\rSimulation completed: {:.2%}, est. finish {:02d}:{:02d}:{:02d}'.format(
                progress, finish_time.hour, finish_time.minute, finish_time.second), end='')

        def save_networks() -> None:
            state = sim.gather_state()
            action = allocator.get_action(state)

            allocator.networks['policy']['target'](state[newaxis])
            allocator.networks['policy']['target'].save(
                join(self.config.model_path, 'actor_allocation_' + training_name))

            allocator.networks['value1']['target'](concatenate([state, action])[newaxis])
            allocator.networks['value1']['target'].save(
                join(self.config.model_path, 'critic_allocation_' + training_name))

            copy2(join(dirname(__file__), 'config.py'), self.config.model_path)

        def anneal_parameters() -> tuple:
            if simulation_step > self.config.exploration_noise_step_start_decay:
                exploration_noise_momentum_new = max(
                    0.0,
                    exploration_noise_momentum - self.config.exploration_noise_linear_decay_per_step
                )
            else:
                exploration_noise_momentum_new = exploration_noise_momentum

            return (
                exploration_noise_momentum_new,
            )

        # setup---------------------------------------------------------------------------------------------------------
        training_name = 'training_random_data' + name
        real_time_start = datetime.now()

        if self.config.toggle_profiling:
            self.profiler.enable()

        sim = Simulation(
            config=self.config,
            rng=self.rng,
        )
        if self.config.verbosity == 1:
            print('Expected load: {:.2f}'.format(sim.get_expected_load()))

        allocator = TD3ActorCritic(
            name='allocator',
            rng=self.rng,
            **self.config.td3_args
        )

        # load pretrained networks
        if value_network_path:
            allocator.load_pretrained_networks(value_path=value_network_path,
                                               policy_path=policy_network_path)
        # load anchoring parameters
        policy_parameters_anchor = None
        policy_parameters_fisher = None
        if anchoring_parameters_path:
            with gzip_open(anchoring_parameters_path, 'rb') as file:
                policy_parameters = pickle_load(file)
            policy_parameters_anchor = policy_parameters['final']
            policy_parameters_fisher = policy_parameters['fisher']

        exploration_noise_momentum = self.config.exploration_noise_momentum_initial

        policy_parameters: dict = {
            'initial': [],
            'final': [],
            'fisher': [],
        }

        # main loop-----------------------------------------------------------------------------------------------------
        per_episode_metrics: dict = {
            'reward_per_step': -infty * ones(self.config.num_episodes),
            'value_loss_mean': +infty * ones(self.config.num_episodes),
        }
        state_shape = sim.gather_state().shape

        for episode_id in range(self.config.num_episodes):
            episode_metrics: dict = {
                'rewards': -infty * ones(self.config.num_steps_per_episode),
                'value_losses': +infty * ones(self.config.num_steps_per_episode),
            }

            step_experience: dict = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}
            state_next = self.rng.random(state_shape)

            # initialize network & log initial params
            if episode_id == 0 and not value_network_path:  # no pre-trained loaded
                allocator.networks['policy']['primary'].initialize_inputs(state_next[newaxis])
                policy_parameters['initial'] = allocator.get_policy_params('primary')

            for step_id in range(self.config.num_steps_per_episode):
                simulation_step = episode_id * self.config.num_steps_per_episode + step_id

                # determine state
                state_current = state_next
                step_experience['state'] = state_current

                # find allocation action based on state
                bandwidth_allocation_solution = allocator.get_action(state_current).numpy()
                noisy_bandwidth_allocation_solution = self.add_random_distribution(
                        action=bandwidth_allocation_solution,
                        tau_momentum=exploration_noise_momentum)
                step_experience['action'] = noisy_bandwidth_allocation_solution

                # step simulation based on action
                (
                    step_reward,
                    unweighted_step_reward_components,
                ) = sim.step(
                    percentage_allocation_solution=noisy_bandwidth_allocation_solution,
                    critical_events_chance=0,
                )
                step_experience['reward'] = self.rng.random()

                # determine new state
                state_next = self.rng.random(state_shape)
                step_experience['next_state'] = state_next

                # save tuple (S, A, r, S_{new})
                allocator.add_experience(**step_experience)

                # train allocator off-policy
                train_policy = False
                if self.policy_training_criterion(simulation_step=simulation_step):
                    train_policy = True
                step_allocation_value1_loss = allocator.train(
                    train_policy=train_policy,
                    policy_parameters_anchor=policy_parameters_anchor,
                    policy_parameters_fisher=policy_parameters_fisher,
                    **self.config.training_args,
                )

                # anneal parameters
                (
                    exploration_noise_momentum,
                ) = anneal_parameters()

                # log step results
                episode_metrics['rewards'][step_id] = step_reward
                episode_metrics['value_losses'][step_id] = step_allocation_value1_loss

                # progress print
                if self.config.verbosity == 1:
                    if step_id % 50 == 0:
                        progress_print()

            # log episode results
            # Remove NaN entries, e.g. steps in which there was no training -> None loss
            episode_metrics['value_losses'] = episode_metrics['value_losses'][~isnan(episode_metrics['value_losses'])]

            per_episode_metrics['reward_per_step'][episode_id] = (
                    sum(episode_metrics['rewards']) / self.config.num_steps_per_episode)
            per_episode_metrics['value_loss_mean'][episode_id] = (
                    mean(episode_metrics['value_losses']))

            # print episode results
            if self.config.verbosity == 1:
                print('\n', end='')
                print('episode per step reward: {:.2f}'.format(
                    sum(episode_metrics['rewards']) / self.config.num_steps_per_episode))
                print('episode mean value loss: {:.2f}'.format(
                    mean(episode_metrics['value_losses'])))

            # reset simulation for next episode
            sim.reset()

        # teardown------------------------------------------------------------------------------------------------------
        policy_parameters['final'] = allocator.get_policy_params('target')
        policy_parameters['fisher'] = allocator.get_policy_fisher_diagonal_estimate('primary')

        # save trained networks locally
        save_networks()

        # save logged results
        with gzip_open(join(self.config.log_path, training_name + '_per_episode_metrics.gzip'), 'wb') as file:
            pickle_dump(per_episode_metrics, file=file)

        with gzip_open(join(self.config.model_path, 'policy_parameters_' + training_name + '.gzip'), 'wb') as file:
            pickle_dump(policy_parameters, file=file)

        # end compute performance profiling
        if self.config.toggle_profiling:
            self.profiler.disable()
            if self.config.verbosity == 1:
                self.profiler.print_stats(sort='cumulative')
            self.profiler.dump_stats(join(self.config.performance_profile_path, training_name + '.profile'))

        # plots
        plot_scatter_plot(per_episode_metrics['reward_per_step'],
                          title='Per Step Reward')
        plot_scatter_plot(per_episode_metrics['value_loss_mean'],
                          title='Mean Value Loss')

    def test(
            self,
            allocator: str,  # 'pretrained', 'random'
            probability_critical_events: float,
            policy_network_path: None or str = None,
            policy_pruning_parameters_path: None or str = None,
            name: str = '',
    ) -> None:
        def progress_print() -> None:
            progress = (episode_id * self.config.num_steps_per_episode + step_id + 1) / self.config.steps_total
            timedelta = datetime.now() - real_time_start
            finish_time = real_time_start + timedelta / progress

            print('\rSimulation completed: {:.2%}, est. finish {:02d}:{:02d}:{:02d}'.format(
                progress, finish_time.hour, finish_time.minute, finish_time.second), end='')

        # setup---------------------------------------------------------------------------------------------------------
        testing_name = 'testing_' + name
        if self.config.verbosity == 1:
            print('\n' + testing_name)
        real_time_start = datetime.now()
        if self.config.toggle_profiling:
            self.profiler.enable()

        sim = Simulation(
            config=self.config,
            rng=self.rng,
        )
        if self.config.verbosity == 1:
            print('Expected load: {:.2f}'.format(sim.get_expected_load()))

        # define allocation function
        if allocator == 'pretrained':
            allocator_network = load_model(policy_network_path)

            def allocate_state_current() -> ndarray:
                return allocator_network.call(state_current[newaxis]).numpy().squeeze()

            if self.config.prune_network:
                with gzip_open(join(self.config.model_path, policy_pruning_parameters_path), 'rb') as file:
                    training_parameters_initial = pickle_load(file=file)['initial']
                parameters_new = prune_network(
                    network=allocator_network,
                    training_parameters_initial=training_parameters_initial,
                    **self.config.pruning_args
                )
                allocator_network.set_weights(parameters_new)
        elif allocator == 'random':
            def allocate_state_current() -> ndarray:
                solution = self.rng.random(self.config.num_actions_policy)
                return solution / sum(solution)
        else:
            def allocate_state_current() -> None: pass
            if self.config.verbosity == 1:
                print('invalid allocator')
            exit()

        # main loop-----------------------------------------------------------------------------------------------------
        per_episode_metrics: dict = {
            'reward_per_step': -infty * ones(self.config.num_episodes),
            'priority_timeouts_per_occurrence': +infty * ones(self.config.num_episodes),
        }
        for episode_id in range(self.config.num_episodes):
            episode_metrics: dict = {
                'rewards': -infty * ones(self.config.num_steps_per_episode),
                'priority_timeouts': +infty * ones(self.config.num_steps_per_episode),
            }

            state_next = sim.gather_state()
            for step_id in range(self.config.num_steps_per_episode):
                simulation_step = episode_id * self.config.num_steps_per_episode + step_id

                # determine state
                state_current = state_next

                # find allocation action based on state
                bandwidth_allocation_solution = allocate_state_current()
                noisy_bandwidth_allocation_solution = bandwidth_allocation_solution

                # step simulation based on action
                (
                    step_reward,
                    unweighted_step_reward_components,
                ) = sim.step(
                    percentage_allocation_solution=noisy_bandwidth_allocation_solution,
                    critical_events_chance=probability_critical_events,
                )

                # determine new state
                state_next = sim.gather_state()

                # log step results
                episode_metrics['rewards'][step_id] = step_reward
                episode_metrics['priority_timeouts'][step_id] = unweighted_step_reward_components['sum_priority_timeouts']

                # progress print
                if self.config.verbosity == 1:
                    if step_id % 50 == 0:
                        progress_print()

            # log episode results
            per_episode_metrics['reward_per_step'][episode_id] = (
                sum(episode_metrics['rewards']) / self.config.num_steps_per_episode
            )
            per_episode_metrics['priority_timeouts_per_occurrence'][episode_id] = (
                sum(episode_metrics['priority_timeouts']) / self.config.num_steps_per_episode / (
                    probability_critical_events + self.config.tiny_numerical_value)
            )

            # print episode results
            if self.config.verbosity == 1:
                print('\n', end='')
                print('episode per step reward: {:.2f}'.format(
                    sum(episode_metrics['rewards']) / self.config.num_steps_per_episode
                ))
                print('episode per occurrence priority timeouts: {:.2f}'.format(
                    sum(episode_metrics['priority_timeouts']) / self.config.num_steps_per_episode / (
                        probability_critical_events + self.config.tiny_numerical_value)
                ))

            # reset simulation for next episode
            sim.reset()

        # teardown------------------------------------------------------------------------------------------------------
        # save logged results
        with gzip_open(join(self.config.log_path, testing_name + '_per_episode_metrics.gzip'), 'wb') as file:
            pickle_dump(per_episode_metrics, file=file)

        # end compute performance profiling
        if self.config.toggle_profiling:
            self.profiler.disable()
            if self.config.verbosity == 1:
                self.profiler.print_stats(sort='cumulative')
            self.profiler.dump_stats('train_critical_events.profile')

        # plots
        plot_scatter_plot(per_episode_metrics['reward_per_step'],
                          title='Per Step Reward')
        plot_scatter_plot(per_episode_metrics['priority_timeouts_per_occurrence'],
                          title='Per Occurrence Priority Timeouts')
