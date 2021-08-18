
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
            self
    ) -> None:
        self.config = Config()
        self.rng = default_rng()

        if self.config.toggle_profiling:
            import cProfile
            self.profiler: cProfile.Profile = cProfile.Profile()

    def policy_training_criterion(
            self,
            simulation_step: int,
    ) -> bool:
        """Train policy networks only every k steps and/or only after j total steps to ensure a good value function"""
        if (
            simulation_step > self.config.train_policy_after_j
            and
            (simulation_step % self.config.train_policy_every_k) == 0
        ):
            return True
        return False

    def add_random_distribution(
            self,
            action: ndarray,  # turns out its much faster to numpy the tensor and then do operations on ndarray
            tau_momentum: float,
    ) -> ndarray:
        if tau_momentum == 0.0:
            return action

        random_distribution = self.rng.random(size=self.config.num_actions_policy, dtype='float32')
        random_distribution = random_distribution / sum(random_distribution)

        noisy_action = tau_momentum * random_distribution + (1 - tau_momentum) * action
        if sum(noisy_action) != 0:
            noisy_action = noisy_action / sum(noisy_action)

        return noisy_action

    def train_critical_events(
            self,
    ) -> None:
        def progress_print():
            progress = (episode_id * self.config.num_steps_per_episode + step_id + 1) / self.config.steps_total
            timedelta = datetime.now() - real_time_start
            finish_time = real_time_start + timedelta / progress

            print('\rSimulation completed: {:.2%}, est. finish {:02d}:{:02d}:{:02d}'.format(
                progress, finish_time.hour, finish_time.minute, finish_time.second), end='')

        def save_networks():
            state = sim.gather_state()
            action = allocator.get_action(state)

            allocator.networks['policy']['target'](state[newaxis])
            allocator.networks['policy']['target'].save(
                join(self.config.model_path, 'actor_allocation_' + training_name))

            allocator.networks['value1']['target'](concatenate([state, action])[newaxis])
            allocator.networks['value1']['target'].save(
                join(self.config.model_path, 'critic_allocation_' + training_name))

            copy2(join(dirname(__file__), 'config.py'), self.config.model_path)

        # setup------------------------------------------
        training_name = 'training_critical_events'
        real_time_start = datetime.now()

        if self.config.toggle_profiling:
            self.profiler.enable()

        sim = Simulation(
            config=self.config,
            rng=self.rng,
        )
        allocator = TD3ActorCritic(
            name='allocator',
            **self.config.td3_args
        )
        print('Expected load: {:.2f}'.format(sim.get_expected_load()))

        exploration_noise_momentum = self.config.exploration_noise_momentum_initial

        policy_parameters: dict = {
            'initial': [],
            'final': [],
            'fisher': [],
        }
        # ###############################################
        # fisher = []

        # main loop--------------------------------------
        per_episode_metrics: dict = {
            'reward_per_step': -infty * ones(self.config.num_episodes),
            'value_loss_mean': +infty * ones(self.config.num_episodes),
        }
        for episode_id in range(self.config.num_episodes):
            episode_metrics: dict = {
                'rewards': -infty * ones(self.config.num_steps_per_episode),
                'value_losses': +infty * ones(self.config.num_steps_per_episode),
            }

            step_experience: dict = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}
            state_next = sim.gather_state()
            if episode_id == 0:  # initialize network to log initial params
                allocator.networks['policy']['primary'].initialize_inputs(state_next[newaxis])
                policy_parameters['initial'] = allocator.get_policy_params('primary')

            for step_id in range(self.config.num_steps_per_episode):
                simulation_step = episode_id * self.config.num_steps_per_episode + step_id

                state_current = state_next
                step_experience['state'] = state_current

                bandwidth_allocation_solution = allocator.get_action(state_current).numpy()
                noisy_bandwidth_allocation_solution = self.add_random_distribution(
                        action=bandwidth_allocation_solution,
                        tau_momentum=exploration_noise_momentum)
                step_experience['action'] = noisy_bandwidth_allocation_solution

                (
                    step_reward,
                    unweighted_step_reward_components,
                ) = sim.step(percentage_allocation_solution=noisy_bandwidth_allocation_solution)
                step_experience['reward'] = step_reward

                state_next = sim.gather_state()
                step_experience['next_state'] = state_next

                allocator.add_experience(**step_experience)

                train_policy = False
                if self.policy_training_criterion(simulation_step=simulation_step):
                    train_policy = True
                step_allocation_value1_loss = allocator.train(
                    training_noise_std=self.config.training_noise_std,
                    training_noise_clip=self.config.training_noise_clip,
                    tau_target_update=self.config.tau_target_update,
                    train_policy=train_policy)

                # ##########################################
                # if simulation_step > 0.8 * self.config.steps_total:
                #     fisher.append(allocator.get_policy_fisher_diagonal_estimate('primary'))

                if simulation_step > self.config.exploration_noise_step_start_decay:
                    exploration_noise_momentum = max(
                        0.0,
                        exploration_noise_momentum - self.config.exploration_noise_linear_decay_per_step
                    )

                # log
                episode_metrics['rewards'][step_id] = step_experience['reward']
                episode_metrics['value_losses'][step_id] = step_allocation_value1_loss

                # Progress print----------------------------------------------------------------------------------------
                if step_id % 50 == 0:
                    progress_print()
                # ------------------------------------------------------------------------------------------------------
            # log
            # Remove NaN entries, e.g. steps in which there was no training -> None loss
            episode_metrics['value_losses'] = episode_metrics['value_losses'][~isnan(episode_metrics['value_losses'])]

            per_episode_metrics['reward_per_step'][episode_id] = (
                    sum(episode_metrics['rewards']) / self.config.num_steps_per_episode)
            per_episode_metrics['value_loss_mean'][episode_id] = (
                    mean(episode_metrics['value_losses']))

            print('\n', end='')
            print('episode per step reward: {:.2f}'.format(
                sum(episode_metrics['rewards']) / self.config.num_steps_per_episode))
            print('episode mean value loss: {:.2f}'.format(
                mean(episode_metrics['value_losses'])))

            sim.reset()

        # teardown---------------------------------------
        policy_parameters['final'] = allocator.get_policy_params('target')
        policy_parameters['fisher'] = allocator.get_policy_fisher_diagonal_estimate('primary')

        save_networks()

        # log
        # TODO: save logs
        with gzip_open(join(self.config.model_path, 'policy_parameters_' + training_name + '.gzip'), 'wb') as file:
            pickle_dump(policy_parameters, file=file)

        if self.config.toggle_profiling:
            self.profiler.disable()
            self.profiler.print_stats(sort='cumulative')
            self.profiler.dump_stats(join(self.config.performance_profile_path, training_name + '.profile'))

        # plots------------------------------------------
        plot_scatter_plot(per_episode_metrics['reward_per_step'], title='Per Step Reward')
        plot_scatter_plot(per_episode_metrics['value_loss_mean'], title='Mean Value Loss')

        ################################################
        # from numpy import array
        # fisher = array(fisher)
        # plot_scatter_plot(fisher[:, 0], title='aaaaaaaaaa')

    def test_critical_events(
            self,
            allocator: str,  # 'pretrained', 'random'
            policy_network_path: None or str = None,
            policy_pruning_parameters_path: None or str = None,
    ) -> None:
        def progress_print():
            progress = (episode_id * self.config.num_steps_per_episode + step_id + 1) / self.config.steps_total
            timedelta = datetime.now() - real_time_start
            finish_time = real_time_start + timedelta / progress

            print('\rSimulation completed: {:.2%}, est. finish {:02d}:{:02d}:{:02d}'.format(
                progress, finish_time.hour, finish_time.minute, finish_time.second), end='')

        # setup----------------------------------------------------
        real_time_start = datetime.now()
        if self.config.toggle_profiling:
            self.profiler.enable()

        sim = Simulation(
            config=self.config,
            rng=self.rng,
        )
        print('Expected load: {:.2f}'.format(sim.get_expected_load()))

        # define allocation function-------------------------------
        if allocator == 'pretrained':
            allocator_network = load_model(policy_network_path)

            def allocate_state_current() -> ndarray:
                return allocator_network.call(state_current[newaxis]).numpy().squeeze()

            if self.config.prune_network:
                # Prune
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
            def allocate_state_current(): pass
            print('invalid allocator')
            exit()

        # main loop------------------------------------------------
        per_episode_metrics: dict = {
            'reward_per_step': -1_000 * ones(self.config.num_episodes),
        }
        for episode_id in range(self.config.num_episodes):
            episode_metrics: dict = {
                'rewards': -1_000 * ones(self.config.num_steps_per_episode),
            }

            state_next = sim.gather_state()
            for step_id in range(self.config.num_steps_per_episode):
                simulation_step = episode_id * self.config.num_steps_per_episode + step_id

                state_current = state_next
                bandwidth_allocation_solution = allocate_state_current()

                noisy_bandwidth_allocation_solution = bandwidth_allocation_solution

                (
                    step_reward,
                    unweighted_step_reward_components,
                ) = sim.step(percentage_allocation_solution=noisy_bandwidth_allocation_solution)
                state_next = sim.gather_state()

                episode_metrics['rewards'][step_id] = step_reward

                # Progress print--------------------------------------------
                if step_id % 50 == 0:
                    progress_print()

            per_episode_metrics['reward_per_step'][episode_id] = (
                    sum(episode_metrics['rewards']) / self.config.num_steps_per_episode)

            print('\n', end='')
            print('episode per step reward: {:.2f}'.format(
                sum(episode_metrics['rewards']) / self.config.num_steps_per_episode))

            sim.reset()

        # teardown-------------------------------------------------
        if self.config.toggle_profiling:
            self.profiler.disable()
            self.profiler.print_stats(sort='cumulative')
            self.profiler.dump_stats('train_critical_events.profile')

        # plots----------------------------------------------------
        plot_scatter_plot(per_episode_metrics['reward_per_step'], title='Per Step Reward')

    def train_no_critical_events(
            self
    ) -> None:
        pass

    def train_normal(
            self
    ) -> None:
        pass

    def train_on_random_data(
            self,
            value_network_path: None or str = None,
            policy_network_path: None or str = None,
            anchoring_parameters_path: None or str = None,
    ) -> None:
        def progress_print():
            progress = (episode_id * self.config.num_steps_per_episode + step_id + 1) / self.config.steps_total
            timedelta = datetime.now() - real_time_start
            finish_time = real_time_start + timedelta / progress

            print('\rSimulation completed: {:.2%}, est. finish {:02d}:{:02d}:{:02d}'.format(
                progress, finish_time.hour, finish_time.minute, finish_time.second), end='')

        def save_networks():
            state = sim.gather_state()
            action = allocator.get_action(state)

            allocator.networks['policy']['target'](state[newaxis])
            allocator.networks['policy']['target'].save(
                join(self.config.model_path, 'actor_allocation_' + training_name))

            allocator.networks['value1']['target'](concatenate([state, action])[newaxis])
            allocator.networks['value1']['target'].save(
                join(self.config.model_path, 'critic_allocation_' + training_name))

            copy2(join(dirname(__file__), 'config.py'), self.config.model_path)

        # setup------------------------------------------
        training_name = 'training_random_data'
        real_time_start = datetime.now()

        if self.config.toggle_profiling:
            self.profiler.enable()

        sim = Simulation(
            config=self.config,
            rng=self.rng,
        )
        print('Expected load: {:.2f}'.format(sim.get_expected_load()))

        allocator = TD3ActorCritic(
            name='allocator',
            **self.config.td3_args
        )
        # load pretrained
        if value_network_path:
            allocator.load_pretrained_networks(value_path=value_network_path,
                                               policy_path=policy_network_path)
        # load anchoring
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
        per_episode_metrics: dict = {
            'reward_per_step': -infty * ones(self.config.num_episodes),
            'value_loss_mean': +infty * ones(self.config.num_episodes),
        }
        state_shape = sim.gather_state().shape

        # main loop--------------------------------------
        for episode_id in range(self.config.num_episodes):
            episode_metrics: dict = {
                'rewards': -infty * ones(self.config.num_steps_per_episode),
                'value_losses': +infty * ones(self.config.num_steps_per_episode),
            }

            step_experience: dict = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}
            state_next = self.rng.random(state_shape)  # TODO: Maybe normalize

            # initialize network to log initial params
            if episode_id == 0 and not value_network_path:  # no pre-trained loaded
                allocator.networks['policy']['primary'].initialize_inputs(state_next[newaxis])
                policy_parameters['initial'] = allocator.get_policy_params('primary')

            for step_id in range(self.config.num_steps_per_episode):
                simulation_step = episode_id * self.config.num_steps_per_episode + step_id

                state_current = state_next
                step_experience['state'] = state_current

                bandwidth_allocation_solution = allocator.get_action(state_current).numpy()
                noisy_bandwidth_allocation_solution = self.add_random_distribution(
                        action=bandwidth_allocation_solution,
                        tau_momentum=exploration_noise_momentum)
                step_experience['action'] = noisy_bandwidth_allocation_solution

                (
                    step_reward,
                    unweighted_step_reward_components,
                ) = sim.step(percentage_allocation_solution=noisy_bandwidth_allocation_solution)
                step_experience['reward'] = self.rng.random()

                state_next = self.rng.random(state_shape)  # TODO: Maybe normalize
                step_experience['next_state'] = state_next

                allocator.add_experience(**step_experience)

                train_policy = False
                if self.policy_training_criterion(simulation_step=simulation_step):
                    train_policy = True
                step_allocation_value1_loss = allocator.train(
                    training_noise_std=self.config.training_noise_std,
                    training_noise_clip=self.config.training_noise_clip,
                    tau_target_update=self.config.tau_target_update,
                    train_policy=train_policy,
                    weight_anchoring_lambda=self.config.anchoring_weight_lambda,
                    policy_parameters_anchor=policy_parameters_anchor,
                    policy_parameters_fisher=policy_parameters_fisher,
                )

                # ##########################################
                # if simulation_step > 0.8 * self.config.steps_total:
                #     fisher.append(allocator.get_policy_fisher_diagonal_estimate('primary'))

                if simulation_step > self.config.exploration_noise_step_start_decay:
                    exploration_noise_momentum = max(
                        0.0,
                        exploration_noise_momentum - self.config.exploration_noise_linear_decay_per_step
                    )

                # log
                episode_metrics['rewards'][step_id] = step_reward
                episode_metrics['value_losses'][step_id] = step_allocation_value1_loss

                # Progress print----------------------------------------------------------------------------------------
                if step_id % 50 == 0:
                    progress_print()

            # log
            # Remove NaN entries, e.g. steps in which there was no training -> None loss
            episode_metrics['value_losses'] = episode_metrics['value_losses'][~isnan(episode_metrics['value_losses'])]

            per_episode_metrics['reward_per_step'][episode_id] = (
                    sum(episode_metrics['rewards']) / self.config.num_steps_per_episode)
            per_episode_metrics['value_loss_mean'][episode_id] = (
                    mean(episode_metrics['value_losses']))

            print('\n', end='')
            print('episode per step reward: {:.2f}'.format(
                sum(episode_metrics['rewards']) / self.config.num_steps_per_episode))
            print('episode mean value loss: {:.2f}'.format(
                mean(episode_metrics['value_losses'])))

            sim.reset()

        # teardown---------------------------------------
        policy_parameters['final'] = allocator.get_policy_params('target')
        policy_parameters['fisher'] = allocator.get_policy_fisher_diagonal_estimate('primary')

        save_networks()

        # log
        # TODO: save logs
        with gzip_open(join(self.config.model_path, 'policy_parameters_' + training_name + '.gzip'), 'wb') as file:
            pickle_dump(policy_parameters, file=file)

        if self.config.toggle_profiling:
            self.profiler.disable()
            self.profiler.print_stats(sort='cumulative')
            self.profiler.dump_stats(join(self.config.performance_profile_path, training_name + '.profile'))

        # plots------------------------------------------
        plot_scatter_plot(per_episode_metrics['reward_per_step'], title='Per Step Reward')
        plot_scatter_plot(per_episode_metrics['value_loss_mean'], title='Mean Value Loss')
