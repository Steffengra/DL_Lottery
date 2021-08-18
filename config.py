
from numpy import (
    ndarray,
    array,
    ceil,
    floor
)
from os.path import (
    join,
    dirname
)
from DL_Lottery_imports.dl_internals_with_expl import (
    optimizer_adam,
    optimizer_nadam
)

# TODO: Remember that batch normalization may be required
# TODO: Look more deeply into the fisher approximation by adam. seems fishy - does it jump around much at every step?


class Config:
    def __init__(
            self
    ) -> None:
        # Tweakable-----------------------------------------------------------------------------------------------------
        # Simulation Environment Parameters-----------------------------------------
        simulation_title: str = 'test'
        simulation_length_seconds: int = 1
        self.num_episodes: int = 10

        self.available_rb_ofdm: int = 10
        self.bandwidth_per_rb_khz: int = 12 * 15  # 5g num0 = 12 subcarriers * 15 khz

        self.num_users: int = 5
        self.ue_snr_db: float = 10
        self.ue_position_range: dict = {'low': -100, 'high': 100}
        self.ue_rayleigh_fading_scale: float = .1
        # self.path_loss_exponent: float = 2
        self.job_creation_probability: float = 0.4
        self.max_job_size_rb: int = 6
        self.timeout_step: dict = {  # after how many time delays is a job timed out?
            'normal': 5,
            'priority': 1,
        }
        self.symbols_per_subframe: int = 14  # see: 5g numerologies, 14=num0
        self.reward_sum_weightings: dict = {
            'sum_capacity_kbit_per_second': + 1.0,
            'sum_normal_timeouts': - 1.0,  # not including priority jobs
            'sum_priority_timeouts': - 1.0,
        }

        self.toggle_profiling: bool = False
        # self.toggle_reward_component_logging: bool = False  # fine grain logging  # TODO
        # self.toggle_position_logging: bool = False  # high computation cost
        # self.toggle_resource_grid_logging: bool = False  # high computation cost
        # self.resource_grid_logging_max_rb: int = 200  # plots only the last x logged rb

        # Neural Network Parameters-------------------------------------------------
        # Architecture-------
        self.hidden_layers_value_net: list = [512, 512, 512, 512, 512]
        self.hidden_layers_policy_net: list = [512, 512, 512, 512, 512]
        # self.hidden_layers_value_net: list = [2, 3]
        # self.hidden_layers_policy_net: list = [2, 3]
        self.hidden_layer_args: dict = {
            'activation_hidden': 'penalized_tanh',  # [>'relu', 'elu', 'penalized_tanh']
            'kernel_initializer': 'glorot_uniform'  # options: tf.keras.initializers, default: >'glorot_uniform'
        }
        self.num_actions_policy: int = self.num_users
        # Training-----------
        self.future_reward_discount_gamma_allocation: float = 0.00
        self.train_policy_every_k: int = 1  # td3 waits a number of value net updates before updating policy net
        train_policy_after_j: float = 0.0  # start training policy after j% of simulation steps
        self.optimizer_critic = optimizer_adam  # 'adam', 'nadam', 'amsgrad'
        self.optimizer_critic_args: dict = {
            'learning_rate': 1e-4,
            'epsilon': 1e-8,
            'amsgrad': True
        }
        self.optimizer_actor = optimizer_adam
        self.optimizer_actor_args: dict = {
            'learning_rate': 1e-4,
            'epsilon': 1e-8,
            'amsgrad': True
        }
        self.training_noise_std: float = 1e-2  # introduce a small amount of noise onto the policy in value training..
        self.training_noise_clip: float = 0.05  # ..to avoid narrow peaks in value function
        # soft update theta_target_new = tau * theta_primary + (1 - tau) * theta_target_old
        self.tau_target_update: float = 1e-1
        self.batch_size: int = 256
        self.batch_normalize: bool = False  # Normalize to 0 mean, unit variance based on mini-batch statistics
        self.anchoring_weight_lambda: float = 1e5  # Multiplier weight to the anchoring penalty in the loss function

        # Experience Buffer--
        self.experience_buffer_max_size: int = 20_000
        self.prioritization_factors: list = [0, 1]  # [alpha, beta], alpha=sample weight scaling, beta=IS correction %

        # Exploration--------
        self.exploration_noise_momentum_initial: float = 1.0
        exploration_noise_decay_start: float = 0.0  # After which % of training to start decay
        exploration_noise_decay_threshold: float = 0.5
        # value_based_exploration_interval: dict[str: float] = {'start': 1.0, 'end': 1.0}
        # self.value_based_exploration_config: dict[str: float] = {
        #     'tau_moving_average': 0.99,
        #     'threshold': .2,  # explore when value estimate is x% below average value
        #     'randomness_intensity': 1.0  # in fractional: how much perturbation should be applied when exploring
        # }

        # Pruning------------
        self.prune_network: bool = False
        self.pruning_args: dict = {
            'magnitude_percentile': 10,  # prune parameters |w_final|
            'magnitude_increase_percentile': 0,  # prune parameters |w_final| - |w_init|
        }

        # Definitions---------------------------------------------------------------------------------------------------
        # Simulation Environment Parameters-----------------------------------------
        # self.mobility_types: dict = {'High': {'range_initial': [-25, 25], 'step_size_max': .8},
        #                              'Low': {'range_initial': [-3, 3], 'step_size_max': .001},
        #                              'Static': {'range_initial': [-3, 3], 'step_size_max': 0}}

        self.duration_frame_s: float = 0.01
        self.duration_subframe_s: float = self.duration_frame_s / 10

        self.five_g_numerologies: dict = {0: {'delta_f_khz': 15,  # fr1
                                              'slots_per_subframe': 1,
                                              'max_carrier_bandwidth_mhz': 50,
                                              'max_amount_resource_blocks': 270,
                                              'symbol_duration_ms': 1 / 14},
                                          1: {'delta_f_khz': 30,  # fr1
                                              'slots_per_subframe': 2,
                                              'max_carrier_bandwidth_mhz': 100,
                                              'max_amount_resource_blocks': 273,
                                              'symbol_duration_ms': 1 / 28},
                                          2: {'delta_f_khz': 60,  # fr1 + fr2
                                              'slots_per_subframe': 4,
                                              'max_carrier_bandwidth_mhz': 200,
                                              'max_amount_resource_blocks': 264,
                                              'symbol_duration_ms': 1 / 56},
                                          3: {'delta_f_khz': 120,  # fr2
                                              'slots_per_subframe': 8,
                                              'max_carrier_bandwidth_mhz': 400,
                                              'max_amount_resource_blocks': 264,
                                              'symbol_duration_ms': 1 / 112}}

        self.five_g_mcs_table: dict = {
            2: {0: {'spectral_efficiency': 0.2344, 'mod_order': 2, 'coding_rate': 120 / 1024},
                1: {'spectral_efficiency': 0.3770, 'mod_order': 2, 'coding_rate': 193 / 1024},
                2: {'spectral_efficiency': 0.6016, 'mod_order': 2, 'coding_rate': 308 / 1024},
                3: {'spectral_efficiency': 0.8779, 'mod_order': 2, 'coding_rate': 449 / 1024},
                4: {'spectral_efficiency': 1.1758, 'mod_order': 2, 'coding_rate': 602 / 1024},
                5: {'spectral_efficiency': 1.4766, 'mod_order': 4, 'coding_rate': 378 / 1024},
                6: {'spectral_efficiency': 1.6953, 'mod_order': 4, 'coding_rate': 434 / 1024},
                7: {'spectral_efficiency': 1.9141, 'mod_order': 4, 'coding_rate': 490 / 1024},
                8: {'spectral_efficiency': 2.1602, 'mod_order': 4, 'coding_rate': 553 / 1024},
                9: {'spectral_efficiency': 2.4063, 'mod_order': 4, 'coding_rate': 616 / 1024},
                10: {'spectral_efficiency': 2.5703, 'mod_order': 4, 'coding_rate': 658 / 1024},
                11: {'spectral_efficiency': 2.7305, 'mod_order': 6, 'coding_rate': 466 / 1024},
                12: {'spectral_efficiency': 3.0293, 'mod_order': 6, 'coding_rate': 517 / 1024},
                13: {'spectral_efficiency': 3.3223, 'mod_order': 6, 'coding_rate': 567 / 1024},
                14: {'spectral_efficiency': 3.6094, 'mod_order': 6, 'coding_rate': 616 / 1024},
                15: {'spectral_efficiency': 3.9023, 'mod_order': 6, 'coding_rate': 666 / 1024},
                16: {'spectral_efficiency': 4.2129, 'mod_order': 6, 'coding_rate': 719 / 1024},
                17: {'spectral_efficiency': 4.5234, 'mod_order': 6, 'coding_rate': 772 / 1024},
                18: {'spectral_efficiency': 4.8164, 'mod_order': 6, 'coding_rate': 822 / 1024},
                19: {'spectral_efficiency': 5.1152, 'mod_order': 6, 'coding_rate': 873 / 1024},
                20: {'spectral_efficiency': 5.3320, 'mod_order': 8, 'coding_rate': 682.5 / 1024},
                21: {'spectral_efficiency': 5.5547, 'mod_order': 8, 'coding_rate': 711 / 1024},
                22: {'spectral_efficiency': 5.8906, 'mod_order': 8, 'coding_rate': 754 / 1024},
                23: {'spectral_efficiency': 6.2266, 'mod_order': 8, 'coding_rate': 797 / 1024},
                24: {'spectral_efficiency': 6.5703, 'mod_order': 8, 'coding_rate': 841 / 1024},
                25: {'spectral_efficiency': 6.9141, 'mod_order': 8, 'coding_rate': 885 / 1024},
                26: {'spectral_efficiency': 7.1602, 'mod_order': 8, 'coding_rate': 916.5 / 1024},
                27: {'spectral_efficiency': 7.4063, 'mod_order': 8, 'coding_rate': 948 / 1024}},
            3: {0: {'spectral_efficiency': 0.0586, 'mod_order': 2, 'coding_rate': 30 / 1024},
                1: {'spectral_efficiency': 0.0781, 'mod_order': 2, 'coding_rate': 40 / 1024},
                2: {'spectral_efficiency': 0.0977, 'mod_order': 2, 'coding_rate': 50 / 1024},
                3: {'spectral_efficiency': 0.1250, 'mod_order': 2, 'coding_rate': 64 / 1024},
                4: {'spectral_efficiency': 0.1523, 'mod_order': 2, 'coding_rate': 78 / 1024},
                5: {'spectral_efficiency': 0.1934, 'mod_order': 2, 'coding_rate': 99 / 1024},
                6: {'spectral_efficiency': 0.2344, 'mod_order': 2, 'coding_rate': 120 / 1024},
                7: {'spectral_efficiency': 0.3066, 'mod_order': 2, 'coding_rate': 157 / 1024},
                8: {'spectral_efficiency': 0.3770, 'mod_order': 2, 'coding_rate': 193 / 1024},
                9: {'spectral_efficiency': 0.4902, 'mod_order': 2, 'coding_rate': 251 / 1024},
                10: {'spectral_efficiency': 0.6016, 'mod_order': 2, 'coding_rate': 308 / 1024},
                11: {'spectral_efficiency': 0.7402, 'mod_order': 2, 'coding_rate': 379 / 1024},
                12: {'spectral_efficiency': 0.8770, 'mod_order': 2, 'coding_rate': 449 / 1024},
                13: {'spectral_efficiency': 1.0273, 'mod_order': 2, 'coding_rate': 526 / 1024},
                14: {'spectral_efficiency': 1.1758, 'mod_order': 2, 'coding_rate': 602 / 1024},
                15: {'spectral_efficiency': 1.3281, 'mod_order': 4, 'coding_rate': 340 / 1024},
                16: {'spectral_efficiency': 1.4766, 'mod_order': 4, 'coding_rate': 378 / 1024},
                17: {'spectral_efficiency': 1.6953, 'mod_order': 4, 'coding_rate': 434 / 1024},
                18: {'spectral_efficiency': 1.9141, 'mod_order': 4, 'coding_rate': 490 / 1024},
                19: {'spectral_efficiency': 2.1602, 'mod_order': 4, 'coding_rate': 553 / 1024},
                20: {'spectral_efficiency': 2.4063, 'mod_order': 4, 'coding_rate': 616 / 1024},
                21: {'spectral_efficiency': 2.5664, 'mod_order': 6, 'coding_rate': 438 / 1024},
                22: {'spectral_efficiency': 2.7305, 'mod_order': 6, 'coding_rate': 466 / 1024},
                23: {'spectral_efficiency': 3.0293, 'mod_order': 6, 'coding_rate': 517 / 1024},
                24: {'spectral_efficiency': 3.3223, 'mod_order': 6, 'coding_rate': 567 / 1024},
                25: {'spectral_efficiency': 3.6094, 'mod_order': 6, 'coding_rate': 616 / 1024},
                26: {'spectral_efficiency': 3.9023, 'mod_order': 6, 'coding_rate': 666 / 1024},
                27: {'spectral_efficiency': 4.2129, 'mod_order': 6, 'coding_rate': 719 / 1024},
                28: {'spectral_efficiency': 4.5234, 'mod_order': 6, 'coding_rate': 772 / 1024}}}

        # Simulation Parameters-----------------------------------------------------
        self.model_path: str = join(dirname(__file__), 'SavedModels', simulation_title)
        self.log_path: str = join(dirname(__file__), 'logs')
        self.performance_profile_path: str = join(dirname(__file__), 'performance_profiles')

        # Internal------------------------------------------------------------------
        self.num_steps_per_episode: int = int(
            simulation_length_seconds / self.duration_subframe_s * self.symbols_per_subframe)
        self.steps_total: int = self.num_episodes * self.num_steps_per_episode
        self.pos_base_station: ndarray = array([0, 0])
        self.ue_snr_linear = 10 ** (self.ue_snr_db / 10)

        # Exploration--------------
        self.exploration_noise_step_start_decay: int = ceil(
            exploration_noise_decay_start * self.num_episodes * self.num_steps_per_episode)

        self.exploration_noise_linear_decay_per_step: float = (
                self.exploration_noise_momentum_initial / (
                    exploration_noise_decay_threshold * (
                        self.num_episodes * self.num_steps_per_episode - (
                            self.exploration_noise_step_start_decay))))

        # self.value_based_exploration_step_start: int = ceil(
        #     value_based_exploration_interval['start'] * self.steps_total)
        # self.value_based_exploration_step_end: int = ceil(
        #     value_based_exploration_interval['end'] * self.steps_total)
        # -------------------------

        self.train_policy_after_j: int = floor(
            train_policy_after_j * self.num_episodes * self.num_steps_per_episode)

        self.td3_args = {
            'batch_size': self.batch_size,
            'num_hidden_c': self.hidden_layers_value_net,
            'num_hidden_a': self.hidden_layers_policy_net,
            'num_actions': self.num_actions_policy,
            'num_max_experiences': self.experience_buffer_max_size,
            'gamma': self.future_reward_discount_gamma_allocation,
            'prioritization_factors': self.prioritization_factors,
            'optimizer_critic': self.optimizer_critic,
            'optimizer_critic_args': self.optimizer_critic_args,
            'optimizer_actor': self.optimizer_actor,
            'optimizer_actor_args': self.optimizer_actor_args,
            'hidden_layer_args': self.hidden_layer_args,
            'batch_normalize': self.batch_normalize
        }

        # Plotting------------------------------------------------------------------
        # Branding Palette
        self.color0: str = '#000000'  # black
        self.color1: str = '#21467a'  # blue
        self.color2: str = '#c4263a'  # red
        self.color3: str = '#008700'  # green
        self.color4: str = '#caa023'  # gold

        # Colorblind Palette
        self.ccolor0: str = '#000000'  # use for lines, black
        self.ccolor1: str = '#d01b88'  # use for lines, pink
        self.ccolor2: str = '#254796'  # use for scatter, blue
        self.ccolor3: str = '#307b3b'  # use for scatter, green
        self.ccolor4: str = '#caa023'  # use for scatter, gold
