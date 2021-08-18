
import tensorflow as tf
from numpy import (
    ndarray,
    array,
    ndim,
    newaxis,
    expand_dims
)

from DL_Lottery_imports.experience_buffer import (
    ExperienceBuffer
)
from DL_Lottery_imports.network_models import (
    ValueNetwork,
    ValueNetworkBatchNorm,
    PolicyNetwork,
    PolicyNetworkBatchNorm
)
from DL_Lottery_imports.dl_internals_with_expl import (
    mse_loss,
    huber_loss
)


class TD3ActorCritic:
    batch_size: int
    gamma: float
    prioritization_factors: list[float]
    networks: dict[str: dict]

    def __init__(
            self,
            name: str,
            num_actions: int,
            batch_size: int,
            num_hidden_c: list,
            num_hidden_a: list,
            num_max_experiences: int,
            prioritization_factors: list,
            gamma: float,
            optimizer_critic,
            optimizer_critic_args: dict,
            optimizer_actor,
            optimizer_actor_args: dict,
            hidden_layer_args: dict,
            batch_normalize: bool = False
    ) -> None:
        """
        TD3 proposes a few changes to DDPG with the goal of reducing variance and increasing stability

        - "Soft" update to the target network theta_target = tau * theta_primary + (1 - tau) * theta_target_old
        - Update Value a few times after Policy update to have a good Value estimate for the next Policy update
        - Adding small noise to the TD error bootstrap to smoothen the Value estimate
        - Two independently trained critics, clipping the value estimate to
            the smaller of the two to avoid overestimation
        """
        self.batch_size = batch_size
        self.gamma = gamma  # future reward discount factor
        self.prioritization_factors = prioritization_factors  # [alpha, beta] for prioritized experience replay
        self.batch_normalize = batch_normalize

        self.experience_buffer = ExperienceBuffer(num_max_experiences=num_max_experiences,
                                                  alpha=self.prioritization_factors[0])

        # Initialize Networks-------------------------------------------------------------------------------------------
        self.initialize_networks(name=name,
                                 num_hidden_c=num_hidden_c,
                                 num_hidden_a=num_hidden_a,
                                 num_actions=num_actions,
                                 optimizer_critic=optimizer_critic,
                                 optimizer_critic_args=optimizer_critic_args,
                                 optimizer_actor=optimizer_actor,
                                 optimizer_actor_args=optimizer_actor_args,
                                 batch_normalize=batch_normalize,
                                 **hidden_layer_args)
        # --------------------------------------------------------------------------------------------------------------

    def initialize_networks(
            self,
            name: str,
            num_hidden_c: list,
            num_hidden_a: list,
            activation_hidden: str,
            kernel_initializer: str,
            num_actions,
            optimizer_critic,
            optimizer_critic_args: dict,
            optimizer_actor,
            optimizer_actor_args: dict,
            batch_normalize: bool
    ) -> None:
        self.networks = {}
        networks = {'value1': 'critic', 'value2': 'critic', 'policy': 'actor'}

        # Create network instances according to settings----------------------------------------------------------------
        for net_name in networks.keys():
            # Hyper parameter selection----------------------------------------
            if networks[net_name] == 'critic':
                if not batch_normalize:
                    network_type = ValueNetwork
                else:
                    network_type = ValueNetworkBatchNorm
                init_arguments = {
                    'num_hidden': num_hidden_c,
                }
            elif networks[net_name] == 'actor':
                if not batch_normalize:
                    network_type = PolicyNetwork
                else:
                    network_type = PolicyNetworkBatchNorm
                init_arguments = {
                    'num_hidden': num_hidden_a,
                    'num_actions': num_actions
                }
            else:
                network_type = None
                init_arguments = None
                exit()
            # -----------------------------------------------------------------

            # Create according to selected hyper parameters--------------------
            self.networks[net_name] = {
                'primary':
                    network_type(name=name + net_name + 'Primary',
                                 activation=activation_hidden,
                                 kernel_initializer=kernel_initializer,
                                 **init_arguments),
                'target':
                    network_type(name=name + net_name + 'Target',
                                 activation=activation_hidden,
                                 kernel_initializer=kernel_initializer,
                                 **init_arguments),
            }
            # -----------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Compile network instances-------------------------------------------------------------------------------------
        for net_name in ['value1', 'value2']:
            self.networks[net_name]['primary'].compile(
                optimizer=optimizer_critic(**optimizer_critic_args),
                loss=mse_loss)
        self.networks['policy']['primary'].compile(
            optimizer=optimizer_actor(**optimizer_actor_args),
            loss=mse_loss)

        # 100% copy primary params onto target at init
        self.update_target_networks(tau_target_update=1)
        # --------------------------------------------------------------------------------------------------------------

    def load_pretrained_networks(
            self,
            value_path: str,
            policy_path: str
    ):
        # TODO: This loads networks with newly initialized optimizers
        for network in ['value1', 'value2']:
            for network_type in ['primary', 'target']:
                if network_type == 'primary':
                    loss = self.networks[network][network_type].loss
                    optimizer = self.networks[network][network_type].optimizer
                self.networks[network][network_type] = tf.keras.models.load_model(value_path)
                if network_type == 'primary':
                    self.networks[network][network_type].compile(
                        optimizer=optimizer,
                        loss=loss
                    )
        for network in ['policy']:
            for network_type in ['primary', 'target']:
                if network_type == 'primary':
                    loss = self.networks[network][network_type].loss
                    optimizer = self.networks[network][network_type].optimizer
                self.networks[network][network_type] = tf.keras.models.load_model(policy_path)
                if network_type == 'primary':
                    self.networks[network][network_type].compile(
                        optimizer=optimizer,
                        loss=loss
                    )

    def freeze_layers(
            self,
            num_layers_value: int,
            num_layers_policy: int
    ) -> None:
        """
        Freezes layer weights for training
        """
        # TODO: Does not account for batch norm layers
        layers_policy = [self.networks['policy']['primary'].layers, self.networks['policy']['target'].layers]
        layers_value = [self.networks['value1']['primary'].layers, self.networks['value1']['target'].layers,
                        self.networks['value2']['primary'].layers, self.networks['value2']['target'].layers]
        if (
                num_layers_value > len(layers_value[0])
                or
                num_layers_policy > len(layers_policy[0])
        ):
            print('Cannot freeze more layers than present: {} value, {} policy'.format(
                len(layers_value[0]), len(layers_policy[0])))
            exit()

        for network_layers in layers_policy:
            for layer_id in range(num_layers_policy):
                network_layers[layer_id].trainable = False

        for network_layers in layers_value:
            for layer_id in range(num_layers_value):
                network_layers[layer_id].trainable = False

    def get_values(
            self,
            state: ndarray,
            action: ndarray
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Wrapper to call value networks
        """

        if ndim(state) == 1:
            state = array(state)[newaxis]
            action = array(action)[newaxis]
        network_input = tf.concat([state, action], axis=1)

        value1 = tf.squeeze(self.networks['value1']['target'].call(network_input))
        value2 = tf.squeeze(self.networks['value2']['target'].call(network_input))

        return value1, value2

    def get_value(
            self,
            state: ndarray,
            action: ndarray
    ) -> tf.Tensor:
        """
        Wrapper to call main value network
        """
        if ndim(state) == 1:
            state = array(state)[newaxis]
        if ndim(action) == 1:
            action = array(action)[newaxis]
        network_input = tf.concat([state, action], axis=1)
        value = tf.squeeze(self.networks['value1']['target'].call(network_input))

        return value

    def get_action(
            self,
            state: ndarray
    ) -> tf.Tensor:
        """
        Wrapper to call policy network
        """
        network_input = array(state, dtype='float32')
        if len(network_input.shape) == 1:
            network_input = expand_dims(network_input, axis=0)

        return tf.squeeze(self.networks['policy']['target'].call(network_input))

    def get_policy_params(
            self,
            network: str  # target or primary
    ) -> list:
        return self.networks['policy'][network].get_weights()

    def get_policy_fisher_diagonal_estimate(
            self,
            network: str  # target or primary
    ) -> list:
        """
        Some optimizers such as adam calculate a fisher diagonal estimate part of the optimization
        """
        fisher_diagonal_estimates = [
            self.networks['policy'][network].optimizer.get_slot(var, 'vhat').numpy()
            for var in self.networks['policy'][network].trainable_variables
        ]

        return fisher_diagonal_estimates

    def add_experience(
            self,
            state,
            action,
            reward,
            next_state
    ) -> None:
        """
        Wrapper to add experience to buffer
        """
        self.experience_buffer.add_experience(state, action, reward, next_state)

    @tf.function
    def update_target_networks(
            self,
            tau_target_update
    ) -> None:
        """
        Performs a soft update theta_target_new = tau * theta_primary + (1 - tau) * theta_target_old
        """
        for network_pair in self.networks.values():
            # trainable variables are a list of tf variables
            variables_primary = network_pair['primary'].trainable_variables
            variables_target = network_pair['target'].trainable_variables
            soft_update_parameters = [tf.add(
                tf.multiply(tau_target_update, variable_primary),
                tf.multiply((1 - tau_target_update), variable_target))
                for variable_primary, variable_target
                in zip(variables_primary, variables_target)]
            for v_old, v_soft in zip(variables_target, soft_update_parameters):
                v_old.assign(v_soft)

    @tf.function
    def train_graph(
            self,
            training_noise_std: float,
            training_noise_clip: float,
            tau_target_update: float,
            train_value: bool,
            train_policy: bool,
            states,
            actions,
            rewards,
            next_states,
            sample_importance_weights,
            weight_anchoring_lambda: None or float = None,
            policy_parameters_anchor: None or list = None,
            policy_parameters_fisher: None or list = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        def call_network(network, inputs):
            if self.batch_normalize:
                return network.call_and_normalize_on_batch(inputs)
            else:
                return network.call(inputs)
        """
        Wraps as much as possible of the training process into a graph for performance
        """

        # Value Networks------------------------------------------------------------------------------------------------
        q1_loss = -10.0
        q1_td_error = -10.0

        if train_value:
            target_q = rewards
            if self.gamma > 0:  # future rewards estimate
                next_actions = call_network(self.networks['policy']['target'], next_states)
                # Add a small amount of random noise to action for smoothing------------------------
                noise = tf.random.normal(shape=next_actions.shape,
                                         mean=0, stddev=training_noise_std, dtype=tf.float32)
                noise = tf.clip_by_value(noise, -training_noise_clip, training_noise_clip)
                next_actions += noise
                next_actions = tf.linalg.normalize(next_actions, axis=1, ord=1)[0]  # re-normalize
                # ----------------------------------------------------------------------------------
                input_vector = tf.concat([next_states, next_actions], axis=1)
                # Clipping so that the extra net wont introduce more overestimation-----------------
                q_estimate_1 = call_network(self.networks['value1']['target'], input_vector)
                q_estimate_2 = call_network(self.networks['value2']['target'], input_vector)
                conservative_q_estimate = tf.squeeze(tf.minimum(q_estimate_1, q_estimate_2))
                target_q = target_q + self.gamma * conservative_q_estimate
                # ----------------------------------------------------------------------------------

            # Gradient step value1--------------------------------------------------------------
            input_vector = tf.concat([states, actions], axis=1)
            with tf.GradientTape() as tape:  # autograd
                estimate = tf.squeeze(call_network(self.networks['value1']['primary'], input_vector))
                td_error = tf.math.subtract(target_q, estimate)
                loss = self.networks['value1']['primary'].loss(td_error, sample_importance_weights)

            gradients = tape.gradient(target=loss,  # d_loss / d_parameters
                                      sources=self.networks['value1']['primary'].trainable_variables)
            self.networks['value1']['primary'].optimizer.apply_gradients(  # apply gradient update
                zip(gradients, self.networks['value1']['primary'].trainable_variables))
            # ----------------------------------------------------------------------------------
            q1_td_error = td_error
            q1_loss = loss

            # Gradient step value2--------------------------------------------------------------
            if self.gamma > 0:
                with tf.GradientTape() as tape:  # autograd
                    estimate = tf.squeeze(call_network(self.networks['value2']['primary'], input_vector))
                    td_error = tf.math.subtract(target_q, estimate)
                    loss = self.networks['value2']['primary'].loss(td_error, sample_importance_weights)

                gradients = tape.gradient(target=loss,  # d_loss / d_parameters
                                          sources=self.networks['value2']['primary'].trainable_variables)
                self.networks['value2']['primary'].optimizer.apply_gradients(  # apply gradient update
                    zip(gradients, self.networks['value2']['primary'].trainable_variables))
            # ----------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Actor Network-------------------------------------------------------------------------------------------------
        if train_policy:
            # Gradient step policy--------------------------------------------------------------
            input_vector = states
            with tf.GradientTape() as tape:  # autograd
                # loss value network
                actor_actions = call_network(self.networks['policy']['primary'], input_vector)
                value_network_input = tf.concat([input_vector, actor_actions], axis=1)
                value_network_score = tf.reduce_mean(
                    call_network(self.networks['value1']['target'], value_network_input))
                # TODO: Original Paper, DDPG Paper and other implementations train on primary network. Why?
                # value_network_score = -tf.reduce_mean(
                #     self.networks['value1']['primary'].call(value_network_input, batch_normalize=True))

                # tf.print(value_network_score)

                # loss anchor
                if policy_parameters_anchor:
                    policy_parameters_current = tf.concat(  # flatten
                        [tf.reshape(self.networks['policy']['primary'].trainable_variables[layer], [-1])
                         for layer in range(len(self.networks['policy']['primary'].trainable_variables))],
                        axis=0)
                    policy_parameters_anchor = tf.concat(  # flatten
                        [tf.reshape(policy_parameters_anchor[layer], [-1])
                         for layer in range(len(policy_parameters_anchor))],
                        axis=0)
                    policy_parameters_fisher = tf.concat(  # flatten
                        [tf.reshape(policy_parameters_fisher[layer], [-1])
                         for layer in range(len(policy_parameters_fisher))],
                        axis=0)
                    policy_parameter_difference = tf.math.squared_difference(policy_parameters_current,
                                                                             policy_parameters_anchor)
                    fisher_weighted_policy_parameter_difference = tf.multiply(policy_parameters_fisher,
                                                                              policy_parameter_difference)
                    policy_parameter_difference_sum = tf.reduce_sum(fisher_weighted_policy_parameter_difference)
                    lambda_weighted_policy_parameter_difference_sum = tf.multiply(weight_anchoring_lambda,
                                                                                  policy_parameter_difference_sum)
                    # tf.print('p', lambda_weighted_policy_parameter_difference_sum)
                else:
                    lambda_weighted_policy_parameter_difference_sum = 0.0

                loss = (
                    - value_network_score
                    + lambda_weighted_policy_parameter_difference_sum
                )

            gradients = tape.gradient(target=loss,  # d_loss / d_parameters
                                      sources=self.networks['policy']['primary'].trainable_variables)
            self.networks['policy']['primary'].optimizer.apply_gradients(
                zip(gradients, self.networks['policy']['primary'].trainable_variables))  # apply gradient update
            # ----------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        self.update_target_networks(tau_target_update=tau_target_update)

        return q1_loss, q1_td_error

    def train(
            self,
            training_noise_std: float,
            training_noise_clip: float,
            tau_target_update: float,
            train_value: bool = True,
            train_policy: bool = True,
            weight_anchoring_lambda: None or float = None,
            policy_parameters_anchor: None or list = None,
            policy_parameters_fisher: None or list = None,
    ) -> float or None:

        if self.experience_buffer.get_len() < self.batch_size:
            return None

        # Sample from experience buffer-------------------------------------------------------------
        (
            sample_experiences,
            experience_ids,
            sample_importance_weights
        ) = self.experience_buffer.sample(
            batch_size=self.batch_size,
            beta=self.prioritization_factors[1]
        )
        states = tf.constant([experience['state'] for experience in sample_experiences], dtype=tf.float32)
        actions = tf.constant([experience['action'] for experience in sample_experiences], dtype=tf.float32)
        rewards = tf.constant([experience['reward'] for experience in sample_experiences], dtype=tf.float32)
        next_states = tf.constant([experience['next_state'] for experience in sample_experiences], dtype=tf.float32)
        # ------------------------------------------------------------------------------------------
        train_value = tf.constant(train_value)
        train_policy = tf.constant(train_policy)

        if policy_parameters_anchor:
            weight_anchoring_lambda = tf.constant(weight_anchoring_lambda, dtype=tf.float32)

        (
            q1_loss,
            q1_td_error
        ) = self.train_graph(
            training_noise_std=training_noise_std,
            training_noise_clip=training_noise_clip,
            tau_target_update=tau_target_update,
            train_value=train_value,
            train_policy=train_policy,
            states=states, actions=actions,
            rewards=rewards,
            next_states=next_states,
            sample_importance_weights=sample_importance_weights,
            weight_anchoring_lambda=weight_anchoring_lambda,
            policy_parameters_anchor=policy_parameters_anchor,
            policy_parameters_fisher=policy_parameters_fisher,
        )

        self.experience_buffer.adjust_priorities(experience_ids=experience_ids, new_priorities=q1_td_error.numpy())

        return q1_loss.numpy()
