
from matplotlib.pyplot import (
    show as plt_show,
)
from os import (
    system,
)
from os.path import (
    join,
)

from config import Config
from runner import Runner

# TODO: Try with weaker anchoring penalty?


def main():
    config = Config()
    runner = Runner()

    # pre-train off environment from scratch
    runner.train_critical_events(
        name='base',
    )

    # pre-train off environment from scratch, extra penalty for critical
    priority_reward_old = runner.config.reward_sum_weightings['sum_priority_timeouts']
    runner.config.reward_sum_weightings['sum_priority_timeouts'] = 2 * priority_reward_old
    runner.train_critical_events(
        name='double_penalty',
    )
    runner.config.reward_sum_weightings['sum_priority_timeouts'] = priority_reward_old

    # train on environment, anchored
    runner.train_normal(
        name='anchored',
        value_network_path=join(config.model_path, 'critic_allocation_training_critical_events_base'),
        policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events_base'),
        anchoring_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events_base.gzip'),
    )

    # train on environment, anchored double penalty
    runner.train_normal(
        name='anchored_double_penalty',
        value_network_path=join(config.model_path, 'critic_allocation_training_critical_events_double_penalty'),
        policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events_double_penalty'),
        anchoring_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events_double_penalty.gzip'),
    )

    # train on environment, anchored, no specific rewarding of priority timeouts
    priority_reward_old = runner.config.reward_sum_weightings['sum_priority_timeouts']
    runner.config.reward_sum_weightings['sum_priority_timeouts'] = +0.0
    runner.train_normal(
        name='anchored_no_priority_reward',
        value_network_path=join(config.model_path, 'critic_allocation_training_critical_events_base'),
        policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events_base'),
        anchoring_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events_base.gzip'),
    )
    runner.config.reward_sum_weightings['sum_priority_timeouts'] = priority_reward_old

    # train on environment, anchored double penalty, no specific rewarding of priority timeouts
    priority_reward_old = runner.config.reward_sum_weightings['sum_priority_timeouts']
    runner.config.reward_sum_weightings['sum_priority_timeouts'] = +0.0
    runner.train_normal(
        name='anchored_double_penalty_no_priority_reward',
        value_network_path=join(config.model_path, 'critic_allocation_training_critical_events_double_penalty'),
        policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events_double_penalty'),
        anchoring_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events_double_penalty.gzip'),
    )
    runner.config.reward_sum_weightings['sum_priority_timeouts'] = priority_reward_old

    # double steps for training from here on
    runner.config.update_num_steps_per_episode(new_steps_per_episode=2*runner.config.num_steps_per_episode)

    # train on environment, from scratch
    runner.train_normal(
        name='base',
    )

    # train off environment, no rare events, from scratch, benchmark
    runner.train_no_critical_events(
        name='base',
    )

    # increase num_steps_per_episode, num_episodes for testing
    runner.config.update_num_steps_per_episode(new_steps_per_episode=100_000)
    runner.config.update_num_episodes(new_num_episodes=30)

    # test random allocation, benchmark
    runner.test(
        allocator='random',
        probability_critical_events=config.normal_priority_job_probability,
        name='normal_prob_priority_random_allocation',
    )

    # test pretrained on environment
    runner.test(
        allocator='pretrained',
        probability_critical_events=config.normal_priority_job_probability,
        policy_network_path=join(config.model_path, 'actor_allocation_training_normal_base'),
        policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_normal_base.gzip'),
        name='normal_prob_priority_pretrained_on_normal',
    )

    # test pretrained off environment, benchmark
    runner.test(
        allocator='pretrained',
        probability_critical_events=config.normal_priority_job_probability,
        policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events_base'),
        policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events_base.gzip'),
        name='normal_prob_priority_pretrained_on_high',
    )

    # test pretrained off & on environment, anchored
    runner.test(
        allocator='pretrained',
        probability_critical_events=config.normal_priority_job_probability,
        policy_network_path=join(config.model_path, 'actor_allocation_training_normal_anchored'),
        policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_normal_anchored.gzip'),
        name='normal_prob_priority_pretrained_on_normal_anchored',
    )

    # test pretrained off & on environment, anchored double penalty
    runner.test(
        allocator='pretrained',
        probability_critical_events=config.normal_priority_job_probability,
        policy_network_path=join(config.model_path, 'actor_allocation_training_normal_anchored_double_penalty'),
        policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_normal_anchored_double_penalty.gzip'),
        name='normal_prob_priority_pretrained_on_normal_anchored_double_penalty',
    )

    # test pretrained off & on environment, anchored, trained without priority weighting
    runner.test(
        allocator='pretrained',
        probability_critical_events=config.normal_priority_job_probability,
        policy_network_path=join(config.model_path, 'actor_allocation_training_normal_anchored_no_priority_reward'),
        policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_normal_anchored_no_priority_reward.gzip'),
        name='normal_prob_priority_pretrained_on_normal_anchored_no_priority_reward',
    )

    # test pretrained off & on environment, anchored double penalty, trained without priority weighting
    runner.test(
        allocator='pretrained',
        probability_critical_events=config.normal_priority_job_probability,
        policy_network_path=join(config.model_path, 'actor_allocation_training_normal_anchored_double_penalty_no_priority_reward'),
        policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_normal_anchored_double_penalty_no_priority_reward.gzip'),
        name='normal_prob_priority_pretrained_on_normal_anchored_double_penalty_no_priority_reward',
    )

    # test pretrained without critical events, benchmark
    runner.test(
        allocator='pretrained',
        probability_critical_events=config.normal_priority_job_probability,
        policy_network_path=join(config.model_path, 'actor_allocation_training_no_critical_events_base'),
        policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_no_critical_events_base.gzip'),
        name='normal_prob_priority_pretrained_on_no_critical_events',
    )

    if config.shutdown_on_complete:
        system('shutdown /h')

    if config.show_plots:
        plt_show()


if __name__ == '__main__':
    main()
