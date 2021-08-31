
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


# todo: possible idea, ruin priority jobs fading coefficient


def main():
    config = Config()
    runner = Runner()

    # Train on critical environment
    sum_priority_timeout_old = runner.config.reward_sum_weightings['sum_priority_timeouts']
    runner.config.reward_sum_weightings['sum_priority_timeouts'] = - 5.0
    runner.train_critical_events(
        name='base',
    )
    runner.config.reward_sum_weightings['sum_priority_timeouts'] = sum_priority_timeout_old

    # Train pretrained on normal environment, no sum priority timeout penalty
    sum_priority_timeout_old = runner.config.reward_sum_weightings['sum_priority_timeouts']
    runner.config.reward_sum_weightings['sum_priority_timeouts'] = - 0.0
    runner.train_normal(
        name='anchored',
        value_network_path=join(config.model_path, 'critic_allocation_training_critical_events_base'),
        policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events_base'),
        anchoring_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events_base.gzip'),
    )
    runner.config.reward_sum_weightings['sum_priority_timeouts'] = sum_priority_timeout_old

    # train on normal environment from scratch
    # sum_priority_timeout_old = runner.config.reward_sum_weightings['sum_priority_timeouts']
    # runner.config.reward_sum_weightings['sum_priority_timeouts'] = - 5.0
    # runner.config.update_num_episodes(new_num_episodes=2*runner.config.num_episodes)
    # runner.train_normal(
    #     name='base',
    # )
    # runner.config.reward_sum_weightings['sum_priority_timeouts'] = sum_priority_timeout_old

    runner.config.update_num_episodes(new_num_episodes=10)
    runner.config.update_num_steps_per_episode(new_steps_per_episode=100_000)
    # test pretrained off environment
    # runner.test(
    #     allocator='pretrained',
    #     probability_critical_events=config.normal_priority_job_probability,
    #     policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events_base'),
    #     policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events_base.gzip'),
    #     name='normal_trained_on_critical',
    # )
    # test pretrained on environment
    # runner.test(
    #     allocator='pretrained',
    #     probability_critical_events=config.normal_priority_job_probability,
    #     policy_network_path=join(config.model_path, 'actor_allocation_training_normal_base'),
    #     policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_normal_base.gzip'),
    #     name='normal_trained_on_normal',
    # )
    # test pretrained on environment, anchored
    runner.test(
        allocator='pretrained',
        probability_critical_events=config.normal_priority_job_probability,
        policy_network_path=join(config.model_path, 'actor_allocation_training_normal_anchored'),
        policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_normal_anchored.gzip'),
        name='normal_trained_on_normal_anchored',
    )
    # test random on environment
    # runner.test(
    #     allocator='random',
    #     probability_critical_events=config.normal_priority_job_probability,
    #     name='random_on_normal_benchmark',
    # )

    if config.shutdown_on_complete:
        system('shutdown /h')

    if config.show_plots:
        plt_show()


if __name__ == '__main__':
    main()
