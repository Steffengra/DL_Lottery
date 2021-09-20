
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
    def train_on_critical():
        runner.train_critical_events(
            name='base',
        )

    def train_on_fifty_percent_critical():
        runner.train_fifty_percent_critical(
            name='base',
        )

    def train_on_normal():
        num_episodes_old = runner.config.num_episodes
        runner.config.update_num_episodes(new_num_episodes=2*runner.config.num_episodes)
        runner.train_normal(
            name='base',
        )
        runner.config.update_num_episodes(new_num_episodes=num_episodes_old)

    def train_on_normal_anchored():
        sum_priority_timeout_old = runner.config.reward_sum_weightings['sum_priority_timeouts']
        runner.config.reward_sum_weightings['sum_priority_timeouts'] = - 0.0
        runner.train_normal(
            name='anchored',
            anchoring_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events_base.gzip'),
        )
        runner.config.reward_sum_weightings['sum_priority_timeouts'] = sum_priority_timeout_old

    def train_on_normal_anchored_pretrained():
        sum_priority_timeout_old = runner.config.reward_sum_weightings['sum_priority_timeouts']
        runner.config.reward_sum_weightings['sum_priority_timeouts'] = - 0.0
        runner.train_normal(
            name='anchored_pretrained',
            value_network_path=join(config.model_path, 'critic_allocation_training_critical_events_base'),
            policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events_base'),
            anchoring_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events_base.gzip'),
        )
        runner.config.reward_sum_weightings['sum_priority_timeouts'] = sum_priority_timeout_old

    def test_critical():
        runner.test(
            allocator='pretrained',
            probability_critical_events=config.normal_priority_job_probability,
            policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events_base'),
            policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events_base.gzip'),
            name='critical'
        )

    def test_fifty_percent_critical():
        runner.test(
            allocator='pretrained',
            probability_critical_events=config.normal_priority_job_probability,
            policy_network_path=join(config.model_path, 'actor_allocation_training_fifty_percent_critical_events_base'),
            policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_fifty_percent_critical_events_base.gzip'),
            name='fifty_percent_critical'
        )

    def test_normal():
        runner.test(
            allocator='pretrained',
            probability_critical_events=config.normal_priority_job_probability,
            policy_network_path=join(config.model_path, 'actor_allocation_training_normal_base'),
            policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_normal_base.gzip'),
            name='baseline'
        )

    def test_normal_anchored():
        runner.test(
            allocator='pretrained',
            probability_critical_events=config.normal_priority_job_probability,
            policy_network_path=join(config.model_path, 'actor_allocation_training_normal_anchored'),
            policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_normal_anchored.gzip'),
            name='anchored'
        )

    def test_normal_anchored_pretrained():
        runner.test(
            allocator='pretrained',
            probability_critical_events=config.normal_priority_job_probability,
            policy_network_path=join(config.model_path, 'actor_allocation_training_normal_anchored_pretrained'),
            policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_normal_anchored_pretrained.gzip'),
            name='anchored_pretrained'
        )

    def test_random_scheduler():
        runner.test(
            allocator='random',
            probability_critical_events=config.normal_priority_job_probability,
            name='random_scheduler',
        )

    config = Config()
    runner = Runner()

    # train_on_normal()
    # train_on_critical()
    train_on_fifty_percent_critical()
    # train_on_normal_anchored()
    # train_on_normal_anchored_pretrained()

    runner.config.update_num_episodes(new_num_episodes=5)
    runner.config.update_num_steps_per_episode(new_steps_per_episode=100_000)

    # test_normal()
    # test_critical()
    test_fifty_percent_critical()
    # test_normal_anchored()
    # test_normal_anchored_pretrained()
    # test_random_scheduler()

    if config.shutdown_on_complete:
        system('shutdown /h')

    if config.show_plots:
        plt_show()


if __name__ == '__main__':
    main()
