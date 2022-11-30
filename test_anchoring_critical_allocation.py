
from matplotlib.pyplot import (
    show as plt_show,
)
from os import (
    system,
)
from pathlib import (
    Path,
)

from config import Config
from runner import Runner


def main():
    # TODO: FIX NAMING EVERYWHERE
    def _train_network(
            probability_crit_events: float,
            anchoring_name: None or str = None,
            preload_policy_parameters: bool = False
    ) -> None:

        if anchoring_name is None:
            training_name = f'training_{probability_crit_events}_crit_events_base'
        else:
            training_name = f'training_{probability_crit_events}_crit_events_anchored_{anchoring_name}'

        if preload_policy_parameters:
            training_name = f'{training_name}_pretrained'

        runner.train(
            training_name=training_name,
            probability_critical_events=probability_crit_events,
            policy_anchoring_parameters_path=
                Path(config.model_path, f'policy_parameters_training_{anchoring_name}.gzip')
                if anchoring_name else None,
            policy_network_path=
                Path(config.model_path, f'actor_allocation_training_{anchoring_name}')
                if preload_policy_parameters else None,
        )

    def _test_network(
            network_name: str,
    ) -> None:
        runner.test(
            allocator='pretrained',
            probability_critical_events=config.normal_priority_job_probability,
            policy_network_path=Path(config.model_path, f'actor_allocation_training_{network_name}'),
            name=network_name
        )

    # TRAINING WRAPPERS-------------------------------------------------------------------------------------------------
    def train_on_twenty_percent_crit_events():
        num_episodes_old = runner.config.num_episodes
        runner.config.update_num_episodes(new_num_episodes=2 * runner.config.num_episodes)
        _train_network(
            probability_crit_events=0.2,
        )
        runner.config.update_num_episodes(new_num_episodes=num_episodes_old)

    def train_on_fifty_percent_crit_events():
        num_episodes_old = runner.config.num_episodes
        runner.config.update_num_episodes(new_num_episodes=2*runner.config.num_episodes)
        _train_network(
            probability_crit_events=0.5,
        )
        runner.config.update_num_episodes(new_num_episodes=num_episodes_old)

    def train_on_hundred_percent_crit_events():
        # sum_priority_timeout_old = runner.config.reward_sum_weightings['sum_priority_timeouts']
        # runner.config.reward_sum_weightings['sum_priority_timeouts'] = -10.0
        _train_network(
            probability_crit_events=1.0,
        )
        # runner.config.reward_sum_weightings['sum_priority_timeouts'] = sum_priority_timeout_old

    def train_on_normal_crit_events():
        num_episodes_old = runner.config.num_episodes
        runner.config.update_num_episodes(new_num_episodes=2*runner.config.num_episodes)
        _train_network(
            probability_crit_events=config.normal_priority_job_probability,
        )
        runner.config.update_num_episodes(new_num_episodes=num_episodes_old)

    def train_on_normal_crit_events_anchored_hundred_percent_crit_events():
        # sum_priority_timeout_old = runner.config.reward_sum_weightings['sum_priority_timeouts']
        # runner.config.reward_sum_weightings['sum_priority_timeouts'] = -0.0
        _train_network(
            probability_crit_events=config.normal_priority_job_probability,
            anchoring_name='1.0_crit_events_base',
        )
        # runner.config.reward_sum_weightings['sum_priority_timeouts'] = sum_priority_timeout_old

    def train_on_normal_crit_events_anchored_hundred_percent_crit_events_pretrained():
        # sum_priority_timeout_old = runner.config.reward_sum_weightings['sum_priority_timeouts']
        # runner.config.reward_sum_weightings['sum_priority_timeouts'] = -0.0
        _train_network(
            probability_crit_events=config.normal_priority_job_probability,
            anchoring_name='1.0_crit_events_base',
            preload_policy_parameters=True,
        )
        # runner.config.reward_sum_weightings['sum_priority_timeouts'] = sum_priority_timeout_old

    def train_on_hundred_percent_crit_events_anchored_normal_crit_events_pretrained():
        _train_network(
            probability_crit_events=1.0,
            anchoring_name=f'{config.normal_priority_job_probability}_crit_events_base',
            preload_policy_parameters=True,
        )

    def train_on_no_crit_events_anchored_pretrained():
        runner.train(
            training_name='training_continued_0.0_crit_events_anchored_1.0',
            probability_critical_events=0.0,
            policy_anchoring_parameters_path=Path(config.model_path, 'policy_parameters_training_1.0_crit_events_base.gzip'),
            policy_network_path=Path(config.model_path, f'actor_allocation_training_{config.normal_priority_job_probability}_crit_events_anchored_1.0_crit_events:base_pretrained'),
        )

    def train_on_no_crit_events_pretrained_twenty_percent():
        runner.train(
            training_name='training_continued_0.0_crit_events_pretrained_twenty',
            probability_critical_events=0.0,
            policy_anchoring_parameters_path=None,
            policy_network_path=Path(config.model_path, 'actor_allocation_training_0.2_crit_events_base'),
        )

    # TESTING WRAPPERS--------------------------------------------------------------------------------------------------
    def test_random_scheduler():
        runner.test(
            allocator='random',
            probability_critical_events=config.normal_priority_job_probability,
            name='random_scheduler',
        )

    def test_twenty_percent_crit_events_scheduler_on_normal_crit_events():
        _test_network(network_name='0.2_crit_events_base')

    def test_fifty_percent_crit_events_scheduler_on_normal_crit_events():
        _test_network(network_name='0.5_crit_events_base')

    def test_hundred_percent_crit_events_scheduler_on_normal_crit_events():
        _test_network(network_name='1.0_crit_events_base')

    def test_normal_crit_events_scheduler_on_normal_crit_events():
        _test_network(network_name=f'{config.normal_priority_job_probability}_crit_events_base')

    def test_normal_crit_events_anchored_hundred_percent_crit_events_scheduler_on_normal_crit_events():
        _test_network(network_name=f'{config.normal_priority_job_probability}_crit_events_anchored_1.0_crit_events_base')

    def test_normal_crit_events_anchored_hundred_percent_crit_events_pretrained_scheduler_on_normal_crit_events():
        _test_network(network_name=f'{config.normal_priority_job_probability}_crit_events_anchored_1.0_crit_events_base_pretrained')

    def test_hundred_percent_crit_events_anchored_normal_crit_events_pretrained__scheduler_on_normal_crit_events():
        _test_network(network_name=f'1.0_crit_events_anchored_{config.normal_priority_job_probability}_crit_events_base_pretrained')

    def test_no_crit_events_continued_anchored():
        _test_network(network_name='continued_0.0_crit_events_anchored_1.0')

    def test_no_crit_events_continued_twenty():
        _test_network(network_name='continued_0.0_crit_events_pretrained_twenty')

    def train():
        pass
        # train_on_normal_crit_events()
        # train_on_twenty_percent_crit_events()
        # train_on_fifty_percent_crit_events()
        # train_on_hundred_percent_crit_events()

        # train_on_normal_crit_events_anchored_hundred_percent_crit_events()
        # train_on_normal_crit_events_anchored_hundred_percent_crit_events_pretrained()
        # train_on_hundred_percent_crit_events_anchored_normal_crit_events_pretrained()

        # train_on_no_crit_events_anchored_pretrained()
        # train_on_no_crit_events_pretrained_twenty_percent()

    def test():
        pass
        runner.config.update_num_episodes(new_num_episodes=5)
        runner.config.update_num_steps_per_episode(new_steps_per_episode=200_000)

        # test_random_scheduler()
        test_normal_crit_events_scheduler_on_normal_crit_events()
        # test_twenty_percent_crit_events_scheduler_on_normal_crit_events()
        # test_fifty_percent_crit_events_scheduler_on_normal_crit_events()
        # test_hundred_percent_crit_events_scheduler_on_normal_crit_events()

        # test_normal_crit_events_anchored_hundred_percent_crit_events_scheduler_on_normal_crit_events()
        # test_normal_crit_events_anchored_hundred_percent_crit_events_pretrained_scheduler_on_normal_crit_events()
        # test_hundred_percent_crit_events_anchored_normal_crit_events_pretrained__scheduler_on_normal_crit_events()

        # test_no_crit_events_continued_anchored()
        # test_no_crit_events_continued_twenty()

    config = Config()
    runner = Runner()

    train()
    test()

    if config.shutdown_on_complete:
        system('shutdown /h')

    if config.show_plots:
        plt_show()


if __name__ == '__main__':
    main()
