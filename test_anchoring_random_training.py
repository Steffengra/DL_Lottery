
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
    config = Config()
    runner = Runner()

    # initial training
    runner.train_critical_events(
        name='base',
    )
    # random allocation baseline
    runner.test(
        allocator='random',
        probability_critical_events=1.0,
        name='random',
    )
    # trained network results
    runner.test(
        allocator='pretrained',
        probability_critical_events=1.0,
        policy_network_path=Path(config.model_path, 'actor_allocation_training_critical_events_base'),
        policy_pruning_parameters_path=Path(config.model_path, 'policy_parameters_training_critical_events_base.gzip'),
        name='pretrained',
    )

    # train on random without anchoring -> should ruin performance
    runner.train_on_random_data(
        value_network_path=Path(config.model_path, 'critic_allocation_training_critical_events_base'),
        policy_network_path=Path(config.model_path, 'actor_allocation_training_critical_events_base'),
        name='_no_anchoring'
    )
    # test if performance is ruined
    runner.test(
        allocator='pretrained',
        probability_critical_events=1.0,
        policy_network_path=Path(config.model_path, 'actor_allocation_training_random_data_no_anchoring'),
        policy_pruning_parameters_path=Path(config.model_path, 'policy_parameters_training_critical_events.gzip'),
        name='trained_random_no_anchoring',
    )

    # train on random with anchoring -> performance should be preserved
    runner.train_on_random_data(
        value_network_path=Path(config.model_path, 'critic_allocation_training_critical_events'),
        policy_network_path=Path(config.model_path, 'actor_allocation_training_critical_events'),
        anchoring_parameters_path=Path(config.model_path, 'policy_parameters_training_critical_events.gzip'),
        name='_anchored'
    )
    # test if performance is preserved
    runner.test(
        allocator='pretrained',
        probability_critical_events=1.0,
        policy_network_path=Path(config.model_path, 'actor_allocation_training_random_data_anchoring'),
        policy_pruning_parameters_path=Path(config.model_path, 'policy_parameters_training_critical_events.gzip'),
        name='trained_random_anchored'
    )

    if config.shutdown_on_complete:
        system('shutdown /h')

    if config.show_plots:
        plt_show()


if __name__ == '__main__':
    main()
