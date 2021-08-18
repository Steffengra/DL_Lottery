
from matplotlib.pyplot import (
    show as plt_show,
)
from os.path import (
    join,
)
from os import (
    system,
)

from config import Config
from runner import Runner


def main():
    config = Config()
    runner = Runner()

    shutdown_on_complete = False
    # shutdown_on_complete = True

    # initial training
    runner.train_critical_events()
    # random allocation baseline
    runner.test_critical_events(
        allocator='random',
        name='_random'
    )
    # trained network results
    runner.test_critical_events(
        allocator='pretrained',  # 'random', 'pretrained'
        policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events'),
        policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events.gzip'),
        name='_trained_1'
    )

    # train on random without anchoring -> should ruin performance
    runner.train_on_random_data(
        value_network_path=join(config.model_path, 'critic_allocation_training_critical_events'),
        policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events'),
        name='_no_anchoring'
    )
    # test if performance is ruined
    runner.test_critical_events(
        allocator='pretrained',
        policy_network_path=join(config.model_path, 'actor_allocation_training_random_data'),
        policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events.gzip'),
        name='_trained_random_no_anchoring'
    )

    # train on random with anchoring -> performance should be preserved
    runner.train_on_random_data(
        value_network_path=join(config.model_path, 'critic_allocation_training_critical_events'),
        policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events'),
        anchoring_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events.gzip'),
        name='_anchoring'
    )
    # test if performance is preserved
    runner.test_critical_events(
        allocator='pretrained',
        policy_network_path=join(config.model_path, 'actor_allocation_training_random_data'),
        policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events.gzip'),
        name='_trained_random_anchoring'
    )

    if shutdown_on_complete:
        system('shutdown /h')

    plt_show()


if __name__ == '__main__':
    main()
