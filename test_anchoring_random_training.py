
from matplotlib.pyplot import (
    show as plt_show,
)
from os.path import (
    dirname,
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

    # shutdown_on_complete = False
    shutdown_on_complete = True

    runner.train_critical_events()

    # runner.test_critical_events(
    #     allocator='random'
    # )
    # runner.test_critical_events(
    #     allocator='pretrained',  # 'random', 'pretrained'
    #     policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events'),
    #     policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events.gzip')
    # )

    # runner.train_on_random_data()
    # runner.train_on_random_data(
    #     value_network_path=join(config.model_path, 'critic_allocation_training_critical_events'),
    #     policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events'),
    # )
    runner.train_on_random_data(
        value_network_path=join(config.model_path, 'critic_allocation_training_critical_events'),
        policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events'),
        anchoring_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events.gzip'),
    )

    runner.test_critical_events(
        allocator='pretrained',
        policy_network_path=join(config.model_path, 'actor_allocation_training_random_data'),
        policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events.gzip')
    )

    if shutdown_on_complete:
        system('shutdown /h')

    plt_show()


if __name__ == '__main__':
    main()
