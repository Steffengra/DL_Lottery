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


def main():
    config = Config()
    runner = Runner()

    shutdown_on_complete = False
    # shutdown_on_complete = True

    # pre-train off environment from scratch
    runner.train_critical_events(
        name='base',
    )

    # train on environment, anchored
    runner.train_normal(
        name='anchored',
        value_network_path=join(config.model_path, 'critic_allocation_training_critical_events_base'),
        policy_network_path=join(config.model_path, 'actor_allocation_training_critical_events_base'),
        anchoring_parameters_path=join(config.model_path, 'policy_parameters_training_critical_events_base.gzip'),
    )

    runner.config.update_num_steps_per_episode(new_steps_per_episode=2*runner.config.num_steps_per_episode)
    # train on environment, from scratch
    runner.train_normal(
        name='base',
    )

    # train off environment, no rare events, from scratch, benchmark
    runner.train_no_critical_events(
        name='base',
    )

    runner.config.update_num_steps_per_episode(new_steps_per_episode=50_000)
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

    # test pretrained without critical events, benchmark
    runner.test(
        allocator='pretrained',
        probability_critical_events=config.normal_priority_job_probability,
        policy_network_path=join(config.model_path, 'actor_allocation_training_no_critical_events_base'),
        policy_pruning_parameters_path=join(config.model_path, 'policy_parameters_training_no_critical_events_base.gzip'),
        name='normal_prob_priority_pretrained_on_no_critical_events',
    )

    if shutdown_on_complete:
        system('shutdown /h')

    plt_show()


if __name__ == '__main__':
    main()