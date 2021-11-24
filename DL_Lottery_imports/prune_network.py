
from numpy import (
    ndarray,
    max as np_max,
    percentile,
)


def prune_network(
        network,
        training_parameters_initial: list,
        magnitude_percentile,
        magnitude_increase_percentile,
) -> list:
    """Identify less important nodes within layers by criterions"""
    # TODO: Currently there is no explicit restriction to max nodes pruned per layer
    training_parameters_final = network.get_weights()

    parameter_magnitude = [abs(training_parameters_final[layer])
                           for layer in range(len(training_parameters_final))]

    parameter_magnitude_increase = [abs(training_parameters_final[layer]) - abs(training_parameters_initial[layer])
                                    for layer in range(len(training_parameters_final))]

    criterion_magnitude = [
        parameter_magnitude[layer] < percentile(parameter_magnitude[layer], magnitude_percentile)
        for layer in range(len(parameter_magnitude))
    ]

    criterion_magnitude_increase = [
        parameter_magnitude_increase[layer] < percentile(parameter_magnitude_increase[layer], magnitude_increase_percentile)
        for layer in range(len(parameter_magnitude_increase))
    ]

    parameters_total = 0
    parameters_changed = 0
    parameters_new = training_parameters_final.copy()
    # this loop would be easier with numpy functions, but this is easy to bugfix
    for layer_id, layer in enumerate(training_parameters_final):
        for node_id, node in enumerate(layer):
            if type(node) == ndarray:
                for parameter_id, parameter in enumerate(node):
                    parameters_total += 1
                    if (
                            criterion_magnitude[layer_id][node_id][parameter_id]
                            |  # bitwise or
                            criterion_magnitude_increase[layer_id][node_id][parameter_id]
                    ):
                        parameters_new[layer_id][node_id][parameter_id] = 0.0
                        parameters_changed += 1
            else:  # case: layer has single parameter -> not a list
                parameters_total += 1
                # if there is just a single node we would not want to prune it

    print(f'{parameters_changed / parameters_total:.2%} nodes pruned')

    return parameters_new
