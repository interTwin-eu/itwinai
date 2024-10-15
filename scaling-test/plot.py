from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def create_stacked_plot(
    values: np.ndarray, strategy_labels: List, gpu_numbers: List
) -> Tuple[Any, Any]:
    """Creates a stacked plot showing values from 0 to 1, where the given value
    will be placed on the bottom and the complement will be placed on top for
    each value in 'values'. Returns the figure and the axis so that the caller can
    do what they want with it, e.g. save to file, change it or just show it.

    Notes:
        - Assumes that the rows of 'values' correspond to the labels in
            'strategy_labels' sorted alphabetically and that the columns
            correspond to the GPU numbers in 'gpu_numbers' sorted numerically
            in ascending order.
    """
    # assert values.shape[0] == len(strategy_labels)
    # assert values.shape[1] == len(gpu_numbers)
    # assert (values >= 0).all() and (values <= 1).all()

    strategy_labels = sorted(strategy_labels)
    gpu_numbers = sorted(gpu_numbers, key=lambda x: int(x))

    width = 1 / (len(strategy_labels) + 1)
    comp_color = "lightblue"
    comm_color = "lightgreen"
    complements = 1 - values

    x = np.arange(len(gpu_numbers))
    # Create the plot
    fig, ax = plt.subplots()

    # Creating an offset to "center" around zero
    static_offset = len(strategy_labels) / 2 - 0.5
    for strategy_idx in range(len(strategy_labels)):
        dynamic_bar_offset = strategy_idx - static_offset


        # Drawing the stacked bars
        ax.bar(
            x=x + dynamic_bar_offset * width,
            height=values[strategy_idx],
            width=width,
            color=comp_color,
        )
        ax.bar(
            x=x + dynamic_bar_offset * width,
            height=complements[strategy_idx],
            width=width,
            bottom=values[strategy_idx],
            color=comm_color,
        )

        for gpu_idx in range(len(gpu_numbers)):
            # Positioning the labels under the stacks
            if np.isnan(values[strategy_idx, gpu_idx]): 
                continue
            dynamic_label_offset = strategy_idx - static_offset
            ax.text(
                x=x[gpu_idx] + dynamic_label_offset * width,
                y=-0.1,
                s=strategy_labels[strategy_idx],
                ha="center",
                va="top",
                fontsize=10,
                rotation=60,
            )


    ax.set_ylabel("Computation fraction")
    ax.set_title("Computation vs Communication Time by Method")
    ax.set_xticks(x)
    ax.set_xticklabels(gpu_numbers)
    ax.set_ylim(0, 1)

    # Setting the appropriate colors since the legend is manual
    legend_elements = [
        Patch(facecolor=comm_color, label="Communication"),
        Patch(facecolor=comp_color, label="Computation"),
    ]

    # Positioning the legend outside of the plot to not obstruct
    ax.legend(
        handles=legend_elements,
        loc="upper left",  
        bbox_to_anchor=(0.80, 1.22),  
        borderaxespad=0.0,  
    )
    fig.subplots_adjust(bottom=0.25)
    fig.subplots_adjust(top=0.85)
    return fig, ax
