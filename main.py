# 2020 - github.com/ufukty
# See the LICENSE file

from typing import List, Dict, NewType, Set, Tuple
from enum import Enum
from random import uniform
from datetime import datetime
import numpy as np

import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation as anim


class Graph(Enum):
    Scale_Free = 0
    Random = 1


# Number of nodes
graph_type = Graph.Scale_Free
N = 5000
total_iteration = 250

# Required for scale-free network
# Probability for adding a new node connected to an existing node
# chosen randomly according to the in-degree distribution.
alpha = 0.98

# Required for scale-free network
# Probability for adding an edge between two existing nodes. One
# existing node is chosen randomly according the in-degree
# distribution and the other chosen randomly according to the
# out-degree distribution.
beta = 0.01

# Required for scale-free network
# Probability for adding a new node connected to an existing node
# chosen randomly according to the out-degree distribution.
gamma = 1.0 - alpha - beta

# Required for random network
# Probability of making links for each match
probability = 0.05

# Animation parameters
enable_animation = True
fps = 25
node_size = 8
title_style = {
    "fontsize": 8
}

# ------------------------------------------------------------- #
# Data
# ------------------------------------------------------------- #


class Stage(Enum):
    Initial = 1
    Knowledge_Awareness = 2
    Persuasion = 3
    Decision = 4
    Decision_Accept = 5
    Decision_Reject = -1
    Implementation = 6
    Confirmation = 7
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __str__(self):
        return {
            1: "Initial",
            2: "Knowledge_Awareness",
            3: "Persuasion",
            4: "Decision",
            5: "Decision_Accept",
            -1: "Decision_Reject",
            6: "Implementation",
            7: "Confirmation",
        }[self.value]


class Phase(Enum):
    Innovators = 1
    Early_Adopters = 2
    Early_Majority = 3
    Late_Majority = 4
    Laggards = 5
    
    def __str__(self):
        return {
            1: "Innovators",
            2: "Early Adopters",
            3: "Early Majority",
            4: "Late Majority",
            5: "Laggards",
        }[self.value]


phase_tresholds = {
    Phase.Innovators: 0.025,
    Phase.Early_Adopters: 0.16,
    Phase.Early_Majority: 0.50,
    Phase.Late_Majority: 0.84,
    Phase.Laggards: 1.00,
}

Time = NewType("Time", int)
NodeIndex = NewType("NodeIndex", int)


class StageList:
    
    _current_iter: Time
    _stage_list: List[Stage]
    
    # Update history structure:
    # _event_history = {
    #     (Stage.Initial, Stage.Knowledge_Awareness): {
    #         0: [3, 5, 8, 2, 9],
    #         4: [3, 5, 8, 2, 9],
    #         7: [3, 5, 8, 2, 9],
    #        ...
    #     },
    #    ...
    # }
    _event_history: Dict[Tuple[Stage, Stage], Dict[Time, List[NodeIndex]]]
    
    # Structure:
    # _distribution_stats_per_round = {
    #     Stage.Initial: [50, 49, 48, ...],
    #     Stage.Knowledge_Awareness: [0, 1, 2, ...],
    #     ...
    # }
    _distribution_stats: Dict[Stage, List[int]]
    _update_stats: Dict[Stage, List[int]]
    
    # Structure:
    # _distribution_stats_per_round = {
    #     Stage.Initial: [[], [1, 2, 2, 2, 3, ...], [2, 3], [1], ...],
    #     Stage.Knowledge_Awareness: [[], [1, 2, 2, 2, 3, ...], [2, 3], [1], ...],
    #     ...
    # }
    _degree_stats: Dict[Stage, List[List[int]]]
    
    _reached_adoption_phases: Dict[Phase, Dict[str, Time]]
    
    def __init__(self, G: nx.Graph):
        
        self._G = G
        self._N = len(G.nodes())
        
        self._stage_list = [Stage.Initial] * self._N
        self._event_history = {}
        self._degree_stats = {}
        self._distribution_stats = {}
        self._update_stats = {}
        
        self._reached_adoption_phases = {}
        for phase in Phase:
            self._reached_adoption_phases.update({phase: {
                "start": None,
                "end": None
            }})
        self._reached_adoption_phases[Phase.Innovators]["start"] = 0
        
        for stage in Stage:
            # Pass current iteration and put a 0 for next iteration
            self._update_stats.update({stage: [0, 0]})
            # For current iteration
            self._distribution_stats.update({stage: [0]})
            # Pass current iteration and put a [] for next iteration
            self._degree_stats.update({stage: [[], []]})
        
        self._distribution_stats[Stage.Initial] = [self._N] # For current iteration
        
        self._current_iter = Time(1)
        self._current_phase = Phase.Innovators
        
        return
    
    def _add_update_to_event_history(self, node: NodeIndex, update_from: Stage, update_to: Stage):
        # Allocate space if it isn't done already
        if (update_from, update_to) not in self._event_history:
            self._event_history.update({(update_from, update_to): {}})
        if self._current_iter not in self._event_history[(update_from, update_to)]:
            self._event_history[(update_from, update_to)].update({self._current_iter: []})
        # Save to history
        self._event_history[(update_from, update_to)][self._current_iter].append(node)
        self._update_stats[update_from][self._current_iter] -= 1
        self._update_stats[update_to][self._current_iter] += 1
        self._degree_stats[update_to][self._current_iter].append(self._G.degree(node))
        return
    
    def proceed_to_next_iteration(self):
        
        # Calculate distributions of current iteration and save them
        for stage, update_stat in self._update_stats.items():
            previous_iter_dists = self._distribution_stats[stage][self._current_iter - 1]
            current_iter_dists = previous_iter_dists + update_stat[self._current_iter]
            self._distribution_stats[stage].append(current_iter_dists)
        
        # Check if any of adopting phases has been reached yet
        confirmed_nodes = self._distribution_stats[Stage.Confirmation][-1]
        rate_of_confirmation = float(confirmed_nodes) / float(self._N)
        
        if rate_of_confirmation > phase_tresholds[self._current_phase]:
            self._reached_adoption_phases[self._current_phase]["end"] = self._current_iter
            self._current_phase = Phase(self._current_phase.value + 1)
            self._reached_adoption_phases[self._current_phase]["start"] = self._current_iter
        
        # Proceed to next iteration
        self._current_iter += 1
        
        # Allocate memory for next iteration
        for stage in Stage:
            self._update_stats[stage].append(0)
            self._degree_stats[stage].append([])
        
        return
    
    def update_node(self, node: NodeIndex, update_to: Stage):
        current_stage = self._stage_list[node]
        self._stage_list[node] = update_to
        self._add_update_to_event_history(node, current_stage, update_to)
    
    def get_event_history(self,
                            update: Tuple[Stage, Stage] = None,
                            iteration_offset: int = 0) -> List[NodeIndex]:
        if update != None:
            try:
                return set(self._event_history[update][self._current_iter + iteration_offset])
            except KeyError:
                return set()
        else:
            ret_set = set()
            for update_history in self._event_history.values():
                for iteration, node_list in update_history.items():
                    ret_set.update(node_list)
            return ret_set
    
    def stage_of(self, node: NodeIndex):
        return self._stage_list[node]
    
    def distribution_stats(self, last_iteration_only: bool = False) -> Dict[Stage, List[int]]:
        if not last_iteration_only:
            return self._distribution_stats
        last_stats = {}
        for stage, dists in self._distribution_stats.items():
            last_stats.update({stage: dists[-1]})
        return last_stats
    
    def update_stats(self,) -> Dict[Stage, Dict[Time, int]]:
        return self._update_stats
    
    def degree_stats(self, last_iteration_only: bool = False) -> Dict[Stage, List[List[int]]]:
        if not last_iteration_only:
            return self._degree_stats
        last_stats = {}
        for stage, degree_lists in self._degree_stats.items():
            last_stats.update({stage: degree_lists[-2]})
        return last_stats
    
    def get_iteration(self):
        return self._current_iter
    
    def reached_adoption_phases(self) -> Dict[Phase, Dict[str, Time]]:
        return self._reached_adoption_phases


# ------------------------------------------------------------- #
# Drawing related
# ------------------------------------------------------------- #

Color = NewType("Color", str)

node_colors = {
    Stage.Initial: Color("#CFCFCF"),
    Stage.Knowledge_Awareness: Color("#B6D9EA"),
    Stage.Persuasion: Color("#5BBBEA"),
    Stage.Decision: Color("#008FD4"),
    Stage.Decision_Accept: Color("#B1DE7F"),
    Stage.Decision_Reject: Color("#E38181"),
    Stage.Implementation: Color("#8BE426"),
    Stage.Confirmation: Color("#519900"),
}


def draw_nodes(
    ax: matplotlib.axes.Axes,
    z_order: int,
    pos: list,
    node_list: List[NodeIndex],
    color_list: List[Color],
):
    scatter = ax.scatter(
        [pos[node][0] for node in node_list],
        y=[pos[node][1] for node in node_list],
        s=node_size,
        c=color_list,
        marker="o",
        cmap=None,
        vmin=None,
        vmax=None,
        alpha=None,
        linewidths=None,
        edgecolors=None,
        label=None,
    )
    scatter.set_zorder(z_order)
    return scatter


def draw_stage_distribution_pie_chart(ax: matplotlib.axes.Axes, stage_list: StageList):
    dist_stats = stage_list.distribution_stats(last_iteration_only=True)
    dist_stats = {index: value for index, value in dist_stats.items() if value > 0.0}
    colors = [node_colors[stage] for stage in dist_stats.keys()]
    
    ax.clear()
    
    ax.pie(
        dist_stats.values(),
        labels=dist_stats.keys(),
        colors=colors,
        startangle=90,
    )


def draw_stage_distribution_plot(ax: matplotlib.axes.Axes, stage_list: StageList):
    iteration = stage_list.get_iteration()
    dist_stats = stage_list.distribution_stats()
    
    ax.clear()
    ax.set_title("Stage Distribution", title_style, loc="left", pad=-0.5)
    ax.set_xlabel("t", loc="right", labelpad=-10)
    ax.set_ylabel("number of nodes", labelpad=-0.3)
    
    for stage in Stage:
        ax.plot(
            list(range(iteration)), # until current iteration (which is empty)
            dist_stats[stage],
            color=node_colors[stage],
        )


def draw_adoption_curve_plot(ax: matplotlib.axes.Axes, stage_list: StageList):
    iteration = stage_list.get_iteration()
    dist_stats = stage_list.distribution_stats()
    update_stats = stage_list.update_stats()
    adoption_phases = stage_list.reached_adoption_phases()
    max_y = max(dist_stats[Stage.Confirmation])
    
    ax.clear()
    ax.set_title("Adoption Curve", title_style, loc="left", pad=-0.5)
    ax.set_xlabel("t", loc="right", labelpad=-10)
    ax.set_ylabel("number of nodes", labelpad=-0.3)
    
    # Updates
    ax.plot(
        list(range(iteration)), # until current iteration (which is empty)
        update_stats[Stage.Confirmation][:-1],
        color=node_colors[Stage.Confirmation],
        dashes=(2, 1)
    )
    
    # Distribution
    ax.plot(
        list(range(iteration)), # until current iteration (which is empty)
        dist_stats[Stage.Confirmation],
        color=node_colors[Stage.Confirmation],
    )
    
    for phase, iters in adoption_phases.items():
        start, end = iters["start"], iters["end"]
        if start is not None:
            ax.text(
                start + 1,
                float(max_y) * (1 - phase.value * 0.1),
                phase,
                rotation="0",
                color="#888888"
            )
        if end is not None:
            ax.axvline(x=end, color="#888888", linewidth=1, dashes=(2, 1))


def draw_degrees_of_updated_nodes(ax: matplotlib.axes.Axes, stage_list: StageList):
    iteration = stage_list.get_iteration()
    degree_stats = stage_list.degree_stats(last_iteration_only=True)
    
    scatted_at_least_one_point = False
    
    for stage in [Stage.Decision_Accept, Stage.Implementation, Stage.Confirmation]:
        if len(degree_stats[stage]) > 0:
            scatter = ax.scatter(
                [iteration - 1 for _ in degree_stats[stage]],
                degree_stats[stage],
                color=node_colors[stage],
                s=node_size,
            )
            scatter.set_zorder(11)
            scatted_at_least_one_point = True
    
    if not scatted_at_least_one_point:
        # If there is no new data for current iteration,
        # draw invisible node to run auto-sizing
        scatter = ax.scatter(iteration - 1, 0, color="#ffffff", s=node_size)
        scatter.set_zorder(10)
    
    # return scatter
    return scatter


# ------------------------------------------------------------- #
# Updaters
# ------------------------------------------------------------- #


def update_all_nodes_from_initial_stage(G: nx.Graph, stage_list: StageList):
    
    last_round_updates = stage_list.get_event_history(
        update=(Stage.Initial, Stage.Knowledge_Awareness), iteration_offset=-1
    )
    
    for node in last_round_updates:
        for neighbor in G.neighbors(node):
            if stage_list.stage_of(neighbor) == Stage.Initial:
                stage_list.update_node(node=neighbor, update_to=Stage.Knowledge_Awareness)


def update_all_nodes_from_knowledge_awareness(G: nx.Graph, stage_list: StageList):
    # Get the history of all awereness updates that happened 3 iterations ago
    history = stage_list.get_event_history(
        update=(Stage.Initial, Stage.Knowledge_Awareness),
        iteration_offset=-20,
    )
    for node in history:
        if stage_list.stage_of(node=node) == Stage.Knowledge_Awareness:
            stage_list.update_node(node=node, update_to=Stage.Persuasion)


def update_all_nodes_from_persuation(G: nx.Graph, stage_list: StageList):
    
    for node in G.nodes():
        if stage_list.stage_of(node=node) == Stage.Persuasion:
            # Activation caused by inner reasons ( < %5 )
            if uniform(0.0, 1.0) < 0.05:
                stage_list.update_node(node=node, update_to=Stage.Decision)
            
            # Activation caused by neighbors advices
            else:
                neighbors = list(G.neighbors(node))
                awered_neighbors = [
                    neighbor for neighbor in neighbors
                    if stage_list.stage_of(neighbor) == Stage.Knowledge_Awareness
                ]
                if len(awered_neighbors) >= 0.5 * len(neighbors):
                    stage_list.update_node(node=node, update_to=Stage.Decision)


def update_all_nodes_from_decision(G: nx.Graph, stage_list: StageList):
    
    for node in G.nodes():
        if stage_list.stage_of(node=node) == Stage.Decision:
            decision = uniform(0.0, 1.0)
            if decision < 0.8:
                stage_list.update_node(node=node, update_to=Stage.Decision_Accept)
            else:
                stage_list.update_node(node=node, update_to=Stage.Decision_Reject)


def update_all_nodes_from_decision_accept(G: nx.Graph, stage_list: StageList):
    
    for node in G.nodes():
        if stage_list.stage_of(node=node) == Stage.Decision_Accept:
            neighbors = list(G.neighbors(node))
            activated_neighbors = [
                neighbor for neighbor in neighbors
                if stage_list.stage_of(neighbor) >= Stage.Decision_Accept
            ]
            if len(activated_neighbors) >= 0.8 * len(neighbors):
                stage_list.update_node(node=node, update_to=Stage.Implementation)


def update_all_nodes_from_decision_reject(G: nx.Graph, stage_list: StageList):
    
    for node in G.nodes():
        if stage_list.stage_of(node=node) == Stage.Decision_Reject:
            if uniform(0.0, 1.0) < 0.2:
                stage_list.update_node(node=node, update_to=Stage.Persuasion)


def update_all_nodes_from_implementation(G: nx.Graph, stage_list: StageList):
    
    for node in G.nodes():
        if stage_list.stage_of(node=node) == Stage.Implementation:
            if uniform(0.0, 1.0) < 0.6:
                stage_list.update_node(node=node, update_to=Stage.Confirmation)


def update_network(
    frame,
    ax_stage_dis_pie,
    ax_stage_dis_plot,
    ax_adoption_curve,
    ax_degree_plo,
    ax_graph,
    G: nx.Graph,
    pos: list,
    stage_list: StageList,
):
    
    print("#{:<5}".format(frame), end=" ")
    
    # Inject the innovation at first round
    if frame == 0:
        random_node = int(uniform(0.0, 1.0) * (len(G.nodes()) - 1))
        stage_list.update_node(node=random_node, update_to=Stage.Knowledge_Awareness)
    
    # At following rounds, run simulation
    else:
        # Run all updaters at reverse order for avoid
        # 1 node to pass all stages in 1 iteration
        update_all_nodes_from_implementation(G, stage_list)
        update_all_nodes_from_decision_reject(G, stage_list)
        update_all_nodes_from_decision_accept(G, stage_list)
        update_all_nodes_from_decision(G, stage_list)
        update_all_nodes_from_persuation(G, stage_list)
        update_all_nodes_from_knowledge_awareness(G, stage_list)
        update_all_nodes_from_initial_stage(G, stage_list)
    
    # Save stats for current iteration and proceed to next iteration
    stage_list.proceed_to_next_iteration()
    print(stage_list.distribution_stats(last_iteration_only=True).values(), end=" ")
    
    node_list = stage_list.get_event_history(iteration_offset=0)
    color_list = [node_colors[stage_list.stage_of(node=node)] for node in node_list]
    
    if enable_animation:
        network_scatter = draw_nodes(
            ax=ax_graph, z_order=frame + 10, pos=pos, node_list=node_list, color_list=color_list
        )
        draw_stage_distribution_pie_chart(ax=ax_stage_dis_pie, stage_list=stage_list)
        draw_stage_distribution_plot(ax=ax_stage_dis_plot, stage_list=stage_list)
        draw_adoption_curve_plot(ax=ax_adoption_curve, stage_list=stage_list)
        degree_scatter = draw_degrees_of_updated_nodes(ax=ax_degree_plo, stage_list=stage_list)
        
        print("passing artisans")
        return [network_scatter, degree_scatter]
    else:
        print() # new-line
        return []


# ------------------------------------------------------------- #
# Main
# ------------------------------------------------------------- #


def merge_matplotlib_axes(fig, axs, start: tuple[int, int], end: tuple[int, int]):
    """
    Example call for merging 6 axes starting from 1,1 to 3,2:
        merge_matplotlib_axes(axs, (1,1), (3,2))
        +-----+-----+-----+
        | 0,0 | 0,1 | 0,2 |
        +-----+-----+-----+
        | 1,0 | 1,1 | 1,2 |
        +-----+-----+-----+
        | 2,0 | 2,1 | 2,2 |
        +-----+-----+-----+
        | 3,0 | 3,1 | 3,2 |
        +-----+-----+-----+
    """
    start_y, start_x = start
    end_y, end_x = end
    # start with starting ax
    grid_spec = axs[start_y][start_x].get_gridspec()
    # remove unused axes
    for y in range(start_y, end_y + 1):
        for x in range(start_x, end_x + 1):
            axs[y][x].remove()
    # for ax in axs[start_y:end_y + 1, start_x:end_x + 1]:
    #     ax.remove()
    # merge them together
    ax_merged = fig.add_subplot(grid_spec[start_y:end_y + 1, start_x:end_x + 1])
    return ax_merged


def main():
    
    # Creating the network
    if graph_type == Graph.Scale_Free:
        G = nx.scale_free_graph(N, alpha=alpha, beta=beta, gamma=gamma).to_undirected()
    elif graph_type == Graph.Random:
        G = nx.erdos_renyi_graph(N, probability)
    
    # Create a list for storing the stages of all nodes
    # stages_of_nodes = [Stage.Initial] * N
    stage_list = StageList(G)
    
    # Save the positions of nodes for use them at each iteration
    pos = nx.spring_layout(G)
    
    # Get the ax and prepare figure
    fig, axs = plt.subplots(figsize=[16, 9], ncols=4, nrows=4)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.95,
        wspace=0,
        hspace=0.2,
    )
    fig.text(
        0.87,
        0.05,
        s="github.com/ufukty/doi",
        color="#888888",
    )
    
    # Axes for small drawings
    ax_degree_dist_plo = axs[0][0]
    ax_stage_dis_pie = axs[0][1]
    
    # Axes for bigger drawings neeeds layout organization
    ax_stage_dis_plot = merge_matplotlib_axes(fig, axs, start=(1, 0), end=(1, 1))
    ax_adoption_curve = merge_matplotlib_axes(fig, axs, start=(2, 0), end=(2, 1))
    ax_degree_plo = merge_matplotlib_axes(fig, axs, start=(3, 0), end=(3, 1))
    ax_graph = merge_matplotlib_axes(fig, axs, start=(0, 2), end=(3, 3))
    
    # Calculations for degree related information
    degree_distribution = nx.degree_histogram(G)
    total_degree = 0
    for p, pk in enumerate(degree_distribution):
        total_degree += p * pk
    avg_deg = float(total_degree) / float(N)
    
    # Initial setup for Network Drawing
    if graph_type == Graph.Scale_Free:
        ax_graph.set_title(
            "N={} <k>={:.2f} a={:.2f}, b={:.2f}, g={:.2f}".format(N, avg_deg, alpha, beta, gamma)
        )
    elif graph_type == Graph.Random:
        ax_graph.set_title("N={} <k>={:.2f} p={:.2f}".format(N, avg_deg, probability))
    ax_graph.axes.set_aspect('equal')
    
    # Turn off axis for pie chart
    ax_stage_dis_pie.axis("off")
    
    # Initial setup for degree distribution plot (static one)
    ax_degree_dist_plo.plot(degree_distribution)
    ax_degree_dist_plo.set_title("Degree Distribution", title_style, loc="left", pad=-0.5)
    ax_degree_dist_plo.set_xlabel("k", loc="right", labelpad=-10)
    ax_degree_dist_plo.set_ylabel("P(k)", labelpad=-0.3)
    
    # Initial setup for degrees of updated nodes plot
    ax_degree_plo.set_title("Degrees of updated nodes", title_style, loc="left", pad=-0.5)
    ax_degree_plo.set_xlabel("t", loc="right", labelpad=-10)
    ax_degree_plo.set_ylabel("degree of node", labelpad=-0.3)
    
    # Add invisible node to start to make auto-scale function in sync with other graphs above
    scatter = ax_degree_plo.scatter(0, 0, color="#ffffff", s=node_size)
    scatter.set_zorder(10)
    
    plt.axis("off")
    
    # Initial drawing of network
    nx.draw_networkx_edges(G, pos=pos, edge_color="#999999", alpha=0.2, width=0.5)
    nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors[Stage.Initial], node_size=node_size)
    
    # Setup animation
    func_animation = anim.FuncAnimation(
        fig,
        update_network,
        init_func=lambda: list(),
        fargs=(
            ax_stage_dis_pie,
            ax_stage_dis_plot,
            ax_adoption_curve,
            ax_degree_plo,
            ax_graph,
            G,
            pos,
            stage_list,
        ),
        frames=total_iteration,
        interval=1,
        blit=True
    )
    
    Writer = anim.writers["ffmpeg"]
    writer = Writer(fps=fps, metadata=dict(artist="Me"), bitrate=16000)
    output_filename = datetime.now().strftime("output/animation %Y.%m.%d %H.%M.%S.mp4")
    func_animation.save(output_filename, writer=writer)


if __name__ == "__main__":
    main()
