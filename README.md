# A Basic Diffusion of Innovation Simulation

## Definition

> Diffusion of innovations is a theory that seeks to explain how, why, and at what rate new ideas and technology spread. Everett Rogers, a professor of communication studies, popularized the theory in his book Diffusion of Innovations; the book was first published in 1962, and is now in its fifth edition (2003). Rogers argues that diffusion is the process by which an innovation is communicated over time among the participants in a social system. The origins of the diffusion of innovations theory are varied and span multiple disciplines. [_Wikipedia_](https://en.wikipedia.org/wiki/Diffusion_of_innovations)

## Adoption process

There are 8 stages (states) for each node. Even though stages are pretty much the same with original theory, conditions are custom for this project. Conditions were designed so that the nodes are exposed to various factors throughout the process.

![State diagram which describes the stages used in application and flow between them](images/diagram.png)

## Explanation

Adoption process involves multiple stages for nodes to pass through. Each stages have different rule for upgrade to following stage. Those rules should designed as they are representing internal and external requirements.

For the rules I choose, there is a requirement for each node to wait until certain iteration passed to upgrade its stage from 1st stage to 2nd stage. I designed this rule to represent the delay between hearing and understanding of new things.

Those rules are designed by me and they are not perfect.

## Outcome of the project and comments

In the time window between iteration 25 and 65 (See output #1), hubs are switching from Decision_Accept to Implementation and Confirmation stages. Massive delay caused by the nature of determination. Nodes look their neighbors to see what they are thinking about innovation. And, more neighbors for a node means more time to wait for that node to adopt.

It looks like it follows a skewed S-Curve. Initial part where it accelerates (iterations 0 to 35) is like super-diffusion. But after there, and the most part of process it follows sub-diffusion behavior. 

It is hard to talk about reason but we can say, the turning point is determined by 2 things: limit of market and characteristics of innovation. Limit of market is more obvious than others, because the amount of new nodes to idea to propagate them is much more at former iterations than at the next iterations. So the rate slows down constantly. At the other hand, characteristics of innovation arises from the decision process of nodes. Some stages like (Knowledge_Awareness, decision, implementation for that example) relies on interpersonal factors to update from. But other stages depends on intrapersonal factors (represented by random numbers in this experiment). 

Maybe we think the stages relies on interpersonal factors helps network to adopt much faster because there is no resistence showed by prejudiced individuals.

## Requirements

-   `matplotlib`
-   `networkx`
-   `ffmpeg`
-   `python`

## Output

**Scale-Free network (`alpha=0.98`)**

![Simulation output for scale-free network with 5000 nodes and alpha = 0.98.](images/scale_free_n_5000_a_098.gif)

**Scale-Free network (`alpha=0.50`)**

![Simulation output for scale-free network with 5000 nodes and alpha = 0.50.](images/scale_free_n_5000_a_050.gif)

**Random network (`p=0.05`)**

![Simulation output for random network with 5000 nodes and 0.05 probability of making links between each node match.](images/random_n_500.gif)

## License

GNU General Public License v3.0  
Read the LICENSE file for details
