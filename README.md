# Monte Carlo Tree Search for Checkers

## Project Description
This project implements Monte Carlo Tree Search for the game of Checkers using C++. Two versions of the algorithm are provided:

1. **CPU-Based MCTS**: A standard implementation running on the CPU.
2. **GPU-Based MCTS with CUDA**: A parallelized version leveraging CUDA for acceleration, employing **leaf parallelization** to speed up simulations.

Monte Carlo Tree Search is a heuristic search algorithm used for decision-making in game AI. The goal of this project is to compare the efficiency and performance of the CPU-based and GPU-based MCTS implementations.

## Features
- **Checkers Game Implementation**: Basic rules of checkers are implemented for the AI to interact with.
- **MCTS Algorithm**: Includes selection, expansion, simulation, and backpropagation phases.
- **CPU-Based Approach**: Runs one simulation at the time.
- **GPU-Based Approach**: Utilizes leaf parallelization, allowing multiple simulations to be executed in parallel on the GPU.

## Requirements
### General Dependencies
- C++17 or later

### GPU Version
- NVIDIA GPU with CUDA support
- CUDA Toolkit

## Performance Evaluation
TODO
<!--
| Method        | Time per Simulation | Nodes Evaluated | Speedup Factor |
|--------------|--------------------|----------------|---------------|
| CPU (Multi-threaded) | TBD ms | TBD | 1x |
| GPU (Leaf Parallelization) | TBD ms | TBD | TBDx |
-->

## Future Improvements
- Implement additional parallelization strategies (e.g., root parallelization, tree parallelization).
- Experiment with different game evaluation heuristics.
