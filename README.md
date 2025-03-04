# EvolveNet
EvolveNet: A self-evolving neural network for limited hardware. Uses evolutionary learning, compression, and continuous adaptation to optimize architecture and performance. A step toward more autonomous, resource-efficient AI.

Repository Structure

— README.md                # This file: project description, instructions, and structure.
— LICENSE                  # License file (e.g., MIT, Apache 2.0, etc.).
— .gitignore               # Configuration to ignore unnecessary files.
— docs/                    # Project documentation.
    — architecture.md      # Description of the architecture and concepts.
    — design_decisions.md  # Design decisions and justifications.
— experiments/             # Experiment scripts and data.
    — data/                # Datasets, samples, and logs.
    — scripts/             # Scripts to run experiments and evaluations.
— notebooks/               # Jupyter Notebooks for prototyping and result visualization.
— src/                     # Main source code.
    — python/              # Initial implementation and prototyping in Python (PyTorch).
        — models/          # Model definitions and evolutionary algorithms.
        — training/        # Training scripts, evolutionary cycles, and evaluations.
        — utils/           # Utility functions (data loaders, metrics, etc.).
    — cpp/                 # C/C++ code for optimizations and performance implementations.
        — include/         # Header files.
        — src/             # Implementation of C/C++ modules.
        — tests/           # Unit tests for the C/C++ code.
— tools/                   # Auxiliary tools (scripts, log analysis, etc.).

Main Scripts
src/python/models/
Contains classes and functions that define the network architecture, including evolutionary modules and adaptive mechanisms.

src/python/training/
Contains scripts responsible for training the network and executing evolutionary cycles, adjusting the network according to performance criteria and hardware constraints.

src/python/utils/
Contains utility functions for data loading, metric evaluation, and other supporting operations.

experiments/scripts/
Contains scripts for running experiments, generating logs and plots, and facilitating the comparative analysis of results.

notebooks/
Contains Jupyter Notebooks for interactive prototyping, testing, and visualizing results.
