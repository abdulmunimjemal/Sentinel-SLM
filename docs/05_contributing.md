# Contributing to Sentinel-SLM

Thank you for your interest in contributing to Sentinel-SLM! We welcome contributions from the community to make AI safety more accessible and robust.

## Getting Started

1.  **Fork the repository** and clone it locally.
2.  **Set up your environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pre-commit install  # Install hooks
    ```

## Development Standards

### Code Quality
We use `pre-commit` to enforce code quality:
*   **Linting**: `ruff` is used for Python linting.
*   **Notebooks**: `nbstripout` automatically clears notebook outputs to keep diffs small.
*   **Formatting**: Standard Python formatting rules apply.

### Code Structure
*   `src/sentinel`: Core library code.
*   `scripts/`: Utility scripts for data prep and training.
*   `notebooks/`: Exploratory work and training logs.
*   `docs/`: Documentation.

## Pull Request Process

1.  Create a new branch for your feature (`git checkout -b feature/amazing-feature`).
2.  Make your changes.
3.  Run tests and checks locally (`make check`).
4.  Commit your changes (`git commit -m "feat: add amazing feature"`).
5.  Push to the branch (`git push origin feature/amazing-feature`).
6.  Open a Pull Request.

## License
By contributing, you agree that your contributions will be licensed under the MIT License.


---

[← Previous: Training Results](04_training_results.md) | [Home: Introduction](00_introduction.md)
