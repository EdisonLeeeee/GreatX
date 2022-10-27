# Contributing to GreatX

If you are interested in contributing to GreatX, your contributions will likely fall into one of the following two categories:

1. You want to implement a new feature:
   - In general, we accept any features as long as they fit the scope of this package. If you are unsure about this or need help on the design/implementation of your feature, post about it in an issue.
2. You want to fix a bug:
   - Feel free to send a Pull Request any time you encounter a bug. Please provide a clear and concise description of what the bug was. If you are unsure about if this is a bug at all or how to fix, post about it in an issue.

Once you finish implementing a feature or bug-fix, please send a Pull Request to https://github.com/EdisonLeeeee/GreatX.

## Developing GreatX

To develop GreatX on your machine, here are some tips:

1. Uninstall all existing GreatX installations:

   ```bash
   pip uninstall greatx
   pip uninstall greatx  # run this command twice
   ```

2. Clone a copy of GreatX from source:

   ```bash
   git clone https://github.com/EdisonLeeeee/GreatX
   cd GreatX
   ```

3. If you already cloned GreatX from source, update it:

   ```bash
   git pull
   ```

4. Install GreatX in editable mode:

   ```bash
   pip install -e ".[dev,full]"
   ```

   This mode will symlink the Python files from the current local source tree into the Python install. Hence, if you modify a Python file, you do not need to reinstall GreatX again and again.

5. (TODO) Ensure that you have a working GreatX installation by running the entire test suite with

   ```bash
   pytest
   ```

6. Install pre-commit hooks:

   ```bash
    pre-commit install
   ```

## Unit Testing (TODO)

The GreatX testing suite is located under `test/`.
Run the entire test suite with

```bash
pytest
```

or test individual files via, _e.g._, `pytest test/utils/test_convert.py`.

## Building Documentation

To build the documentation:

1. [Build and install](#developing-GreatX) GreatX from source.
2. Install [Sphinx](https://www.sphinx-doc.org/en/master/) theme via
   ```bash
   pip install git+https://github.com/pyg-team/pyg_sphinx_theme.git
   ```
3. Generate the documentation via:
   ```bash
   cd docs
   make html
   ```

The documentation is now available to view by opening `docs/build/html/index.html`.
