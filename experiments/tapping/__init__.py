try:
    import pytest
    # The following code ensures that `assert` calls in `experiment.py`
    # are rewritten by pytest such that they display more useful information
    # on failure.
    pytest.register_assert_rewrite("dallinger_experiment.experiment")
except ImportError:
    pass  # pytest not installed (e.g. provisioning venv); assert rewriting skipped
