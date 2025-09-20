import pytest

@pytest.fixture
def setup_environment():
    # Set up any necessary environment or parameters for physics loss tests
    pass

@pytest.fixture(params=[1, 4, 10, 20, 40])
def batch_size(request):
    return request.param