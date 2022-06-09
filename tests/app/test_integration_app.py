import pytest

# Can use @pytest.mark.integration marker but preference is to use
# test_integration prefixed files for integration.
# So, marking the entire module as integration, as it uses the autouse fixture
pytestmark = pytest.mark.integration


def test_fake_integration():
    pass
