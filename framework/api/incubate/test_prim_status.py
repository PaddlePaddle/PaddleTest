"""
test prim_status
"""
import sys
import pytest
import paddle
from paddle.incubate.autograd import enable_prim, prim_enabled, disable_prim

sys.path.append("../../utils/")
from interceptor import skip_branch_not_develop


@pytest.mark.api_incubate_autograd_enable_prim
def test_enable_prim():
    """
    test_enable_prim
    """
    paddle.enable_static()
    assert prim_enabled() is False
    enable_prim()
    assert prim_enabled() is True


@pytest.mark.api_incubate_autograd_disable_prim
def test_disable_prim():
    """
    test_disable_prim
    """
    paddle.enable_static()
    enable_prim()
    assert prim_enabled() is True
    disable_prim()
    assert prim_enabled() is False
