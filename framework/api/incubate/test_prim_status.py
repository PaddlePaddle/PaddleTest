"""
test prim status
"""
import sys
import pytest
import paddle
import paddle.incubate.autograd as augograd

sys.path.append("../../utils/")
from interceptor import skip_branch_not_develop


@skip_branch_not_develop
@pytest.mark.api_incubate_autograd_enable_prim
def test_enable_prim():
    """
    test01
    """
    paddle.enable_static()
    assert augograd.prim_enabled() is False
    augograd.enable_prim()
    assert augograd.prim_enabled() is True


@skip_branch_not_develop
@pytest.mark.api_incubate_autograd_disable_prim
def test_disable_prim():
    """
    test02
    """
    paddle.enable_static()
    augograd.enable_prim()
    assert augograd.prim_enabled() is True
    augograd.disable_prim()
    assert augograd.prim_enabled() is False
