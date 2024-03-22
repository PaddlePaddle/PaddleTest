#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
conftest
"""
import os
import pytest
import allure


def pytest_addoption(parser):
    """pytest addoption"""
    # parser.addoption("--all_dir", type=str, default="dir", help="yaml dir")
    # parser.addoption("--yaml", type=str, default="det", help="yaml path")
    # parser.addoption("--case", type=str, default="Con", help="case name")

    parser.addoption("--title", type=str, default="demo_case", help="title name")
    parser.addoption("--layerfile", type=str, default="diy/layer/demo_case/SIR_252.py", help="sublayer.py path")
    parser.addoption("--testing", type=str, default="testing", help="testing yml path")


@pytest.fixture
def title(request):
    """title"""
    return request.config.getoption("--title")


@pytest.fixture
def layerfile(request):
    """layerfile"""
    return request.config.getoption("--layerfile")


@pytest.fixture
def testing(request):
    """testing"""
    return request.config.getoption("--testing")
