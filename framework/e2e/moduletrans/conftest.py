#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
conftest
"""
import os
import pytest
import allure
from yaml_loader import YamlLoader
import controller


def pytest_addoption(parser):
    """pytest addoption"""
    parser.addoption("--yaml", type=str, default="det", help="yaml path")
    parser.addoption("--case", type=str, default="Con", help="case name")


@pytest.fixture
def yaml(request):
    """yaml"""
    return request.config.getoption("--yaml")


@pytest.fixture
def case(request):
    """case"""
    return request.config.getoption("--case")
