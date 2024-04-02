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
    parser.addoption("--all_dir", type=str, default="dir", help="yaml dir")
    parser.addoption("--yaml", type=str, default="det", help="yaml path")
    parser.addoption("--case", type=str, default="Con", help="case name")
    parser.addoption("--testing", type=str, default="testing", help="testing yml path")


@pytest.fixture
def all_dir(request):
    """yaml"""
    return request.config.getoption("--all_dir")


@pytest.fixture
def yaml(request):
    """yaml"""
    return request.config.getoption("--yaml")


@pytest.fixture
def case(request):
    """case"""
    return request.config.getoption("--case")


@pytest.fixture
def testing(request):
    """testing"""
    return request.config.getoption("--testing")
