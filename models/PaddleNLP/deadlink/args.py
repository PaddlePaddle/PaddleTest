#!/usr/bin.env python
"""
args
author:zhengya01
"""

import argparse

def parse_args():
    """
    args
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--func", 
        type=str, 
        default='all', 
        help="func name. \
              support [all | get_link | check_link | diff_link]. \
              only need for func.all and func.get_link")

    parser.add_argument(
        "--repo",
        type=str,
        default='',
        help="repo name. \
              only need for func.all and func.get_link")

    parser.add_argument(
        "--branch", 
        type=str, 
        default='develop', 
        help="branch name. \
              only need for func.all and func.get_link")

    parser.add_argument(
        "--code_path", 
        type=str, 
        default='', 
        help="local path of repo code. \
              only need for func.all and func.get_link")
    parser.add_argument(
        "--link_new_file", 
        type=str, 
        default='', 
        help="new link file name. \
              only need for func.diff_link")

    parser.add_argument(
        "--link_old_file", 
        type=str, 
        default='', 
        help="old link file name. \
              only need for func.diff_link")

    parser.add_argument(
        "--link_diff_file", 
        type=str, 
        default='link_diff.file', 
        help="diff link file name. \
              diff links of link_new_file and link_old_file. \
              only need for func.diff_link")

    parser.add_argument(
        "--link_file", 
        type=str, 
        default='', 
        help="link file name. \
             only need for func.check_link")

    parser.add_argument(
        "--link_res_file", 
        type=str, 
        default='', 
        help="link result file name. \
              only need for func.check_link")

    args = parser.parse_args()
    return args
