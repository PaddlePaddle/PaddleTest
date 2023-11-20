# encoding: utf-8
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""
This module provides functions for analyzing error cases and performing problem analysis.
"""

import os
import re
import json
import time
import logging
import argparse
import requests


def run_flow(message, flow_id):
    """
    Send a request to iplayground and return the response result.
    :param message: The message to be processed.
    :param flow_id: The flow ID of iplayground.
    :return: The response result, or None if the request fails.
    """
    api_url = f"http://iplayground-dev.cloudapi.baidu-int.com/api/rest/v1/flow/{flow_id}/build/predict"
    payload = {"message": message}
    headers = {"Content-Type": "application/json", "X-release": "false"}
    proxies = {"http": None, "https": None}
    try:
        response = requests.post(api_url, json=payload, headers=headers, proxies=proxies)
        response.raise_for_status()
    except Exception as e:
        print(f"Error requesting iplayground: {e}")
        return None
    if response.json().get("status_code") == 200:
        return response.json()
    else:
        message = response.json().get("message")
        print(f"Error requesting iplayground: {message}")
        return None


def read_json(path):
    """
    Read a JSON file and return the parsed data.
    :param path: The path of the JSON file.
    :return: The parsed JSON data, or an empty dictionary if reading fails.
    """
    try:
        with open(path, "r", encoding="utf8") as f:
            json_data = json.load(f)
            return json_data
    except Exception as e:
        logging.error(f"Error reading JSON file: {e}")
        return {}


def write_json(path, data):
    """
    Write data to a JSON file.
    :param path: The path of the JSON file.
    :param data: The data to be written.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error writing JSON file: {e}")
        return {}


def analysis_error_case(path, flow_id):
    """
    Analyze error cases and perform problem analysis.
    :param path: The path of the report folder.
    :param flow_id: The flow ID of iplayground.
    """
    result_list = [i for i in os.listdir(path) if "-result.json" in i]

    for result_file in result_list:
        result_path = os.path.join(path, result_file)
        result_data = read_json(result_path)
        status = result_data.get("status", "")
        status_details = result_data.get("statusDetails", {})

        if status_details and status == "failed":
            # Get error information and keep valid error information for problem analysis
            trace = status_details.get("trace", "")
            trace_data = "".join(trace.split("\n")[-10:])
            err_info = trace_data
            analysis_result = {}

            if err_info:
                analysis_response = run_flow(err_info, flow_id)
                if analysis_response:
                    analysis_result = analysis_response.get("result", None)
                    analysis_result = analysis_result.replace("""```""", "")
                    analysis_result = analysis_result.replace("json", "")
                    if "{" in analysis_result and "}" not in analysis_result:
                        analysis_result = analysis_result + "}"
                    matches = re.findall(r"\{[^{}]*\}", analysis_result)
                    if matches:
                        analysis_result = matches[0]
                else:
                    analysis_result["error_type"] = "Accessing the LLM failed."
            else:
                analysis_result["error_type"] = "Failed to retrieve corresponding error information from the log."

            print("analysis_result:", analysis_result)
            result_data["analysis_result"] = analysis_result

            write_json(result_path, result_data)


def parse_args():
    """
    Parse command-line arguments.
    :return: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report_path",
        type=str,
        default="./report",
        help="the pytest report_path of all cases",
    )
    parser.add_argument(
        "--flow_id",
        type=int,
        default="",
        help="the flow_id of iplayground",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analysis_error_case(args.report_path, args.flow_id)
