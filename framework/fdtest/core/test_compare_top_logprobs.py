#!/bin/env python3
# -*- coding: utf-8 -*-
# @author yubaoku

from core import *
import requests
import json


def get_response(data):
    """
    Get the response from the API using the given data.
    Args:
        data (dict): The input data to be sent to the API.

        Returns:
            dict: The JSON response from the API.
    """
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload)
    return resp.json()


def compare_top_logprobs(base_data, top_logprobs_values=[5, 10]):
    """
    Compare the top logprobs of two different values and check if they match.

    Args:
        base_data (dict): The base data used for generating the responses.
        top_logprobs_values (list): A list of integers representing the top logprobs values to compare.

    Raises:
        AssertionError: If any mismatches are found between the top logprobs values.
    """
    responses = {}

    for val in top_logprobs_values:
        data = base_data.copy()
        data.update({
            "top_logprobs": val,
            "logprobs": True,
            "stream": False,
            "temperature": 0,
            "top_p": 0,
            "max_tokens": 10,
        })

        response = get_response(data)
        responses[val] = response

    for val in top_logprobs_values:
        output = responses[val]["choices"][0]
        token_logprobs_list = output.get("logprobs", {}).get("content", [])
        print("\nTop {} LogProbs:".format(val))
        for i, token_info in enumerate(token_logprobs_list):
            top_items = token_info.get("top_logprobs", [])
            print("  Token {}: {}".format(i + 1, [item['token'] for item in top_items]))

    # Assertion for prefix consistency
    if len(top_logprobs_values) >= 2:
        small = top_logprobs_values[0]
        large = top_logprobs_values[1]

        min_len = min(len(responses[small]["choices"][0]["logprobs"]["content"]),
                      len(responses[large]["choices"][0]["logprobs"]["content"]))

        for i in range(min_len):
            small_top_tokens = [item["token"] for item in
                                responses[small]["choices"][0]["logprobs"]["content"][i]["top_logprobs"]]
            large_top_tokens = [item["token"] for item in
                                responses[large]["choices"][0]["logprobs"]["content"][i]["top_logprobs"]]
            for j, token in enumerate(small_top_tokens):
                assert token == large_top_tokens[j], \
                    "Mismatch at token {} pos {}: '{}' != '{}'".format(i + 1, j + 1, token, large_top_tokens[j])


def test_compare_top_logprobs():
    """
    Test the compare_top_logprobs function with a sample input data.
    Returns:
        None
        AssertionError: If there is a mismatch between the top logprobs values.

    """
    data = {
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ]
    }

    compare_top_logprobs(data, top_logprobs_values=[5, 10])


if __name__ == '__main__':
    test_compare_top_logprobs()