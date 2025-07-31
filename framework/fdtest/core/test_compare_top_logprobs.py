#!/bin/env python3
# -*- coding: utf-8 -*-
# @author yubaoku

from core import *


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


def assert_top_logprobs_prefix_match(small_top, large_top, token_index):
    """
    Assert that all entries in small_top are a prefix of large_top,
    comparing both token and logprob values.
    """
    for j in range(len(small_top)):
        s_token = small_top[j]["token"]
        l_token = large_top[j]["token"]
        assert s_token == l_token, \
            "Token mismatch at token {} pos {}: '{}' != '{}'".format(token_index + 1, j + 1, s_token, l_token)

        s_prob = small_top[j]["logprob"]
        l_prob = large_top[j]["logprob"]
        assert s_prob == l_prob, \
            "Logprob mismatch at token {} pos {}: {} != {}".format(token_index + 1, j + 1, s_prob, l_prob)


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

    # Assertion for prefix consistency
    if len(top_logprobs_values) >= 2:
        small = top_logprobs_values[0]
        large = top_logprobs_values[1]

        small_contents = responses[small]["choices"][0]["logprobs"]["content"]
        large_contents = responses[large]["choices"][0]["logprobs"]["content"]
        min_len = min(len(small_contents), len(large_contents))

        for i in range(min_len):
            small_top = small_contents[i]["top_logprobs"]
            large_top = large_contents[i]["top_logprobs"]
            assert_top_logprobs_prefix_match(small_top, large_top, i)


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
    """
    Test the compare_top_logprobs function with a sample input data.
    Returns:
        None
        AssertionError: If there is a mismatch between the top logprobs values.

    """
    test_compare_top_logprobs()