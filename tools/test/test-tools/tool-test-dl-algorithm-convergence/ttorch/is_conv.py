#!/usr/env/bin python
"""
check conv
"""

import sys

def is_number(v):
    """
    check number
    """
    try:
        float(v)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def is_conv(v, base_v):
    """
    check conv
    """
    if not is_number(v):
        print("train error", end="")
        return 1
    gap = abs(float(v) - float(base_v))
    p = gap / float(base_v)
    if p > 0.1:
        print("not convergent", end="")
        return 2
    else:
        print("{" + '"value": {}, "base_value": {}, "gap": {}'.format(v, base_v, str(p)) + "}", end="")
    return 0

if __name__ == "__main__":
    v = sys.argv[1]
    base_v = sys.argv[2]
    flag = is_conv(v, base_v)
    exit(flag)
