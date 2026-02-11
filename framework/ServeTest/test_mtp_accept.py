import os
import glob
import re
import statistics


def collect_files(root):
    pattern_path = os.path.join(root, "**", "speculate.log.20*")
    files = glob.glob(pattern_path, recursive=True)
    return [f for f in files if os.path.isfile(f)]


def percentile(data, p):
    k = int(len(data) * p)
    k = min(k, len(data) - 1)
    return data[k]


def print_stats(name, values):
    values.sort()
    mean = sum(values) / len(values)

    print(f"\n================ {name} stats ================")
    print(f"Total samples: {len(values)}")
    print("-" * 40)
    print(f"mean: {mean:.6f}")
    if len(values) > 1:
        print(f"std : {statistics.stdev(values):.6f}")
    print(f"min : {values[0]:.6f}")
    print(f"p50 : {percentile(values, 0.50):.6f}")
    print(f"p95 : {percentile(values, 0.95):.6f}")
    print(f"p99 : {percentile(values, 0.99):.6f}")
    print(f"max : {values[-1]:.6f}")
    print("================================================\n")

    return mean


def test_accept_ratio_stats():
    if os.getenv("test_branch") == "master":
        per_head_baseline = 0.984689
        accept_ratio_baseline = 0.4988832952561951
    else:
        per_head_baseline = 0.984689
        accept_ratio_baseline = 0.4988832952561951

    max_diff_ratio = 0.05

    per_head_pattern = re.compile(r"accept_ratio_per_head:\s*\[([^\]]+)\]")
    accept_pattern = re.compile(r"accept_ratio:\s*([0-9.eE+-]+)")

    log_dir = os.getenv("LOG_DIR", ".")  # 默认当前目录

    files = collect_files(log_dir)
    assert files, f"No speculate.log.20* found under {log_dir}"

    per_head_values = []
    accept_values = []

    for fname in files:
        with open(fname, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m1 = per_head_pattern.search(line)
                if m1:
                    per_head_values.extend(float(x.strip()) for x in m1.group(1).split(","))

                m2 = accept_pattern.search(line)
                if m2:
                    accept_values.append(float(m2.group(1)))

    assert per_head_values, "No accept_ratio_per_head found"
    assert accept_values, "No accept_ratio found"

    print(f"\nFiles scanned: {len(files)}")

    # per_head 统计
    mean_per_head = print_stats("accept_ratio_per_head", per_head_values)

    diff1 = abs(mean_per_head - per_head_baseline) / per_head_baseline
    print(
        f"[per_head check] baseline={per_head_baseline:.6f} | "
        f"mean={mean_per_head:.6f} | diff={diff1 * 100:.2f}%"
    )

    assert diff1 <= max_diff_ratio, (
        f"[per_head] drift too large: "
        f"mean={mean_per_head:.6f}, baseline={per_head_baseline:.6f}, "
        f"diff={diff1 * 100:.2f}%"
    )

    # 整体accept_ratio 统计
    mean_accept = print_stats("accept_ratio", accept_values)

    diff2 = abs(mean_accept - accept_ratio_baseline) / accept_ratio_baseline
    print(
        f"[accept_ratio check] baseline={accept_ratio_baseline:.6f} | "
        f"mean={mean_accept:.6f} | diff={diff2 * 100:.2f}%"
    )

    assert diff2 <= max_diff_ratio, (
        f"[accept_ratio] drift too large: "
        f"mean={mean_accept:.6f}, baseline={accept_ratio_baseline:.6f}, "
        f"diff={diff2 * 100:.2f}%"
    )


def test_entropy_stats():
    """
    校验 data_processor.log.20* 中的 entropy
    日志格式:
    entropy_utils.py[line:103] ... entropy: 0.9563972363643617
    """

    if os.getenv("test_branch") == "master":
        entropy_baseline = 0.425077
    else:
        entropy_baseline = 0.425077

    max_diff_ratio = 0.05

    entropy_pattern = re.compile(r"entropy:\s*([0-9.eE+-]+)")

    log_dir = os.getenv("LOG_DIR", ".")

    pattern_path = os.path.join(log_dir, "**", "data_processor.log.20*")
    files = [f for f in glob.glob(pattern_path, recursive=True) if os.path.isfile(f)]

    assert files, f"No data_processor.log.20* found under {log_dir}"

    entropy_values = []

    for fname in files:
        with open(fname, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = entropy_pattern.search(line)
                if m:
                    entropy_values.append(float(m.group(1)))

    assert entropy_values, "No entropy found"

    print(f"\nFiles scanned (entropy): {len(files)}")

    mean_entropy = print_stats("entropy", entropy_values)

    diff = abs(mean_entropy - entropy_baseline) / entropy_baseline

    print(
        f"[entropy check] baseline={entropy_baseline:.6f} | "
        f"mean={mean_entropy:.6f} | diff={diff*100:.2f}%"
    )

    assert diff <= max_diff_ratio, (
        f"[entropy] drift too large: "
        f"mean={mean_entropy:.6f}, baseline={entropy_baseline:.6f}, "
        f"diff={diff*100:.2f}%"
    )
