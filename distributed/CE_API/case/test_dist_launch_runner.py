# test_dist_launch_runner.py
import pytest
import subprocess

def pytest_generate_tests(metafunc):
    if "script_name" in metafunc.fixturenames:
        scripts = metafunc.config.getoption("script")
        if not scripts:
            pytest.skip("No script provided with --script")
        metafunc.parametrize("script_name", scripts)

def test_run_dist_script(script_name):
    cmd = [
        "python",
        "-m",
        "paddle.distributed.launch",
        "--gpus=0,1",
        script_name
    ]
    print(f"\nRunning command: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    assert result.returncode == 0, f"Script {script_name} failed with return code {result.returncode}"

