# conftest.py

def pytest_addoption(parser):
    parser.addoption(
        "--script",
        action="append",
        default=[],
        help="Distributed script(s) to run with paddle.distributed.launch",
    )

