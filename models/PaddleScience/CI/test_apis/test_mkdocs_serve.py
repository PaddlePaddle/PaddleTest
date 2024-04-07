import os
import sys
import pytest
import time

def test_mkdocs_serve():
    """
    检查mkdocs serve命令是否正常工作
    """
    os.system("mkdocs serve -a 127.0.0.1:6007 -f ../../mkdocs.yml >test_mkdocs_serve.log 2>&1 &")
    time.sleep(50)
    with open("test_mkdocs_serve.log", 'r') as file:
        log_content = file.read()
        assert "WARNING" not in log_content and "ERROR" not in log_content

if __name__ == '__main__':
    # 使用 pytest 模块运行测试函数
    code = pytest.main([sys.argv[0]])
    sys.exit(code)