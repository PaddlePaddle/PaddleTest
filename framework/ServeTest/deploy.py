import os
import subprocess
import signal
import socket
import requests
import time
import yaml
import ast
import re
import json
import sys
from flask import Flask, jsonify, request, Response

app = Flask(__name__)

# 默认参数值
PID_FILE = "pid_port"
LOG_FILE = "server.log"
FD_PORT = 8128
FD_WORKER_QUEUE_PORT = 8133
FD_METRICS_PORT = 8135

DEFAULT_PARAMS = {
    "--port": FD_PORT,
    "--engine-worker-queue-port": FD_WORKER_QUEUE_PORT,
    "--metrics-port": FD_METRICS_PORT,
    "--enable-logprob": True,
}


def build_command(config):
    """根据配置构建启动命令"""
    # 基础命令
    cmd = [
        "python", "-m", "fastdeploy.entrypoints.openai.api_server",
    ]

    # 添加配置参数
    for key, value in config.items():
        if "--enable" in key:
            if value:
                cmd.append(key)
        else:
            cmd.extend([key, str(value)])

    return cmd


def merge_configs(base_config, override_config):
    """合并配置，优先级：override_config > base_config"""
    merged = base_config.copy()

    if override_config:
        for key in override_config:
            merged[key] = override_config[key]

    return merged


def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def get_server_pid():
    """获取服务进程ID PORT"""
    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            data = yaml.safe_load(f)
            return data
    return None


def is_server_running():
    """检查服务是否正在运行"""
    pid_port = get_server_pid()
    if pid_port is None:
        return False, "Server not running..."

    server_pid, port = pid_port["PID"], pid_port["PORT"]
    health_check_endpoint = f"http://0.0.0.0:{port}/health"

    try:
        response = requests.get(
            health_check_endpoint,
            timeout=2
        )
        return response.status_code == 200, response.text
    except requests.exceptions.RequestException as e:
        return False, str(e)


def parse_tqdm_progress(log_lines):
    """
    解析 tqdm 风格的进度条
    """
    tqdm_pattern = re.compile(
        r"(?P<prefix>.+?):\s+(?P<percent>\d+)%\|(?P<bar>.+?)\|\s+(?P<step>\d+/\d+)\s+\[(?P<elapsed>\d+:\d+)<(?P<eta>\d+:\d+),\s+(?P<speed>[\d\.]+it/s)\]"
    )

    for line in reversed(log_lines):
        match = tqdm_pattern.search(line)
        if match:
            data = match.groupdict()
            return {
                "status": "服务启动中",
                "progress": {
                    "percent": int(data["percent"]),
                    "step": data["step"],
                    "speed": data["speed"],
                    "eta": data["eta"],
                    "elapsed": data["elapsed"],
                    "bar": data["bar"].strip()
                },
                "raw_line": line.strip()
            }
    return {
        "status": "服务启动中",
        "progress": {},
        "raw_line": log_lines[-1] if log_lines else "server.log为空"
    }


def stop_server(signum=None, frame=None):
    """停止大模型推理服务"""
    pid_port = get_server_pid()
    if pid_port is None:
        if signum:
            sys.exit(0)
        return jsonify({"status": "error", "message": "Service is not running"}), 400

    server_pid, port = pid_port["PID"], pid_port["PORT"]

    # 清理PID文件
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

    try:
        # 终止进程组（包括所有子进程）
        os.killpg(os.getpgid(pid_port["PID"]), signal.SIGTERM)

        output = subprocess.check_output(f"lsof -i:{port} -t", shell=True).decode().strip()
        for pid in output.splitlines():
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed process on port {port}, pid={pid}")
    except Exception as e:
        print(f"Failed to stop server: {e}")
    # 若log目录存在，则重命名为log_pid
    if os.path.isdir('./log'):
        os.rename('./log', './log_{}'.format(time.strftime("%Y%m%d%H%M%S")))

    if signum:
        sys.exit(0)

    return jsonify({"status": "success", "message": "Service stopped", "pid": server_pid}), 200


# 捕获 SIGINT (Ctrl+C) 和 SIGTERM (kill)
signal.signal(signal.SIGINT, stop_server)
signal.signal(signal.SIGTERM, stop_server)


@app.route('/start', methods=['POST'])
def start_service():
    """启动大模型推理服务"""
    # 检查服务是否已在运行
    if is_server_running()[0]:
        return jsonify({"status": "error", "message": "服务已启动，无需start"}), 400

    try:
        base_config = DEFAULT_PARAMS

        override_config = request.get_json() or {}

        final_config = merge_configs(base_config, override_config)

        global FD_PORT
        global FD_WORKER_QUEUE_PORT
        global FD_METRICS_PORT
        FD_PORT = final_config["--port"]
        FD_WORKER_QUEUE_PORT = final_config["--engine-worker-queue-port"]
        FD_METRICS_PORT = final_config["--metrics-port"]

        # 构建命令
        cmd = build_command(final_config)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    print("cmd", cmd)

    try:
        # 设置环境变量并启动进程
        env = os.environ.copy()

        with open(LOG_FILE, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=log,
                env=env,
                start_new_session=True
            )

        # 保存进程ID,port到yaml文件
        with open(PID_FILE, 'w') as f:
            yaml.dump({"PID": process.pid, "PORT": final_config["--port"]}, f)

        return jsonify({
            "status": "success",
            "message": "Service started",
            "pid": process.pid,
            "config": final_config,
            "log_file": LOG_FILE
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/switch', methods=['POST'])
def switch_service():
    """切换模型服务"""
    # kill掉已有服务
    stop_server()

    try:
        base_config = DEFAULT_PARAMS

        override_config = request.get_json() or {}

        final_config = merge_configs(base_config, override_config)

        global FD_PORT
        global FD_WORKER_QUEUE_PORT
        global FD_METRICS_PORT
        FD_PORT = final_config["--port"]
        FD_WORKER_QUEUE_PORT = final_config["--engine-worker-queue-port"]
        FD_METRICS_PORT = final_config["--metrics-port"]

        # 构建命令
        cmd = build_command(final_config)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    print("cmd", cmd)

    try:
        # 设置环境变量并启动进程
        env = os.environ.copy()

        with open(LOG_FILE, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=log,
                env=env,
                start_new_session=True
            )

        # 保存进程ID,port到yaml文件
        with open(PID_FILE, 'w') as f:
            yaml.dump({"PID": process.pid, "PORT": final_config["--port"]}, f)

        return jsonify({
            "status": "success",
            "message": "Service started",
            "pid": process.pid,
            "config": final_config,
            "log_file": LOG_FILE
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/status', methods=['GET', 'POST'])
def service_status():
    """检查服务状态"""
    health, msg = is_server_running()

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            msg = f.readlines()
    result = parse_tqdm_progress(msg)

    if not health:
        return Response(
            json.dumps(result, ensure_ascii=False),
            status=500,
            content_type='application/json'
        )

    # 检查端口是否监听
    ports_status = {
        "api_port": FD_PORT if is_port_in_use(FD_PORT) else None,
        "queue_port": FD_WORKER_QUEUE_PORT if is_port_in_use(FD_WORKER_QUEUE_PORT) else None,
        "metrics_port": FD_METRICS_PORT if is_port_in_use(FD_METRICS_PORT) else None
    }

    result["status"] = "服务启动完成"
    result["ports_status"] = ports_status

    return Response(
        json.dumps(result, ensure_ascii=False),
        status=200,
        content_type='application/json'
    )


@app.route('/stop', methods=['POST'])
def stop_service():
    """停止大模型推理服务"""
    res, status_code = stop_server()

    return res, status_code


@app.route('/config', methods=['GET'])
def get_config():
    """获取当前server配置"""
    health, msg = is_server_running()

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            msg = f.readlines()
    result = parse_tqdm_progress(msg)

    if not health:
        return Response(
            json.dumps(result, ensure_ascii=False),
            status=500,
            content_type='application/json'
        )

    if not os.path.exists("log/api_server.log"):
        return Response(
            json.dumps({"message": "api_server.log不存在"}, ensure_ascii=False),
            status=500,
            content_type='application/json'
        )

    try:
        # 筛选出包含"args:"的行
        with open("log/api_server.log", 'r') as f:
            lines = [line for line in f.readlines() if "args:" in line]

        last_line = lines[-1] if lines else ""

        # 使用正则表达式提取JSON格式的配置
        match = re.search(r'args\s*[:：]\s*(.*)', last_line)
        if not match:
            return Response(
                json.dumps({"message": "api_server.log中没有args信息，请检查log"}, ensure_ascii=False),
                status=500,
                content_type='application/json'
            )

        # 尝试解析JSON
        config_json = match.group(1).strip()
        config_data = ast.literal_eval(config_json)
        print("config_data", config_data, type(config_data))
        return Response(
            json.dumps({"server_config": config_data}, ensure_ascii=False),
            status=200,
            content_type='application/json'
        )

    except Exception as e:
        return Response(
            json.dumps({"message": "api_server.log解析失败，请检查log", "error": str(e)}, ensure_ascii=False),
            status=500,
            content_type='application/json'
        )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

