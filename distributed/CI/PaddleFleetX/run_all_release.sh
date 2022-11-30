#!/usr/bin/env bash
set -e

bash PaddleTest/distributed/CI/PaddleFleetX/before_hook.sh
bash PaddleTest/distributed/CI/PaddleFleetX/case_chain.sh
bash PaddleTest/distributed/CI/PaddleFleetX/end_hook.sh
