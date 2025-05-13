echo "启动reward服务"
cd PaddleLLM/llm/alignment/rl/reward
python reward_server.py > reward_server.log 2>&1 &
sleep 60s
curl -X 'POST' \
  'http://10.174.137.209:8731/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "src": [
    "test"
  ],
  "tgt": [
    "test"
  ],
  "response": [
    "test"
  ]
}' >> reward_server.log
cd -

# echo "kill reward 服务"
# pkill -9 -f reward_server.py