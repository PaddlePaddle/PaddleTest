# git clone https://github.com/PaddlePaddle/Paddle.git
unset http_proxy && unset https_proxy
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
python -c "import paddle;print(paddle.__version__);print(paddle.version.show())"
cd Paddle && python ./tools/print_signatures.py paddle > get_all_api_new.spec
cat get_all_api_new.spec  | grep "paddle.distributed." > dist_api_list_new.spec

file1="/ce_dist/Paddle/tools/dist_api_list_new.spec"
file2="/ce_dist/PaddleTest/distributed/CE_API/scripts/dist_api_list_old.spec"
output_file="/ce_dist/diff.spec"

# 使用diff命令比较文件
diff_result=$(diff "$file1" "$file2")

# 判断diff命令的输出是否为空
if [ -n "$diff_result" ]; then
    # 如果diff命令的输出不为空，表示文件之间存在差异
    echo "文件之间存在差异" > "$output_file"
    cat "$output_file"
    exit 1
else
    # 如果diff命令的输出为空，表示文件之间不存在差异
    echo "文件之间不存在差异" > "$output_file"
    exit 0
fi
