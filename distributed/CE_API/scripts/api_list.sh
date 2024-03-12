# git clone https://github.com/PaddlePaddle/Paddle.git
unset http_proxy && unset https_proxy
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
python -c "import paddle;print(paddle.__version__);print(paddle.version.show())"
python Paddle/tools/print_signatures.py paddle > get_all_api_new.spec
cat get_all_api_new.spec  | grep "paddle.distributed." > dist_api_list_new.spec

file1="./dist_api_list_new.spec"
file2="./PaddleTest/distributed/CE_API/scripts/dist_api_list_old.spec"
output_file="./diff.spec"

# 使用diff命令比较文件
diff "$file1" "$file2" > "$output_file"

# 判断diff命令的输出是否为空
if [ -s "$output_file" ]; then
    echo "文件之间存在差异"
else
    echo "文件之间不存在差异"
fi
mkdir result_file
mv diff.spec result
mv dist_api_list_new.spec result
