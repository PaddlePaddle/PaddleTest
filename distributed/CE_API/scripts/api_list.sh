# git clone https://github.com/PaddlePaddle/Paddle.git
unset http_proxy && unset https_proxy
# python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
python -c "import paddle;print(paddle.__version__);print(paddle.version.show())"
python Paddle/tools/print_signatures.py paddle > get_all_api_new.spec
cat get_all_api_new.spec  | grep "paddle.distributed." > dist_api_list_new.spec

mkdir result_file
file1="./dist_api_list_new.spec"
file2="./PaddleTest/distributed/CE_API/scripts/dist_api_list_old.spec"
file1_dist="./result_file/dist_list_new.spec"
file2_dist="./result_file/dist_list_old.spec"
output_file="./result_file/diff.spec"

> "$file1_dist"
> "$file2_dist"
> "$output_file"

# 逐行读取源文件，并提取空格前的内容  
while IFS= read -r line; do  
    # 使用awk提取空格前的部分  
    echo "$line" | awk '{print $1}' >> "$file1_dist"  
done < "$file1" 
while IFS= read -r line; do  
    # 使用awk提取空格前的部分  
    echo "$line" | awk '{print $1}' >> "$file2_dist"  
done < "$file2" 

# 使用diff命令比较文件
diff "$file1_dist" "$file2_dist" > "$output_file"

# 判断diff命令的输出是否为空
if [ -s "$output_file" ]; then
    echo "文件之间存在差异"
    exit 1
else
    echo "文件之间不存在差异"
    exit 0
fi
