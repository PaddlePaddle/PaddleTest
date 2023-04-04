if [ -f models_list_function.txt ]; then
rm -rf  models_list_function.txt
fi

# 功能回归models list
for i in `ls cases | grep -v big `
do
    echo ${i} >> models_list_function.txt
done

#精度对齐回归 models list （因流水线任务时长8h限制，需拆分成两个）
if [ -f models_list_precision_01.txt ]; then
    rm -rf  models_list_precision_01.txt
fi

for i in `ls cases | grep big | grep -v mmoe | grep -v dnn`
do
    echo ${i} >> models_list_precision_01.txt
done

if [ -f models_list_precision_02.txt ]; then
    rm -rf  models_list_precision_02.txt
fi

for i in `ls cases | grep big | grep -E 'mmoe|dnn'`
do
    echo ${i} >> models_list_precision_02.txt
done
