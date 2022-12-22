<<<<<<< HEAD
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
=======
if [ -f models_list.txt ]; then
rm -rf  models_list.txt
fi

for i in `ls cases | grep -v big `
do
    echo ${i} >> models_list.txt
done

if [ -f models_list_precision.txt ]; then
    rm -rf  models_list_precision.txt
fi

for i in `ls cases | grep big`
do
    echo ${i} >> models_list_precision.txt
>>>>>>> 212f7ade4e0a5d169c2f958d3346af7d81a424a9
done
