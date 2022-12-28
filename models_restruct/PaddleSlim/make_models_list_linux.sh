if [ -f models_list_linux_function.txt ]; then
rm -rf  models_list_linux_function.txt
fi

for i in `ls cases | grep -v precision `
do
    echo ${i} >> models_list_linux_function.txt
done

if [ -f models_list_linux_precision.txt ]; then
    rm -rf  models_list_linux_precision.txt
fi

for i in `ls cases | grep precision`
do
    echo ${i} >> models_list_linux_precision.txt
done
