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
done
