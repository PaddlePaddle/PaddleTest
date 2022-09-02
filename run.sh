
#!/bin/bash
function getdir(){
    for element in `ls $1`
    do  
        dir_or_file=$1"/"$element
        if [ -d $dir_or_file ]
        then 
            getdir $dir_or_file
        else
            echo $dir_or_file
            /usr/local/bin/pre-commit run --file  $dir_or_file
        fi  
    done
}
root_dir="/paddle/PaddleTest/models/AutomaticTestSystem"
getdir $root_dir
