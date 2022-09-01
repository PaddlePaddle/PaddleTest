apt install openjdk-8-jdk
wget https://registry.npmjs.org/allure-commandline/-/allure-commandline-2.13.0.tgz
tar -zxvf allure-commandline-2.13.0.tgz
rm -f allure-commandline-2.13.0.tgz
mv package allure
chmod -R 777 allure
export PATH=allure/bin/allure:$PATH
ln -s /paddle/no_one_ocr/function/AutomaticTestSystem/allure/bin/allure /usr/bin/allure
