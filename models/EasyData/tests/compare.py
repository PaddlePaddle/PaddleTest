
def compare(result, str, standard):
    result_value = ""
    standard_value = ""
    with open(result, "r", encoding="utf-8") as f:
        for line in f:
            if str in line:
                index = line[line.find(str) :]
                result_value += index

    with open(standard, "r", encoding="utf-8") as f:
        for line in f:
            standard_value += line
    assert result_value==standard_value

    
if __name__ == '__main__':
    obj = compare("test_PPDC1.log","ClasOutput INFO","PPDC1_standard.txt")



