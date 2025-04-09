
'''
====================================================================================
ПАРАМЕТРЫ ТАБЛИЦЫ КОНКРЕТНОГО ЖК
'''
# Параметры веб-страницы
house_info = 'house__additional-info'                   # <-- класс прочих параметров, которые расположены после описания основной информации
col_name_parameter = 'left-column flex flex-middle'     # <-- столбец наименования параметра в таблице с описанием ЖК
col_value_parameter = 'f-b-35 descr'                    # <-- столбец значения параметра в таблице с описанием ЖК

# Формат преобразования параметров в XPATH
class_house = "//div[@class='{}']".format(house_info)
class_name = "//div[@class='{}']".format(col_name_parameter)
class_values = "//div[@class='{}']".format(col_value_parameter)


