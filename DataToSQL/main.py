'''
Created on 29.04.2019

@author: florian
'''
from DataToSQL.DataToSQL import *
from PlotData.PlotData import *

def main():
    database = DataToSQL(db_name="DatabaseName")
    database.create_table("DataTable", ['ID', 'Name', 'Age'], ['INTEGER PRIMARY KEY', 'TEXT', 'INTEGER'])
    database.add_row_to_table("DataTable", [1, "XYZ", 10])
    database.add_row_to_table("DataTable", [2, "ABC", 5])
    print(database.get_from_query("SELECT * FROM DataTable"))
    print(database.get_column_names("DataTable"))
    database.add_column("DataTable", index = 100, column_name = 'NewColumn', column_type = 'TEXT')
    database.add_column("DataTable", index = -1, column_name = 'NewColumn2', column_type = 'TEXT')
    database.add_row_to_table("DataTable", ["x", 1, "ZZZ", 10, "y"])
    
    print(database.sql_to_numpy_2d("DataTable", 'ID', 'Name', [1, 2], ['XYZ', 'ABC'], 'Age', function = "AVG"))
    database.sql_plot_3d_colorbar("DataTable", 'ID', 'Name', [1, 2], ['XYZ', 'ABC'], 'Age', max_val=10, function="AVG")

if __name__ == '__main__':
    main()