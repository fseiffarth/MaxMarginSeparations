'''
Created on 29.04.2019

@author: florian
'''
import os
import sqlite3
import numpy as np
from DataToSQL.PlotData.PlotData import plot_3d_data_colorbar


class DataToSQL(object):
    '''
    object for saving and getting data in SQL files
    '''

    def __init__(self, file_path="", db_name="database"):
        '''
        Constructor
        '''

        self.file_path = file_path
        self.db_name = db_name + ".sqlite"
        """Create db if it does not exist"""
        if not os.path.isfile(self.file_path + self.db_name):
            con = sqlite3.connect(self.file_path + self.db_name + ".sqlite")

    """Check if table exists otherwise create"""

    def create_table(self, table_name, column_list, type_list=[]):
        string = ""
        for i, entry in enumerate(column_list, 0):
            string += entry
            if type_list and len(type_list) == len(column_list):
                string += " " + type_list[i]
            string += ","
        string = string[:-1]
        self.set_with_query("CREATE TABLE if not exists " + table_name + " (" + string + ");")

    """Add a new column at specific position to the table"""

    def add_column(self, table_name, index=-1, column_name="", column_type="TEXT"):
        if column_name and column_name not in self.get_column_names(table_name):
            if index == -1:
                self.set_with_query(
                    "ALTER TABLE " + table_name + " ADD COLUMN " + column_name + " " + column_type + ";")
            else:
                column_list = self.get_column_names(table_name)

                string1 = "("
                string2 = ""
                for i, entry in enumerate(column_list, 0):
                    string1 += entry
                    string2 += entry
                    string1 += ","
                    string2 += ","
                string1 = string1[:-1]
                string2 = string2[:-1]
                string1 += ")"
                print(string1, string2)

                column_list.insert(index, column_name)
                type_list = self.get_column_types(table_name)
                type_list.insert(index, column_type)
                print(column_list, type_list)
                self.set_with_query("ALTER TABLE " + table_name + " RENAME TO TEMPORARYUSEDTABLE;")
                self.create_table(table_name, column_list, type_list)

                self.set_with_query(
                    "INSERT INTO " + table_name + " " + string1 + " SELECT " + string2 + " FROM TEMPORARYUSEDTABLE;")
                self.set_with_query("DROP TABLE TEMPORARYUSEDTABLE")

    """Add a new row with values to a table"""

    def add_row_to_table(self, table_name, row_value_list):
        con = sqlite3.connect(self.file_path + self.db_name)
        cur = con.cursor()

        column_names = self.get_column_names(table_name)

        if len(row_value_list) != len(column_names) - self.get_auto_increment_column_number(table_name):
            print("No row inserted. Value list and column number do not fit together!")


        else:
            values = "("
            for x, i in enumerate(row_value_list, 0):
                values += "?"
                values += ","
            values = values[:-1]
            values += ")"
            string = "("

            for entry in column_names:
                if entry not in self.get_auto_increment_column_names(table_name):
                    string += entry
                    string += ","
            string = string[:-1]
            string += ")"

            to_db = [row_value_list]

            cur.executemany("INSERT INTO " + table_name + " " + string + " VALUES " + values + " ;", to_db)
            con.commit()
            con.close()

    """Get data from database via query string"""

    def get_from_query(self, query):
        con = sqlite3.connect(self.file_path + self.db_name)
        cur = con.cursor()
        cur.execute(query)
        result = cur.fetchall()
        con.commit()
        con.close()
        return result

    """Change data in the database via a query"""

    def set_with_query(self, query):
        con = sqlite3.connect(self.file_path + self.db_name)
        cur = con.cursor()
        cur.execute(query)
        con.commit()
        con.close()

    """Get all column names of a table"""

    def get_column_names(self, table_name):
        info = self.get_column_info(table_name)
        column_names = []
        for x in info:
            column_names.append(x[1])
        return column_names

    def get_auto_increment_column_names(self, table_name):
        info = self.get_column_info(table_name)
        column_names = []
        for x in info:
            if x[5]:
                column_names.append(x[1])
        return column_names

    def get_auto_increment_column_number(self, table_name):
        return len(self.get_auto_increment_column_names(table_name))

    """Get all column types of a table"""

    def get_column_types(self, table_name):
        info = self.get_column_info(table_name)
        column_types = []
        for x in info:
            column_types.append(x[2])
        return column_types

    """Get all column info of a table"""

    def get_column_info(self, table_name):
        return self.get_from_query("PRAGMA table_info(" + table_name + ");")

    def sql_to_numpy_2d(self, table_name, column_name_A, column_name_B, column_name_A_values, column_name_B_values,
                        column, function=None):
        result = np.zeros((len(column_name_A_values), len(column_name_B_values)))
        for i, A in enumerate(column_name_A_values, 0):
            for j, B in enumerate(column_name_B_values, 0):
                if function is None:
                    val = self.get_from_query("SELECT " + str(column) + " FROM " + str(table_name) + " WHERE " + str(
                        column_name_A) + " = '" + str(A) + "'" + " AND " + str(column_name_B) + " = '" + str(B) + "'")
                else:
                    val = self.get_from_query("SELECT " + str(column) + " FROM " + str(table_name) + " WHERE " + str(
                        column_name_A) + " = '" + str(A) + "'" + " AND " + str(column_name_B) + " = '" + str(B) + "'")
                    if val:
                        val = self.get_from_query("SELECT " + function + "(" + str(column) + ")" + " FROM " + str(
                            table_name) + " WHERE " + str(column_name_A) + " = '" + str(A) + "'" + " AND " + str(
                            column_name_B) + " = '" + str(B) + "'")
                if val:
                    result[i][j] = val[0][0]

        return result

    def sql_plot_3d_colorbar(self, table_name, column_name_A, column_name_B, column_name_A_values, column_name_B_values,
                             column, function=None, column_A_ticks="class", column_B_ticks="class", min_val=0,
                             max_val=100, heading="Test", x_label_name="xLabel", y_label_name="yLabel", colormap="Reds",
                             colorbar_label="", tikz_save=None, name="Test"):
        data = self.sql_to_numpy_2d(table_name, column_name_A, column_name_B, column_name_A_values,
                                    column_name_B_values, column, function)
        plot_3d_data_colorbar(data, column_A_ticks, column_B_ticks, column_name_A_values, column_name_B_values, min_val,
                              max_val, heading, x_label_name, y_label_name, colormap, colorbar_label, tikz_save, name)

    def experiment_to_database(self, experiment_name, experiment_attributes, experiment_attributes_type,
                               experiment_values):
        """

        :param experiment_name:
        :param experiment_attributes:
        :param experiment_attributes_type:
        :param experiment_values:
        """
        self.create_table(experiment_name, ["Id"] + experiment_attributes,
                                   ['INTEGER PRIMARY KEY AUTOINCREMENT'] + experiment_attributes_type)

        for i, values in enumerate(experiment_values, 0):
            self.add_row_to_table(experiment_name, values)
