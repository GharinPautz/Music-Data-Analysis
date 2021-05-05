import mysklearn.myutils
import copy
import csv 
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        num_rows = 0
        num_cols = 0
        for row in self.data:
            num_rows += 1
        num_cols = len(self.data[0])

        return num_rows, num_cols 

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        if col_identifier not in self.column_names:
            print("invalid col_identifier")
            raise ValueError

        col_index = self.column_names.index(col_identifier)
        column = []

        for row in self.data:
            if (include_missing_values):
                column.append(row[col_index])
            elif (not include_missing_values and row[col_index] != "NA"):
                column.append(row[col_index])
            
        return column 

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for rowNum,row in enumerate(self.data):
            for colNum, value in enumerate(row):
                try:
                    self.data[rowNum][colNum] = float(value)
                except ValueError:
                    pass

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        temp_table = []
        for row in self.data:
            if row not in rows_to_drop:
                temp_table.append(row)
        self.data = temp_table

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        # empty data table
        self.data = []
    
        # open file for reading
        infile = open(filename, "r", encoding='utf-8')

        with infile as data:
            read_header = False
            for line in csv.reader(data):
                if not read_header:
                    self.column_names = line
                    read_header = True
                else:
                    # add each line in file to table
                    self.data.append(line)
        
        # close file
        infile.close()

        self.convert_to_numeric()
        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """

        with open(filename, mode='w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
            writer.writerow(self.column_names)
            for row in self.data:
                writer.writerow(row)

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        col_name_indexes = []
        for col_name in key_column_names:
            col_name_indexes.append(self.column_names.index(col_name))

            
        simplified_table = []
        for row in self.data:
            simplified_row = []
            for col_index in col_name_indexes:
                simplified_row.append(row[col_index])
            simplified_table.append(simplified_row)


        read_rows = []
        duplicate_rows = []
        for rowNum, row in enumerate(simplified_table):      
            if row in read_rows:
                duplicate_rows.append(self.data[rowNum])
            else:
                read_rows.append(row)
        return duplicate_rows

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        temp_table = []
        for row in self.data:
            contains_missing_value = False
            for value in row:
                if value == "NA":
                    contains_missing_value = True
            if not contains_missing_value:
                temp_table.append(row)
        self.data = temp_table

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        column = self.get_column(col_name)

        num_values = 0
        sum_values = 0
        for value in column:
            if value != "NA":
                num_values += 1
            try:
                sum_values += float(value)
            except ValueError:
                pass
        average = sum_values / num_values

        col_index = self.column_names.index(col_name)
        for rowNum, row in enumerate(self.data):
            if self.data[rowNum][col_index] == "NA":
                self.data[rowNum][col_index] = average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. 
            The column names and their order is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        if self.data == []:
            return MyPyTable(col_names, [])

        stats_table = []
        self.convert_to_numeric()

        for col_name in col_names:
            row = []
            row.append(col_name)
            col = self.get_column(col_name)

            sum_vals = 0
            num_vals = 0
            
            minimum = col[0]
            maximum = col[0]
            # calculate min and max
            for val in col:
                num_vals += 1
                sum_vals += val
                minimum = min(minimum, val)
                maximum = max(maximum, val)
            row.append(minimum)
            row.append(maximum)

            # calculate mid 
            mid = (minimum + maximum) / 2
            row.append(mid)

            # calculate average
            avg = sum_vals / num_vals
            row.append(avg)

            # calculate median
            col.sort()
        
            # if odd num_vals, get middle index
            # if even num_vals, get average of 2 middle values
            if num_vals % 2 == 0:
                # even
                index = int(num_vals / 2)
                median = (col[index] + col[index - 1]) / 2
            else:
                # odd
                index = int(num_vals / 2)
                median = col[index]
            row.append(median)
            
            stats_table.append(row)

        return MyPyTable(["attribute", "min", "max", "mid", "avg", "median"], stats_table)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # create table column names for joined table
        joined_table_columns = []
        for table_1_col in self.column_names:
            joined_table_columns.append(table_1_col)
        
        for table_2_col in other_table.column_names:
            if table_2_col not in joined_table_columns:
                joined_table_columns.append(table_2_col)

        # join data from table 1 and table 2
        joined_table_data = []
        if len(key_column_names) == 1:
            table_1_key_index = self.column_names.index(key_column_names[0])
            table_2_key_index = other_table.column_names.index(key_column_names[0])
            for table_1_row in self.data:
                for table_2_row in other_table.data:
                    if table_1_row[table_1_key_index] == table_2_row[table_2_key_index]:
                        # matching keys
                        row = []
                        for val in table_1_row:
                            row.append(val)
                        for val in table_2_row:
                            if val not in row:
                                row.append(val)
                        joined_table_data.append(row)
        else: # more than 1 key to join on
            # get 2 tables of composite key values for table 1 and table 2
            table_1_composite_key = []
            table_2_composite_key = []
            # table 1
            for row_num, row in enumerate(self.data):
                row = []
                for key_col_name in key_column_names:
                    table_1_key_index = self.column_names.index(key_col_name)
                    row.append(self.data[row_num][table_1_key_index])
                table_1_composite_key.append(row)

            # table 2
            for row_num, row in enumerate(other_table.data):
                row = []
                for key_col_name in key_column_names:
                    table_2_key_index = other_table.column_names.index(key_col_name)
                    row.append(other_table.data[row_num][table_2_key_index])
                table_2_composite_key.append(row)

            # do join
            for table_1_row_num, table_1_row in enumerate(self.data):
                for table_2_row_num, table_2_row in enumerate(other_table.data):
                    if table_1_composite_key[table_1_row_num] == table_2_composite_key[table_2_row_num]:
                        # matching keys
                        row = []
                        for val in table_1_row:
                            row.append(val)
                        for val in table_2_row:
                            if val not in row:
                                row.append(val)
                        joined_table_data.append(row) 


        return MyPyTable(joined_table_columns, joined_table_data) 

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        # create table column names for joined table
        joined_table_columns = []
        for table_1_col in self.column_names:
            joined_table_columns.append(table_1_col)
        
        for table_2_col in other_table.column_names:
            if table_2_col not in joined_table_columns:
                joined_table_columns.append(table_2_col)

        # join data from table 1 and table 2
        joined_table = self.data.copy()
        if len(key_column_names) == 1:
            joined_table_key_index = self.column_names.index(key_column_names[0])
            table_2_key_index = other_table.column_names.index(key_column_names[0])
            for row in joined_table:
                found_match = False
                for table_2_row in other_table.data:
                    if row[joined_table_key_index] == table_2_row[table_2_key_index]:
                        found_match = True
                        # add table 2 data to joined table
                        for val in table_2_row:
                            if val not in row:
                                row.append(val)
                if not found_match:
                    # fill in missing table 2 values with 'NA'
                    for index, val in enumerate(table_2_row):
                        if index != table_2_key_index:
                            row.append("NA")
            # fill in rows in table 2 not in table 1
            for row in other_table.data:
                new_row = []
                found_match = False
                for joined_row in joined_table:
                    if row[table_2_key_index] == joined_row[joined_table_key_index]:
                        found_match = True
                if not found_match:
                    # fill in NAs and add row to joined_table
                    for col_index, col_name in enumerate(joined_table_columns):
                        try:
                            table_2_col_index = other_table.column_names.index(col_name)
                            new_row.append(row[table_2_col_index])
                        except ValueError:
                            new_row.append("NA")
                    joined_table.append(new_row)
        else: # composite key
            # get 2 tables of composite key values for table 1 and table 2
            table_1_composite_key = []
            table_2_composite_key = []

            # get indexes of composite keys in table 1 and table 2
            table_1_key_indexes = []
            table_2_key_indexes = []

            # table 1
            for row_num, row in enumerate(self.data):
                row = []
                for key_col_name in key_column_names:
                    table_1_key_index = self.column_names.index(key_col_name)
                    table_1_key_indexes.append(table_1_key_index)
                    row.append(self.data[row_num][table_1_key_index])
                table_1_composite_key.append(row)

            # table 2
            for row_num, row in enumerate(other_table.data):
                row = []
                for key_col_name in key_column_names:
                    table_2_key_index = other_table.column_names.index(key_col_name)
                    table_2_key_indexes.append(table_2_key_index)
                    row.append(other_table.data[row_num][table_2_key_index])
                table_2_composite_key.append(row)

            # do join
            for table_1_row_num, row in enumerate(joined_table):
                found_match = False
                for table_2_row_num, table_2_row in enumerate(other_table.data):
                    if table_1_composite_key[table_1_row_num] == table_2_composite_key[table_2_row_num]:
                        found_match = True
                        # add table 2 data to joined table
                        for val in table_2_row:
                            if val not in row:
                                row.append(val)
                if not found_match:
                    # fill in missing table 2 values with 'NA'
                    for index, val in enumerate(table_2_row):
                        if index not in table_2_key_indexes:
                            row.append("NA")
            # fill in rows in table 2 not in table 1
            for table_2_row_num, row in enumerate(other_table.data):
                new_row = []
                found_match = False
                for table_1_row_num, joined_row in enumerate(self.data): 
                    if table_1_composite_key[table_1_row_num] == table_2_composite_key[table_2_row_num]:
                        found_match = True
                if not found_match:
                    # fill in NAs and add row to joined_table
                    for col_index, col_name in enumerate(joined_table_columns):
                        try:
                            table_2_col_index = other_table.column_names.index(col_name)
                            new_row.append(row[table_2_col_index])
                        except ValueError:
                            new_row.append("NA")
                    joined_table.append(new_row)

        return MyPyTable(joined_table_columns, joined_table)