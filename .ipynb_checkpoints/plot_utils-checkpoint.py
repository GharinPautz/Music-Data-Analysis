"""
Programmer: Ghar Pautz
Class: CPSC 322-02, Spring 2021
Programming Assignment #3
02/25/2021
I did not attempt the bonus.

Description: This program writes utility functions for manipulating and
    showcasing data in a jupyter notebook.
"""
import matplotlib.pyplot as plt

def make_frequency_diagram(x, y, title, xlabel):
    """Displays a frequency diagram as a bar chart given a list of values.

    Args:
        x (list): List of categories
        y (list): List of heights for bar charts
        title (str): The title that appears at the top of the visualization
        xlabel (str): The label for the x axis units
    """
    indices = []
    for i in range(0, len(x)):
        indices.append(i)

    # Make figure
    plt.figure()
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel(xlabel)
    plt.bar(indices, y)
    plt.xticks(indices, x, rotation=45, horizontalalignment="right")
    plt.grid(True)
    plt.show()

def pie_chart(x, y, title):
    """Makes a pie chart to display percentages

    Args:
        x (list): list of labels for wedges in pie charts
        y (list): list of percentages to display in pie chart
        title (str): The title that appears at the top of the visualization
    """
    plt.figure()
    plt.title(title)
    plt.pie(y, labels=x, autopct="%1.1f%%")
    plt.show()

def make_box_plot(distributions, labels, title, x_units, y_units):
    """Makes box plot(s) to display distribution data

    Args:
        distributions (list of list): data to show in box plots
        labels (list of str): The labels for each box plot
        x_units (str): Label for x axis
        y_units (str): Label for y axis
    """
    plt.figure()
    plt.boxplot(distributions)
    plt.title(title)
    plt.ylabel(y_units)
    plt.xlabel(x_units)
    plt.xticks(list(range(1, len(labels) + 1)), labels, rotation=90)
    plt.show()