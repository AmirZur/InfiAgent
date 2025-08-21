import logging
from typing import List
from mcp.server.fastmcp import FastMCP
import pandas as pd

mcp = FastMCP("DataAgent")
logger = logging.getLogger("mcp")
logger.setLevel(logging.INFO)

df = None

# simple tool for testing
# @mcp.tool()
# def add(a : int, b : int) -> int:
#     """Add together two numbers."""
#     return a + b

@mcp.tool()
def load_data(file_name : str) -> dict:
    """Load data from `file_name`. Must be executed before all other tools."""
    global df
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    if not file_name.startswith("examples/DA-Agent/data/da-dev-tables/"):
        file_name = "examples/DA-Agent/data/da-dev-tables/" + file_name
    df = pd.read_csv(file_name)
    return {
        'success': True,
        'information': {
            'data size': df.shape[0]
        }
    }

@mcp.tool()
def get_column_names() -> List[str]:
    """Returns all column names of dataset. Must execute `load_data` before using this tool."""
    assert df is not None, "Must execute `load_data` first."
    return df.columns.tolist()

@mcp.tool()
def describe_column(column_name : str) -> dict:
    """Uses the `describe` function from the `pandas` library to describe the `column_name`. Must execute `load_data` before using this tool."""
    assert df is not None, "Must execute `load_data` first."
    return df[column_name].describe().to_dict()

@mcp.tool()
def get_value_counts(column_name : str) -> dict:
    """Uses the `value_counts` function from the `pandas` library to get the value counts from `column_name`. Must execute `load_data` before using this tool."""
    assert df is not None, "Must execute `load_data` first."
    return df[column_name].value_counts().to_dict()

@mcp.tool()
def filter(column_name : str, value : float, by : str) -> dict:
    """Filters the column `column_name`. Must execute `load_data` before using this tool.
     
    Arguments:
    column_name : str
        Name of column to filter.
    value : float
        Value to filter by.
    by : str
        Method by which to filter. Must be one of the following options.
        "equal": keep only values equal to `value`.
        "less": keep only values strictly less than `value`.
        "less/equal": keep only values less than or equal to `value`.
        "greater": keep only values strictly greater than `value`.
        "greater/equal": keep only values greater than or equal to `value`.
    """
    global df
    assert df is not None, "Must execute `load_data` first."
    if by == "equal":
        df = df[df[column_name] == value]
    elif by == "less":
        df = df[df[column_name] < value]
    elif by == "less/equal":
        df = df[df[column_name] <= value]
    elif by == "greater":
        df = df[df[column_name] > value]
    elif by == "greater/equal":
        df = df[df[column_name] >= value]
    else:
        return {
            "success": False,
            "information": {
                "error message": f"Unrecognized value for `by`: \"{by}\"."
            }
        }
    return {
        'success': True,
        'information': {
            'data size': df.shape[0]
        }
    }

@mcp.tool()
def remove_outliers(column_name : str) -> dict:
    "Removes outliers with interquartile range method for provided `column_name`. Must execute `load_data` before using this tool."
    global df
    assert df is not None, "Must execute `load_data` first."
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df = df[(lower <= df[column_name]) & (df[column_name] <= upper)]
    return {
        'success': True,
        'information': {
            'data size': df.shape[0]
        }
    }

@mcp.tool()
def compute_mean(column_name : str) -> float:
    """Computes mean of provided `column_name`. Must execute `load_data` before using this tool."""
    assert df is not None, "Must execute `load_data` first."
    return df[column_name].mean()

@mcp.tool()
def compute_standard_deviation(column_name : str) -> float:
    """Computes standard deviation of provided `column_name`. Must execute `load_data` before using this tool."""
    assert df is not None, "Must execute `load_data` first."
    return df[column_name].std()


if __name__ == "__main__":
    mcp.run()