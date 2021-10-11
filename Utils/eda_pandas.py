from typing import Optional, List
from pandas import Dataframe
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt


def show_null_info(df: DataFrame, threshold: float=20.0) -> None:
    """
    This will display all the columns having null values beyond the provided theshold.

    Parameters:
        df: Dataframe object
        threshhold: Threshold percentage for checking null values
    """

    col_names, null_percentages = [], []
    n_datapoints = df.shape[0] 
    for col in df.columns:
        cur_null_percentage = round((df[col].isna().sum()/n_datapoints)*100, 2)
        if cur_null_percentage >= threshold:
            col_names.append(col)
            null_percentages.append(cur_null_percentage)
    
    if len(col_names) > 0:
        plt.bar(col_names, null_percentages)
        plt.xticks(col_names, col_names, rotation="vertical")
        plt.xlabel("Column names")
        plt.ylabel("Null values %")
        plt.title("Null values % per column")
        plt.show()
    else:
        print(f"No null values or all null values under {threshold}% threshold")

def draw_heatmap(data: np.ndarray, row_labels: List[str], col_labels: List[str], ax: Axes, fig_title: str="Heatmap", 
    **kwargs) -> None:
    """
    This will display heatmap based on values inside the data for every row-column pair in the input data.

    Parameters:
        data: Numpy array representing data
        row_labels: Labels for the row
        col_labels: Labels for the column
        ax: Axes instance to which the heatmap will be plotted
        fig_title: Title for the heatmap figure
        **kwargs: Arguments for `imshow`
    Returns:

    """

    hm = ax.imshow(data, **kwargs)
    
    # Creating colorbar to show mapping of color codings
    cbar = ax.figure.colorbar(hm, ax=ax)
    
    # Showing all ticks
    ax.set_xticks(np.arange(len(row_labels)))
    ax.set_yticks(np.arange(len(col_labels)))
    # Showing all labels
    ax.set_xticklabels(row_labels)
    ax.set_yticklabels(col_labels)
    # Rotating the row labels to 45 degree
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Looping over data dimensions and displaying heatmap values.
    for idx in range(len(col_labels)):
        for jdx in range(len(row_labels)):
            ax.text(jdx, idx, data[idx, jdx], ha="center", va="center", color="w")
            
    ax.set_title(fig_title)

def remove_outliers(train_df: DataFrame, train_labels_df: DataFrame, delta: float=0.0, inplace: bool=True
	) -> Optional[DataFrame, DataFrame]:
    """
    This will remove outliers (rows) from the input dataframe.

    Parameters:
        df: Dataframe object
        delta: Offset to add for checking outliers. i.e. Q3+(1.5+delta)*IQR
    """

    outliers_row_idxs = np.array([])
    for col in train_df.columns:
        Q1 = np.percentile(train_df[col], 25, interpolation = "midpoint")
        Q3 = np.percentile(train_df[col], 75, interpolation = "midpoint")
        IQR = Q3 - Q1
        # Right hand side outliers
        outliers_row_idxs = np.union1d(
            outliers_row_idxs, np.where(train_df[col] >= (Q3+(1.5+delta)*IQR))
        )
        # Left hand side outliers
        outliers_row_idxs = np.union1d(
            outliers_row_idxs, np.where(train_df[col] <= (Q1-(1.5+delta)*IQR))
        )
    if inplace:
    	train_df.drop(outliers_row_idxs, inplace=True)
    	train_labels_df.drop(outliers_row_idxs, inplace=True)
    else:
    	train_df.drop(outliers_row_idxs), train_labels_df.drop(outliers_row_idxs)

def _cal_skewness(df: Dataframe, cols: List[str], threshold: float=1.0) -> List[str]:
    """
    Private function to return skewed columns in a dataframe.

    Parameters:
        df: Dataframe object
        cols: List of column names to check for skewness
        threshold: Threshold to use for checking skewness.

    Returns:
        skewed_cols: List of skewed columns
    """

    skewed_cols = []
    for col in cols:
        if abs(df[col].skew()) >= threshold:
            skewed_cols.append(col)
            
    return skewed_cols

def show_skewness(df: DataFrame, threshold: float=1.0) -> Optional[List[str]]:
    """
    Function to return & display skewed columns in a dataframe if present.

    Parameters:
        df: Dataframe object
        cols: List of column names to check for skewness
        threshold: Threshold to use for checking skewness.

    Returns:
        skewed_cols: List of skewed columns
    """
    skewed_cols = _cal_skewness(df, df.columns, threshold)
    n_skewed_cols = len(skewed_cols)
    if n_skewed_cols > 0:
        start_col_idx = 0
        n_subplots = sum(divmod(n_skewed_cols, 4))
        n_rows = sum(divmod(n_subplots, 2))
        figure, ax = plt.subplots(n_rows, 2, figsize=(6.4*2, 4.8*n_subplots))
        for row in range(n_rows):
            for col in range(2):
                if start_col_idx+3 < n_skewed_cols:
                    df.boxplot(column=skewed_cols[start_col_idx:start_col_idx+4], ax=ax[row][col])
                elif start_col_idx < n_skewed_cols:
                    df.boxplot(column=skewed_cols[start_col_idx:], ax=ax[row][col])
                ax[row][col].set_title("Skewness")
                start_col_idx += 4
                
        plt.show()
    else:
        print(f"Skewness of all the columns are less that the given threshold: {threshold}")
        
    return skewed_cols

def fill_nulls(df: DataFrame, mode: str="mean") -> None:
    """
    This will fill/replace null values with the specified mode.

    Parameters:
        df: Dataframe object
        mode: Mode to use for filling null values. Options are: `mean`, `median`, `min` & `max`
    """

    fill_null_map = dict()
    for col in df.columns:
        if mode == "mean":
            fill_null_map[col] = df[col].mean()
        elif mode == "median":
            fill_null_map[col] = df[col].median()
        elif mode == "min":
            fill_null_map[col] = df[col].min()
        elif mode == "max":
            fill_null_map[col] = df[col].max()
        else:
            raise ValueError("Invalid mode provided for filling missing values")
            
    df.fillna(fill_null_map, inplace=True)
    print(f"Filled missing values with mode as {mode}")