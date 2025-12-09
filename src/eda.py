"""
Exploratory Data Analysis (EDA) functions for insurance risk analytics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_numeric_distribution(df: pd.DataFrame, column: str, bins: int = 50, 
                              log_scale: bool = False, ax=None):
    """
    Plot distribution of a numeric column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to plot
    bins : int
        Number of bins for histogram
    log_scale : bool
        Whether to use log scale
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    data = df[column].dropna()
    
    if log_scale and (data > 0).all():
        data = np.log1p(data)
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel(f'Log({column})')
    else:
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel(column)
    
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {column}')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_categorical_distribution(df: pd.DataFrame, column: str, top_n: int = 20, ax=None):
    """
    Plot distribution of a categorical column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to plot
    top_n : int
        Number of top categories to show
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    value_counts = df[column].value_counts().head(top_n)
    
    ax.barh(range(len(value_counts)), value_counts.values, edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(value_counts)))
    ax.set_yticklabels(value_counts.index)
    ax.set_xlabel('Count')
    ax.set_title(f'Distribution of {column} (Top {top_n})')
    ax.grid(True, alpha=0.3, axis='x')
    
    return ax


def plot_boxplot(df: pd.DataFrame, column: str, by: Optional[str] = None, ax=None):
    """
    Create box plot for outlier detection.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to plot
    by : str, optional
        Group by this column
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if by:
        df.boxplot(column=column, by=by, ax=ax)
        ax.set_title(f'Box Plot of {column} by {by}')
    else:
        df.boxplot(column=column, ax=ax)
        ax.set_title(f'Box Plot of {column}')
    
    ax.set_ylabel(column)
    plt.xticks(rotation=45)
    
    return ax


def plot_correlation_matrix(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None, 
                           figsize: Tuple[int, int] = (12, 10)):
    """
    Plot correlation matrix for numeric columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numeric_cols : list, optional
        List of numeric columns. If None, uses all numeric columns.
    figsize : tuple
        Figure size
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, ax


def plot_temporal_trends(df: pd.DataFrame, date_col: str, value_col: str, 
                        agg_func: str = 'sum', figsize: Tuple[int, int] = (14, 6)):
    """
    Plot temporal trends over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    date_col : str
        Date column name
    value_col : str
        Value column to aggregate
    agg_func : str
        Aggregation function ('sum', 'mean', 'count')
    figsize : tuple
        Figure size
    """
    df_temporal = df.copy()
    df_temporal[date_col] = pd.to_datetime(df_temporal[date_col], errors='coerce')
    df_temporal = df_temporal.dropna(subset=[date_col])
    
    if agg_func == 'sum':
        grouped = df_temporal.groupby(date_col)[value_col].sum()
    elif agg_func == 'mean':
        grouped = df_temporal.groupby(date_col)[value_col].mean()
    else:
        grouped = df_temporal.groupby(date_col)[value_col].count()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Date')
    ax.set_ylabel(value_col)
    ax.set_title(f'Temporal Trend: {value_col} ({agg_func})')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig, ax


def plot_loss_ratio_by_category(df: pd.DataFrame, category_col: str, 
                               top_n: int = 15, figsize: Tuple[int, int] = (12, 6)):
    """
    Plot loss ratio by category.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    category_col : str
        Category column to group by
    top_n : int
        Number of top categories to show
    figsize : tuple
        Figure size
    """
    grouped = df.groupby(category_col).agg({
        'TotalClaims': 'sum',
        'TotalPremium': 'sum'
    }).reset_index()
    
    grouped['LossRatio'] = grouped['TotalClaims'] / grouped['TotalPremium']
    grouped = grouped.sort_values('LossRatio', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(grouped)), grouped['LossRatio'].values, 
                   edgecolor='black', alpha=0.7, 
                   cmap='RdYlGn_r' if grouped['LossRatio'].max() > 1 else 'viridis')
    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels(grouped[category_col].values)
    ax.set_xlabel('Loss Ratio (TotalClaims / TotalPremium)')
    ax.set_title(f'Loss Ratio by {category_col}')
    ax.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='Break-even (1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    return fig, ax


def create_insightful_visualization_1(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Creative visualization 1: Risk-return scatter plot by Province.
    Shows relationship between TotalPremium and TotalClaims by Province.
    """
    province_stats = df.groupby('Province').agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum',
        'PolicyID': 'count'
    }).reset_index()
    
    province_stats['LossRatio'] = province_stats['TotalClaims'] / province_stats['TotalPremium']
    province_stats['AvgPremium'] = province_stats['TotalPremium'] / province_stats['PolicyID']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    scatter = ax.scatter(province_stats['AvgPremium'], province_stats['LossRatio'],
                        s=province_stats['PolicyID']/10, alpha=0.6, 
                        c=province_stats['LossRatio'], cmap='RdYlGn_r',
                        edgecolors='black', linewidth=1.5)
    
    # Add labels
    for idx, row in province_stats.iterrows():
        ax.annotate(row['Province'], 
                   (row['AvgPremium'], row['LossRatio']),
                   fontsize=9, alpha=0.8)
    
    ax.set_xlabel('Average Premium per Policy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Risk-Return Analysis by Province\n(Bubble size = Number of Policies)', 
                fontsize=14, fontweight='bold')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Break-even')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.colorbar(scatter, ax=ax, label='Loss Ratio')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def create_insightful_visualization_2(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Creative visualization 2: Temporal analysis of claims and premiums with trend.
    Shows monthly trends and seasonality.
    """
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    monthly_stats = df.groupby('TransactionMonth').agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum',
        'PolicyID': 'count'
    }).reset_index()
    
    monthly_stats['LossRatio'] = monthly_stats['TotalClaims'] / monthly_stats['TotalPremium']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Premium and Claims over time
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(monthly_stats['TransactionMonth'], monthly_stats['TotalPremium'], 
                     'b-', marker='o', label='Total Premium', linewidth=2)
    line2 = ax1_twin.plot(monthly_stats['TransactionMonth'], monthly_stats['TotalClaims'], 
                          'r-', marker='s', label='Total Claims', linewidth=2)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Premium (ZAR)', color='b', fontweight='bold')
    ax1_twin.set_ylabel('Total Claims (ZAR)', color='r', fontweight='bold')
    ax1.set_title('Monthly Premium vs Claims Trend', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Loss Ratio over time
    ax2 = axes[1]
    ax2.plot(monthly_stats['TransactionMonth'], monthly_stats['LossRatio'], 
             'g-', marker='o', linewidth=2, markersize=6)
    ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Break-even')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Loss Ratio', fontweight='bold')
    ax2.set_title('Monthly Loss Ratio Trend', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 3: Number of policies
    ax3 = axes[2]
    ax3.bar(monthly_stats['TransactionMonth'], monthly_stats['PolicyID'], 
            alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Number of Policies', fontweight='bold')
    ax3.set_title('Monthly Policy Count', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def create_insightful_visualization_3(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Creative visualization 3: Multi-dimensional analysis of vehicle risk factors.
    Shows relationship between vehicle characteristics and risk metrics.
    """
    # Group by vehicle make and type
    vehicle_risk = df.groupby(['Make', 'VehicleType']).agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum',
        'PolicyID': 'count'
    }).reset_index()
    
    vehicle_risk['LossRatio'] = vehicle_risk['TotalClaims'] / vehicle_risk['TotalPremium']
    vehicle_risk = vehicle_risk[vehicle_risk['PolicyID'] >= 50]  # Filter for sufficient sample size
    vehicle_risk = vehicle_risk.sort_values('LossRatio', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create color map based on VehicleType
    unique_types = vehicle_risk['VehicleType'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
    color_map = dict(zip(unique_types, colors))
    
    for vtype in unique_types:
        data = vehicle_risk[vehicle_risk['VehicleType'] == vtype]
        ax.scatter(data['PolicyID'], data['LossRatio'], 
                  s=data['TotalPremium']/1000, alpha=0.6,
                  c=[color_map[vtype]], label=vtype,
                  edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Number of Policies (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Vehicle Risk Profile: Make vs Type\n(Bubble size = Total Premium)', 
                fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Break-even')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

