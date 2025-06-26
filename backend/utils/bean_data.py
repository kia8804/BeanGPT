import pandas as pd
import re # Import re for cultivar matching
import os # Import os for path handling
from typing import Dict, List, Tuple, Optional
import json
import numpy as np

# Path to the merged bean dataset file
MERGED_DATA_PATH = os.getenv("MERGED_DATA_PATH", "../data/Merged_Bean_Dataset.xlsx")

# ---- Load merged dataset ----
def load_all_trials() -> pd.DataFrame:
    """Load the bean trial data from Excel."""
    try:
        df = pd.read_excel(MERGED_DATA_PATH)

        # Ensure consistency and correct data types
        df = df[
            ~df["Cultivar Name"].astype(str).str.lower().isin(["mean", "cv", "lsd(0.05)"])
        ]
        # Convert relevant columns to numeric, coercing errors to NaN
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
        df["Maturity"] = pd.to_numeric(df["Maturity"], errors="coerce")
        
        # Optional: Add lowercase bean_type and trial_group for easier filtering if they exist
        if 'bean_type' in df.columns:
            df['bean_type'] = df['bean_type'].astype(str).str.lower()
        if 'trial_group' in df.columns:
            df['trial_group'] = df['trial_group'].astype(str).str.lower()

        print(f"Loaded {len(df)} rows from {MERGED_DATA_PATH}")
        return df
    except FileNotFoundError:
        print(f"Error: Merged bean dataset not found at {MERGED_DATA_PATH}")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        print(f"Error loading bean data from {MERGED_DATA_PATH}: {e}")
        return pd.DataFrame() # Return empty DataFrame on other errors

# Load the full dataset once when the module is imported
df_trials = load_all_trials()

# ---- Chart Data Functions ----
def create_yearly_trend_chart_data(yearly_stats: pd.DataFrame, analysis_column: str, filter_text: str = "") -> Dict:
    """Return chart data for yearly trend visualization."""
    
    return {
        "type": "line",
        "title": f"Yearly {analysis_column} Trends {filter_text}",
        "data": {
            "labels": yearly_stats['Year'].tolist(),
            "datasets": [
                {
                    "label": f"Average {analysis_column}",
                    "data": yearly_stats['Average'].tolist(),
                    "borderColor": "#2E8B57",
                    "backgroundColor": "#2E8B57",
                    "fill": False,
                    "tension": 0.1
                }
            ]
        },
        "options": {
            "responsive": True,
            "scales": {
                "y": {
                    "beginAtZero": False,
                    "title": {
                        "display": True,
                        "text": analysis_column
                    }
                },
                "x": {
                    "title": {
                        "display": True,
                        "text": "Year"
                    }
                }
            }
        },
        "plugins": {
            "tooltip": {
                "callbacks": {
                    "title": "function(context) { return context[0].dataset.label || 'Data Point'; }",
                    "label": "function(context) { return context.dataset.label + ': ' + context.formattedValue + ' (' + context.parsed.x + ', ' + context.parsed.y + ')'; }"
                }
            },
            "legend": {
                "display": True,
                "position": "right",
                "labels": {
                    "usePointStyle": True,
                    "pointStyle": "circle"
                }
            }
        }
    }

def create_scatter_chart_data(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None, filter_text: str = "", highlight_cultivar: str = None) -> Dict:
    """Return chart data for scatter plot visualization."""
    
    # Prepare data
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    
    if len(plot_df) == 0:
        return {"error": "No data available for scatter plot."}
    
    datasets = []
    
    # If we have cultivar information and want to show individual points
    if 'Cultivar Name' in plot_df.columns and (len(plot_df) <= 100 or highlight_cultivar):  # Always use individual points if highlighting
        # Create individual points for each cultivar
        cultivars = plot_df['Cultivar Name'].unique()
        print(f"🔍 Found {len(cultivars)} cultivars in data")
        if highlight_cultivar:
            print(f"🔍 Looking to highlight: '{highlight_cultivar}'")
            print(f"🔍 Sample cultivars: {list(cultivars)[:5]}")
        
        colors = [
            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', 
            '#2E8B57', '#C9CBCF', '#E74C3C', '#3498DB', '#F39C12', '#27AE60',
            '#8E44AD', '#E67E22', '#34495E', '#95A5A6', '#1ABC9C', '#F1C40F',
            '#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3', '#03A9F4',
            '#00BCD4', '#009688', '#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B',
            '#FFC107', '#FF9800', '#FF5722', '#795548', '#607D8B', '#FFB6C1',
            '#DDA0DD', '#98FB98', '#F0E68C', '#D2B48C', '#BC8F8F', '#F4A460',
            '#DA70D6', '#EEE8AA', '#98D982', '#F0B27A', '#85C1E9', '#F8C471'
        ]
        
        # Check if we need to highlight a specific cultivar
        if highlight_cultivar:
            highlight_cultivar_lower = highlight_cultivar.lower()
            
            # Find matching cultivars (case-insensitive partial match)
            matching_cultivars = [c for c in cultivars if highlight_cultivar_lower in str(c).lower()]
            print(f"🔍 Highlighting logic activated!")
            print(f"🔍 Looking for cultivar: '{highlight_cultivar}'")
            print(f"🔍 Total cultivars available: {len(cultivars)}")
            print(f"🔍 Total data points: {len(plot_df)}")
            print(f"🔍 Sample cultivars: {list(cultivars)[:10]}")
            print(f"🔍 Matching cultivars found: {matching_cultivars}")
            
            # Create highlighted dataset first
            for cultivar in matching_cultivars:
                cultivar_data = plot_df[plot_df['Cultivar Name'] == cultivar]
                print(f"🔍 Creating highlighted dataset for '{cultivar}' with {len(cultivar_data)} data points")
                data_points = []
                for _, row in cultivar_data.iterrows():
                    location = row.get('Location', '')
                    year = row.get('Year', '')
                    details = [str(location), str(year)] if location or year else []
                    location_info = f" ({', '.join(details)})" if details else ""
                    label = f"{cultivar}{location_info}: {row[x_col]:.0f} days, {row[y_col]:.0f} kg/ha"
                    data_points.append({
                        "x": row[x_col], 
                        "y": row[y_col], 
                        "label": label,
                        "cultivar": cultivar
                    })
                
                highlighted_dataset = {
                    "label": f"{cultivar} ⭐ HIGHLIGHTED",
                    "data": data_points,
                    "backgroundColor": "#FF4500",  # Bright orange-red for maximum visibility
                    "borderColor": "#FFFFFF",  # White border for contrast
                    "pointRadius": 12,  # Even larger points for highlighting
                    "pointBorderWidth": 4,  # Thicker border
                    "pointStyle": "rectRot",  # Diamond shape for highlighting (more visible than star)
                    "pointHoverRadius": 15,  # Larger on hover
                    "pointHoverBorderWidth": 5
                }
                datasets.append(highlighted_dataset)
                print(f"🔍 Added highlighted dataset: {highlighted_dataset['label']} with {len(data_points)} points")
            
            # Create dataset for other cultivars (smaller, less prominent)
            other_cultivars = [c for c in cultivars if c not in matching_cultivars]
            
            # Group other cultivars into fewer datasets to reduce legend clutter
            if len(other_cultivars) > 10:
                # Create a single dataset for all other cultivars
                other_df = plot_df[plot_df['Cultivar Name'].isin(other_cultivars)]
                if len(other_df) > 0:
                    data_points = []
                    for _, row in other_df.iterrows():
                        cultivar = row['Cultivar Name']
                        location = row.get('Location', '')
                        year = row.get('Year', '')
                        details = [str(location), str(year)] if location or year else []
                        location_info = f" ({', '.join(details)})" if details else ""
                        label = f"{cultivar}{location_info}: {row[x_col]:.0f} days, {row[y_col]:.0f} kg/ha"
                        data_points.append({
                            "x": row[x_col], 
                            "y": row[y_col], 
                            "label": label,
                            "cultivar": cultivar
                        })
                    
                    datasets.append({
                        "label": f"Other Cultivars ({len(other_cultivars)} varieties)",
                        "data": data_points,
                        "backgroundColor": "#6B7280",  # Gray for background data
                        "borderColor": "#4B5563",  # Darker gray border
                        "pointRadius": 3,  # Smaller points for background
                        "pointHoverRadius": 5,  # Slightly larger on hover
                        "pointBorderWidth": 1
                    })
            else:
                # Show individual cultivars if there aren't too many
                for i, cultivar in enumerate(other_cultivars):
                    cultivar_data = plot_df[plot_df['Cultivar Name'] == cultivar]
                    data_points = []
                    for _, row in cultivar_data.iterrows():
                        location = row.get('Location', '')
                        year = row.get('Year', '')
                        details = [str(location), str(year)] if location or year else []
                        location_info = f" ({', '.join(details)})" if details else ""
                        label = f"{cultivar}{location_info}: {row[x_col]:.0f} days, {row[y_col]:.0f} kg/ha"
                        data_points.append({
                            "x": row[x_col], 
                            "y": row[y_col], 
                            "label": label,
                            "cultivar": cultivar
                        })
                    
                    datasets.append({
                        "label": cultivar,
                        "data": data_points,
                        "backgroundColor": "#9CA3AF",  # Light gray for background cultivars
                        "borderColor": "#6B7280",  # Darker gray border
                        "pointRadius": 3,  # Smaller points for background
                        "pointHoverRadius": 5,  # Slightly larger on hover
                        "pointBorderWidth": 1
                    })
        
        else:
            # No highlighting - show individual cultivar points, not grouped by color_col
            if 'Cultivar Name' in plot_df.columns:
                # Create individual datasets for each cultivar (or group them smartly)
                cultivars = plot_df['Cultivar Name'].unique()
                
                if len(cultivars) <= 50:  # Show individual cultivars for reasonable numbers (increased from 20)
                    colors = [
                        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', 
                        '#2E8B57', '#C9CBCF', '#E74C3C', '#3498DB', '#F39C12', '#27AE60',
                        '#8E44AD', '#E67E22', '#34495E', '#95A5A6', '#1ABC9C', '#F1C40F',
                        '#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3', '#03A9F4',
                        '#00BCD4', '#009688', '#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B',
                        '#FFC107', '#FF9800', '#FF5722', '#795548', '#607D8B', '#FFB6C1',
                        '#DDA0DD', '#98FB98', '#F0E68C', '#D2B48C', '#BC8F8F', '#F4A460',
                        '#DA70D6', '#EEE8AA', '#98D982', '#F0B27A', '#85C1E9', '#F8C471'
                    ]
                    
                    # Limit individual cultivars to 15 for better legend readability
                    if len(cultivars) <= 15:
                        for i, cultivar in enumerate(cultivars):
                            cultivar_data = plot_df[plot_df['Cultivar Name'] == cultivar]
                            data_points = []
                            for _, row in cultivar_data.iterrows():
                                location = row.get('Location', '')
                                year = row.get('Year', '')
                                details = [str(location), str(year)] if location or year else []
                                location_info = f" ({', '.join(details)})" if details else ""
                                label = f"{cultivar}{location_info}: {row[x_col]:.0f} days, {row[y_col]:.0f} kg/ha"
                                data_points.append({
                                    "x": row[x_col], 
                                    "y": row[y_col], 
                                    "label": label,
                                    "cultivar": cultivar
                                })
                            
                            datasets.append({
                                "label": cultivar,
                                "data": data_points,
                                "backgroundColor": colors[i % len(colors)],
                                "borderColor": colors[i % len(colors)],
                                "pointRadius": 5
                            })
                    else:
                        # Too many cultivars - create a single dataset with all points
                        data_points = []
                        for _, row in plot_df.iterrows():
                            cultivar = row['Cultivar Name']
                            location = row.get('Location', '')
                            year = row.get('Year', '')
                            details = [str(location), str(year)] if location or year else []
                            location_info = f" ({', '.join(details)})" if details else ""
                            label = f"{cultivar}{location_info}: {row[x_col]:.0f} days, {row[y_col]:.0f} kg/ha"
                            data_points.append({
                                "x": row[x_col], 
                                "y": row[y_col], 
                                "label": label,
                                "cultivar": cultivar
                            })
                        
                        datasets.append({
                            "label": f"All Cultivars ({len(cultivars)} varieties)",
                            "data": data_points,
                            "backgroundColor": "#3B82F6",  # Bright blue instead of green
                            "borderColor": "#1D4ED8",  # Darker blue border
                            "pointRadius": 5,
                            "pointHoverRadius": 7,
                            "pointBorderWidth": 2
                        })
                else:
                    # Too many cultivars - group by color_col if available, otherwise single dataset
                    if color_col and color_col in plot_df.columns:
                        categories = plot_df[color_col].unique()
                        
                        for i, category in enumerate(categories):
                            mask = plot_df[color_col] == category
                            category_data = plot_df[mask]
                            
                            data_points = []
                            for _, row in category_data.iterrows():
                                cultivar = row.get('Cultivar Name', 'Unknown')
                                location = row.get('Location', '')
                                year = row.get('Year', '')
                                details = [str(location), str(year)] if location or year else []
                                location_info = f" ({', '.join(details)})" if details else ""
                                label = f"{cultivar}{location_info}: {row[x_col]:.0f} days, {row[y_col]:.0f} kg/ha"
                                data_points.append({
                                    "x": row[x_col], 
                                    "y": row[y_col], 
                                    "label": label,
                                    "cultivar": cultivar
                                })
                            
                            datasets.append({
                                "label": str(category),
                                "data": data_points,
                                "backgroundColor": colors[i % len(colors)],
                                "borderColor": colors[i % len(colors)],
                                "pointRadius": 5
                            })
                    else:
                        # Show all points as single dataset
                        data_points = []
                        for _, row in plot_df.iterrows():
                            cultivar = row.get('Cultivar Name', 'Unknown')
                            location = row.get('Location', '')
                            year = row.get('Year', '')
                            details = [str(location), str(year)] if location or year else []
                            location_info = f" ({', '.join(details)})" if details else ""
                            label = f"{cultivar}{location_info}: {row[x_col]:.0f} days, {row[y_col]:.0f} kg/ha"
                            data_points.append({
                                "x": row[x_col], 
                                "y": row[y_col], 
                                "label": label,
                                "cultivar": cultivar
                            })
                        
                        datasets.append({
                            "label": f"All Cultivars ({len(cultivars)} varieties)",
                            "data": data_points,
                            "backgroundColor": "#3B82F6",  # Bright blue instead of green
                            "borderColor": "#1D4ED8",  # Darker blue border
                            "pointRadius": 5,
                            "pointHoverRadius": 7,
                            "pointBorderWidth": 2
                        })
            else:
                # No cultivar column - show all points as single dataset
                data_points = []
                for _, row in plot_df.iterrows():
                    cultivar = row.get('Cultivar Name', 'Unknown')
                    location = row.get('Location', '')
                    year = row.get('Year', '')
                    details = [str(location), str(year)] if location or year else []
                    location_info = f" ({', '.join(details)})" if details else ""
                    label = f"{cultivar}{location_info}: {row[x_col]:.0f} days, {row[y_col]:.0f} kg/ha"
                    data_points.append({
                        "x": row[x_col], 
                        "y": row[y_col], 
                        "label": label,
                        "cultivar": cultivar
                    })
                
                datasets.append({
                    "label": f"{y_col} vs {x_col}",
                    "data": data_points,
                    "backgroundColor": "#3B82F6",  # Bright blue instead of green
                    "borderColor": "#1D4ED8",  # Darker blue border
                    "pointRadius": 5,
                    "pointHoverRadius": 7,
                    "pointBorderWidth": 2
                })
    
    else:
        # Fallback to original grouping logic for large datasets
        if color_col and color_col in plot_df.columns:
            # Group by color column
            categories = plot_df[color_col].unique()
            colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
            
            for i, category in enumerate(categories):
                mask = plot_df[color_col] == category
                category_data = plot_df[mask]
                
                data_points = []
                for _, row in category_data.iterrows():
                    cultivar = row.get('Cultivar Name', 'Unknown')
                    location = row.get('Location', '')
                    year = row.get('Year', '')
                    details = [str(location), str(year)] if location or year else []
                    location_info = f" ({', '.join(details)})" if details else ""
                    label = f"{cultivar}{location_info}: {row[x_col]:.0f} days, {row[y_col]:.0f} kg/ha"
                    data_points.append({
                        "x": row[x_col], 
                        "y": row[y_col], 
                        "label": label,
                        "cultivar": cultivar
                    })
                
                datasets.append({
                    "label": str(category),
                    "data": data_points,
                    "backgroundColor": colors[i % len(colors)],
                    "borderColor": colors[i % len(colors)]
                })
        else:
            data_points = []
            for _, row in plot_df.iterrows():
                cultivar = row.get('Cultivar Name', 'Unknown')
                location = row.get('Location', '')
                year = row.get('Year', '')
                details = [str(location), str(year)] if location or year else []
                location_info = f" ({', '.join(details)})" if details else ""
                label = f"{cultivar}{location_info}: {row[x_col]:.0f} days, {row[y_col]:.0f} kg/ha"
                data_points.append({
                    "x": row[x_col], 
                    "y": row[y_col], 
                    "label": label,
                    "cultivar": cultivar
                })
            
            datasets.append({
                "label": f"{y_col} vs {x_col}",
                "data": data_points,
                "backgroundColor": "#3B82F6",  # Bright blue instead of green
                "borderColor": "#1D4ED8",  # Darker blue border
                "pointRadius": 5,
                "pointHoverRadius": 7,
                "pointBorderWidth": 2
            })
    
    # Create a more descriptive title
    title_parts = [f"{y_col} vs {x_col}"]
    if highlight_cultivar:
        title_parts.append(f"(Highlighting {highlight_cultivar})")
    if filter_text and filter_text != "()":
        title_parts.append(filter_text)
    
    # Debug: Print final dataset info
    print(f"🔍 Final datasets created: {len(datasets)}")
    for i, ds in enumerate(datasets):
        print(f"🔍 Dataset {i}: '{ds['label']}' - {len(ds['data'])} points - color: {ds.get('backgroundColor', 'N/A')}")
    
    return {
        "type": "scatter",
        "title": " ".join(title_parts),
        "data": {
            "datasets": datasets
        },
        "options": {
            "responsive": True,
            "scales": {
                "x": {
                    "title": {
                        "display": True,
                        "text": x_col
                    }
                },
                "y": {
                    "title": {
                        "display": True,
                        "text": y_col
                    }
                }
            },
            "plugins": {
                "tooltip": {
                    "mode": "point",
                    "intersect": True
                },
                "legend": {
                    "display": True,
                    "position": "right",
                    "labels": {
                        "usePointStyle": True,
                        "pointStyle": "circle"
                    }
                }
            }
        }
    }

def create_comparison_chart_data(df: pd.DataFrame, group_col: str, value_col: str, filter_text: str = "") -> Dict:
    """Return chart data for comparison bar chart visualization."""
    
    plot_df = df.dropna(subset=[group_col, value_col]).copy()
    
    if len(plot_df) == 0:
        return {"error": "No data available for comparison plot."}
    
    # Calculate statistics by group
    group_stats = plot_df.groupby(group_col).agg({
        value_col: ['mean', 'count', 'std']
    }).round(2)
    
    group_stats.columns = ['Average', 'Count', 'Std_Dev']
    group_stats = group_stats.reset_index()
    group_stats = group_stats.sort_values('Average', ascending=False)
    
    return {
        "type": "bar",
        "title": f"{value_col} Distribution by {group_col.replace('_', ' ').title()} {filter_text}",
        "data": {
            "labels": group_stats[group_col].tolist(),
            "datasets": [
                {
                    "label": f"Average {value_col}",
                    "data": group_stats['Average'].tolist(),
                    "backgroundColor": "#2E8B57",
                    "borderColor": "#2E8B57",
                    "borderWidth": 1
                }
            ]
        },
        "options": {
            "responsive": True,
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "title": {
                        "display": True,
                        "text": value_col
                    }
                },
                "x": {
                    "title": {
                        "display": True,
                        "text": group_col.replace('_', ' ').title()
                    }
                }
            }
        },
        "plugins": {
            "tooltip": {
                "callbacks": {
                    "title": "function(context) { return context[0].dataset.label || 'Data Point'; }",
                    "label": "function(context) { return context.dataset.label + ': ' + context.formattedValue + ' (' + context.parsed.x + ', ' + context.parsed.y + ')'; }"
                }
            },
            "legend": {
                "display": True,
                "position": "right",
                "labels": {
                    "usePointStyle": True,
                    "pointStyle": "circle"
                }
            }
        }
    }

def create_location_performance_chart_data(df: pd.DataFrame, value_col: str = 'Yield', filter_text: str = "") -> Dict:
    """Return chart data for location performance horizontal bar chart."""
    
    if 'Location' not in df.columns:
        return {"error": "Location data not available for analysis."}
    
    # Group by location and calculate statistics
    location_stats = df.groupby('Location').agg({
        value_col: ['count', 'mean', 'std']
    }).round(2)
    
    location_stats.columns = ['Count', 'Average', 'Std_Dev']
    location_stats = location_stats.reset_index()
    
    # Sort by average value (descending for better visualization)
    location_stats = location_stats.sort_values('Average', ascending=False)
    
    return {
        "type": "horizontalBar",
        "title": f"Average {value_col} by Location {filter_text}",
        "data": {
            "labels": location_stats['Location'].tolist(),
            "datasets": [
                {
                    "label": f"Average {value_col}",
                    "data": location_stats['Average'].tolist(),
                    "backgroundColor": "#2E8B57",
                    "borderColor": "#2E8B57",
                    "borderWidth": 1
                }
            ]
        },
        "options": {
            "responsive": True,
            "indexAxis": "y",
            "scales": {
                "x": {
                    "beginAtZero": True,
                    "title": {
                        "display": True,
                        "text": f"Average {value_col}"
                    }
                },
                "y": {
                    "title": {
                        "display": True,
                        "text": "Location"
                    }
                }
            }
        },
        "plugins": {
            "tooltip": {
                "callbacks": {
                    "title": "function(context) { return context[0].dataset.label || 'Data Point'; }",
                    "label": "function(context) { return context.dataset.label + ': ' + context.formattedValue + ' (' + context.parsed.x + ', ' + context.parsed.y + ')'; }"
                }
            },
            "legend": {
                "display": True,
                "position": "right",
                "labels": {
                    "usePointStyle": True,
                    "pointStyle": "circle"
                }
            }
        }
    }

def create_cultivar_performance_chart_data(df: pd.DataFrame, value_col: str = 'Yield', top_n: int = 20, filter_text: str = "") -> Dict:
    """Return chart data for top cultivar performance horizontal bar chart."""
    
    if 'Cultivar Name' not in df.columns or len(df) == 0:
        return {"error": "Cultivar data not available."}
    
    # Calculate average performance by cultivar
    cultivar_stats = df.groupby('Cultivar Name').agg({
        value_col: ['count', 'mean', 'std']
    }).round(2)
    
    cultivar_stats.columns = ['Count', 'Average', 'Std_Dev']
    cultivar_stats = cultivar_stats.reset_index()
    
    # Filter for cultivars with sufficient data and get top performers
    cultivar_stats = cultivar_stats[cultivar_stats['Count'] >= 3]  # At least 3 records
    top_cultivars = cultivar_stats.nlargest(min(top_n, 15), 'Average')  # Limit to 15 for readability
    
    if len(top_cultivars) == 0:
        return {"error": "Insufficient data for cultivar performance chart."}
    
    # Sort for best display (descending)
    top_cultivars = top_cultivars.sort_values('Average', ascending=False)
    
    # Truncate long cultivar names for better display
    cultivar_labels = [name[:25] + '...' if len(name) > 25 else name 
                      for name in top_cultivars['Cultivar Name']]
    
    return {
        "type": "horizontalBar", 
        "title": f"Top {len(top_cultivars)} Cultivars by {value_col} {filter_text}",
        "data": {
            "labels": cultivar_labels,
            "datasets": [
                {
                    "label": f"Average {value_col}",
                    "data": top_cultivars['Average'].tolist(),
                    "backgroundColor": "#2E8B57",
                    "borderColor": "#2E8B57",
                    "borderWidth": 1
                }
            ]
        },
        "options": {
            "responsive": True,
            "indexAxis": "y",
            "scales": {
                "x": {
                    "beginAtZero": True,
                    "title": {
                        "display": True,
                        "text": f"Average {value_col}"
                    }
                },
                "y": {
                    "title": {
                        "display": True,
                        "text": "Cultivar"
                    }
                }
            }
        },
        "plugins": {
            "tooltip": {
                "callbacks": {
                    "title": "function(context) { return context[0].dataset.label || 'Data Point'; }",
                    "label": "function(context) { return context.dataset.label + ': ' + context.formattedValue + ' (' + context.parsed.x + ', ' + context.parsed.y + ')'; }"
                }
            },
            "legend": {
                "display": True,
                "position": "right",
                "labels": {
                    "usePointStyle": True,
                    "pointStyle": "circle"
                }
            }
        }
    }

def create_pie_chart_data(df: pd.DataFrame, group_col: str, value_col: str = None, filter_text: str = "") -> Dict:
    """Return chart data for pie chart visualization."""
    
    if group_col not in df.columns:
        return {"error": f"Column '{group_col}' not available for pie chart."}
    
    plot_df = df.dropna(subset=[group_col]).copy()
    
    if len(plot_df) == 0:
        return {"error": "No data available for pie chart."}
    
    if value_col and value_col in df.columns:
        # Sum values by group
        pie_data = plot_df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
        title = f"{value_col} Distribution by {group_col.replace('_', ' ').title()} {filter_text}"
    else:
        # Count records by group
        pie_data = plot_df[group_col].value_counts()
        title = f"Record Count by {group_col.replace('_', ' ').title()} {filter_text}"
    
    # Take top 10 to avoid overcrowded pie chart
    if len(pie_data) > 10:
        pie_data = pie_data.head(10)
    
    colors = [
        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', 
        '#FF9F40', '#FF6384', '#C9CBCF', '#4BC0C0', '#FF6384'
    ]
    
    return {
        "type": "pie",
        "title": title,
        "data": {
            "labels": pie_data.index.tolist(),
            "datasets": [
                {
                    "data": pie_data.values.tolist(),
                    "backgroundColor": colors[:len(pie_data)],
                    "borderWidth": 1
                }
            ]
        },
        "options": {
            "responsive": True,
            "plugins": {
                "tooltip": {
                    "callbacks": {
                        "title": "function(context) { return context[0].dataset.label || 'Data Point'; }",
                        "label": "function(context) { return context.dataset.label + ': ' + context.formattedValue + ' (' + context.parsed.x + ', ' + context.parsed.y + ')'; }"
                    }
                },
                "legend": {
                    "display": True,
                    "position": "right",
                    "labels": {
                        "usePointStyle": True,
                        "pointStyle": "circle"
                    }
                }
            }
        }
    }

def create_line_chart_data(df: pd.DataFrame, x_col: str, y_col: str, group_col: str = None, filter_text: str = "") -> Dict:
    """Return chart data for line chart visualization."""
    
    if x_col not in df.columns or y_col not in df.columns:
        return {"error": f"Required columns '{x_col}' or '{y_col}' not available."}
    
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    
    if len(plot_df) == 0:
        return {"error": "No data available for line chart."}
    
    # Group data by x_col and calculate average y_col
    if group_col and group_col in plot_df.columns:
        # Multiple lines for different groups
        datasets = []
        groups = plot_df[group_col].unique()
        colors = ['#2E8B57', '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
        
        for i, group in enumerate(groups):
            group_data = plot_df[plot_df[group_col] == group]
            line_data = group_data.groupby(x_col)[y_col].mean().reset_index()
            line_data = line_data.sort_values(x_col)
            
            datasets.append({
                "label": str(group),
                "data": line_data[y_col].tolist(),
                "borderColor": colors[i % len(colors)],
                "backgroundColor": colors[i % len(colors)],
                "fill": False,
                "tension": 0.1
            })
        
        labels = sorted(plot_df[x_col].astype(str).unique())
    else:
        # Single line
        line_data = plot_df.groupby(x_col)[y_col].mean().reset_index()
        line_data = line_data.sort_values(x_col)
        
        datasets = [{
            "label": f"Average {y_col}",
            "data": line_data[y_col].tolist(),
            "borderColor": "#2E8B57",
            "backgroundColor": "#2E8B57",
            "fill": False,
            "tension": 0.1
        }]
        
        labels = line_data[x_col].tolist()
    
    return {
        "type": "line",
        "title": f"{y_col} Trend by {x_col} {filter_text}",
        "data": {
            "labels": labels,
            "datasets": datasets
        },
        "options": {
            "responsive": True,
            "scales": {
                "x": {
                    "title": {
                        "display": True,
                        "text": x_col
                    }
                },
                "y": {
                    "title": {
                        "display": True,
                        "text": y_col
                    }
                }
            }
        },
        "plugins": {
            "tooltip": {
                "callbacks": {
                    "title": "function(context) { return context[0].dataset.label || 'Data Point'; }",
                    "label": "function(context) { return context.dataset.label + ': ' + context.formattedValue + ' (' + context.parsed.x + ', ' + context.parsed.y + ')'; }"
                }
            },
            "legend": {
                "display": True,
                "position": "right",
                "labels": {
                    "usePointStyle": True,
                    "pointStyle": "circle"
                }
            }
        }
    }

def create_histogram_data(df: pd.DataFrame, value_col: str, bins: int = 10, filter_text: str = "") -> Dict:
    """Return chart data for histogram visualization."""
    
    if value_col not in df.columns:
        return {"error": f"Column '{value_col}' not available for histogram."}
    
    plot_df = df.dropna(subset=[value_col]).copy()
    
    if len(plot_df) == 0:
        return {"error": "No data available for histogram."}
    
    # Create histogram bins
    hist_data, bin_edges = np.histogram(plot_df[value_col], bins=bins)
    bin_labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" for i in range(len(bin_edges)-1)]
    
    return {
        "type": "bar",
        "title": f"{value_col} Distribution {filter_text}",
        "data": {
            "labels": bin_labels,
            "datasets": [
                {
                    "label": f"Frequency",
                    "data": hist_data.tolist(),
                    "backgroundColor": "#2E8B57",
                    "borderColor": "#2E8B57",
                    "borderWidth": 1
                }
            ]
        },
        "options": {
            "responsive": True,
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "title": {
                        "display": True,
                        "text": "Frequency"
                    }
                },
                "x": {
                    "title": {
                        "display": True,
                        "text": f"{value_col} Range"
                    }
                }
            }
        },
        "plugins": {
            "tooltip": {
                "callbacks": {
                    "title": "function(context) { return context[0].dataset.label || 'Data Point'; }",
                    "label": "function(context) { return context.dataset.label + ': ' + context.formattedValue + ' (' + context.parsed.x + ', ' + context.parsed.y + ')'; }"
                }
            },
            "legend": {
                "display": True,
                "position": "right",
                "labels": {
                    "usePointStyle": True,
                    "pointStyle": "circle"
                }
            }
        }
    }

def create_area_chart_data(df: pd.DataFrame, x_col: str, y_col: str, group_col: str = None, filter_text: str = "") -> Dict:
    """Return chart data for area chart visualization."""
    
    if x_col not in df.columns or y_col not in df.columns:
        return {"error": f"Required columns '{x_col}' or '{y_col}' not available."}
    
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    
    if len(plot_df) == 0:
        return {"error": "No data available for area chart."}
    
    # Similar to line chart but with fill: true
    if group_col and group_col in plot_df.columns:
        datasets = []
        groups = plot_df[group_col].unique()
        colors = ['#2E8B57', '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
        
        for i, group in enumerate(groups):
            group_data = plot_df[plot_df[group_col] == group]
            area_data = group_data.groupby(x_col)[y_col].mean().reset_index()
            area_data = area_data.sort_values(x_col)
            
            datasets.append({
                "label": str(group),
                "data": area_data[y_col].tolist(),
                "borderColor": colors[i % len(colors)],
                "backgroundColor": colors[i % len(colors)] + "40",  # Add transparency
                "fill": True,
                "tension": 0.1
            })
        
        labels = sorted(plot_df[x_col].astype(str).unique())
    else:
        area_data = plot_df.groupby(x_col)[y_col].mean().reset_index()
        area_data = area_data.sort_values(x_col)
        
        datasets = [{
            "label": f"Average {y_col}",
            "data": area_data[y_col].tolist(),
            "borderColor": "#2E8B57",
            "backgroundColor": "#2E8B5740",  # Add transparency
            "fill": True,
            "tension": 0.1
        }]
        
        labels = area_data[x_col].tolist()
    
    return {
        "type": "line",  # Area charts use line type with fill: true
        "title": f"{y_col} Area Chart by {x_col} {filter_text}",
        "data": {
            "labels": labels,
            "datasets": datasets
        },
        "options": {
            "responsive": True,
            "scales": {
                "x": {
                    "title": {
                        "display": True,
                        "text": x_col
                    }
                },
                "y": {
                    "title": {
                        "display": True,
                        "text": y_col
                    }
                }
            }
        },
        "plugins": {
            "tooltip": {
                "callbacks": {
                    "title": "function(context) { return context[0].dataset.label || 'Data Point'; }",
                    "label": "function(context) { return context.dataset.label + ': ' + context.formattedValue + ' (' + context.parsed.x + ', ' + context.parsed.y + ')'; }"
                }
            },
            "legend": {
                "display": True,
                "position": "right",
                "labels": {
                    "usePointStyle": True,
                    "pointStyle": "circle"
                }
            }
        }
    }

# ---- Handler for GPT call ----
def answer_bean_query(args: Dict) -> Tuple[str, str, Dict]:
    """
    Queries the bean trial data based on provided arguments.
    Can perform filtering, display, statistical analysis, and comparative analysis.

    Args dict may contain:
      - year: integer (single year)
      - year_start: integer (start year for range)
      - year_end: integer (end year for range) 
      - location: string (e.g. "WOOD", "ELOR")
      - min_yield: number
      - max_maturity: number
      - cultivar: string (partial match)
      - bean_type: string ("coloured bean" or "white bean")
      - trial_group: string ("major" or "minor")
      - sort: string ("highest" or "lowest")
      - limit: integer (how many rows total to return)
      - analysis_type: string ("average", "sum", "count", "max", "min", "median", "std", "similar", "compare", "yearly_average", "trend")
      - analysis_column: string (column to analyze, e.g., "Yield", "Maturity")
      - compare_to: string (what to compare against, e.g., "white beans", "2020 average")
      - similarity_threshold: number (percentage for similarity, default 15%)
      - group_by: string ("year", "location", "bean_type", "cultivar") - for grouping analysis
    """

    # Check if data was loaded successfully
    if df_trials.empty:
        return "Bean trial data could not be loaded.", "", {}

    # Debug: Print the arguments received
    print(f"🔍 Bean query args received: {args}")
    
    # Debug: Print some dataset info
    if not df_trials.empty:
        print(f"🔍 Dataset has {len(df_trials)} rows")
        # Convert to string to avoid mixed type sorting issues
        sample_cultivars = sorted([str(c) for c in df_trials['Cultivar Name'].dropna().unique()])[:20]
        print(f"🔍 Sample cultivars: {sample_cultivars}")
        
        # Check specifically for OAC cultivars
        oac_cultivars = [str(c) for c in df_trials['Cultivar Name'].dropna().unique() if 'oac' in str(c).lower()]
        print(f"🔍 OAC cultivars found: {oac_cultivars}")
    
    df = df_trials.copy()

    # 1) Extract filters and analysis parameters from args
    year = args.get("year", None)
    year_start = args.get("year_start", None)
    year_end = args.get("year_end", None)
    location = args.get("location", None)
    min_yield = args.get("min_yield", None)
    max_maturity = args.get("max_maturity", None)
    cultivar = args.get("cultivar", None)
    bean_type = args.get("bean_type", None)
    trial_group = args.get("trial_group", None)
    sort_order = args.get("sort", None)  # "highest" or "lowest"
    top_n = args.get("limit", None)  # e.g. 50
    
    # Analysis parameters
    analysis_type = args.get("analysis_type", None)
    analysis_column = args.get("analysis_column", "Yield")  # Default to Yield
    compare_to = args.get("compare_to", None)
    similarity_threshold = args.get("similarity_threshold", 15.0)  # Default 15% similarity
    group_by = args.get("group_by", None)  # For grouping analysis
    chart_type = args.get("chart_type", None)  # Specific chart type requested
    
    # Fallback: Check if cultivar name is mentioned in args but not extracted
    if not cultivar:
        # Look for common cultivar patterns in the arguments string and original question
        args_str = str(args).lower()
        original_question = args.get('original_question', '').lower()
        search_text = args_str + ' ' + original_question
        
        if 'oac seal' in search_text:
            cultivar = 'OAC Seal'
            print(f"🔧 Detected cultivar '{cultivar}' from args context")
        elif 'oac steam' in search_text:
            cultivar = 'OAC Steam'
            print(f"🔧 Detected cultivar '{cultivar}' from args context")
        elif 'oac' in search_text:
            cultivar = 'OAC'
            print(f"🔧 Detected cultivar '{cultivar}' from args context")
    
    # Special handling for global comparison requests
    if analysis_type == "compare" and cultivar and "global" in str(args).lower():
        print(f"🎯 Triggering global comparison for: {cultivar}")
        return handle_global_comparison(df, cultivar, analysis_column, args)
    
    # Also handle ranking/comparison requests that mention cultivars
    if cultivar and (analysis_type in ["compare", "cultivar_analysis"] or 
                    any(term in str(args).lower() for term in ["rank", "ranking", "production", "compare"])):
        print(f"🎯 Triggering global comparison (fallback) for: {cultivar}")
        return handle_global_comparison(df, cultivar, analysis_column, args)

    # 2) Apply year filtering - handle both single year and year ranges
    if year is not None:
        # Single year filter
        df = df[pd.to_numeric(df["Year"], errors="coerce") == int(year)].dropna(subset=["Year"])
    elif year_start is not None or year_end is not None:
        # Year range filter
        if year_start is not None:
            df = df[pd.to_numeric(df["Year"], errors="coerce") >= int(year_start)].dropna(subset=["Year"])
        if year_end is not None:
            df = df[pd.to_numeric(df["Year"], errors="coerce") <= int(year_end)].dropna(subset=["Year"])

    # Apply other filters
    if location:
        # Ensure Location column is string before comparison
        df = df[df["Location"].astype(str).str.upper() == location.upper()]

    if min_yield is not None:
        # Ensure Yield column is numeric before comparison
        df = df[pd.to_numeric(df["Yield"], errors="coerce") >= float(min_yield)].dropna(subset=["Yield"])

    if max_maturity is not None:
        # Ensure Maturity column is numeric before comparison
        df = df[pd.to_numeric(df["Maturity"], errors="coerce") <= float(max_maturity)].dropna(subset=["Maturity"])

    if cultivar:
        # Use .str.contains for partial matching, handle potential non-string types
        original_df_size = len(df)
        df = df[df["Cultivar Name"].astype(str).str.contains(cultivar, case=False, na=False)]
        
        # If no exact matches found and the search term has multiple words, try just the first word
        if df.empty and len(cultivar.split()) > 1:
            first_word = cultivar.split()[0]
            df = df_trials.copy()  # Reset to original data with previous filters
            # Reapply all previous filters
            if year is not None:
                df = df[pd.to_numeric(df["Year"], errors="coerce") == int(year)].dropna(subset=["Year"])
            elif year_start is not None or year_end is not None:
                if year_start is not None:
                    df = df[pd.to_numeric(df["Year"], errors="coerce") >= int(year_start)].dropna(subset=["Year"])
                if year_end is not None:
                    df = df[pd.to_numeric(df["Year"], errors="coerce") <= int(year_end)].dropna(subset=["Year"])
            if location:
                df = df[df["Location"].astype(str).str.upper() == location.upper()]
            if min_yield is not None:
                df = df[pd.to_numeric(df["Yield"], errors="coerce") >= float(min_yield)].dropna(subset=["Yield"])
            if max_maturity is not None:
                df = df[pd.to_numeric(df["Maturity"], errors="coerce") <= float(max_maturity)].dropna(subset=["Maturity"])
            if bean_type:
                if 'bean_type' in df.columns:
                    df = df[df["bean_type"] == bean_type.lower()]
            if trial_group:
                if 'trial_group' in df.columns:
                    df = df[df["trial_group"] == trial_group.lower()]
            
            # Try partial match with first word (e.g., "OAC" from "OAC Steam")
            df = df[df["Cultivar Name"].astype(str).str.contains(first_word, case=False, na=False)]

    if bean_type:
        # Ensure bean_type column exists and is lowercase string before comparison
        if 'bean_type' in df.columns:
            df = df[df["bean_type"] == bean_type.lower()]
        else:
            print("Warning: 'bean_type' column not found in data for filtering.")

    if trial_group:
        # Ensure trial_group column exists and is lowercase string before comparison
        if 'trial_group' in df.columns:
            df = df[df["trial_group"] == trial_group.lower()]
        else:
             print("Warning: 'trial_group' column not found in data for filtering.")

    # 4) If empty after filtering, return a "no results" message + empty full-table
    if df.empty:
        # Build a description of the filters that were applied
        applied_filters = []
        if year is not None:
            applied_filters.append(f"year: {year}")
        if year_start is not None or year_end is not None:
            year_range = f"{year_start or 'start'}-{year_end or 'end'}"
            applied_filters.append(f"year range: {year_range}")
        if location:
            applied_filters.append(f"location: {location}")
        if bean_type:
            applied_filters.append(f"bean type: {bean_type}")
        if cultivar:
            applied_filters.append(f"cultivar: {cultivar}")
        if trial_group:
            applied_filters.append(f"trial group: {trial_group}")
        if min_yield is not None:
            applied_filters.append(f"minimum yield: {min_yield}")
        if max_maturity is not None:
            applied_filters.append(f"maximum maturity: {max_maturity} days")
        
        filter_description = ", ".join(applied_filters) if applied_filters else "specified criteria"
        
        # Try to provide helpful suggestions, especially for cultivar searches
        suggestions = ""
        if cultivar:
            # Find similar cultivars in the dataset
            available_cultivars = df_trials['Cultivar Name'].dropna().unique()
            
            # For cultivar searches, show related cultivars
            if len(cultivar.split()) > 1:
                first_word = cultivar.split()[0].lower()
                related_cultivars = [c for c in available_cultivars if first_word in str(c).lower()]
                if related_cultivars:
                    suggestions = f"""

### 🌱 **Available {first_word.upper()} Cultivars:**
{', '.join(sorted(related_cultivars)[:10])}
{f"... and {len(related_cultivars)-10} more" if len(related_cultivars) > 10 else ""}
"""
            else:
                # Show cultivars that contain the search term
                related_cultivars = [c for c in available_cultivars if cultivar.lower() in str(c).lower()]
                if related_cultivars:
                    suggestions = f"""

### 🌱 **Available Cultivars containing '{cultivar}':**
{', '.join(sorted(related_cultivars)[:10])}
{f"... and {len(related_cultivars)-10} more" if len(related_cultivars) > 10 else ""}
"""
            
            # If still no related cultivars and bean_type is specified, show other cultivars of that type
            if not suggestions and bean_type:
                bean_type_cultivars = df_trials[df_trials['bean_type'] == bean_type.lower()]['Cultivar Name'].dropna().unique() if 'bean_type' in df_trials.columns else []
                if len(bean_type_cultivars) > 0:
                    suggestions = f"""

### 🌱 **Available {bean_type.title()} Cultivars:**
{', '.join(sorted(bean_type_cultivars)[:15])}
{f"... and {len(bean_type_cultivars)-15} more" if len(bean_type_cultivars) > 15 else ""}
"""
        
        # Return a more informative message that signals the system should fall back to research papers
        no_data_message = f"""## 🔍 **Dataset Query Results**

No matching cultivar performance data found for: **{filter_description}**

Our structured dataset contains {len(df_trials)} total records spanning multiple years and locations, but none match your specific query parameters.{suggestions}

*The system will now search scientific literature for relevant information...*
"""
        
        return no_data_message, "", {}

    # 5) PERFORM SPECIAL MULTI-YEAR ANALYSIS
    if analysis_type in ["yearly_average", "trend"]:
        if analysis_column not in df.columns:
            available_columns = ", ".join([col for col in df.columns if col not in ['Cultivar Name', 'Location', 'Year']])
            error_msg = f"""## 🔍 **Dataset Analysis Error**

Column '{analysis_column}' is not available in our cultivar performance dataset.

**Available analysis columns:** {available_columns}

*The system will now search scientific literature for relevant information...*
"""
            return error_msg, "", {}
        
        # Group by year and calculate statistics
        yearly_stats = df.groupby('Year').agg({
            analysis_column: ['count', 'mean', 'std', 'min', 'max']
        }).round(2)
        
        # Flatten column names
        yearly_stats.columns = ['Count', 'Average', 'Std_Dev', 'Min', 'Max']
        yearly_stats = yearly_stats.reset_index()
        
        # Build filter description
        filters = []
        if bean_type:
            filters.append(f"**{bean_type}**")
        if location:
            filters.append(f"**{location}**")
        if cultivar:
            filters.append(f"cultivars matching **{cultivar}**")
        if trial_group:
            filters.append(f"**{trial_group}** trials")
        
        filter_text = f"({', '.join(filters)})" if filters else "(all data)"
        
        # Create response
        if analysis_type == "yearly_average":
            # Create interactive visualization
            chart_data = create_yearly_trend_chart_data(yearly_stats, analysis_column, filter_text)
            
            response = f"""## 📈 **Yearly Average {analysis_column} Analysis**

**Filter:** {filter_text}

### 📋 **Year-by-Year Data:**

"""
            # Add the yearly data as a table
            response += yearly_stats.to_markdown(index=False)
            
            # Add summary statistics
            overall_avg = df[analysis_column].mean()
            total_years = len(yearly_stats)
            total_records = df.shape[0]
            
            response += f"""

### 🎯 **Summary:**
- **Overall Average:** {overall_avg:.0f}
- **Years Covered:** {total_years} years ({yearly_stats['Year'].min():.0f} - {yearly_stats['Year'].max():.0f})
- **Total Records:** {total_records}
- **Best Year:** {yearly_stats.loc[yearly_stats['Average'].idxmax(), 'Year']:.0f} (avg: {yearly_stats['Average'].max():.0f})
- **Lowest Year:** {yearly_stats.loc[yearly_stats['Average'].idxmin(), 'Year']:.0f} (avg: {yearly_stats['Average'].min():.0f})

💡 **Insight:** {"Yields have been relatively stable across years." if yearly_stats['Average'].std() < 200 else "Significant year-to-year variation in yields."}
"""
            
            return response, "", chart_data  # Return chart data for frontend rendering
        
        elif analysis_type == "trend":
            # Calculate trend analysis
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(yearly_stats['Year'], yearly_stats['Average'])
            except ImportError:
                error_msg = """## 🔍 **Dataset Analysis Limitation**

Advanced trend analysis requires additional statistical libraries that are not currently available.

*The system will now search scientific literature for trend information...*
"""
                return error_msg, "", {}
            
            # Create enhanced trend visualization
            chart_data = create_yearly_trend_chart_data(yearly_stats, analysis_column, filter_text)
            
            response = f"""## 📈 **Trend Analysis for {analysis_column}**

**Filter:** {filter_text}

### 📋 **Yearly Data:**

"""
            response += yearly_stats.to_markdown(index=False)
            
            # Trend analysis
            trend_direction = "increasing" if slope > 0 else "decreasing"
            trend_strength = "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.4 else "weak"
            
            response += f"""

### 🎯 **Trend Analysis:**
- **Direction:** {trend_direction.title()} by {abs(slope):.1f} units per year
- **Correlation:** {r_value:.3f} ({trend_strength} correlation)
- **Statistical Significance:** {"Significant" if p_value < 0.05 else "Not significant"} (p = {p_value:.3f})

💡 **Interpretation:** The data shows a {trend_strength} {trend_direction} trend over the analyzed period.
"""
            
            return response, "", chart_data  # Return chart data for frontend rendering

    # 5) PERFORM STATISTICAL ANALYSIS if requested
    if analysis_type:
        if analysis_column not in df.columns:
            available_columns = ", ".join([col for col in df.columns if col not in ['Cultivar Name', 'Location', 'Year']])
            error_msg = f"""## 🔍 **Dataset Analysis Error**

Column '{analysis_column}' is not available in our cultivar performance dataset.

**Available analysis columns:** {available_columns}

*The system will now search scientific literature for relevant information...*
"""
            return error_msg, "", {}
        
        # Build filter description helper function
        def build_filter_description():
            filters = []
            if year is not None:
                filters.append(f"**{year}**")
            if location:
                filters.append(f"**{location}**")
            if bean_type:
                filters.append(f"**{bean_type}**")
            if cultivar:
                filters.append(f"**{cultivar}**")
            if trial_group:
                filters.append(f"**{trial_group}** trials")
            if min_yield is not None:
                filters.append(f"yield ≥ **{min_yield}**")
            if max_maturity is not None:
                filters.append(f"maturity ≤ **{max_maturity}** days")
            return filters
        
        # Handle SIMILARITY ANALYSIS (e.g., "which yields are similar to white beans in 2020")
        if analysis_type == "similar":
            # First, calculate the reference value from the filtered data
            analysis_data = pd.to_numeric(df[analysis_column], errors="coerce").dropna()
            
            if analysis_data.empty:
                error_msg = """## 🔍 **Dataset Analysis Results**

No data found matching the specified criteria for similarity analysis.

*The system will now search scientific literature for relevant comparative information...*
"""
                return error_msg, "", {}
            
            reference_mean = analysis_data.mean()
            reference_std = analysis_data.std()
            
            # Calculate similarity range (using percentage threshold)
            threshold_value = reference_mean * (similarity_threshold / 100.0)
            lower_bound = reference_mean - threshold_value
            upper_bound = reference_mean + threshold_value
            
            # Now find ALL data (removing previous filters) that falls within this range
            full_df = df_trials.copy()
            full_df = full_df.dropna(subset=[analysis_column])
            full_df[analysis_column] = pd.to_numeric(full_df[analysis_column], errors="coerce")
            
            # Find similar records
            similar_df = full_df[
                (full_df[analysis_column] >= lower_bound) & 
                (full_df[analysis_column] <= upper_bound)
            ].copy()
            
            if similar_df.empty:
                return f"🔍 **No similar records found** within {similarity_threshold}% of the reference value.", "", {}
            
            # Group by bean type, location, or cultivar for better insights
            filters_desc = build_filter_description()
            filter_text = f"({', '.join(filters_desc)})" if filters_desc else "(all data)"
            
            # Analyze the similar records
            similar_by_type = similar_df.groupby(['bean_type'] if 'bean_type' in similar_df.columns else ['Cultivar Name']).agg({
                analysis_column: ['count', 'mean', 'min', 'max']
            }).round(2)
            
            # Build response
            response = f"""## 🎯 **Similar {analysis_column} Analysis**

**Reference Group:** {filter_text}
- **Average {analysis_column}:** {reference_mean:.0f}
- **Similarity Range:** {lower_bound:.0f} - {upper_bound:.0f} (±{similarity_threshold}%)

---

## 📊 **Groups with Similar Performance:**

"""
            
            # Add insights about different groups
            if 'bean_type' in similar_df.columns:
                for bean_type in similar_df['bean_type'].unique():
                    if pd.isna(bean_type):
                        continue
                    type_data = similar_df[similar_df['bean_type'] == bean_type]
                    count = len(type_data)
                    avg_yield = type_data[analysis_column].mean()
                    response += f"• **{bean_type.title()} Beans:** {count} cultivars (avg: {avg_yield:.0f})\n"
            
            # Show top cultivars
            top_similar = similar_df.nlargest(5, analysis_column)[['Cultivar Name', analysis_column, 'Year', 'Location']]
            response += f"\n## 🏆 **Top Similar Performers:**\n\n"
            response += top_similar.to_markdown(index=False)
            
            response += f"\n\n💡 **Found {len(similar_df)} records** with {analysis_column.lower()} similar to your reference group."
            
            return response, "", {}
        
        # Handle COMPARISON ANALYSIS
        elif analysis_type == "compare":
            # Check if this is a cultivar vs others comparison
            if cultivar and bean_type:
                # This is a "cultivar X vs other cultivars of same type" comparison
                filters_desc = build_filter_description()
                filter_text = f"({', '.join(filters_desc)})" if filters_desc else "(all data)"
                
                # Start with fresh bean type data (ignore previous cultivar filtering)
                bean_type_data = df_trials.copy()
                
                # Apply non-cultivar filters to get all data of the requested bean type
                if year is not None:
                    bean_type_data = bean_type_data[pd.to_numeric(bean_type_data["Year"], errors="coerce") == int(year)].dropna(subset=["Year"])
                elif year_start is not None or year_end is not None:
                    if year_start is not None:
                        bean_type_data = bean_type_data[pd.to_numeric(bean_type_data["Year"], errors="coerce") >= int(year_start)].dropna(subset=["Year"])
                    if year_end is not None:
                        bean_type_data = bean_type_data[pd.to_numeric(bean_type_data["Year"], errors="coerce") <= int(year_end)].dropna(subset=["Year"])
                if location:
                    bean_type_data = bean_type_data[bean_type_data["Location"].astype(str).str.upper() == location.upper()]
                if min_yield is not None:
                    bean_type_data = bean_type_data[pd.to_numeric(bean_type_data["Yield"], errors="coerce") >= float(min_yield)].dropna(subset=["Yield"])
                if max_maturity is not None:
                    bean_type_data = bean_type_data[pd.to_numeric(bean_type_data["Maturity"], errors="coerce") <= float(max_maturity)].dropna(subset=["Maturity"])
                if bean_type and 'bean_type' in bean_type_data.columns:
                    bean_type_data = bean_type_data[bean_type_data["bean_type"] == bean_type.lower()]
                if trial_group and 'trial_group' in bean_type_data.columns:
                    bean_type_data = bean_type_data[bean_type_data["trial_group"] == trial_group.lower()]
                
                # Now look for the specific cultivar in this bean type data
                cultivar_data = bean_type_data[bean_type_data["Cultivar Name"].astype(str).str.contains(cultivar, case=False, na=False)]
                search_term = cultivar
                
                if cultivar_data.empty:
                    # Try just the first word if the full cultivar name wasn't found
                    if len(cultivar.split()) > 1:
                        first_word = cultivar.split()[0]
                        cultivar_data = bean_type_data[bean_type_data["Cultivar Name"].astype(str).str.contains(first_word, case=False, na=False)]
                        search_term = first_word
                
                if cultivar_data.empty:
                    # Show available cultivars in this bean type
                    available_cultivars = bean_type_data['Cultivar Name'].dropna().unique()
                    if len(cultivar.split()) > 1:
                        first_word = cultivar.split()[0].lower()
                        related_cultivars = [c for c in available_cultivars if first_word in str(c).lower()]
                        if related_cultivars:
                            suggestions = f"Available {first_word.upper()} cultivars: {', '.join(sorted(related_cultivars)[:10])}"
                        else:
                            suggestions = f"Available {bean_type} cultivars: {', '.join(sorted(available_cultivars)[:10])}"
                    else:
                        suggestions = f"Available {bean_type} cultivars: {', '.join(sorted(available_cultivars)[:10])}"
                    
                    return f"""❌ **No data found** for '{cultivar}' in {bean_type}s.

### 🌱 **{suggestions}**
{f"... and {len(available_cultivars)-10} more" if len(available_cultivars) > 10 else ""}""", "", {}
                
                # Get data for other cultivars (those that DON'T match our search term)
                other_cultivars_data = bean_type_data[
                    ~bean_type_data["Cultivar Name"].astype(str).str.contains(search_term, case=False, na=False)
                ]
                
                if other_cultivars_data.empty:
                    return f"❌ **No comparison data** found for other {bean_type}s.", "", {}
                
                # Calculate statistics
                cultivar_stats = {
                    'count': len(cultivar_data),
                    'mean': cultivar_data[analysis_column].mean(),
                    'std': cultivar_data[analysis_column].std(),
                    'min': cultivar_data[analysis_column].min(),
                    'max': cultivar_data[analysis_column].max()
                }
                
                others_stats = {
                    'count': len(other_cultivars_data),
                    'mean': other_cultivars_data[analysis_column].mean(),
                    'std': other_cultivars_data[analysis_column].std(),
                    'min': other_cultivars_data[analysis_column].min(),
                    'max': other_cultivars_data[analysis_column].max()
                }
                
                # Calculate difference and statistical significance
                difference = cultivar_stats['mean'] - others_stats['mean']
                percent_diff = (difference / others_stats['mean']) * 100 if others_stats['mean'] != 0 else 0
                
                # Simple statistical test (assuming normal distribution)
                from math import sqrt
                pooled_std = sqrt(((cultivar_stats['std']**2 * cultivar_stats['count']) + 
                                 (others_stats['std']**2 * others_stats['count'])) / 
                                (cultivar_stats['count'] + others_stats['count']))
                
                # Find cultivar names that match
                matching_cultivars = cultivar_data["Cultivar Name"].unique()
                cultivar_display = ', '.join(matching_cultivars[:3])
                if len(matching_cultivars) > 3:
                    cultivar_display += f" (and {len(matching_cultivars)-3} more)"
                
                # Update the title to show what was actually found
                found_description = f"{search_term} cultivars" if search_term != cultivar else cultivar
                
                response = f"""## 📊 **{analysis_column} Comparison: {found_description} vs Other {bean_type.title()}s**

**Filter:** {filter_text}

### 🎯 **Key Findings:**

**{cultivar_display}:**
- **Records:** {cultivar_stats['count']}
- **Average {analysis_column}:** **{cultivar_stats['mean']:.0f}**
- **Range:** {cultivar_stats['min']:.0f} - {cultivar_stats['max']:.0f}
- **Std Dev:** {cultivar_stats['std']:.0f}

**Other {bean_type.title()}s:**
- **Records:** {others_stats['count']} (from {other_cultivars_data['Cultivar Name'].nunique()} different cultivars)
- **Average {analysis_column}:** **{others_stats['mean']:.0f}**
- **Range:** {others_stats['min']:.0f} - {others_stats['max']:.0f}
- **Std Dev:** {others_stats['std']:.0f}

### 📈 **Statistical Analysis:**
- **Difference:** {difference:+.0f} {analysis_column.lower()} ({percent_diff:+.1f}%)
- **Performance:** {cultivar_display} {'outperforms' if difference > 0 else 'underperforms'} other {bean_type}s by {abs(percent_diff):.1f}%
- **Significance:** {'Potentially significant' if abs(percent_diff) > 5 else 'Minimal'} difference

### 💡 **Interpretation:**
{f"{cultivar_display} shows {'superior' if difference > 0 else 'lower'} yield performance compared to other {bean_type}s." if abs(percent_diff) > 5 else f"{cultivar_display} performs similarly to other {bean_type}s."}

**Possible explanations for differences:**
- Genetic traits and breeding objectives
- Maturity differences affecting yield potential
- Disease resistance characteristics
- Adaptation to specific growing conditions
- Seed size and plant architecture differences
"""
                
                # Also create comparison chart
                chart_data = create_comparison_chart_data(df, 'Cultivar Name', analysis_column, filter_text)
                
                return response, "", chart_data
                
            else:
                # Original bean type comparison logic
                if 'bean_type' in df.columns:
                    comparison = df.groupby('bean_type').agg({
                        analysis_column: ['count', 'mean', 'std', 'min', 'max']
                    }).round(2)
                    
                    response = f"## 📊 **{analysis_column} Comparison by Bean Type**\n\n"
                    for bean_type in comparison.index:
                        if pd.isna(bean_type):
                            continue
                        stats = comparison.loc[bean_type]
                        response += f"""
### {bean_type.title()} Beans
- **Records:** {int(stats[(analysis_column, 'count')])}
- **Average:** {stats[(analysis_column, 'mean')]:.0f}
- **Range:** {stats[(analysis_column, 'min')]:.0f} - {stats[(analysis_column, 'max')]:.0f}
- **Std Dev:** {stats[(analysis_column, 'std')]:.0f}
"""
                    
                    return response, "", {}
        
        # Handle STANDARD STATISTICAL ANALYSIS with improved formatting
        else:
            analysis_data = pd.to_numeric(df[analysis_column], errors="coerce").dropna()
            
            if analysis_data.empty:
                error_msg = """## 🔍 **Dataset Analysis Results**

No data found matching the specified criteria for similarity analysis.

*The system will now search scientific literature for relevant comparative information...*
"""
                return error_msg, "", {}
            
            filters_desc = build_filter_description()
            filter_text = f"**Analyzing:** {', '.join(filters_desc)}" if filters_desc else "**Analyzing:** All available data"
            
            # Perform the requested analysis with better formatting
            if analysis_type == "average":
                result_value = analysis_data.mean()
                
                # Add context and insights
                response = f"""## 📈 **Average {analysis_column} Analysis**

{filter_text}

### 🎯 **Key Results:**
- **Average {analysis_column}:** **{result_value:.0f}**
- **Data Range:** {analysis_data.min():.0f} - {analysis_data.max():.0f}
- **Standard Deviation:** {analysis_data.std():.0f}
- **Total Records:** {len(analysis_data)}

### 📊 **Performance Distribution:**
- **Top 25%:** Above {analysis_data.quantile(0.75):.0f}
- **Bottom 25%:** Below {analysis_data.quantile(0.25):.0f}
- **Median:** {analysis_data.median():.0f}
"""
                
                # Add performance categories
                if analysis_column == "Yield":
                    if result_value > 4000:
                        response += "\n💪 **Performance:** Excellent yield performance!"
                    elif result_value > 3000:
                        response += "\n✅ **Performance:** Good yield performance."
                    else:
                        response += "\n📊 **Performance:** Moderate yield performance."
                
                return response, "", {}
                
            elif analysis_type == "count":
                result_value = len(analysis_data)
                response = f"""## 📊 **Data Count Analysis**

{filter_text}

### 🎯 **Results:**
- **Total Records:** **{result_value}**
- **Data Coverage:** {analysis_data.min():.0f} - {analysis_data.max():.0f} {analysis_column.lower()}
"""
                return response, "", {}
                
            elif analysis_type in ["max", "min"]:
                result_value = analysis_data.max() if analysis_type == "max" else analysis_data.min()
                record = df.loc[df[analysis_column] == result_value].iloc[0]
                
                response = f"""## 🏆 **{analysis_type.title()} {analysis_column} Analysis**

{filter_text}

### 🎯 **{analysis_type.title()} Value:** **{result_value:.0f}**

**Record Details:**
- **Cultivar:** {record.get('Cultivar Name', 'Unknown')}
- **Year:** {record.get('Year', 'Unknown')}
- **Location:** {record.get('Location', 'Unknown')}
"""
                return response, "", {}
                
            elif analysis_type == "scatter":
                # Handle scatter plot requests directly
                filters_desc = build_filter_description()
                filter_text = f"({', '.join(filters_desc)})" if filters_desc else "(all data)"
                
                # Use dynamic axes if specified, otherwise default to Maturity vs Yield
                x_col = args.get('x_axis', 'Maturity')
                y_col = args.get('y_axis', 'Yield')
                
                # Validate and fallback for columns
                if x_col not in df.columns:
                    if 'Maturity' in df.columns:
                        x_col = 'Maturity'
                    elif 'Year' in df.columns:
                        x_col = 'Year'
                    else:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        x_col = numeric_cols[0] if numeric_cols else df.columns[0]
                
                if y_col not in df.columns:
                    if 'Yield' in df.columns:
                        y_col = 'Yield'
                    elif 'Maturity' in df.columns:
                        y_col = 'Maturity'
                    else:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        y_col = numeric_cols[1] if len(numeric_cols) > 1 else (analysis_column if analysis_column in df.columns else numeric_cols[0])
                
                # Check if there's a specific cultivar to highlight
                highlight_cultivar = args.get('cultivar', None) or args.get('highlight_cultivar', None)
                
                if len(df) < 3:
                    return f"❌ **Insufficient data** for scatter plot. Need at least 3 data points, found {len(df)}.", "", {}
                
                # Create scatter plot
                chart_data = create_scatter_chart_data(df, x_col, y_col, None, filter_text, highlight_cultivar)
                
                response = f"""## 📊 **Scatter Plot: {y_col} vs {x_col}**

**Filter:** {filter_text}

### 📋 **Data Summary:**
- **Total Records:** {len(df)}
- **X-axis ({x_col}):** {df[x_col].min():.0f} - {df[x_col].max():.0f}
- **Y-axis ({y_col}):** {df[y_col].min():.0f} - {df[y_col].max():.0f}
- **Correlation:** {df[x_col].corr(df[y_col]):.3f}

### 📊 **Interactive Scatter Plot:**
Each point represents one trial record.
{f"**{highlight_cultivar}** cultivar is highlighted in red." if highlight_cultivar else ""}
"""
                return response, "", chart_data
                
            else:
                # Handle other analysis types with basic formatting
                if analysis_type == "sum":
                    result_value = analysis_data.sum()
                    title = f"Total {analysis_column}"
                elif analysis_type == "median":
                    result_value = analysis_data.median()
                    title = f"Median {analysis_column}"
                elif analysis_type == "std":
                    result_value = analysis_data.std()
                    title = f"Standard Deviation of {analysis_column}"
                elif analysis_type == "visualization":
                    # Handle visualization requests - create specific chart type if requested
                    filters_desc = build_filter_description()
                    filter_text = f"({', '.join(filters_desc)})" if filters_desc else "(all data)"
                    
                    chart_data = None
                    chart_type_desc = "Data Visualization"
                    
                    # Check what dimensions we have available
                    has_cultivars = 'Cultivar Name' in df.columns and df['Cultivar Name'].nunique() > 1
                    has_locations = 'Location' in df.columns and df['Location'].nunique() > 1
                    has_bean_types = 'bean_type' in df.columns and df['bean_type'].nunique() > 1
                    has_years = 'Year' in df.columns and df['Year'].nunique() > 1
                    has_numeric_cols = analysis_column in df.columns
                    
                    # Create specific chart type if requested
                    if chart_type == "pie":
                        # Pie chart - choose best categorical grouping
                        if has_bean_types and df['bean_type'].nunique() <= 8:
                            chart_data = create_pie_chart_data(df, 'bean_type', analysis_column, filter_text)
                            chart_type_desc = f"Pie Chart: {analysis_column} by Bean Type"
                        elif has_locations and df['Location'].nunique() <= 8:
                            chart_data = create_pie_chart_data(df, 'Location', analysis_column, filter_text)
                            chart_type_desc = f"Pie Chart: {analysis_column} by Location"
                        elif has_cultivars and df['Cultivar Name'].nunique() <= 8:
                            chart_data = create_pie_chart_data(df, 'Cultivar Name', analysis_column, filter_text)
                            chart_type_desc = f"Pie Chart: {analysis_column} by Cultivar"
                    
                    elif chart_type == "bar":
                        # Bar chart - categorical comparison
                        if has_cultivars and df['Cultivar Name'].nunique() <= 20:
                            chart_data = create_cultivar_performance_chart_data(df, analysis_column, min(15, df['Cultivar Name'].nunique()), filter_text)
                            chart_type_desc = f"Bar Chart: {analysis_column} by Cultivar"
                        elif has_locations:
                            chart_data = create_location_performance_chart_data(df, analysis_column, filter_text)
                            chart_type_desc = f"Bar Chart: {analysis_column} by Location"
                        elif has_bean_types:
                            chart_data = create_comparison_chart_data(df, 'bean_type', analysis_column, filter_text)
                            chart_type_desc = f"Bar Chart: {analysis_column} by Bean Type"
                    
                    elif chart_type == "scatter":
                        # Handle scatter plot requests
                        filters_desc = build_filter_description()
                        filter_text = f"({', '.join(filters_desc)})" if filters_desc else "(all data)"
                        
                        # Use dynamic axes if specified, otherwise default to Maturity vs Yield
                        x_col = args.get('x_axis', 'Maturity')
                        y_col = args.get('y_axis', 'Yield')
                        
                        # Validate and fallback for columns
                        if x_col not in df.columns:
                            # Try common alternatives
                            if 'Maturity' in df.columns:
                                x_col = 'Maturity'
                            elif 'Year' in df.columns:
                                x_col = 'Year'
                            else:
                                # Use first numeric column
                                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                                x_col = numeric_cols[0] if numeric_cols else df.columns[0]
                        
                        if y_col not in df.columns:
                            # Try common alternatives
                            if 'Yield' in df.columns:
                                y_col = 'Yield'
                            elif 'Maturity' in df.columns:
                                y_col = 'Maturity'
                            else:
                                # Use second numeric column or analysis_column
                                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                                y_col = numeric_cols[1] if len(numeric_cols) > 1 else (analysis_column if analysis_column in df.columns else numeric_cols[0])
                        
                        # Check if there's a specific cultivar to highlight in the original query
                        if 'cultivar' in args and args['cultivar']:
                            highlight_cultivar = args['cultivar']
                        elif 'highlight_cultivar' in args and args['highlight_cultivar']:
                            highlight_cultivar = args['highlight_cultivar']
                        else:
                            highlight_cultivar = None
                        
                        if len(df) < 3:
                            return f"❌ **Insufficient data** for scatter plot. Need at least 3 data points, found {len(df)}.", "", {}
                        
                        # For scatter plots, don't auto-assign color_col - let function handle cultivars
                        color_col = None
                        chart_data = create_scatter_chart_data(df, x_col, y_col, color_col, filter_text, highlight_cultivar)
                        
                        response = f"""## 📊 **Scatter Plot: {y_col} vs {x_col}**

**Filter:** {filter_text}

### 📋 **Data Summary:**
- **Total Records:** {len(df)}
- **X-axis ({x_col}):** {df[x_col].min():.0f} - {df[x_col].max():.0f}
- **Y-axis ({y_col}):** {df[y_col].min():.0f} - {df[y_col].max():.0f}
- **Correlation:** {df[x_col].corr(df[y_col]):.3f}

### 📊 **Interactive Scatter Plot:**
Each point represents one trial record.
{f"**{highlight_cultivar}** cultivar is highlighted in red." if highlight_cultivar else ""}
"""
                        return response, "", chart_data
                        
                    elif chart_type == "location_analysis":
                        # Handle location-based analysis
                        filters_desc = build_filter_description()
                        filter_text = f"({', '.join(filters_desc)})" if filters_desc else "(all data)"
                        
                        if 'Location' not in df.columns:
                            return f"❌ **Location data not available** in the dataset.", "", {}
                        
                        if df['Location'].nunique() < 2:
                            return f"❌ **Insufficient location diversity** for analysis. Found only {df['Location'].nunique()} location(s).", "", {}
                        
                        chart_data = create_location_performance_chart_data(df, analysis_column, filter_text)
                        
                        # Calculate location statistics
                        location_stats = df.groupby('Location').agg({
                            analysis_column: ['count', 'mean', 'std', 'min', 'max']
                        }).round(2)
                        location_stats.columns = ['Count', 'Average', 'Std_Dev', 'Min', 'Max']
                        location_stats = location_stats.sort_values('Average', ascending=False)
                        
                        response = f"""## 📊 **Location Performance Analysis: {analysis_column}**

**Filter:** {filter_text}

### 📋 **Location Statistics:**

"""
                        response += location_stats.to_markdown()
                        
                        response += f"""

### 🏆 **Top Performing Locations:**
1. **{location_stats.index[0]}**: {location_stats.iloc[0]['Average']:.0f} average {analysis_column.lower()}
2. **{location_stats.index[1] if len(location_stats) > 1 else 'N/A'}**: {location_stats.iloc[1]['Average']:.0f if len(location_stats) > 1 else 0} average {analysis_column.lower()}

### 📊 **Interactive Chart:**
The chart below compares performance across all locations.
"""
                        return response, "", chart_data
                        
                    else:
                        # Handle any other analysis type dynamically - use same intelligent selection as visualization
                        filters_desc = build_filter_description()
                        filter_text = f"({', '.join(filters_desc)})" if filters_desc else "(all data)"
                        
                        # Use the same dynamic chart selection logic as visualization
                        chart_data = None
                        chart_type_desc = analysis_type.replace('_', ' ').title()
                        
                        # Check what dimensions we have available
                        has_cultivars = 'Cultivar Name' in df.columns and df['Cultivar Name'].nunique() > 1
                        has_locations = 'Location' in df.columns and df['Location'].nunique() > 1
                        has_bean_types = 'bean_type' in df.columns and df['bean_type'].nunique() > 1
                        has_numeric_cols = analysis_column in df.columns
                        
                        # Intelligently choose the best visualization based on data structure
                        if has_numeric_cols and len(df) >= 3:
                            # Priority 1: Scatter plots for correlation analysis
                            if 'Maturity' in df.columns and 'Yield' in df.columns and len(df) >= 10:
                                # Use dynamic axes if specified, otherwise default to Maturity vs Yield
                                x_col = args.get('x_axis', 'Maturity')
                                y_col = args.get('y_axis', 'Yield')
                                
                                # Validate columns exist
                                if x_col not in df.columns:
                                    x_col = 'Maturity'
                                if y_col not in df.columns:
                                    y_col = 'Yield'
                                
                                # For scatter plots, don't auto-assign color_col - let function handle cultivars
                                color_col = None
                                highlight_cultivar = args.get('cultivar', None) or args.get('highlight_cultivar', None)
                                chart_data = create_scatter_chart_data(df, x_col, y_col, color_col, filter_text, highlight_cultivar)
                            
                            # Priority 2: Categorical comparisons based on data diversity
                            elif has_bean_types and df['bean_type'].nunique() <= 6:
                                chart_data = create_comparison_chart_data(df, 'bean_type', analysis_column, filter_text)
                            
                            elif has_locations and df['Location'].nunique() <= 15:
                                chart_data = create_location_performance_chart_data(df, analysis_column, filter_text)
                            
                            elif has_cultivars and df['Cultivar Name'].nunique() <= 20:
                                chart_data = create_cultivar_performance_chart_data(df, analysis_column, min(15, df['Cultivar Name'].nunique()), filter_text)
                            
                            # Priority 3: Pie charts for small categorical sets
                            elif (has_bean_types and df['bean_type'].nunique() <= 4) or (has_locations and df['Location'].nunique() <= 6):
                                group_col = 'bean_type' if has_bean_types else 'Location'
                                chart_data = create_pie_chart_data(df, group_col, analysis_column, filter_text)
                        
                        if chart_data:
                            response = f"""## 📊 **{chart_type_desc} Analysis**

**Filter:** {filter_text}

### 📋 **Data Summary:**
- **Total Records:** {len(df)}
- **Average {analysis_column}:** {df[analysis_column].mean():.0f}
- **Range:** {df[analysis_column].min():.0f} - {df[analysis_column].max():.0f}

### 📊 **Interactive Visualization:**
Chart automatically selected based on your data structure and analysis request.
"""
                            return response, "", chart_data
                        else:
                            # If we can't create a chart, provide basic statistics
                            result_value = analysis_data.mean()  # Default to average
                            title = f"{analysis_type.replace('_', ' ').title()} {analysis_column}"
                
                        response = f"""## 📊 **{chart_type_desc} Analysis**

**Filter:** {filter_text}

### 📋 **Data Summary:**
- **Total Records:** {len(df)}
- **Average {analysis_column}:** {df[analysis_column].mean():.0f}
- **Range:** {df[analysis_column].min():.0f} - {df[analysis_column].max():.0f}

### 📊 **Analysis Result:**
Chart generation completed successfully.
"""
                        return response, "", chart_data
                
                else:
                    # For non-visualization analysis types, set default values
                    result_value = analysis_data.mean()  # Default to average
                    title = f"{analysis_type.replace('_', ' ').title()} {analysis_column}"
                
                response = f"""## 📊 **{title} Analysis**

{filter_text}

### 🎯 **Result:** **{result_value:.2f}**
- **Data Range:** {analysis_data.min():.0f} - {analysis_data.max():.0f}
- **Total Records:** {len(analysis_data)}
"""
                return response, "", {}

    # 6) REGULAR TABLE DISPLAY (non-analysis queries) with visualizations
    # Sort by Yield if requested, only if Yield column exists and is numeric
    if sort_order in ["highest", "lowest"] and "Yield" in df.columns:
        # Drop rows with NaN in Yield before sorting if necessary
        df_sorted = df.dropna(subset=["Yield"])
        if not df_sorted.empty:
            df = df_sorted.sort_values(by="Yield", ascending=(sort_order == "lowest"))

    # Select the columns to display (ensure columns exists)
    display_cols = [
        c for c in ["Year", "Location", "Cultivar Name", "Yield", "Maturity", "bean_type", "trial_group"] 
        if c in df.columns
    ]
    
    # Remove trial_group from display if all values in the filtered data are null/empty
    if 'trial_group' in display_cols and df['trial_group'].isna().all():
        display_cols.remove('trial_group')
        
    if not display_cols:
         return "Error: Could not find relevant columns in data to display results.", "", {}

    display_df = df[display_cols]
    
    # Generate interactive visualizations based on the data
    visualizations = []
    filter_desc = []
    if bean_type:
        filter_desc.append(f"{bean_type}")
    if location:
        filter_desc.append(f"at {location}")
    if year:
        filter_desc.append(f"in {year}")
    filter_text = f"({', '.join(filter_desc)})" if filter_desc else ""

    # Generate chart data for frontend rendering
    chart_data = {}
    
    # 1. Scatter plot: Yield vs Maturity (if both columns exist and sufficient data)
    if 'Yield' in display_df.columns and 'Maturity' in display_df.columns and len(display_df) > 5:
        highlight_cultivar = args.get('cultivar', None) or args.get('highlight_cultivar', None)
        # Don't use bean_type as color_col for scatter plots - let function handle individual cultivars
        chart_data['scatter'] = create_scatter_chart_data(display_df, 'Maturity', 'Yield', None, filter_text, highlight_cultivar)

    # 2. Location comparison (if multiple locations)
    if 'Location' in display_df.columns and display_df['Location'].nunique() > 1:
        chart_data['location'] = create_location_performance_chart_data(display_df, 'Yield', filter_text)

    # 3. Bean type comparison (if multiple bean types)
    if 'bean_type' in display_df.columns and display_df['bean_type'].nunique() > 1:
        chart_data['comparison'] = create_comparison_chart_data(display_df, 'bean_type', 'Yield', filter_text)

    # 4. Top cultivars (if enough cultivars and not already filtered by cultivar)
    if 'Cultivar Name' in display_df.columns and display_df['Cultivar Name'].nunique() > 5 and not cultivar:
        chart_data['cultivars'] = create_cultivar_performance_chart_data(display_df, 'Yield', 15, filter_text)

    # Build response with data
    response = f"## 📊 **Bean Data Analysis Results**\n\n"
    
    if filter_desc:
        response += f"**Filters Applied:** {', '.join(filter_desc)}\n\n"
    
    response += f"**Total Records Found:** {len(display_df)}\n\n"

    # Add data table
    response += "### 📋 **Data Table:**\n\n"

    # Build a Markdown preview for the first 10 rows
    preview_count = min(args.get("limit", 10), len(display_df))
    preview_df = display_df.head(preview_count)

    # Use to_markdown for consistent table formatting
    response += preview_df.to_markdown(index=False)

    # If there are more rows than preview_count, append "show more" message and build full table
    full_md = "" # Initialize full_md
    if len(display_df) > preview_count:
        response += "\n\n**Show more in app to see the full list.**"
        # Build the full markdown table
        full_md = display_df.to_markdown(index=False)
    
    return response, full_md, chart_data


# ---- GPT-compatible JSON Schema ----
function_schema = {
    "name": "query_bean_data",
    "description": "Query dry bean trial data with filtering, analysis, and visualization options",
    "parameters": {
        "type": "object",
        "properties": {
            "year": {"type": "integer", "description": "Single year to filter by"},
            "year_start": {"type": "integer", "description": "Start year for range filtering"},
            "year_end": {"type": "integer", "description": "End year for range filtering"},
            "location": {"type": "string", "description": "Location code (e.g., WOOD, ELOR, HARR)"},
            "bean_type": {"type": "string", "description": "Type of bean: 'white bean' or 'coloured bean'"},
            "trial_group": {"type": "string", "description": "Trial group: 'major' or 'minor'"},
            "cultivar": {"type": "string", "description": "Cultivar name or partial name to search for"},
            "min_yield": {"type": "number", "description": "Minimum yield threshold"},
            "max_maturity": {"type": "number", "description": "Maximum maturity days"},
            "sort": {"type": "string", "enum": ["highest", "lowest"], "description": "Sort order for results"},
            "limit": {"type": "integer", "description": "Maximum number of results to return"},
            "analysis_type": {
                "type": "string",
                "enum": ["average", "sum", "count", "max", "min", "median", "std", "similar", "compare", "yearly_average", "trend", "visualization", "scatter", "location_analysis", "cultivar_analysis"],
                "description": "Type of analysis to perform"
            },
            "analysis_column": {"type": "string", "description": "Column to analyze (Yield, Maturity, etc.)"},
            "compare_to": {"type": "string", "description": "What to compare against"},
            "similarity_threshold": {"type": "number", "description": "Percentage threshold for similarity (default 15%)"},
            "group_by": {"type": "string", "description": "Column to group analysis by"},
            "chart_type": {"type": "string", "enum": ["pie", "bar", "line", "histogram", "area", "scatter"], "description": "Specific chart type for visualization"}
        }
    }
}

def handle_global_comparison(df: pd.DataFrame, cultivar: str, analysis_column: str, args: Dict) -> Tuple[str, str, Dict]:
    """Handle global comparison requests that compare cultivar performance with broader context."""
    
    print(f"🔍 Global comparison called with cultivar: '{cultivar}', column: '{analysis_column}'")
    print(f"🔍 Available cultivars in dataset: {sorted(df['Cultivar Name'].dropna().astype(str).unique())[:10]}...")
    
    # Find cultivar data
    cultivar_data = df[df["Cultivar Name"].astype(str).str.contains(cultivar, case=False, na=False)]
    print(f"🔍 Found {len(cultivar_data)} records for '{cultivar}'")
    
    if cultivar_data.empty and len(cultivar.split()) > 1:
        # Try first word
        first_word = cultivar.split()[0]
        print(f"🔍 Trying first word: '{first_word}'")
        cultivar_data = df[df["Cultivar Name"].astype(str).str.contains(first_word, case=False, na=False)]
        print(f"🔍 Found {len(cultivar_data)} records for first word '{first_word}'")
        cultivar = first_word  # Update cultivar name for display
    
    if cultivar_data.empty:
        # Try to find similar cultivars and provide helpful response
        all_cultivars = df['Cultivar Name'].dropna().unique()
        similar_cultivars = [c for c in all_cultivars if cultivar.lower() in str(c).lower()]
        
        if not similar_cultivars:
            # Try partial matching with first 3 characters
            similar_cultivars = [c for c in all_cultivars if str(c).lower().startswith(cultivar[:3].lower())]
        
        response = f"""❌ **No exact match found** for cultivar '{cultivar}' in our dataset.

## 📊 **Available Dataset Statistics**

**Total Cultivars:** {len(all_cultivars)}
**Total Trials:** {len(df)}
**Average Yield Across All Cultivars:** {df['Yield'].mean():.0f} kg/ha
**Yield Range:** {df['Yield'].min():.0f} - {df['Yield'].max():.0f} kg/ha

"""
        
        if similar_cultivars:
            response += f"""### 🌱 **Similar Cultivars Available:**
{', '.join(sorted(similar_cultivars)[:10])}
{f"... and {len(similar_cultivars)-10} more" if len(similar_cultivars) > 10 else ""}

"""
        
        # Get top performing cultivars for context
        top_cultivars = df.groupby('Cultivar Name')['Yield'].mean().sort_values(ascending=False).head(10)
        response += f"""### 🏆 **Top 10 Performing Cultivars:**

"""
        for i, (cv, yield_val) in enumerate(top_cultivars.items(), 1):
            response += f"{i}. **{cv}**: {yield_val:.0f} kg/ha\n"
        
        response += f"""
💡 **This provides context from our cultivar trial dataset for comparison with global production estimates.**"""
        
        return response, "", {}
    
    # Calculate cultivar statistics
    cultivar_stats = {
        'name': cultivar,
        'count': len(cultivar_data),
        'mean': cultivar_data[analysis_column].mean(),
        'std': cultivar_data[analysis_column].std(),
        'min': cultivar_data[analysis_column].min(),
        'max': cultivar_data[analysis_column].max(),
        'median': cultivar_data[analysis_column].median()
    }
    
    # Get all data for comparison
    all_data_stats = {
        'count': len(df),
        'mean': df[analysis_column].mean(),
        'std': df[analysis_column].std(),
        'min': df[analysis_column].min(),
        'max': df[analysis_column].max(),
        'median': df[analysis_column].median()
    }
    
    # Calculate performance ranking
    all_cultivars = df.groupby('Cultivar Name')[analysis_column].mean().sort_values(ascending=False)
    cultivar_rank = None
    for i, (cv, _) in enumerate(all_cultivars.items(), 1):
        if cultivar.lower() in str(cv).lower():
            cultivar_rank = i
            break
    
    # Get bean type breakdown if available
    bean_type_stats = {}
    if 'bean_type' in df.columns:
        for bt in df['bean_type'].unique():
            if pd.notna(bt):
                bt_data = df[df['bean_type'] == bt]
                bean_type_stats[bt] = {
                    'count': len(bt_data),
                    'mean': bt_data[analysis_column].mean(),
                    'std': bt_data[analysis_column].std()
                }
    
    # Build comprehensive response
    response = f"""## 🌍 **Global Dry Bean Production Context**

### 📊 **{cultivar} Performance Data**

**Records Found:** {cultivar_stats['count']} trials
**Average {analysis_column}:** {cultivar_stats['mean']:.0f} kg/ha
**Range:** {cultivar_stats['min']:.0f} - {cultivar_stats['max']:.0f} kg/ha
**Standard Deviation:** {cultivar_stats['std']:.0f}
{f"**Ranking:** #{cultivar_rank} out of {len(all_cultivars)} cultivars" if cultivar_rank else ""}

### 🌾 **Dataset Comparison Context**

| Metric | {cultivar} | All Cultivars | Difference |
|--------|------------|---------------|------------|
| Average {analysis_column} | {cultivar_stats['mean']:.0f} | {all_data_stats['mean']:.0f} | {cultivar_stats['mean'] - all_data_stats['mean']:+.0f} |
| Median {analysis_column} | {cultivar_stats['median']:.0f} | {all_data_stats['median']:.0f} | {cultivar_stats['median'] - all_data_stats['median']:+.0f} |
| Maximum {analysis_column} | {cultivar_stats['max']:.0f} | {all_data_stats['max']:.0f} | {cultivar_stats['max'] - all_data_stats['max']:+.0f} |

### 📈 **Performance Analysis**

"""
    
    # Add performance analysis
    performance_vs_avg = ((cultivar_stats['mean'] - all_data_stats['mean']) / all_data_stats['mean']) * 100
    
    if performance_vs_avg > 10:
        performance_desc = f"**Excellent** - {performance_vs_avg:.1f}% above dataset average"
    elif performance_vs_avg > 0:
        performance_desc = f"**Above Average** - {performance_vs_avg:.1f}% above dataset average"
    elif performance_vs_avg > -10:
        performance_desc = f"**Below Average** - {abs(performance_vs_avg):.1f}% below dataset average"
    else:
        performance_desc = f"**Poor** - {abs(performance_vs_avg):.1f}% below dataset average"
    
    response += f"• **{cultivar} Performance:** {performance_desc}\n"
    response += f"• **Consistency:** {'High' if cultivar_stats['std'] < all_data_stats['std'] else 'Moderate' if cultivar_stats['std'] < all_data_stats['std'] * 1.2 else 'Variable'} (CV: {(cultivar_stats['std']/cultivar_stats['mean']*100):.1f}%)\n"
    
    # Add bean type context if available
    if bean_type_stats:
        response += f"\n### 🫘 **Bean Type Context**\n\n"
        for bt, stats in bean_type_stats.items():
            cultivar_vs_type = ((cultivar_stats['mean'] - stats['mean']) / stats['mean']) * 100
            response += f"• **{bt.title()} Beans:** Avg {stats['mean']:.0f} kg/ha ({stats['count']} cultivars) - {cultivar} is {cultivar_vs_type:+.1f}% vs this group\n"
    
    # Add detailed trial information
    response += f"\n### 📋 **{cultivar} Trial Details**\n\n"
    
    # Group by year and location for detailed breakdown
    detailed_data = cultivar_data.groupby(['Year', 'Location']).agg({
        analysis_column: ['mean', 'count']
    }).round(0)
    detailed_data.columns = ['Avg_Yield', 'Trials']
    detailed_data = detailed_data.reset_index()
    
    if len(detailed_data) <= 20:  # Show details if not too many
        response += detailed_data.to_markdown(index=False)
    else:
        # Show summary by year
        yearly_summary = cultivar_data.groupby('Year').agg({
            analysis_column: ['mean', 'count']
        }).round(0)
        yearly_summary.columns = ['Avg_Yield', 'Trials']
        yearly_summary = yearly_summary.reset_index()
        response += "**Yearly Summary:**\n\n" + yearly_summary.to_markdown(index=False)
    
    response += f"\n\n💡 **This provides actual yield data from our cultivar trials to compare with global production estimates.**"
    
    # Create visualization data
    chart_data = create_scatter_chart_data(
        df=df,
        x_col="Maturity", 
        y_col="Yield",
        highlight_cultivar=cultivar,
        filter_text=f"(Highlighting {cultivar})"
    )
    
    return response, detailed_data.to_markdown(index=False) if len(detailed_data) <= 20 else yearly_summary.to_markdown(index=False), chart_data