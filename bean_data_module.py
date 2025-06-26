import pandas as pd
import re


# ---- Load merged dataset ----
def load_all_trials():
    merged_path = (
        r"C:\Users\mirka\Documents\Guelph Research\data\Merged_Bean_Dataset.xlsx"
    )
    df = pd.read_excel(merged_path)

    # Ensure consistency
    df = df[
        ~df["Cultivar Name"].astype(str).str.lower().isin(["mean", "cv", "lsd(0.05)"])
    ]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
    df["Maturity"] = pd.to_numeric(df["Maturity"], errors="coerce")
    return df


# load the full dataset once
df_trials = load_all_trials()


# ---- Query Logic ----
def query_bean_data(
    year=None,
    location=None,
    min_yield=None,
    max_maturity=None,
    cultivar=None,
    bean_type=None,
    trial_group=None,
    sort_by_yield=False,
    limit=10,
):
    df = df_trials.copy()

    if year is not None:
        df = df[df["Year"] == int(year)]
    if location:
        df = df[df["Location"].str.upper() == location.upper()]
    if min_yield is not None:
        df = df[df["Yield"] >= float(min_yield)]
    if max_maturity is not None:
        df = df[df["Maturity"] <= float(max_maturity)]
    if cultivar:
        df = df[df["Cultivar Name"].str.contains(cultivar, case=False, na=False)]
    if bean_type:
        df = df[df["bean_type"].str.lower() == bean_type.lower()]
    if trial_group:
        df = df[df["trial_group"].str.lower() == trial_group.lower()]

    if sort_by_yield:
        df = df.sort_values(by="Yield", ascending=False)

    if df.empty:
        return "No matching cultivars found for your criteria."

    return (
        df[
            [
                "Year",
                "Location",
                "Cultivar Name",
                "Yield",
                "Maturity",
                "bean_type",
                "trial_group",
            ]
        ],
        limit,
    )


# ---- GPT-compatible JSON Schema ----
# function_schema = {
#     "name": "query_bean_data",
#     "description": "Query dry bean cultivar trial data by year, location, yield, or type.",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "year": {
#                 "type": "integer",
#                 "description": "Year of the trial (e.g., 2022)",
#             },
#             "location": {
#                 "type": "string",
#                 "description": "Location code (e.g., WOOD, ELOR)",
#             },
#             "min_yield": {
#                 "type": "number",
#                 "description": "Minimum yield (e.g., 4000)",
#             },
#             "max_maturity": {
#                 "type": "number",
#                 "description": "Maximum maturity in days",
#             },
#             "cultivar": {
#                 "type": "string",
#                 "description": "Cultivar name or partial match (e.g., 'OAC')",
#             },
#             "bean_type": {
#                 "type": "string",
#                 "enum": ["coloured bean", "white bean"],
#                 "description": "Type of bean",
#             },
#             "trial_group": {
#                 "type": "string",
#                 "enum": ["major", "minor"],
#                 "description": "Major or minor performance trial",
#             },
#         },
#     },
# }


# ---- Handler for GPT call ----
# ---- Query Logic ----
def answer_bean_query(args: dict):
    """
    Args dict may contain:
      - year: integer
      - location: string (e.g. "WOOD", "ELOR")
      - min_yield: number
      - max_maturity: number
      - cultivar: string (partial match)
      - bean_type: string ("coloured bean" or "white bean")
      - trial_group: string ("major" or "minor")
      - sort: string ("highest" or "lowest")
      - limit: integer (how many rows total to return)
    """

    # 1) Extract filters from args
    year = args.get("year", None)
    location = args.get("location", None)
    min_yield = args.get("min_yield", None)
    max_maturity = args.get("max_maturity", None)
    cultivar = args.get("cultivar", None)
    bean_type = args.get("bean_type", None)
    trial_group = args.get("trial_group", None)

    sort_order = args.get("sort", None)  # "highest" or "lowest"
    top_n = args.get("limit", None)  # e.g. 50

    # 2) Start with a fresh copy of the full DataFrame
    df = df_trials.copy()

    # 3) Apply each filter if provided
    if year is not None:
        df = df[df["Year"] == int(year)]

    if location:
        df = df[df["Location"].str.upper() == location.upper()]

    if min_yield is not None:
        df = df[df["Yield"] >= float(min_yield)]

    if max_maturity is not None:
        df = df[df["Maturity"] <= float(max_maturity)]

    if cultivar:
        df = df[df["Cultivar Name"].str.contains(cultivar, case=False, na=False)]

    if bean_type:
        df = df[df["bean_type"].str.lower() == bean_type.lower()]

    if trial_group:
        df = df[df["trial_group"].str.lower() == trial_group.lower()]

    # 4) Sort by Yield if requested
    if sort_order == "highest":
        df = df.sort_values(by="Yield", ascending=False)
    elif sort_order == "lowest":
        df = df.sort_values(by="Yield", ascending=True)

    # 5) Apply limit if provided
    if top_n is not None:
        df = df.head(int(top_n))

    # 6) If empty, return a “no results” message + empty full‐table
    if df.empty:
        return "No matching results found.", ""

    # 7) Select the columns to display
    display_df = df[
        [
            "Year",
            "Location",
            "Cultivar Name",
            "Yield",
            "Maturity",
            "bean_type",
            "trial_group",
        ]
    ]

    # 8) Convert to list of row‐dicts
    rows = display_df.to_dict("records")

    # 9) Build a Markdown preview for the first 10 rows
    preview_count = min(10, len(rows))
    preview_rows = rows[:preview_count]

    cols = list(display_df.columns)
    preview_md = "| " + " | ".join(cols) + " |\n"
    preview_md += "| " + " | ".join(["---"] * len(cols)) + " |\n"
    for row in preview_rows:
        preview_md += "| " + " | ".join(str(row[c]) for c in cols) + " |\n"

    # 10) If there are more rows than preview_count, append “show more” message and build full table
    if len(rows) > preview_count:
        preview_md += "\n**Show more in app to see the full list.**\n"

        full_md = "| " + " | ".join(cols) + " |\n"
        full_md += "| " + " | ".join(["---"] * len(cols)) + " |\n"
        for row in rows:
            full_md += "| " + " | ".join(str(row[c]) for c in cols) + " |\n"

        return preview_md, full_md

    # 11) Otherwise, all rows fit in the preview
    return preview_md, ""


# ---- GPT‐compatible JSON Schema ----
function_schema = {
    "name": "query_bean_data",
    "description": "Query dry bean cultivar trial data by year, location, yield, or type.",
    "parameters": {
        "type": "object",
        "properties": {
            "year": {
                "type": "integer",
                "description": "Year of the trial (e.g., 2022)",
            },
            "location": {
                "type": "string",
                "description": "Location code (e.g., WOOD, ELOR)",
            },
            "min_yield": {
                "type": "number",
                "description": "Minimum yield (e.g., 4000)",
            },
            "max_maturity": {
                "type": "number",
                "description": "Maximum maturity in days",
            },
            "cultivar": {
                "type": "string",
                "description": "Cultivar name or partial match (e.g., 'OAC')",
            },
            "bean_type": {
                "type": "string",
                "enum": ["coloured bean", "white bean"],
                "description": "Type of bean",
            },
            "trial_group": {
                "type": "string",
                "enum": ["major", "minor"],
                "description": "Major or minor trial",
            },
            "sort": {
                "type": "string",
                "enum": ["highest", "lowest"],
                "description": "Sort by yield (highest=descending, lowest=ascending)",
            },
            "limit": {
                "type": "integer",
                "description": "Number of rows to return (e.g., 50 for top 50)",
            },
        },
        "required": [],
    },
}
