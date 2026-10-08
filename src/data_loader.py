from pathlib import Path
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw data directory
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Main project workbook
WORKBOOK_PATH = RAW_DATA_DIR / "Budget_Variance_Project_Data.xlsx"


REQUIRED_SHEETS = [
    "Budget",
    "Actuals_Raw",
    "Departments",
    "Categories",
    "Business_Units",
    "Regions",
    "Materiality_Rules",
    "Data_Dictionary",
]


def load_workbook(path: Path = WORKBOOK_PATH) -> dict[str, pd.DataFrame]:
    """
    Load the complete project workbook.

    Returns:
        Dictionary where:
        key   = sheet name
        value = pandas DataFrame
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Project workbook not found at: {path}"
        )

    excel_file = pd.ExcelFile(path)

    available_sheets = excel_file.sheet_names

    missing_sheets = [
        sheet
        for sheet in REQUIRED_SHEETS
        if sheet not in available_sheets
    ]

    if missing_sheets:
        raise ValueError(
            f"Missing required sheets: {missing_sheets}"
        )

    data = {
        sheet: pd.read_excel(path, sheet_name=sheet)
        for sheet in REQUIRED_SHEETS
    }

    return data


def load_budget(path: Path = WORKBOOK_PATH) -> pd.DataFrame:
    """Load budget data only."""
    return load_workbook(path)["Budget"]


def load_actuals(path: Path = WORKBOOK_PATH) -> pd.DataFrame:
    """Load raw actual transaction data."""
    return load_workbook(path)["Actuals_Raw"]


if __name__ == "__main__":
    data = load_workbook()

    print("\nVARIA DATA LOADER")
    print("=" * 50)

    for sheet_name, df in data.items():
        print(
            f"{sheet_name:<20} "
            f"{df.shape[0]:>6} rows × "
            f"{df.shape[1]:>3} columns"
        )