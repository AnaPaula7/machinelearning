import pandas as pd

def extract_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=0, header=7)
    df = df[1:]  # Remove cabeçalho extra
    df.columns = [str(col).strip() for col in df.columns]
    return df
