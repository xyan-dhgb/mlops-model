import pandas as pd
def preprocess_csv_data(df):
    """
    Preprocess the CSV data for model input
    """
    # Create a copy of the dataframe to avoid modifying the original
    processed_df = df.copy()

    # Fill missing coordinates with -1 (for normal cases)
    for col in ['X', 'Y', 'RADIUS']:
        processed_df[col] = processed_df[col].fillna(-1)

    # Convert categorical variables to numeric
    # Background tissue
    bg_map = {'F': 0, 'G': 1, 'D': 2}
    processed_df['BG'] = processed_df['BG'].map(bg_map)

    # Class mapping
    class_map = {
        'NORM': 0,
        'CIRC': 1,
        'SPIC': 2,
        'ARCH': 3,
        'ASYM': 4,
        'CALC': 5,
        'MISC': 6
    }
    processed_df['CLASS_NUM'] = processed_df['CLASS'].map(class_map)

    # Severity mapping
    severity_map = {'Normal': 0, 'Benign': 1, 'Malignant': 2}
    processed_df['SEVERITY_NUM'] = processed_df['SEVERITY'].map(severity_map)

    # Density mapping
    density_map = {'A': 1, 'B': 2, 'C/D': 3}
    processed_df['DENSITY_NUM'] = processed_df['DENSITY'].map(density_map)

    # BI-RADS mapping
    birads_map = {
        'BI-RADS 1': 1,
        'BI-RADS 2': 2,
        'BI-RADS 3': 3,
        'BI-RADS 4': 4,
        'BI-RADS 5': 5
    }
    processed_df['BIRADS_NUM'] = processed_df['BI-RADS'].map(birads_map)

    return processed_df