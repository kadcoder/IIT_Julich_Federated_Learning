import pandas as pd
from sklearn.model_selection import train_test_split

# File paths
file_paths = {
    "CamCAN": "/home/tanurima/germany/brain_age_parcels/CamCAN/CamCAN_features_cleaned.csv",
    "SALD": "/home/tanurima/germany/brain_age_parcels/SALD/SALD_features_cleaned.csv",
    "eNki": "/home/tanurima/germany/brain_age_parcels/eNki/eNki_features_cleaned.csv"
}

# Function to split data while maintaining age distribution
def stratified_split(file_path, test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)
    
    # Create age bins for stratification
    df['AgeGroup'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 100], labels=False)
    
    # Split data while maintaining age distribution
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['AgeGroup'], random_state=random_state)
    
    # Drop AgeGroup column after stratification
    train_df = train_df.drop(columns=['AgeGroup'])
    test_df = test_df.drop(columns=['AgeGroup'])
    
    return train_df, test_df

# Process each dataset
for name, path in file_paths.items():
    train, test = stratified_split(path)
    
    # Save the split datasets
    train.to_csv(f"/home/tanurima/germany/brain_age_parcels/{name}/{name}_train.csv", index=False)
    test.to_csv(f"/home/tanurima/germany/brain_age_parcels/{name}/{name}_test.csv", index=False)
    
    print(f"{name} dataset split completed. Train and Test sets saved.")