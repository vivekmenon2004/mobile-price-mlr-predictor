import pandas as pd
import json

df = pd.read_excel('Mobile_MLR_Dataset.xlsx')
brands = sorted(df['Brand'].unique().tolist())

metadata = {
    "brands": brands,
    "num_samples": len(df),
    "features": df.columns.tolist()
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("Metadata saved to model_metadata.json")
print(f"Brands found: {len(brands)}")
