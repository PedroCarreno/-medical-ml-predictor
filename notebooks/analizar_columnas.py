import pandas as pd

df = pd.read_csv(r'd:\OneDrive\UTN\5\CienciaDeDatos_\medical-ml-predictor\data\dataset.csv')

print("COLUMNAS NUMERICAS CON VALORES FALTANTES:")
print("="*80)

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    nulls = df[col].isnull().sum()
    if nulls > 0:
        median_val = df[col].median()
        mean_val = df[col].mean()
        print(f"{col:30s} | Nulls: {nulls:6,} | Mediana: {median_val:8.1f} | Media: {mean_val:8.1f}")

print("\n" + "="*80)
print("\nCOLUMNAS CATEGORICAS CON VALORES FALTANTES:")
print("="*80)

cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    nulls = df[col].isnull().sum()
    if nulls > 0:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'N/A'
        print(f"{col:30s} | Nulls: {nulls:6,} | Moda: {mode_val}")
