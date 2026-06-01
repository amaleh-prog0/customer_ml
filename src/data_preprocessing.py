import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, encoding='latin1')
    # basic cleaning
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    return df

def create_customer_features(df):
    customer_features = df.groupby('CustomerID').agg(
        total_spend=('TotalSpend', 'sum'),
        total_quantity=('Quantity', 'sum'),
        avg_unit_price=('UnitPrice', 'mean'),
        num_transactions=('InvoiceNo', 'nunique'),
        num_unique_products=('StockCode', 'nunique'),
        most_common_country=('Country', lambda x: x.mode()[0])
    ).reset_index()
    # target: high value (spend > median)
    median_spend = customer_features['total_spend'].median()
    customer_features['high_value'] = (customer_features['total_spend'] > median_spend).astype(int)
    return customer_features

def encode_country(df):
    le = LabelEncoder()
    df['country_code'] = le.fit_transform(df['most_common_country'])
    return df, le

def prepare_features(df):
    X = df.drop(['CustomerID', 'most_common_country', 'high_value'], axis=1)
    y = df['high_value']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)