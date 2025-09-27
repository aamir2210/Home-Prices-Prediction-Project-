# home_prices.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Home Prices (Linear Regression)", page_icon="🏠", layout="centered")
st.title("🏠 Home Prices Prediction (Linear Regression)")
st.write("Interactively predict home prices from square footage and inspect model performance.")

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "home_dataset.csv"   # expects columns: sqft, price

@st.cache_data
def load_data():
    if CSV.exists():
        df = pd.read_csv(CSV)
        # try to be tolerant with column names
        cols = {c.lower(): c for c in df.columns}
        sqft_col  = cols.get("sqft")  or cols.get("square_feet") or cols.get("area") or "sqft"
        price_col = cols.get("price") or "price"
        # rename for consistency if present
        if sqft_col in df and price_col in df:
            df = df.rename(columns={sqft_col: "sqft", price_col: "price"})
            df = df[["sqft", "price"]].dropna()
            df = df[(df["sqft"] > 0) & (df["price"] > 0)]
            source = "csv"
            return df, source
    # fallback: synthetic data like your notebook
    rng = np.random.default_rng(0)
    house_sizes = rng.integers(800, 5000, 200)
    house_prices = 150 * house_sizes + rng.normal(scale=10000, size=len(house_sizes))
    df = pd.DataFrame({"sqft": house_sizes, "price": house_prices})
    source = "synthetic"
    return df, source

df, source = load_data()
st.caption(f"Data source: **{source}** ({len(df):,} rows)")

# ── Sidebar controls
st.sidebar.header("Model Settings")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

# ── Train/test split & model
X = df[["sqft"]].values
y = df["price"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = float(mean_squared_error(y_test, y_pred))
r2  = float(r2_score(y_test, y_pred))

c1, c2 = st.columns(2)
c1.metric("R² (test)", f"{r2:.3f}")
c2.metric("MSE (test)", f"{mse:,.0f}")

# ── User prediction
st.subheader("Try a prediction")
default_sqft = int(np.percentile(df["sqft"], 50))
sqft = st.number_input("Square footage", min_value=int(df["sqft"].min()),
                       max_value=int(df["sqft"].max()), value=default_sqft, step=50)
if st.button("Predict price"):
    pred = model.predict([[sqft]])[0]
    st.success(f"Estimated price: **${pred:,.0f}**")

# ── Plots (use st.pyplot instead of plt.show)
st.subheader("Data & Regression Line")
fig, ax = plt.subplots()
ax.scatter(df["sqft"], df["price"], s=12, alpha=0.5, label="Data")
# regression line across the range
x_line = np.linspace(df["sqft"].min(), df["sqft"].max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
ax.plot(x_line, y_line, linewidth=2, label="Linear fit")
ax.set_xlabel("Square footage")
ax.set_ylabel("Price ($)")
ax.legend()
st.pyplot(fig)

st.subheader("Sample of the dataset")
st.dataframe(df.head(20))
