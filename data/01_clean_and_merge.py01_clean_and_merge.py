# %% [markdown]
# # Late Deliveries, Low Stars: Olist Operations Analysis
# **Client:** Olist COO  |  **Decision:** where to act to cut late deliveries  |
# **Success measure:** raise avg review score / cut late rate
#
# ## Part 1: Load, clean, merge  (supports Q1-Q5)

# %%
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent   # put the 9 Olist CSV files in a folder called "data"

orders    = pd.read_csv(DATA / "olist_orders_dataset.csv")
items     = pd.read_csv(DATA / "olist_order_items_dataset.csv")
reviews   = pd.read_csv(DATA / "olist_order_reviews_dataset.csv")
customers = pd.read_csv(DATA / "olist_customers_dataset.csv")
sellers   = pd.read_csv(DATA / "olist_sellers_dataset.csv")
products  = pd.read_csv(DATA / "olist_products_dataset.csv")
cat_tr    = pd.read_csv(DATA / "product_category_name_translation.csv")

for name, df in [("orders", orders), ("items", items), ("reviews", reviews),
                 ("customers", customers), ("sellers", sellers), ("products", products)]:
    print(f"{name:10s} {df.shape}")

# %% [markdown]
# ## Cleaning log
# Every step records: what, why, rows before, rows after.

# %%
cleaning_log = []

def log_step(step, reason, before, after):
    cleaning_log.append({"Step": step, "Reason": reason,
                         "Rows before": before, "Rows after": after,
                         "Rows removed": before - after})

# 1. Convert timestamps to datetime (no rows change)
date_cols = ["order_purchase_timestamp", "order_approved_at",
             "order_delivered_carrier_date", "order_delivered_customer_date",
             "order_estimated_delivery_date"]
orders[date_cols] = orders[date_cols].apply(pd.to_datetime)
log_step("Convert timestamps to datetime",
         "Needed for date maths (delivery days, monthly trend)", len(orders), len(orders))

# 2. Keep delivered orders only
before = len(orders)
print(orders["order_status"].value_counts())
orders = orders[orders["order_status"] == "delivered"].copy()
log_step("Keep order_status == 'delivered'",
         "Only delivered orders have a real delivery date to judge lateness", before, len(orders))

# 3. Drop missing customer delivery date
before = len(orders)
orders = orders.dropna(subset=["order_delivered_customer_date"])
log_step("Drop missing order_delivered_customer_date",
         "Cannot compute delivery days or lateness without it", before, len(orders))

# 4. Create delivery features
orders["delivery_days"] = (orders["order_delivered_customer_date"]
                           - orders["order_purchase_timestamp"]).dt.days
orders["days_late"] = (orders["order_delivered_customer_date"]
                       - orders["order_estimated_delivery_date"]).dt.days
orders["is_late"] = orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]
orders["order_month"] = orders["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
log_step("Create delivery_days, days_late, is_late, order_month",
         "Core variables for Q1-Q5 (late = delivered after estimated date)", len(orders), len(orders))

# 5. Remove delivery-time outliers
before = len(orders)
print(orders["delivery_days"].describe())
p99 = orders["delivery_days"].quantile(0.99)
print("99th percentile delivery days:", p99)
orders = orders[(orders["delivery_days"] >= 0) & (orders["delivery_days"] <= 120)]
log_step("Remove delivery_days < 0 or > 120",
         "Impossible/extreme values (check against the 99th percentile above) distort averages",
         before, len(orders))

# 6. Deduplicate reviews: keep latest review per order
before = len(reviews)
reviews["review_answer_timestamp"] = pd.to_datetime(reviews["review_answer_timestamp"])
reviews = (reviews.sort_values("review_answer_timestamp")
                  .drop_duplicates("order_id", keep="last"))
log_step("Keep 1 review per order (latest)",
         "Some orders have several reviews; would duplicate rows on merge", before, len(reviews))

# 7. Merge orders + customers + reviews (order level)
before = len(orders)
order_df = (orders.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
                  .merge(reviews[["order_id", "review_score"]], on="order_id", how="left"))
assert len(order_df) == before, "Merge changed row count!"   # row-count check
log_step("Merge customers (state) and reviews",
         "Need state and review score per order; row count checked unchanged", before, len(order_df))

# 8. Drop orders with no review score
before = len(order_df)
order_df = order_df.dropna(subset=["review_score"])
log_step("Drop orders without a review score",
         "Review-based questions (Q3, Q4) need a score", before, len(order_df))

# %% [markdown]
# ## Items level table (for revenue, categories, sellers)
# `order_items` has several rows per order, so keep it separate and aggregate to order level
# before counting orders or revenue.

# %%
items_df = items[items["order_id"].isin(order_df["order_id"])].copy()

products = products.merge(cat_tr, on="product_category_name", how="left")
n_missing_en = products["product_category_name_english"].isna().sum()
products["category"] = products["product_category_name_english"].fillna("unknown")
log_step("Fill missing English category with 'unknown'",
         f"{n_missing_en} products have no English name (translation table is incomplete)",
         len(products), len(products))

before = len(items_df)
items_df = (items_df.merge(products[["product_id", "category"]], on="product_id", how="left")
                    .merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
                    .merge(order_df[["order_id", "order_month", "customer_state",
                                     "is_late", "review_score"]], on="order_id", how="left"))
assert len(items_df) == before
items_df["category"] = items_df["category"].fillna("unknown")

# Order-level revenue (price only; freight kept separately)
rev = items_df.groupby("order_id").agg(revenue=("price", "sum"),
                                       freight=("freight_value", "sum")).reset_index()
order_df = order_df.merge(rev, on="order_id", how="left")

# %%
log_df = pd.DataFrame(cleaning_log)
print(log_df.to_string(index=False))
print("\nFinal order-level table:", order_df.shape, "| items table:", items_df.shape)
print("Late rate:", round(order_df["is_late"].mean() * 100, 1), "%")

# %% [markdown]
# ## Q1: Monthly orders and revenue + 3-month rolling average

# %%
monthly = (order_df.groupby("order_month")
                   .agg(orders=("order_id", "nunique"), revenue=("revenue", "sum"))
                   .reset_index())
print(monthly.head(6)); print(monthly.tail(6))   # look for thin 2016 / truncated late 2018

# keep only complete, reasonably populated months for the trend
monthly = monthly[(monthly["order_month"] >= "2017-01-01") & (monthly["order_month"] <= "2018-08-01")].copy()
monthly["orders_roll3"]  = monthly["orders"].rolling(3).mean()
monthly["revenue_roll3"] = monthly["revenue"].rolling(3).mean()
monthly.head()

# %% [markdown]
# ## Q2: Late rate overall and by state  (own function)

# %%
def calc_late_rate(df, group_col):
    """Late-order share and order count per group, sorted worst first."""
    out = (df.groupby(group_col)
             .agg(orders=("order_id", "nunique"), late_rate=("is_late", "mean"))
             .reset_index())
    out["late_rate"] = (out["late_rate"] * 100).round(2)
    return out.sort_values("late_rate", ascending=False)

print("Overall late share:", round(order_df["is_late"].mean() * 100, 2), "%")
calc_late_rate(order_df, "customer_state").head(10)
