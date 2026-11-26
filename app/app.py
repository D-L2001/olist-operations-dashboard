"""Operations Dashboard: Late Deliveries, Low Stars (Olist).
Run with:  streamlit run app/app.py
Needs data/orders_clean.csv.gz and data/seller_orders_clean.csv.gz (created by the notebook).
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Olist Operations Dashboard", layout="wide")

# ---------------------------------------------------------------- data
HERE = Path(__file__).resolve().parent
DATA_DIRS = [HERE.parent / "data", HERE / "data", Path("data")]


@st.cache_data
def load_data():
    for d in DATA_DIRS:
        o, s = d / "orders_clean.csv.gz", d / "seller_orders_clean.csv.gz"
        if o.exists() and s.exists():
            orders = pd.read_csv(o, parse_dates=["order_purchase_timestamp", "order_month"])
            sellers = pd.read_csv(s, parse_dates=["order_month"])
            return orders, sellers
    return None, None


orders, seller_orders = load_data()
if orders is None:
    st.error("Cleaned data not found. Run the notebook first so that data/orders_clean.csv.gz "
             "and data/seller_orders_clean.csv.gz exist.")
    st.stop()

orders["is_late"] = orders["is_late"].astype(bool)
seller_orders["is_late"] = seller_orders["is_late"].astype(bool)

page = st.sidebar.radio("Page", ["Operations dashboard", "Seller look-up"])

# ---------------------------------------------------------------- page 1
if page == "Operations dashboard":
    st.title("Operations Dashboard: Late Deliveries, Low Stars")
    st.caption("Olist Brazilian e-commerce, delivered orders with a review, 2016-2018")

    st.sidebar.header("Filters")
    min_d = orders["order_purchase_timestamp"].min().date()
    max_d = orders["order_purchase_timestamp"].max().date()
    date_range = st.sidebar.date_input("Order date range", (min_d, max_d),
                                       min_value=min_d, max_value=max_d)
    states = sorted(orders["customer_state"].unique())
    sel_states = st.sidebar.multiselect("Customer state", states, default=states)
    cats = sorted(orders["category"].fillna("unknown").unique())
    sel_cats = st.sidebar.multiselect("Product category (main item)", cats, default=cats)

    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.info("Pick both a start and an end date.")
        st.stop()

    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    df = orders[(orders["order_purchase_timestamp"] >= start)
                & (orders["order_purchase_timestamp"] < end)
                & (orders["customer_state"].isin(sel_states))
                & (orders["category"].fillna("unknown").isin(sel_cats))]

    if df.empty:
        st.warning("No orders match these filters. Widen the filters.")
        st.stop()

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders", f"{df['order_id'].nunique():,}")
    c2.metric("Revenue (BRL)", f"{df['revenue'].sum():,.0f}")
    c3.metric("On-time rate", f"{(1 - df['is_late'].mean()) * 100:.1f}%",
              delta=f"{(df['is_late'].mean() - orders['is_late'].mean()) * -100:+.1f} pts vs all orders")
    c4.metric("Average review", f"{df['review_score'].mean():.2f} stars",
              delta=f"{df['review_score'].mean() - orders['review_score'].mean():+.2f} vs all orders")

    st.divider()
    left, right = st.columns(2)

    # Chart 1: monthly trend + 3-month rolling average
    monthly = (df.groupby("order_month").agg(orders=("order_id", "nunique")).reset_index()
                 .sort_values("order_month"))
    monthly["3-month rolling average"] = monthly["orders"].rolling(3).mean()
    fig1 = px.line(monthly, x="order_month", y=["orders", "3-month rolling average"],
                   title="Monthly orders", labels={"order_month": "Month", "value": "Orders", "variable": ""})
    left.plotly_chart(fig1, use_container_width=True)
    left.caption("Takeaway: shows how order volume moves over the selected period.")

    # Chart 2: delivery-delay chart (days late, negative = early)
    fig2 = px.histogram(df[df["days_late"].between(-40, 30)], x="days_late", nbins=70,
                        title="Delivery delay vs estimated date (days; below 0 = early)",
                        labels={"days_late": "Days late (negative = early)"})
    fig2.add_vline(x=0, line_dash="dash", annotation_text="estimated date")
    fig2.update_yaxes(title="Orders")
    right.plotly_chart(fig2, use_container_width=True)
    right.caption("Takeaway: most orders arrive before the estimate; the right tail is the late orders.")

    left2, right2 = st.columns(2)

    # Chart 3: late rate by state
    by_state = (df.groupby("customer_state")
                  .agg(orders=("order_id", "nunique"), late_rate=("is_late", "mean")).reset_index())
    by_state["late_rate"] *= 100
    fig3 = px.bar(by_state.sort_values("late_rate", ascending=False), x="customer_state", y="late_rate",
                  hover_data=["orders"], title="Late-delivery rate by state",
                  labels={"customer_state": "State", "late_rate": "Late orders (%)"})
    left2.plotly_chart(fig3, use_container_width=True)
    left2.caption("Takeaway: late rates differ a lot by state; the tallest bars are where to act first.")

    # Chart 4: review score by delivery status
    df = df.assign(status=df["is_late"].map({True: "Late", False: "On time"}))
    avg_rev = df.groupby("status")["review_score"].mean().reset_index()
    fig4 = px.bar(avg_rev, x="status", y="review_score", title="Average review: on-time vs late orders",
                  labels={"status": "Delivery status", "review_score": "Average review (stars)"})
    fig4.update_yaxes(range=[0, 5])
    right2.plotly_chart(fig4, use_container_width=True)
    right2.caption("Takeaway: late orders get much lower reviews than on-time orders.")

    # Chart 5: revenue by category
    by_cat = (df.assign(category=df["category"].fillna("unknown")).groupby("category")["revenue"].sum()
                .nlargest(10).reset_index().sort_values("revenue"))
    fig5 = px.bar(by_cat, x="revenue", y="category", orientation="h",
                  title="Top 10 categories by revenue", labels={"revenue": "Revenue (BRL)", "category": "Category"})
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Takeaway: a few categories bring most of the revenue.")

    # CSV download of the filtered data
    st.download_button("Download filtered data (CSV)",
                       df.drop(columns=["status"]).to_csv(index=False).encode("utf-8"),
                       file_name="olist_filtered_orders.csv", mime="text/csv")

# ---------------------------------------------------------------- page 2
else:
    st.title("Seller look-up")
    counts = seller_orders.groupby("seller_id").size().sort_values(ascending=False)
    seller_id = st.selectbox("Choose a seller (sorted by number of orders; type to search)",
                             counts.index.tolist(),
                             format_func=lambda s: f"{s[:8]}...  ({counts[s]} orders)")

    s = seller_orders[seller_orders["seller_id"] == seller_id]
    plat_late = seller_orders["is_late"].mean()
    plat_review = seller_orders["review_score"].mean()
    plat_rev_per_order = seller_orders["revenue"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders", f"{len(s):,}")
    c2.metric("Late rate", f"{s['is_late'].mean() * 100:.1f}%",
              delta=f"{(s['is_late'].mean() - plat_late) * 100:+.1f} pts vs platform ({plat_late * 100:.1f}%)",
              delta_color="inverse")
    c3.metric("Average review", f"{s['review_score'].mean():.2f}",
              delta=f"{s['review_score'].mean() - plat_review:+.2f} vs platform ({plat_review:.2f})")
    c4.metric("Revenue (BRL)", f"{s['revenue'].sum():,.0f}",
              delta=f"{s['revenue'].mean() - plat_rev_per_order:+.0f} per order vs platform")

    if len(s) < 30:
        st.info("This seller has fewer than 30 orders, so its rates are noisy.")

    monthly = s.groupby("order_month").agg(orders=("order_id", "nunique"),
                                           late_orders=("is_late", "sum")).reset_index()
    fig = px.bar(monthly, x="order_month", y=["orders", "late_orders"], barmode="group",
                 title="Seller orders and late orders by month",
                 labels={"order_month": "Month", "value": "Orders", "variable": ""})
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("Download this seller's orders (CSV)", s.to_csv(index=False).encode("utf-8"),
                       file_name=f"seller_{seller_id[:8]}_orders.csv", mime="text/csv")
