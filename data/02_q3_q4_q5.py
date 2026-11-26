# %% [markdown]
# ## Part 2: Q3 t-test, Q4 categories, Q5 sellers
# Runs Part 1 first so all tables (order_df, items_df) exist.

# %%
from pathlib import Path
exec(open(Path(__file__).parent / "01_clean_and_merge.py", encoding="utf-8").read())

from scipy import stats

# %% [markdown]
# ## Q3: Do review scores differ between on-time and late orders?

# %%
on_time = order_df.loc[~order_df["is_late"], "review_score"]
late    = order_df.loc[order_df["is_late"], "review_score"]

print(f"On-time: n={len(on_time):,}  mean={on_time.mean():.2f}")
print(f"Late:    n={len(late):,}  mean={late.mean():.2f}")

# Welch's t-test (does not assume equal variances)
t_stat, p_value = stats.ttest_ind(on_time, late, equal_var=False)

# Effect size: Cohen's d (with 100k orders, p is tiny almost always, so report size too)
pooled_sd = np.sqrt((on_time.var(ddof=1) + late.var(ddof=1)) / 2)
cohens_d = (on_time.mean() - late.mean()) / pooled_sd
diff = on_time.mean() - late.mean()

print(f"\nt = {t_stat:.1f}, p = {p_value:.2e}, mean difference = {diff:.2f} stars, Cohen's d = {cohens_d:.2f}")

# One-sentence manager version (check the numbers printed above before using)
print(f"\nManager sentence: Late orders score about {diff:.1f} stars lower than on-time orders "
      f"({late.mean():.1f} vs {on_time.mean():.1f}), and a gap this big is extremely unlikely to be chance.")
# Limitation to mention: this shows association, not proof that lateness alone causes low scores.

# %% [markdown]
# ## Q4: Revenue and review score by English category

# %%
# One row per (order, category) so multi-item orders do not count a review twice
cat_orders = items_df.drop_duplicates(["order_id", "category"])

cat_stats = (items_df.groupby("category").agg(revenue=("price", "sum")).reset_index()
             .merge(cat_orders.groupby("category")
                              .agg(orders=("order_id", "nunique"),
                                   avg_review=("review_score", "mean")).reset_index(),
                    on="category"))

print("Top 10 categories by revenue:")
print(cat_stats.sort_values("revenue", ascending=False).head(10).round(2).to_string(index=False))

# Worst reviews: require at least 100 orders so tiny categories do not dominate
print("\nWorst 10 categories by avg review (min 100 orders):")
print(cat_stats[cat_stats["orders"] >= 100].sort_values("avg_review").head(10).round(2).to_string(index=False))

# %% [markdown]
# ## Q5: Seller late rates (sellers with at least 30 orders)

# %%
# One row per (order, seller): an order with 2 sellers counts once for each
seller_orders = items_df.drop_duplicates(["order_id", "seller_id"])

seller_stats = (seller_orders.groupby("seller_id")
                .agg(orders=("order_id", "nunique"),
                     late_orders=("is_late", "sum"),
                     late_rate=("is_late", "mean"),
                     avg_review=("review_score", "mean")).reset_index())

qualified = seller_stats[seller_stats["orders"] >= 30].copy()
print(f"Sellers total: {len(seller_stats):,} | with >= 30 orders: {len(qualified):,}")

print("\nTop 10 sellers by late rate:")
print(qualified.sort_values("late_rate", ascending=False).head(10).round(3).to_string(index=False))

# Worst 10% of qualified sellers by late rate
cutoff = qualified["late_rate"].quantile(0.90)
worst = qualified[qualified["late_rate"] >= cutoff]

total_late_all  = seller_stats["late_orders"].sum()      # all late seller-orders
total_late_qual = qualified["late_orders"].sum()         # only qualified sellers

print(f"\nWorst 10% = {len(worst)} sellers (late rate >= {cutoff:.1%})")
print(f"They cause {worst['late_orders'].sum() / total_late_all:.1%} of ALL late orders")
print(f"They cause {worst['late_orders'].sum() / total_late_qual:.1%} of late orders among qualified sellers")
# State which definition you use in the report (the first matches the question wording).

# %% [markdown]
# ## Quick sanity checks

# %%
assert order_df["order_id"].is_unique
print("Orders:", order_df["order_id"].nunique(), "| Revenue total:", round(order_df["revenue"].sum()))
print("Category revenue total matches:", round(cat_stats["revenue"].sum()) == round(items_df["price"].sum()))
