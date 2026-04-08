"""
ML Regression — Interactive Session App
ReDI School | Regression session

Three modes, one URL:
  A — Find the Line         : adjust slope/intercept, beat sklearn
  B — Steer the Descent     : step through gradient descent on the loss landscape
  C — Break the Regression  : inject failure modes, group activity
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="ML Regression Explorer",
    page_icon="📈",
    layout="wide",
)

RNG_SEED = 42
N_BASE   = 250  # synthetic data points


# ── Data ───────────────────────────────────────────────────────────────────────

@st.cache_data
def make_base_data():
    """Clean synthetic housing-like data: linear, low noise, no pathologies."""
    rng = np.random.default_rng(RNG_SEED)
    sqft    = rng.uniform(400, 4000, N_BASE)
    price_k = 80 + 0.18 * sqft + rng.normal(0, 50, N_BASE)
    return sqft.astype(float), price_k.astype(float)


def make_break_data(n_out, out_sev, hetero, rel, noise, add_rooms):
    """Synthetic data with controllable failure modes for Mode C."""
    rng = np.random.default_rng(RNG_SEED)
    sqft_base = rng.uniform(400, 4000, N_BASE)

    # ── Relationship type ──────────────────────────────────────────────────────
    if rel == "Quadratic":
        s = (sqft_base - 2200) / 900
        price_k = 400 + 200 * s + 130 * s ** 2 + rng.normal(0, 40, N_BASE)
    elif rel == "Logarithmic":
        price_k = 50 + 280 * np.log(sqft_base / 400) + rng.normal(0, 40, N_BASE)
    else:
        price_k = 80 + 0.18 * sqft_base + rng.normal(0, 50, N_BASE)

    # ── Extra noise ────────────────────────────────────────────────────────────
    if noise > 0:
        price_k = price_k + rng.normal(0, noise, N_BASE)

    # ── Heteroscedasticity: fan shape — variance grows with sqft ──────────────
    if hetero:
        pct = (sqft_base - sqft_base.min()) / (sqft_base.max() - sqft_base.min())
        price_k = price_k + rng.normal(0, 1, N_BASE) * pct * 280

    # ── Counter-trend outliers: tiny flats overpriced, huge flats underpriced ──
    # Placed in the body of the x-distribution but far off the trend line,
    # pulling the fitted line in the wrong direction.
    sqft = sqft_base.copy()
    if n_out > 0:
        rng2 = np.random.default_rng(RNG_SEED + 99)
        n_hi = (n_out + 1) // 2   # small sqft, absurdly expensive
        n_lo = n_out // 2          # large sqft, absurdly cheap

        sqft_hi  = rng2.uniform(400, 800, n_hi)
        price_hi = 520 + (out_sev - 1) * 70 + rng2.normal(0, 15, n_hi)  # 520–800 k$

        sqft_lo  = rng2.uniform(3400, 4000, n_lo)
        price_lo = np.maximum(20, 190 - (out_sev - 1) * 28 + rng2.normal(0, 15, n_lo))  # 190→50 k$

        sqft    = np.concatenate([sqft,    sqft_hi,  sqft_lo])
        price_k = np.concatenate([price_k, price_hi, price_lo])

    # ── Rooms: synthetic correlated feature (rooms ≈ sqft / 500) ──────────────
    if add_rooms:
        rooms_base = sqft_base / 500 + rng.normal(0, 0.4, N_BASE) + 1
        rooms_base = np.clip(rooms_base, 1, 12).round()
        if n_out > 0:
            n_out_actual = len(sqft) - N_BASE
            rooms = np.concatenate([rooms_base, np.full(n_out_actual, rooms_base.mean())])
        else:
            rooms = rooms_base
        return sqft, price_k, np.column_stack([sqft, rooms]), ["sqft (m²)", "rooms"]

    return sqft, price_k, sqft.reshape(-1, 1), ["sqft (m²)"]


def fit_ols(X, y):
    return LinearRegression().fit(X, y)


def r2_score(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def mse_score(y, yhat):
    return float(np.mean((y - yhat) ** 2))



# ── Mode A ─────────────────────────────────────────────────────────────────────

def mode_a():
    sqft, price_k = make_base_data()

    ols     = fit_ols(sqft.reshape(-1, 1), price_k)
    w1_opt  = float(ols.coef_[0])
    w0_opt  = float(ols.intercept_)
    mse_opt = mse_score(price_k, ols.predict(sqft.reshape(-1, 1)))
    r2_opt  = r2_score(price_k,  ols.predict(sqft.reshape(-1, 1)))

    with st.sidebar:
        st.subheader("Your line")
        w1 = st.slider("Slope  w₁  (k$ per sqft²)", -0.1, 0.5, 0.0, 0.005)
        w0 = st.slider("Intercept  w₀  (k$)", -200.0, 400.0, 100.0, 5.0)
        show_opt = st.button("✅ Show Optimal Line")
        st.markdown("---")
        with st.expander("ℹ️ What are MSE and R²?"):
            st.markdown(
                "**MSE** — average *squared* distance from each point to the line.  \n"
                "Lower = better. Squaring punishes big misses much more than small ones.\n\n"
                "**R²** — fraction of price variation your line explains.  \n"
                "· 1.0 = perfect  · 0.0 = no better than the mean  · <0 = worse than the mean"
            )

    y_man   = w1 * sqft + w0
    mse_man = mse_score(price_k, y_man)
    r2_man  = r2_score(price_k,  y_man)

    if "best_mse_a" not in st.session_state or mse_man < st.session_state.best_mse_a:
        st.session_state.best_mse_a = mse_man

    # Live formula — makes slope/intercept concrete
    sign = "+" if w1 >= 0 else ""
    st.markdown(
        f"**Your model:** &nbsp;"
        f"$\\hat{{price}} = {w0:.1f} + {w1:.3f} \\times sqft$"
        f"&emsp;→&emsp; *for every extra m², price changes by {sign}{w1:.3f} k$*",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Your MSE",      f"{mse_man:,.1f}")
    c2.metric("Personal Best", f"{st.session_state.best_mse_a:,.1f}")
    c3.metric("Your R²",       f"{r2_man:.3f}")
    if show_opt:
        c4.metric("Optimal R²", f"{r2_opt:.3f}",
                  delta=f"MSE = {mse_opt:,.1f}", delta_color="off")

    x_rng = np.linspace(sqft.min() - 100, sqft.max() + 100, 300)
    resid = price_k - y_man

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Scatter + Your Line", "Residuals (your line)"))
    fig.add_trace(go.Scatter(x=sqft, y=price_k, mode="markers",
        marker=dict(color="#4C9BE8", size=6, opacity=0.55)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_rng, y=w1 * x_rng + w0, mode="lines",
        line=dict(color="#E84C4C", width=3)), row=1, col=1)
    if show_opt:
        fig.add_trace(go.Scatter(x=x_rng, y=w1_opt * x_rng + w0_opt, mode="lines",
            line=dict(color="#2ECC71", width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sqft, y=resid, mode="markers",
        marker=dict(color="#E84C4C", size=5, opacity=0.5)), row=1, col=2)
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"), row=1, col=2)
    fig.update_xaxes(title_text="sqft (m²)",      row=1, col=1)
    fig.update_yaxes(title_text="price (k$)",      row=1, col=1)
    fig.update_xaxes(title_text="sqft (m²)",      row=1, col=2)
    fig.update_yaxes(title_text="Residual (k$)",   row=1, col=2)
    fig.update_layout(showlegend=False, height=430, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Good fit → residuals randomly scattered around 0 with no pattern. "
               "Any structure means the model is missing something.")


# ── Mode B ─────────────────────────────────────────────────────────────────────

def mode_b():
    sqft, price_k = make_base_data()
    mu_x, sig_x = sqft.mean(), sqft.std()
    Xn      = (sqft - mu_x) / sig_x
    ols     = fit_ols(Xn.reshape(-1, 1), price_k)
    w1_opt  = float(ols.coef_[0])
    w0_opt  = float(ols.intercept_)
    mse_opt = mse_score(price_k, ols.predict(Xn.reshape(-1, 1)))

    for k, v in [("gd_w0", 0.0), ("gd_w1", 0.0), ("gd_hist", [])]:
        if k not in st.session_state:
            st.session_state[k] = v

    with st.sidebar:
        st.subheader("Gradient descent")
        lr = st.select_slider("Learning rate  α",
            options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0], value=0.05)
        cb1, cb2 = st.columns(2)
        step1 = cb1.button("Step ×1")
        step10 = cb2.button("Step ×10")
        reset = st.button("🔄 Reset")
        st.markdown("---")
        with st.expander("ℹ️ How does this work?"):
            st.markdown(
                "1. Start at (w₀=0, w₁=0)\n"
                "2. Compute MSE\n"
                "3. Compute the **gradient** — which direction raises MSE fastest?\n"
                "4. Step in the **opposite** direction (downhill)\n"
                "5. Repeat\n\n"
                "The **orange arrow** shows your next step on the loss map.\n\n"
                "sklearn's `.fit()` runs this automatically.\n\n"
                "Try α=2.0 to see what happens when the step is too big."
            )

    def do_gd(w0, w1, alpha, steps=1):
        n_pts = len(Xn)
        for _ in range(steps):
            err = w1 * Xn + w0 - price_k
            w0 -= alpha * (2 / n_pts) * err.sum()
            w1 -= alpha * (2 / n_pts) * (err * Xn).sum()
            st.session_state.gd_hist.append((w0, w1, mse_score(price_k, w1 * Xn + w0)))
        return w0, w1

    if reset:
        st.session_state.gd_w0 = 0.0
        st.session_state.gd_w1 = 0.0
        st.session_state.gd_hist = []

    if step1:
        st.session_state.gd_w0, st.session_state.gd_w1 = do_gd(
            st.session_state.gd_w0, st.session_state.gd_w1, lr)
    if step10:
        st.session_state.gd_w0, st.session_state.gd_w1 = do_gd(
            st.session_state.gd_w0, st.session_state.gd_w1, lr, steps=10)

    w0c = st.session_state.gd_w0
    w1c = st.session_state.gd_w1
    hist = st.session_state.gd_hist
    n_steps = len(hist)
    mse_cur = mse_score(price_k, w1c * Xn + w0c)

    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Steps taken", n_steps)
    cm2.metric("Current MSE", f"{mse_cur:,.1f}")
    cm3.metric("Optimal MSE", f"{mse_opt:,.1f}")
    if n_steps > 0 and mse_cur <= mse_opt * 1.02:
        st.success(f"🎯 Converged in {n_steps} steps!")

    # Loss landscape (vectorised)
    all_w0 = [0.0, w0c, w0_opt] + [h[0] for h in hist]
    all_w1 = [0.0, w1c, w1_opt] + [h[1] for h in hist]
    pad0 = max(60, (max(all_w0) - min(all_w0)) * 0.4)
    pad1 = max(30, (max(all_w1) - min(all_w1)) * 0.4)
    w0g = np.linspace(min(all_w0) - pad0, max(all_w0) + pad0, 60)
    w1g = np.linspace(min(all_w1) - pad1, max(all_w1) + pad1, 60)
    W0, W1 = np.meshgrid(w0g, w1g)
    yhat_grid = W1[:, :, None] * Xn[None, None, :] + W0[:, :, None]
    Z = np.mean((price_k[None, None, :] - yhat_grid) ** 2, axis=2)

    # Gradient arrow at current position
    n_pts = len(Xn)
    err_c = w1c * Xn + w0c - price_k
    dw0 = (2 / n_pts) * err_c.sum()
    dw1 = (2 / n_pts) * (err_c * Xn).sum()
    gm = np.sqrt(dw0 ** 2 + dw1 ** 2) + 1e-9
    alen = (w0g[-1] - w0g[0]) * 0.1
    aw0 = -dw0 / gm * alen
    aw1 = -dw1 / gm * alen

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("MSE Loss Landscape (bowl shape)", "Scatter + Current Line"))

    fig.add_trace(go.Contour(x=w0g, y=w1g, z=Z,
        colorscale="Blues_r", showscale=False, ncontours=25), row=1, col=1)

    if hist:
        path_w0 = [0.0] + [h[0] for h in hist]
        path_w1 = [0.0] + [h[1] for h in hist]
        fig.add_trace(go.Scatter(x=path_w0, y=path_w1, mode="lines+markers",
            line=dict(color="#E84C4C", width=2),
            marker=dict(size=5, color="#E84C4C")), row=1, col=1)

    fig.add_trace(go.Scatter(x=[w0c], y=[w1c], mode="markers",
        marker=dict(color="#E84C4C", size=14, symbol="x")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[w0_opt], y=[w1_opt], mode="markers",
        marker=dict(color="#2ECC71", size=14, symbol="star")), row=1, col=1)

    fig.add_annotation(
        x=w0c + aw0, y=w1c + aw1, ax=w0c, ay=w1c,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowcolor="#FFA500", arrowwidth=3, arrowsize=1.5,
    )

    fig.update_xaxes(title_text="w₀ (intercept)", row=1, col=1)
    fig.update_yaxes(title_text="w₁ (slope)", row=1, col=1)

    x_rng  = np.linspace(sqft.min() - 100, sqft.max() + 100, 300)
    xn_rng = (x_rng - mu_x) / sig_x
    fig.add_trace(go.Scatter(x=sqft, y=price_k, mode="markers",
        marker=dict(color="#4C9BE8", size=7, opacity=0.6)), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_rng, y=w1c * xn_rng + w0c, mode="lines",
        line=dict(color="#E84C4C", width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_rng, y=w1_opt * xn_rng + w0_opt, mode="lines",
        line=dict(color="#2ECC71", width=1.5, dash="dot")), row=1, col=2)
    fig.update_xaxes(title_text="sqft (m²)",  row=1, col=2)
    fig.update_yaxes(title_text="price (k$)",  row=1, col=2)
    fig.update_layout(showlegend=False, height=450, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    if hist:
        mse_vals = [h[2] for h in hist]
        fig2 = go.Figure(go.Scatter(x=list(range(1, n_steps + 1)), y=mse_vals,
            mode="lines+markers", line=dict(color="#E84C4C", width=2)))
        fig2.update_layout(title="MSE over steps", xaxis_title="Step", yaxis_title="MSE",
            height=220, margin=dict(t=35, b=30))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Press **Step ×1** to start. Watch the red dot move on the loss landscape toward the green ★.")


# ── Mode C ─────────────────────────────────────────────────────────────────────

def mode_c():
    with st.sidebar:
        st.subheader("Failure modes")
        n_out     = st.slider("Outlier count",    0, 20, 0)
        out_sev   = st.slider("Outlier severity", 1.0, 6.0, 3.0, 0.5)
        hetero    = st.toggle("Heteroscedasticity", value=False)
        rel       = st.radio("Relationship", ["Linear", "Quadratic", "Logarithmic"])
        noise     = st.slider("Noise level (k$)", 0.0, 200.0, 0.0, step=10.0)
        add_rooms = st.toggle("Add 'rooms' feature (multicollinearity)", value=False)
        st.markdown("---")
        with st.expander("ℹ️ What to look for"):
            st.markdown(
                "**Scatter:** does the line follow the cloud?\n\n"
                "**Residuals (right plot):** should be random noise around 0\n"
                "- Fan shape → heteroscedasticity\n"
                "- Curve → non-linear relationship\n"
                "- Clusters at edges → high-leverage outliers\n\n"
                "**R²:** closer to 1 = better — but a high R² doesn't mean assumptions are met."
            )

    sqft, price_k, X, names = make_break_data(n_out, out_sev, hetero, rel, noise, add_rooms)
    model  = fit_ols(X, price_k)
    yhat   = model.predict(X)
    resid  = price_k - yhat
    r2_val = r2_score(price_k, yhat)

    # ── Metrics + coefficient table — always visible ──────────────────────────
    cr, cw = st.columns([1, 2])
    with cr:
        col = "#2ECC71" if r2_val > 0.7 else ("#F39C12" if r2_val > 0.4 else "#E74C3C")
        st.markdown(
            f"<div style='font-size:2.5rem;font-weight:bold;color:{col}'>R² = {r2_val:.3f}</div>",
            unsafe_allow_html=True)
    with cw:
        coef_rows = "".join(
            f"<tr><td><b>{n}</b></td><td style='padding-left:1em'>{c:.4f}</td></tr>"
            for n, c in zip(names, model.coef_)
        )
        coef_rows += (f"<tr><td><b>intercept</b></td>"
                      f"<td style='padding-left:1em'>{model.intercept_:.2f}</td></tr>")
        st.markdown(
            f"<table style='font-size:0.95rem;border-collapse:collapse'>"
            f"<tr><th style='text-align:left'>Feature</th>"
            f"<th style='text-align:left;padding-left:1em'>Coefficient</th></tr>"
            f"{coef_rows}"
            f"</table>"
            f"<div style='font-size:0.78rem;color:gray;margin-top:4px'>"
            f"price (k$) = intercept + Σ coef × feature</div>",
            unsafe_allow_html=True)

    # ── Contextual warnings ────────────────────────────────────────────────────
    if add_rooms:
        st.warning(
            "**Multicollinearity active** — `rooms` and `sqft` are correlated (larger flats have more rooms).  \n"
            "Toggle `rooms` off and watch the `sqft` coefficient shift — R² barely changes.  \n"
            "The model's *predictions* are unaffected. But the coefficients are no longer trustworthy:  \n"
            "the model can't tell whether price comes from size or room count, so it splits the credit "
            "arbitrarily between them.  \n"
            "_When features are correlated, individual coefficients can't be interpreted — "
            "even when R² looks great._"
        )
    if hetero:
        st.info(
            "**Heteroscedasticity** is normal in real housing data — cheap flats cluster "
            "tightly, expensive ones vary wildly.  \n"
            "Predictions are still usable. What breaks: your **confidence intervals** — "
            "the model is systematically overconfident about expensive houses."
        )

    x_rng  = np.linspace(sqft.min() - 50, sqft.max() + 50, 300)
    X_line = (np.column_stack([x_rng, np.full_like(x_rng, X[:, 1].mean())])
              if add_rooms else x_rng.reshape(-1, 1))

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Scatter + Fitted Line", "Residual Plot"))
    fig.add_trace(go.Scatter(x=sqft, y=price_k, mode="markers",
        marker=dict(color="#4C9BE8", size=6, opacity=0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_rng, y=model.predict(X_line), mode="lines",
        line=dict(color="#E84C4C", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=yhat, y=resid, mode="markers",
        marker=dict(color="#9B59B6", size=6, opacity=0.5)), row=1, col=2)
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"), row=1, col=2)
    fig.update_xaxes(title_text="sqft (m²)",         row=1, col=1)
    fig.update_yaxes(title_text="price (k$)",         row=1, col=1)
    fig.update_xaxes(title_text="Fitted price (k$)",  row=1, col=2)
    fig.update_yaxes(title_text="Residual (k$)",      row=1, col=2)
    fig.update_layout(showlegend=False, height=440, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Missions — try all of these"):
        st.markdown("""
**1 · Outliers**
Add 10 outliers at severity 5. Watch the line tilt. Does R² drop a lot?
Look at the residuals — do you see clusters at the edges pulling the line?
Can you make the scatter *look* OK but residuals reveal a problem?

**2 · Heteroscedasticity**
Toggle on. What pattern appears in the residuals?
What does this mean for predicting expensive vs cheap houses?
*(This pattern is actually common in real housing data.)*

**3 · Non-linearity**
Switch to Quadratic. What shape appears in the residuals?
Would you trust predictions for a 200 m² apartment? What would you do instead?

**4 · Noise**
Push noise to 60–80. At what level does R² fall below 0.5?
Is the regression still useful here — and for what?

**5 · Multicollinearity**
Enable `rooms`. Look at the warning — both coefficients are shown.  
Try to explain both numbers to a manager: "each extra m² adds X k$, each extra room adds Y k$."  
Now disable `rooms` — how much does the `sqft` coefficient change?  
R² barely moves. What does that tell you about prediction vs interpretation?
""")
    st.caption(
        "Residual tip: random scatter around 0 = assumptions roughly met. "
        "Any pattern (fan, curve, clusters) = the model is missing something."
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.title("📈 ML Regression Explorer")
    st.caption("ReDI School · Regression session  |  synthetic housing data")

    with st.sidebar:
        mode = st.selectbox(
            "Mode",
            ["A — Find the Line", "B — Steer the Descent", "C — Break the Regression"],
        )
        st.markdown("---")

    if mode.startswith("A"):
        st.header("Mode A — Find the Line")
        st.markdown(
            "Adjust **slope** and **intercept** to fit the line through the data by eye. "
            "Hit **Show Optimal** to see sklearn's answer and compare."
        )
        mode_a()
    elif mode.startswith("B"):
        st.header("Mode B — Steer the Descent")
        st.markdown(
            "Pick a **learning rate** and click **Step** to run gradient descent manually.  \n"
            "**Mission:** get to the green ★ in fewer than 15 steps."
        )
        mode_b()
    else:
        st.header("Mode C — Break the Regression")
        st.markdown(
            "Use the controls on the left to introduce failures into the data. "
            "Watch what happens to the line, the residuals, and R²."
        )
        mode_c()


if __name__ == "__main__":
    main()
