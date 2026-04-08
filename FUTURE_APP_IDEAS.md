# Future App Ideas — Regression Interactive Explorer

Ideas for future modes or standalone mini-apps. Grouped by concept.
Current app (v1) lives in `app.py` and covers: Find the Line · Steer the Descent · Break the Regression.

---

## High priority — high payoff, low build complexity

### Anscombe's Quartet / Datasaurus

Show only summary statistics first (mean, SD, R², slope) — students guess which dataset is "best".
Reveal the scatter plots. All four (or twelve) datasets have identical numbers, completely different shapes.
**Key message:** R² alone tells you nothing. Always plot. Always check residuals.
**Build:** trivial — hardcoded data, one plotly facet grid, a "reveal" button.

### Confidence vs Prediction Intervals

Two bands around the fitted line, always visible on the scatter plot.

- Narrow band = where the *mean line* might truly sit (CI)
- Wide band = where the *next individual point* will land (PI)
Slider: increase N → CI shrinks, PI barely moves (irreducible noise).
Toggle heteroscedasticity → PI becomes a lie at the high end, CI is still roughly OK.
**Key message:** "What will this house sell for?" needs a PI, not a CI. They're different questions.
**Build:** `scipy.stats.t` for exact intervals, ~30 lines.

### Residual Histogram + Normality Check

Live histogram of residuals, always beside the residual plot.
Overlaid normal curve. Toggle: add skew, fat tails, bimodal distribution.
**Key message:** OLS coefficients stay unbiased with non-normal noise (Gauss-Markov).
What breaks: p-values and confidence intervals. The line is still useful — the uncertainty estimates lie.
**Build:** trivial, just `go.Histogram` + overlay.

### Polynomial Degree Slider (Bias-Variance Preview)

One dataset, one slider: degree 1 → 2 → 3 → 7 → 15.
Side by side: fitted curve on data vs fitted curve on a held-out test set.
Watch train R² go to 1.0 while test R² collapses.
**Key message:** more complexity ≠ better model. Natural lead-in to regularization session.
**Build:** `numpy.polyfit` or sklearn `PolynomialFeatures`, simple.
**Note:** fits better in the regularization session than here, but could be a 2-min demo slide.

---

## Medium priority — strong concept, moderate build

### Bootstrap Confidence Intervals (Animated)

Start: scatter + one fitted line.
Press "Sample": resample 200 of 300 points with replacement, fit line, draw faintly in grey. Repeat.
After 100 samples: a bundle of grey lines. Width = standard error of the slope.
Toggle multicollinearity → bundle for sqft coefficient explodes even though the lines barely move.
**Key message:** coefficients are random variables. Their spread IS the standard error. No formula needed.
**Build:** `st.button("Add sample")` + session state accumulating lines. ~50 lines.
**Bonus:** show histogram of slope estimates building up alongside — connects to "unbiased" concept.

### Leverage & Cook's Distance (Draggable Point)

Scatter plot with regression line. One point is draggable (use plotly `selectedpoints` or a slider for x/y).
Move it to extreme X at normal Y → high leverage, low influence (line barely moves).
Move it to extreme X at extreme Y → high influence (line chases it).
Cook's distance indicator: traffic-light colour on the dragged point.
**Key message:** one point can dominate your entire model. This is why you check for influential observations before trusting coefficients.
**Build:** moderate — need to handle drag interaction cleanly in Streamlit.

### Feature Scaling → Gradient Descent Landscape

Run GD on unscaled features (sqft 400–4000, rooms 1–12 side by side):
Loss landscape is a stretched ellipse → GD zigzags, takes many steps.
Toggle "normalise features" → circular bowl → GD goes straight to minimum in far fewer steps.
**Key message:** sklearn scales internally for some solvers — but you need to know why, and why your coefficients change when you scale.
**Build:** extends Mode B in current app — add a normalise toggle + redraw landscape. Natural extension.

---

## Lower priority — deep or niche, but visually satisfying

### Geometric Interpretation (Projection)

3D scatter: two features (sqft, rooms) on X/Y axes, price on Z.
Show the regression *plane* fitted through the cloud.
Residuals as vertical lines from each point to the plane.
Rotate the view to show that residuals are orthogonal to the feature plane — that's what OLS minimises.
**Key message:** regression = projection. The hat matrix H = X(XᵀX)⁻¹Xᵀ is just projecting y onto the column space of X.
**Build:** `go.Surface` + `go.Scatter3d`. Moderate. Best as a demo, not interactive.

### Simpson's Paradox

Dataset where overall regression slope is positive, but negative within every subgroup.
Dropdown: "Show all data" vs "Colour by group". Slope visibly reverses.
**Key message:** aggregate trends can be completely misleading. Always ask "is there a lurking variable?"
**Build:** easy — synthetic data, plotly colour grouping.

### Regression to the Mean

Historical concept (Galton's height data). Tall parents → children shorter than parents on average.
Interactive: plot parent height vs child height. Show that predictions always "regress" toward the mean.
Slider: extremeness of parent → see how much the prediction moves toward centre vs the parent value.
**Key message:** the name "regression" comes from here. Extreme observations tend to be followed by less extreme ones — not because of any mechanism, just statistics.
**Build:** trivial.

### Prediction vs Extrapolation

Clean dataset with a clear range (sqft 400–4000). Extend X axis to 6000, 10000.
Show confidence and prediction intervals widening rapidly outside the training range.
Toggle: "what if the true relationship is quadratic?" → inside range looks fine, outside explodes.
**Key message:** your model is only valid inside (approximately) the training distribution. Extrapolation is an assumption, not a calculation.
**Build:** easy.

### QQ Plot for Residuals

Theoretical quantiles of a normal distribution vs actual residual quantiles.
Points should lie on the diagonal for normally distributed residuals.
Toggle: fat tails, skew, outliers → watch the QQ plot curve at the ends.
**Key message:** QQ plot is the standard diagnostic for normality of residuals. Learn the shapes.
**Build:** trivial with `scipy.stats.probplot`.

---

## Story arc these could form (if ever expanded to a full tool)

```
Find the Line           → what is regression doing?
Steer the Descent       → how does it find the answer? (current app Mode B)
Bootstrap the Slope     → how certain is that answer?
Confidence vs PI        → what can you promise a client?
Break the Regression    → when can't you trust it? (current app Mode C)
Polynomial Degree       → what happens when you make it too flexible?
```

Each step answers one natural "but wait..." question from the previous one.
