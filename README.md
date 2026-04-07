# ML Regression Explorer

Interactive Streamlit app built for the **ReDI School ML Regression** session.  
Designed for classroom use — no install required for students, just a URL.

## What it does

Three modes, one app:

| Mode | Purpose | When to use |
|------|---------|-------------|
| **A — Find the Line** | Manually adjust slope & intercept to fit a regression line. Personal best MSE tracked. | Demo: before introducing sklearn |
| **B — Steer the Descent** | Step through gradient descent manually. Pick a learning rate, click Step, watch the path trace across the MSE loss landscape. | Demo: after introducing the cost function |
| **C — Break the Regression** | Toggle outliers, heteroscedasticity, non-linearity, noise, and multicollinearity. Scatter + residual plots always visible side by side. | Group activity: last ~25 min of session |

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (only `app.py`, `requirements.txt`, `.streamlit/config.toml` needed)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo, branch `main`, main file `app.py`
4. Deploy → share the URL with students

## Classroom use

**Mode A** (~5 min): Show the scatter. Ask students to guess the slope before clicking "Show Optimal".

**Mode B** (~10 min): Demo with `α = 2.0` first (watch it diverge), then `α = 0.1`. Ask: *"What did sklearn's `.fit()` just do?"*

**Mode C** (~25 min group activity): Split into groups of 4–5. Each group works through the missions in the sidebar expander. Debrief: each group picks one finding and explains it to the class.

## Data

Fully synthetic housing data (`sqft` × `price`). No external files or API calls — the app is self-contained.

## Requirements

- Python 3.9+
- See `requirements.txt`
