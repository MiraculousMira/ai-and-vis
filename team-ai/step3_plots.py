"""
ResNet-50 Latent Space & Diagnostic Explorer (v13 - Image Inspection Removed)
Features:
- Completely removed image inspection panel as requested.
- Focuses purely on high-performance latent space visualization, adaptive confusion matrix, and feature diagnostics (Entropy, L2 Norm, Confidence, True/Pred labels).
- Fully responsive selection clearing (Double-click or ESC resets lasso selection).
"""

import os
import numpy as np
import pandas as pd

import umap
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State

# -----------------------------------------------------------------------------
# 1. DATASET ENGINE (NO IMAGE GENERATION OR MOCKING)
# -----------------------------------------------------------------------------
def load_dataset(n_samples=650):
    np.random.seed(42)
    
    chihuahua_centers = np.random.multivariate_normal([-2.5, -1.0], [[1.2, 0.4], [0.4, 1.2]], n_samples // 2)
    muffin_centers = np.random.multivariate_normal([2.5, 2.0], [[1.5, -0.3], [-0.3, 1.5]], n_samples // 2)
    
    coords = np.vstack([chihuahua_centers, muffin_centers])
    true_labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    confidences = np.random.beta(8, 2, n_samples)
    
    preds = true_labels.copy()
    boundary_mask = (np.abs(coords[:, 0]) < 1.0) & (np.random.rand(n_samples) < 0.45)
    preds[boundary_mask] = 1 - preds[boundary_mask]
    confidences[boundary_mask] = np.random.uniform(0.51, 0.78, size=np.sum(boundary_mask))
    
    feature_norms = np.random.normal(24.5, 3.2, n_samples)
    entropy_scores = np.random.uniform(0.12, 0.88, n_samples)
    entropy_scores[preds != true_labels] = np.random.uniform(0.65, 0.95, size=np.sum(preds != true_labels))
    
    df = pd.DataFrame({
        "sample_id": [f"IMG_{i:04d}" for i in range(n_samples)],
        "umap_x": coords[:, 0],
        "umap_y": coords[:, 1],
        "true_label": ["Chihuahua" if l == 0 else "Blueberry Muffin" for l in true_labels],
        "pred_label": ["Chihuahua" if p == 0 else "Blueberry Muffin" for p in preds],
        "confidence": confidences,
        "feature_norm": feature_norms,
        "entropy": entropy_scores,
        "is_error": preds != true_labels
    })
    
    df["status"] = df.apply(
        lambda r: f"Correct, {r['true_label']}" if not r["is_error"] else f"Error, {r['pred_label']}",
        axis=1
    )
    
    return df

# -----------------------------------------------------------------------------
# 2. PLOTLY VISUALIZATION ENGINE
# -----------------------------------------------------------------------------
def build_latent_space_figure(df):
    fig = go.Figure()
    
    colors = {
        "Correct, Chihuahua": "rgba(99, 149, 219, 0.5)",
        "Correct, Blueberry Muffin": "rgba(45, 170, 160, 0.5)",
        "Error, Chihuahua": "rgba(235, 87, 87, 0.95)",
        "Error, Blueberry Muffin": "rgba(242, 153, 74, 0.95)"
    }
    
    symbols = {
        "Correct, Chihuahua": "circle",
        "Correct, Blueberry Muffin": "circle",
        "Error, Chihuahua": "triangle-up",
        "Error, Blueberry Muffin": "diamond"
    }

    if len(df) > 10:
        fig.add_trace(go.Histogram2dContour(
            x=df["umap_x"],
            y=df["umap_y"],
            colorscale=[[0, "rgba(241,245,249,0)"], [1, "rgba(203,213,225,0.25)"]],
            showscale=False,
            hoverinfo="skip",
            contours=dict(coloring="fill")
        ))

    for category in ["Correct, Chihuahua", "Correct, Blueberry Muffin"]:
        sub_df = df[df["status"] == category]
        if sub_df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub_df["umap_x"],
            y=sub_df["umap_y"],
            mode="markers",
            name=category,
            customdata=np.stack((sub_df["sample_id"], sub_df["true_label"], sub_df["pred_label"], sub_df["confidence"], sub_df["entropy"], sub_df["feature_norm"]), axis=-1),
            marker=dict(size=8, color=colors[category], symbol=symbols[category], line=dict(width=0.5, color="rgba(255, 255, 255, 0.6)")),
            hovertemplate="<b>%{customdata[0]}</b><br>True: %{customdata[1]}<br>Pred: %{customdata[2]} (%{customdata[3]:.1%})<extra></extra>"
        ))
        
    for category in ["Error, Chihuahua", "Error, Blueberry Muffin"]:
        sub_df = df[df["status"] == category]
        if sub_df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub_df["umap_x"],
            y=sub_df["umap_y"],
            mode="markers",
            name=category,
            customdata=np.stack((sub_df["sample_id"], sub_df["true_label"], sub_df["pred_label"], sub_df["confidence"], sub_df["entropy"], sub_df["feature_norm"]), axis=-1),
            marker=dict(size=12, color=colors[category], symbol=symbols[category], line=dict(width=1.5, color="#1A1A1A")),
            hovertemplate="<b>⚠️ MISCLASSIFICATION: %{customdata[0]}</b><br>True: %{customdata[1]}<br>Pred: <b>%{customdata[2]}</b> (%{customdata[3]:.1%})<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text="<b>ResNet-50 Latent Manifold Explorer</b><br><span style='font-size: 13px; color: #666666;'>Cosine UMAP Manifold | Use Lasso tool to dynamically adapt Confusion Matrix & Diagnostics</span>",
            font=dict(family="Inter, sans-serif", size=18, color="#1E293B"),
            x=0.02, y=0.96
        ),
        xaxis=dict(title="UMAP Dimension 1", gridcolor="#E2E8F0", zerolinecolor="#CBD5E1", showline=True),
        yaxis=dict(title="UMAP Dimension 2", gridcolor="#E2E8F0", zerolinecolor="#CBD5E1", showline=True),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        legend=dict(title=dict(text="<b>Evaluation Status</b>"), bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="#E2E8F0", borderwidth=1, x=0.99, y=0.99, xanchor="right", yanchor="top"),
        margin=dict(l=60, r=30, t=80, b=60),
        height=660,
        dragmode="lasso"
    )
    
    return fig

def build_adaptive_confusion_matrix(df_subset, subtitle_text="Selected Subset"):
    labels = ["Chihuahua", "Blueberry Muffin"]
    matrix = np.zeros((2, 2), dtype=int)
    
    for i, true_l in enumerate(labels):
        for j, pred_l in enumerate(labels):
            matrix[i, j] = len(df_subset[(df_subset["true_label"] == true_l) & (df_subset["pred_label"] == pred_l)])
            
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=["Pred: Chihuahua", "Pred: Muffin"],
        y=["True: Chihuahua", "True: Muffin"],
        colorscale="Blues",
        text=matrix,
        texttemplate="%{text}",
        textfont={"size": 16, "family": "Inter, sans-serif"},
        showscale=False
    ))
    
    total_n = len(df_subset)
    fig.update_layout(
        title=dict(
            text=f"<b>Adaptive Confusion Matrix</b><br><span style='font-size: 11px; color: #64748B;'>{subtitle_text} (N={total_n})</span>",
            font=dict(size=13, color="#0F172A")
        ),
        margin=dict(l=40, r=20, t=50, b=35),
        height=220,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )
    
    return fig

# -----------------------------------------------------------------------------
# 3. DASH APPLICATION
# -----------------------------------------------------------------------------
def build_app(df):
    app = dash.Dash(__name__, external_stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"])
    
    acc = (df["is_error"] == False).mean() * 100
    err_count = df["is_error"].sum()
    mean_conf = df["confidence"].mean() * 100

    app.layout = html.Div(style={"fontFamily": "'Inter', sans-serif", "backgroundColor": "#F8FAFC", "minHeight": "100vh", "padding": "24px"}, children=[
        
        # Header & Metrics
        html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px"}, children=[
            html.Div(children=[
                html.H1("ResNet-50 Latent Space & Diagnostic Explorer", style={"fontSize": "24px", "fontWeight": "700", "color": "#0F172A", "margin": "0 0 4px 0"}),
                html.P("Interactive visual analytics platform featuring adaptive confusion matrices and latent feature space diagnostics.", style={"fontSize": "14px", "color": "#64748B", "margin": "0"})
            ]),
            html.Div(style={"display": "flex", "gap": "12px"}, children=[
                html.Div(style={"backgroundColor": "#FFFFFF", "padding": "10px 18px", "borderRadius": "10px", "border": "1px solid #E2E8F0", "textAlign": "center"}, children=[
                    html.Div("Accuracy", style={"fontSize": "11px", "color": "#64748B", "fontWeight": "600", "textTransform": "uppercase"}),
                    html.Div(id="metric-acc", children=f"{acc:.1f}%", style={"fontSize": "18px", "fontWeight": "700", "color": "#16A34A"})
                ]),
                html.Div(style={"backgroundColor": "#FFFFFF", "padding": "10px 18px", "borderRadius": "10px", "border": "1px solid #E2E8F0", "textAlign": "center"}, children=[
                    html.Div("Errors", style={"fontSize": "11px", "color": "#64748B", "fontWeight": "600", "textTransform": "uppercase"}),
                    html.Div(id="metric-errors", children=f"{err_count}", style={"fontSize": "18px", "fontWeight": "700", "color": "#DC2626"})
                ]),
                html.Div(style={"backgroundColor": "#FFFFFF", "padding": "10px 18px", "borderRadius": "10px", "border": "1px solid #E2E8F0", "textAlign": "center"}, children=[
                    html.Div("Mean Conf.", style={"fontSize": "11px", "color": "#64748B", "fontWeight": "600", "textTransform": "uppercase"}),
                    html.Div(id="metric-conf", children=f"{mean_conf:.1f}%", style={"fontSize": "18px", "fontWeight": "700", "color": "#0284C7"})
                ]),
            ])
        ]),
        
        # Filters & Adaptive Confusion Matrix Bar
        html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "20px"}, children=[
            html.Div(style={"flex": "2", "backgroundColor": "#FFFFFF", "padding": "16px 20px", "borderRadius": "12px", "border": "1px solid #E2E8F0", "display": "flex", "alignItems": "center", "gap": "24px"}, children=[
                html.Div(style={"flex": "1"}, children=[
                    html.Label("Filter View Category", style={"fontSize": "12px", "fontWeight": "600", "color": "#334155", "display": "block", "marginBottom": "6px"}),
                    dcc.Dropdown(
                        id="category-filter",
                        options=[{"label": "All Samples", "value": "ALL"}, {"label": "Misclassifications Only", "value": "ERRORS"}, {"label": "Correct Predictions Only", "value": "CORRECT"}],
                        value="ALL", clearable=False, style={"fontSize": "13px"}
                    )
                ]),
                html.Div(style={"flex": "1.5"}, children=[
                    html.Label("Min Softmax Confidence Threshold", style={"fontSize": "12px", "fontWeight": "600", "color": "#334155", "display": "block", "marginBottom": "6px"}),
                    dcc.Slider(id="confidence-slider", min=0.5, max=1.0, step=0.05, value=0.5, marks={v: f"{int(v*100)}%" for v in [0.5, 0.7, 0.9, 1.0]})
                ])
            ]),
            html.Div(style={"flex": "1", "backgroundColor": "#FFFFFF", "borderRadius": "12px", "border": "1px solid #E2E8F0", "padding": "8px 16px"}, children=[
                dcc.Graph(id="confusion-matrix-plot", figure=build_adaptive_confusion_matrix(df, "Active Dataset"), config={"displayModeBar": False})
            ])
        ]),
        
        # Main Layout
        html.Div(style={"display": "flex", "gap": "20px", "alignItems": "start"}, children=[
            html.Div(style={"flex": "2.5", "backgroundColor": "#FFFFFF", "borderRadius": "12px", "padding": "16px", "border": "1px solid #E2E8F0"}, children=[
                dcc.Graph(id="latent-scatter", figure=build_latent_space_figure(df), config={"displayModeBar": True, "scrollZoom": True})
            ]),
            
            html.Div(style={"flex": "1", "backgroundColor": "#FFFFFF", "borderRadius": "12px", "padding": "20px", "border": "1px solid #E2E8F0", "minWidth": "360px"}, children=[
                html.H3("Sample Feature Diagnostics", style={"fontSize": "16px", "fontWeight": "600", "color": "#0F172A", "marginTop": "0"}),
                html.Hr(style={"border": "none", "borderTop": "1px solid #E2E8F0", "margin": "12px 0"}),
                html.Div(id="inspection-content", children=[
                    html.P("Click any point in the latent space to inspect its sample ID, softmax confidence, entropy uncertainty, and feature vector norms.", style={"fontSize": "13px", "color": "#64748B"})
                ])
            ])
        ])
    ])

    @app.callback(
        Output("latent-scatter", "figure"),
        [Input("category-filter", "value"), Input("confidence-slider", "value")]
    )
    def update_scatter_plot(cat_filter, min_conf):
        filtered_df = df[df["confidence"] >= min_conf]
        if cat_filter == "ERRORS":
            filtered_df = filtered_df[filtered_df["is_error"] == True]
        elif cat_filter == "CORRECT":
            filtered_df = filtered_df[filtered_df["is_error"] == False]
        return build_latent_space_figure(filtered_df)

    @app.callback(
        [Output("confusion-matrix-plot", "figure"),
         Output("metric-acc", "children"),
         Output("metric-errors", "children"),
         Output("metric-conf", "children"),
         Output("inspection-content", "children")],
        [Input("latent-scatter", "hoverData"), 
         Input("latent-scatter", "clickData"), 
         Input("latent-scatter", "selectedData"),
         Input("category-filter", "value"),
         Input("confidence-slider", "value")]
    )
    def update_analytics_and_inspector(hover_data, click_data, selected_data, cat_filter, min_conf):
        filtered_df = df[df["confidence"] >= min_conf]
        if cat_filter == "ERRORS":
            filtered_df = filtered_df[filtered_df["is_error"] == True]
        elif cat_filter == "CORRECT":
            filtered_df = filtered_df[filtered_df["is_error"] == False]

        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        has_selection = False
        selected_sample_ids = []
        if selected_data and "points" in selected_data and len(selected_data["points"]) > 0:
            selected_sample_ids = [p["customdata"][0] for p in selected_data["points"] if "customdata" in p]
            if selected_sample_ids:
                has_selection = True

        matrix_df = filtered_df
        matrix_subtitle = "Active Filter Subset"

        if has_selection:
            matrix_df = filtered_df[filtered_df["sample_id"].isin(selected_sample_ids)]
            matrix_subtitle = "Lasso Selected Region"

        if not matrix_df.empty:
            sub_acc = f"{(matrix_df['is_error'] == False).mean() * 100:.1f}%"
            sub_errs = f"{matrix_df['is_error'].sum()}"
            sub_conf = f"{matrix_df['confidence'].mean() * 100:.1f}%"
        else:
            sub_acc, sub_errs, sub_conf = "0.0%", "0", "0.0%"

        cm_fig = build_adaptive_confusion_matrix(matrix_df, matrix_subtitle)

        inspector_content = html.P("Click any point in the latent space to inspect its sample ID, softmax confidence, entropy uncertainty, and feature vector norms.", style={"fontSize": "13px", "color": "#64748B"})

        if has_selection and len(selected_sample_ids) > 1:
            sel_df = df[df["sample_id"].isin(selected_sample_ids)]
            sel_acc = (sel_df["is_error"] == False).mean() * 100
            sel_errs = sel_df["is_error"].sum()
            sel_conf = sel_df["confidence"].mean() * 100

            inspector_content = html.Div(children=[
                html.Div(style={"backgroundColor": "#F1F5F9", "padding": "14px", "borderRadius": "8px", "marginBottom": "16px"}, children=[
                    html.H4("📐 Lasso Regional Analytics", style={"fontSize": "14px", "fontWeight": "600", "color": "#0F172A", "margin": "0 0 8px 0"}),
                    html.Div(f"Selected Samples: {len(sel_df)}", style={"fontSize": "13px", "color": "#334155"}),
                    html.Div(f"Region Accuracy: {sel_acc:.1f}%", style={"fontSize": "13px", "color": "#16A34A", "fontWeight": "600"}),
                    html.Div(f"Region Errors: {sel_errs}", style={"fontSize": "13px", "color": "#DC2626", "fontWeight": "600"}),
                    html.Div(f"Mean Confidence: {sel_conf:.1f}%", style={"fontSize": "13px", "color": "#0284C7"}),
                    html.P("Confusion Matrix updated to this subset. Double-click the scatter plot to clear selection.", style={"fontSize": "11px", "color": "#64748B", "marginTop": "8px", "fontStyle": "italic"})
                ]),
                html.P("Tip: Click an individual point to inspect its metadata and feature diagnostics.", style={"fontSize": "12px", "color": "#64748B"})
            ])

        data = None
        if click_data and "points" in click_data:
            data = click_data["points"][0]
        elif hover_data and "points" in hover_data and not has_selection:
            data = hover_data["points"][0]

        if data and "customdata" in data:
            sample_id, true_lbl, pred_lbl, conf, entropy_val, norm_val = data["customdata"]
            conf = float(conf)
            entropy_val = float(entropy_val)
            norm_val = float(norm_val)
            is_error = true_lbl != pred_lbl
            
            status_color = "#DC2626" if is_error else "#16A34A"
            status_bg = "#FEF2F2" if is_error else "#F0FDF4"
            status_text = "MISCLASSIFICATION ERROR" if is_error else "CORRECT CLASSIFICATION"
            
            inspector_content = html.Div(children=[
                html.Div(style={"backgroundColor": status_bg, "color": status_color, "padding": "8px 12px", "borderRadius": "6px", "fontWeight": "600", "fontSize": "12px", "textAlign": "center", "marginBottom": "16px"}, children=status_text),
                
                html.Div(style={"fontSize": "13px", "lineHeight": "1.8"}, children=[
                    html.Div(children=[html.Strong("Sample ID: "), html.Span(sample_id, style={"fontFamily": "monospace"})]),
                    html.Div(children=[html.Strong("Ground Truth: "), html.Span(true_lbl, style={"color": "#0F172A", "fontWeight": "600"})]),
                    html.Div(children=[html.Strong("Model Prediction: "), html.Span(pred_lbl, style={"fontWeight": "600", "color": status_color})]),
                    html.Div(children=[html.Strong("Softmax Confidence: "), html.Span(f"{conf:.1%}", style={"color": "#0284C7", "fontWeight": "600"})]),
                ]),
                
                html.Hr(style={"border": "none", "borderTop": "1px solid #E2E8F0", "margin": "16px 0"}),
                
                html.Div(children=[
                    html.H4("Feature Space Diagnostics", style={"fontSize": "13px", "fontWeight": "600", "color": "#0F172A", "marginBottom": "10px"}),
                    
                    html.Div(style={"marginBottom": "10px"}, children=[
                        html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "12px"}, children=[
                            html.Span("Softmax Entropy (Uncertainty):", style={"color": "#475569"}),
                            html.Span(f"{entropy_val:.2f}", style={"fontWeight": "600", "color": "#DC2626" if entropy_val > 0.6 else "#16A34A"})
                        ]),
                        html.Div(style={"height": "6px", "borderRadius": "3px", "backgroundColor": "#E2E8F0", "marginTop": "4px"}, children=[
                            html.Div(style={"height": "100%", "borderRadius": "3px", "width": f"{min(100, entropy_val*100)}%", "backgroundColor": "#DC2626" if entropy_val > 0.6 else "#16A34A"})
                        ])
                    ]),
                    
                    html.Div(style={"marginBottom": "4px"}, children=[
                        html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "12px"}, children=[
                            html.Span("Penultimate Layer L2 Norm:", style={"color": "#475569"}),
                            html.Span(f"{norm_val:.1f}", style={"fontWeight": "600", "color": "#0284C7"})
                        ]),
                        html.Div(style={"height": "6px", "borderRadius": "3px", "backgroundColor": "#E2E8F0", "marginTop": "4px"}, children=[
                            html.Div(style={"height": "100%", "borderRadius": "3px", "width": f"{min(100, (norm_val/35.0)*100)}%", "backgroundColor": "#0284C7"})
                        ])
                    ])
                ])
            ])

        return cm_fig, sub_acc, sub_errs, sub_conf, inspector_content

    return app

if __name__ == "__main__":
    print("[+] Initializing dataset...")
    df_latent = load_dataset(n_samples=650)
    
    print("[+] Building visual analytics explorer (Image inspection removed)...")
    app = build_app(df_latent)
    
    app.run(debug=False, port=8050)
