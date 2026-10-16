"""Pure CPU Plotly reports from normalized scientific tables."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def report(tables, path):
    summaries, components, roles = tables["checkpoints"], tables["components"], tables["roles"]
    fig = make_subplots(rows=3, cols=2, subplot_titles=["Edges", "Density", "Clean task score", "Component in-strength", "Role turnover", "Final-set Jaccard"])
    for (trajectory, threshold), frame in summaries.groupby(["trajectory_id", "threshold"]):
        frame = frame.sort_values("step")
        name = f"{frame.iloc[0].model_id} / {frame.iloc[0].task} / {frame.iloc[0].replacement} / {threshold}"
        for metric, row, col in [("num_edges", 1, 1), ("density", 1, 2), ("clean_score", 2, 1)]:
            fig.add_trace(go.Scatter(x=frame.step, y=frame[metric], mode="lines+markers", name=name,
                customdata=frame[["checkpoint", "measurement_status"]].values,
                hovertemplate="step=%{x}<br>value=%{y}<br>%{customdata}<extra>%{fullData.name}</extra>", showlegend=(row==1 and col==1)), row, col)
    strength = components[components.metric == "in_strength"]
    for (run, threshold), frame in strength.groupby(["run_id", "threshold"]):
        fig.add_trace(go.Box(y=frame.value, name=f"{run[:8]} / {threshold}", showlegend=False), 2, 2)
    for (trajectory, threshold, metric), frame in roles.groupby(["run_id", "threshold", "metric"]):
        frame = frame.sort_values("step")
        for column, col in [("churn", 1), ("final_jaccard", 2)]:
            fig.add_trace(go.Scatter(x=frame.step, y=frame[column], mode="lines+markers", showlegend=False,
                                    name=f"{trajectory[:8]} {metric} {threshold}"), 3, col)
    fig.update_layout(height=1050, title="Fresh experiment analysis — measured checkpoints; no inherited phase boundaries")
    fig.update_xaxes(title_text="Training step", row=1)
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return fig
