import plotly.graph_objects as go


def reliability_gauge(score):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "green"},
                "steps": [
                    {"range": [0, 50], "color": "#7f1d1d"},
                    {"range": [50, 75], "color": "#ca8a04"},
                    {"range": [75, 100], "color": "#15803d"},
                ],
            },
        )
    )

    fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))

    return fig