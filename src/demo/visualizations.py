"""
VantageFlow AI - Visualization Functions for Streamlit

Comprehensive chart generation functions using Plotly with:
- Consistent color scheme
- Professional styling
- Mobile-responsive layouts
- Clear labels and titles
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ==================== COLOR SCHEME ====================

# VantageFlow AI Color Palette
COLORS = {
    'primary': '#1e3a8a',        # Dark blue
    'secondary': '#3b82f6',      # Medium blue
    'success': '#059669',        # Green
    'warning': '#f59e0b',        # Orange
    'danger': '#dc2626',         # Red
    'neutral': '#6b7280',        # Gray
    'light_bg': '#f3f4f6',       # Light gray
    'light_blue': '#eff6ff',     # Light blue
    'positive': '#10b981',       # Bright green
    'negative': '#ef4444',       # Bright red
}

# Score range colors
SCORE_COLORS = {
    'excellent': '#059669',      # 750-850 (Green)
    'good': '#10b981',           # 700-749 (Light green)
    'fair': '#f59e0b',           # 650-699 (Orange)
    'poor': '#ef4444',           # 600-649 (Light red)
    'very_poor': '#dc2626'       # 300-599 (Dark red)
}

# Standard layout settings
LAYOUT_CONFIG = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': {'family': 'Arial, sans-serif', 'color': '#1f2937'},
    'margin': dict(l=40, r=40, t=60, b=40),
    'hovermode': 'closest',
    'showlegend': True
}


# ==================== CHART FUNCTIONS ====================

def create_score_gauge(
    score: float,
    risk_tier: Optional[str] = None,
    min_score: int = 300,
    max_score: int = 850
) -> go.Figure:
    """
    Create Plotly gauge chart for credit score (300-850 scale).

    Args:
        score: Credit score value (300-850)
        risk_tier: Optional risk tier ('LOW', 'MEDIUM', 'HIGH', 'VERY HIGH')
        min_score: Minimum score value (default 300)
        max_score: Maximum score value (default 850)

    Returns:
        Plotly Figure object

    Example:
        >>> from src.demo.visualizations import create_score_gauge
        >>> fig = create_score_gauge(score=720, risk_tier='LOW')
        >>> fig.show()
    """
    # Determine color based on score
    if score >= 750:
        color = SCORE_COLORS['excellent']
        tier_text = "EXCELLENT"
    elif score >= 700:
        color = SCORE_COLORS['good']
        tier_text = "GOOD"
    elif score >= 650:
        color = SCORE_COLORS['fair']
        tier_text = "FAIR"
    elif score >= 600:
        color = SCORE_COLORS['poor']
        tier_text = "POOR"
    else:
        color = SCORE_COLORS['very_poor']
        tier_text = "VERY POOR"

    # Use provided risk_tier if available
    if risk_tier:
        tier_text = risk_tier

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<b>Credit Score</b><br><span style='font-size:0.8em;color:{COLORS['neutral']}'>{tier_text}</span>",
            'font': {'size': 20, 'color': COLORS['primary']}
        },
        number={
            'font': {'size': 72, 'color': color, 'family': 'Arial Black'},
            'suffix': ""
        },
        delta={
            'reference': 650,  # National average reference
            'increasing': {'color': COLORS['success']},
            'decreasing': {'color': COLORS['danger']},
            'font': {'size': 18}
        },
        gauge={
            'axis': {
                'range': [min_score, max_score],
                'tickwidth': 2,
                'tickcolor': COLORS['neutral'],
                'tickmode': 'linear',
                'tick0': min_score,
                'dtick': 50
            },
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': COLORS['neutral'],
            'steps': [
                {'range': [300, 600], 'color': '#fee2e2'},   # Very poor (light red)
                {'range': [600, 650], 'color': '#fed7aa'},   # Poor (light orange)
                {'range': [650, 700], 'color': '#fef3c7'},   # Fair (light yellow)
                {'range': [700, 750], 'color': '#d1fae5'},   # Good (light green)
                {'range': [750, 850], 'color': '#a7f3d0'}    # Excellent (bright green)
            ],
            'threshold': {
                'line': {'color': color, 'width': 6},
                'thickness': 0.8,
                'value': score
            }
        }
    ))

    fig.update_layout(
        height=400,
        **LAYOUT_CONFIG,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    return fig


def create_income_chart(
    monthly_data: pd.DataFrame,
    date_column: str = 'month',
    income_column: str = 'income',
    show_ma: bool = True,
    ma_window: int = 3
) -> go.Figure:
    """
    Create time series chart of monthly income with moving average.

    Args:
        monthly_data: DataFrame with monthly income data
        date_column: Name of date column
        income_column: Name of income column
        show_ma: Whether to show moving average line
        ma_window: Moving average window size (default 3 months)

    Returns:
        Plotly Figure object

    Example:
        >>> import pandas as pd
        >>> from src.demo.visualizations import create_income_chart
        >>>
        >>> data = pd.DataFrame({
        ...     'month': pd.date_range('2023-01-01', periods=12, freq='M'),
        ...     'income': [3200, 3400, 3100, 3500, 3300, 3600, 3400, 3700, 3500, 3800, 3600, 3900]
        ... })
        >>> fig = create_income_chart(data)
        >>> fig.show()
    """
    fig = go.Figure()

    # Income line with markers
    fig.add_trace(go.Scatter(
        x=monthly_data[date_column],
        y=monthly_data[income_column],
        mode='lines+markers',
        name='Monthly Income',
        line=dict(color=COLORS['secondary'], width=3),
        marker=dict(size=8, color=COLORS['primary'], line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor=f"rgba(59, 130, 246, 0.1)",
        hovertemplate='<b>%{x|%b %Y}</b><br>Income: $%{y:,.2f}<extra></extra>'
    ))

    # Moving average
    if show_ma and len(monthly_data) >= ma_window:
        ma = monthly_data[income_column].rolling(window=ma_window).mean()
        fig.add_trace(go.Scatter(
            x=monthly_data[date_column],
            y=ma,
            mode='lines',
            name=f'{ma_window}-Month MA',
            line=dict(color=COLORS['success'], width=2, dash='dash'),
            hovertemplate='<b>%{x|%b %Y}</b><br>MA: $%{y:,.2f}<extra></extra>'
        ))

    fig.update_layout(
        title={
            'text': '<b>Monthly Income Trend</b>',
            'font': {'size': 18, 'color': COLORS['primary']},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Month',
        yaxis_title='Income ($)',
        xaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            showline=True,
            linecolor=COLORS['neutral']
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            showline=True,
            linecolor=COLORS['neutral'],
            tickformat='$,.0f'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=400,
        **LAYOUT_CONFIG
    )

    return fig


def create_spending_breakdown(
    category_data: Dict[str, float],
    title: str = "Spending Breakdown by Category"
) -> go.Figure:
    """
    Create pie chart of spending by category.

    Args:
        category_data: Dictionary mapping category names to amounts
        title: Chart title

    Returns:
        Plotly Figure object

    Example:
        >>> from src.demo.visualizations import create_spending_breakdown
        >>>
        >>> spending = {
        ...     'RENT': 1200,
        ...     'GROCERIES': 450,
        ...     'UTILITIES': 200,
        ...     'TRANSPORTATION': 180,
        ...     'ENTERTAINMENT': 150,
        ...     'DINING': 300,
        ...     'SHOPPING': 250
        ... }
        >>> fig = create_spending_breakdown(spending)
        >>> fig.show()
    """
    categories = list(category_data.keys())
    amounts = list(category_data.values())

    # Color palette for categories
    colors = px.colors.qualitative.Set3[:len(categories)]

    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=amounts,
        hole=0.4,  # Donut chart
        marker=dict(
            colors=colors,
            line=dict(color='white', width=2)
        ),
        textposition='outside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
    )])

    # Calculate total
    total = sum(amounts)

    fig.update_layout(
        title={
            'text': f'<b>{title}</b>',
            'font': {'size': 18, 'color': COLORS['primary']},
            'x': 0.5,
            'xanchor': 'center'
        },
        annotations=[dict(
            text=f'<b>Total</b><br>${total:,.0f}',
            x=0.5,
            y=0.5,
            font_size=16,
            showarrow=False
        )],
        height=450,
        **LAYOUT_CONFIG,
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.05
        )
    )

    return fig


def create_cashflow_chart(
    monthly_data: pd.DataFrame,
    date_column: str = 'month',
    income_column: str = 'income',
    expense_column: str = 'expenses',
    net_column: Optional[str] = None
) -> go.Figure:
    """
    Create bar + line combo chart for monthly cashflow analysis.

    Args:
        monthly_data: DataFrame with monthly cashflow data
        date_column: Name of date column
        income_column: Name of income column
        expense_column: Name of expenses column
        net_column: Optional name of net cashflow column (calculated if not provided)

    Returns:
        Plotly Figure object

    Example:
        >>> import pandas as pd
        >>> from src.demo.visualizations import create_cashflow_chart
        >>>
        >>> data = pd.DataFrame({
        ...     'month': pd.date_range('2023-01-01', periods=12, freq='M'),
        ...     'income': [3200, 3400, 3100, 3500, 3300, 3600, 3400, 3700, 3500, 3800, 3600, 3900],
        ...     'expenses': [2800, 2900, 2850, 2950, 3000, 2900, 3100, 3050, 2950, 3000, 3100, 3050]
        ... })
        >>> fig = create_cashflow_chart(data)
        >>> fig.show()
    """
    # Calculate net cashflow if not provided
    if net_column is None:
        monthly_data = monthly_data.copy()
        monthly_data['net_cashflow'] = monthly_data[income_column] - monthly_data[expense_column]
        net_column = 'net_cashflow'

    fig = go.Figure()

    # Income bars
    fig.add_trace(go.Bar(
        x=monthly_data[date_column],
        y=monthly_data[income_column],
        name='Income',
        marker_color=COLORS['success'],
        opacity=0.8,
        hovertemplate='<b>%{x|%b %Y}</b><br>Income: $%{y:,.2f}<extra></extra>'
    ))

    # Expense bars
    fig.add_trace(go.Bar(
        x=monthly_data[date_column],
        y=monthly_data[expense_column],
        name='Expenses',
        marker_color=COLORS['danger'],
        opacity=0.8,
        hovertemplate='<b>%{x|%b %Y}</b><br>Expenses: $%{y:,.2f}<extra></extra>'
    ))

    # Net cashflow line
    fig.add_trace(go.Scatter(
        x=monthly_data[date_column],
        y=monthly_data[net_column],
        name='Net Cashflow',
        mode='lines+markers',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8, symbol='diamond'),
        yaxis='y2',
        hovertemplate='<b>%{x|%b %Y}</b><br>Net: $%{y:,.2f}<extra></extra>'
    ))

    # Add zero line for net cashflow
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=COLORS['neutral'],
        opacity=0.5,
        yref='y2'
    )

    fig.update_layout(
        title={
            'text': '<b>Monthly Cashflow Analysis</b>',
            'font': {'size': 18, 'color': COLORS['primary']},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Month',
        yaxis=dict(
            title='Amount ($)',
            showgrid=True,
            gridcolor='#e5e7eb',
            tickformat='$,.0f'
        ),
        yaxis2=dict(
            title='Net Cashflow ($)',
            overlaying='y',
            side='right',
            showgrid=False,
            tickformat='$,.0f'
        ),
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=450,
        **LAYOUT_CONFIG
    )

    return fig


def create_shap_waterfall(
    shap_values: Dict[str, float],
    base_value: float,
    prediction: float,
    feature_values: Optional[Dict[str, float]] = None,
    max_display: int = 10
) -> go.Figure:
    """
    Create SHAP waterfall chart showing feature contributions.

    Args:
        shap_values: Dictionary mapping feature names to SHAP values
        base_value: Base prediction value (expected value)
        prediction: Final prediction value
        feature_values: Optional dictionary of actual feature values
        max_display: Maximum number of features to display

    Returns:
        Plotly Figure object

    Example:
        >>> from src.demo.visualizations import create_shap_waterfall
        >>>
        >>> shap_vals = {
        ...     'income_cv': -0.045,
        ...     'savings_rate': -0.032,
        ...     'overdraft_count_3mo': 0.058,
        ...     'expense_income_ratio': 0.023,
        ...     'avg_balance': -0.018
        ... }
        >>> fig = create_shap_waterfall(
        ...     shap_values=shap_vals,
        ...     base_value=0.25,
        ...     prediction=0.236
        ... )
        >>> fig.show()
    """
    # Sort by absolute value
    sorted_features = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:max_display]

    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]

    # Calculate cumulative sum for waterfall
    cumulative = [base_value]
    for val in values:
        cumulative.append(cumulative[-1] + val)

    # Prepare data for waterfall
    y_labels = ['Base Value'] + [f.replace('_', ' ').title() for f in features] + ['Prediction']

    # Measure values (absolute changes)
    measure_values = [base_value] + values + [prediction - cumulative[-1]]

    # Text labels
    text_labels = [f'${base_value:.3f}']
    for i, val in enumerate(values):
        feature_val = feature_values.get(features[i], None) if feature_values else None
        if feature_val is not None:
            text_labels.append(f'{val:+.4f}<br>val={feature_val:.2f}')
        else:
            text_labels.append(f'{val:+.4f}')
    text_labels.append(f'${prediction:.3f}')

    # Colors based on increase/decrease
    colors = [COLORS['neutral']]  # Base value
    for val in values:
        colors.append(COLORS['danger'] if val > 0 else COLORS['success'])
    colors.append(COLORS['primary'])  # Prediction

    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=['absolute'] + ['relative'] * len(values) + ['total'],
        x=y_labels,
        textposition="outside",
        text=text_labels,
        y=measure_values,
        connector={"line": {"color": COLORS['neutral'], "width": 2, "dash": "dot"}},
        increasing={"marker": {"color": COLORS['danger']}},
        decreasing={"marker": {"color": COLORS['success']}},
        totals={"marker": {"color": COLORS['primary']}}
    ))

    fig.update_layout(
        title={
            'text': '<b>SHAP Waterfall Plot - Feature Contributions</b>',
            'font': {'size': 18, 'color': COLORS['primary']},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='',
        yaxis_title='Prediction Value',
        xaxis=dict(
            tickangle=-45,
            showgrid=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            tickformat='.3f'
        ),
        height=500,
        **LAYOUT_CONFIG,
        showlegend=False
    )

    return fig


def create_feature_importance_bar(
    importances: Dict[str, float],
    top_n: int = 15,
    title: str = "Feature Importance"
) -> go.Figure:
    """
    Create horizontal bar chart of feature importances.

    Args:
        importances: Dictionary mapping feature names to importance values
        top_n: Number of top features to display
        title: Chart title

    Returns:
        Plotly Figure object

    Example:
        >>> from src.demo.visualizations import create_feature_importance_bar
        >>>
        >>> importances = {
        ...     'overdraft_count_3mo': 0.142,
        ...     'income_cv': 0.118,
        ...     'savings_rate': 0.095,
        ...     'avg_balance': 0.087,
        ...     'expense_income_ratio': 0.076,
        ...     'discretionary_pct': 0.064,
        ...     'income_trend_3mo': 0.052,
        ...     'spending_volatility': 0.048
        ... }
        >>> fig = create_feature_importance_bar(importances)
        >>> fig.show()
    """
    # Sort by importance
    sorted_features = sorted(
        importances.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    features = [f[0].replace('_', ' ').title() for f in sorted_features]
    values = [f[1] for f in sorted_features]

    # Colors based on value (positive/negative)
    colors = [COLORS['success'] if v >= 0 else COLORS['danger'] for v in values]

    fig = go.Figure(go.Bar(
        y=features[::-1],  # Reverse for top-to-bottom
        x=values[::-1],
        orientation='h',
        marker=dict(
            color=colors[::-1],
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f"{v:.4f}" for v in values[::-1]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': f'<b>{title}</b>',
            'font': {'size': 18, 'color': COLORS['primary']},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Importance Value',
        yaxis_title='',
        xaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            zeroline=True,
            zerolinecolor=COLORS['neutral'],
            zerolinewidth=2
        ),
        yaxis=dict(
            showgrid=False
        ),
        height=max(400, top_n * 30),
        **LAYOUT_CONFIG,
        showlegend=False,
        margin=dict(l=150, r=100, t=60, b=40)
    )

    return fig


def create_fairness_dashboard(
    fairness_metrics: Dict[str, Any],
    protected_attribute: str = 'gender'
) -> go.Figure:
    """
    Create multi-panel fairness metrics dashboard.

    Args:
        fairness_metrics: Dictionary with fairness metric results
            Expected structure:
            {
                'groups': ['male', 'female'],
                'selection_rates': [0.45, 0.42],
                'tpr': [0.78, 0.75],
                'fpr': [0.12, 0.14],
                'disparate_impact': [1.0, 0.93],
                'demographic_parity': [1.0, 0.07],
                'equalized_odds': [1.0, 0.03]
            }
        protected_attribute: Name of protected attribute being analyzed

    Returns:
        Plotly Figure with subplots

    Example:
        >>> from src.demo.visualizations import create_fairness_dashboard
        >>>
        >>> metrics = {
        ...     'groups': ['Group A', 'Group B', 'Group C'],
        ...     'selection_rates': [0.45, 0.42, 0.48],
        ...     'tpr': [0.78, 0.75, 0.80],
        ...     'fpr': [0.12, 0.14, 0.11],
        ...     'disparate_impact': [1.0, 0.93, 1.07],
        ...     'demographic_parity': [0.0, 0.03, 0.03],
        ...     'equalized_odds': [0.0, 0.03, 0.02]
        ... }
        >>> fig = create_fairness_dashboard(metrics)
        >>> fig.show()
    """
    # Create subplots: 2 rows, 2 columns
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            'Selection Rates by Group',
            'TPR vs FPR by Group',
            'Disparate Impact (80% Rule)',
            'Fairness Metrics Summary'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'scatter'}],
            [{'type': 'bar'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )

    groups = fairness_metrics.get('groups', [])

    # 1. Selection Rates Bar Chart
    selection_rates = fairness_metrics.get('selection_rates', [])
    fig.add_trace(
        go.Bar(
            x=groups,
            y=selection_rates,
            name='Selection Rate',
            marker_color=COLORS['secondary'],
            text=[f'{sr:.1%}' for sr in selection_rates],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Selection Rate: %{y:.1%}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )

    # Add reference line at average
    avg_selection = np.mean(selection_rates)
    fig.add_hline(
        y=avg_selection,
        line_dash="dash",
        line_color=COLORS['success'],
        annotation_text=f"Avg: {avg_selection:.1%}",
        row=1, col=1
    )

    # 2. TPR vs FPR Scatter Plot
    tpr = fairness_metrics.get('tpr', [])
    fpr = fairness_metrics.get('fpr', [])

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode='markers+text',
            name='Groups',
            marker=dict(
                size=15,
                color=list(range(len(groups))),
                colorscale='Viridis',
                showscale=False,
                line=dict(width=2, color='white')
            ),
            text=groups,
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>FPR: %{x:.2%}<br>TPR: %{y:.2%}<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )

    # Add diagonal reference line (ideal: TPR high, FPR low)
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color=COLORS['neutral'], dash="dash", width=1),
        row=1, col=2
    )

    # 3. Disparate Impact Bar Chart (80% Rule)
    disparate_impact = fairness_metrics.get('disparate_impact', [])

    # Colors: green if >= 0.8, red if < 0.8
    di_colors = [COLORS['success'] if di >= 0.8 else COLORS['danger'] for di in disparate_impact]

    fig.add_trace(
        go.Bar(
            x=groups,
            y=disparate_impact,
            name='Disparate Impact',
            marker_color=di_colors,
            text=[f'{di:.2f}' for di in disparate_impact],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>DI Ratio: %{y:.2f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )

    # Add 80% rule threshold line
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color=COLORS['warning'],
        annotation_text="80% Threshold",
        row=2, col=1
    )

    # 4. Fairness Metrics Summary (grouped bar)
    demographic_parity = fairness_metrics.get('demographic_parity', [])
    equalized_odds = fairness_metrics.get('equalized_odds', [])

    fig.add_trace(
        go.Bar(
            x=groups,
            y=demographic_parity,
            name='Demographic Parity Δ',
            marker_color=COLORS['secondary'],
            hovertemplate='<b>%{x}</b><br>DP Difference: %{y:.3f}<extra></extra>'
        ),
        row=2, col=2
    )

    fig.add_trace(
        go.Bar(
            x=groups,
            y=equalized_odds,
            name='Equalized Odds Δ',
            marker_color=COLORS['primary'],
            hovertemplate='<b>%{x}</b><br>EO Difference: %{y:.3f}<extra></extra>'
        ),
        row=2, col=2
    )

    # Update axes
    fig.update_xaxes(title_text="Group", row=1, col=1)
    fig.update_yaxes(title_text="Selection Rate", tickformat='.0%', row=1, col=1)

    fig.update_xaxes(title_text="False Positive Rate", tickformat='.0%', row=1, col=2)
    fig.update_yaxes(title_text="True Positive Rate", tickformat='.0%', row=1, col=2)

    fig.update_xaxes(title_text="Group", row=2, col=1)
    fig.update_yaxes(title_text="DI Ratio", row=2, col=1)

    fig.update_xaxes(title_text="Group", row=2, col=2)
    fig.update_yaxes(title_text="Difference", row=2, col=2)

    # Overall layout
    fig.update_layout(
        title={
            'text': f'<b>Fairness Metrics Dashboard - {protected_attribute.title()}</b>',
            'font': {'size': 20, 'color': COLORS['primary']},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        **LAYOUT_CONFIG,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5
        )
    )

    return fig


# ==================== HELPER FUNCTIONS ====================

def get_color_for_score(score: float) -> str:
    """
    Get color for a given credit score.

    Args:
        score: Credit score (300-850)

    Returns:
        Hex color code
    """
    if score >= 750:
        return SCORE_COLORS['excellent']
    elif score >= 700:
        return SCORE_COLORS['good']
    elif score >= 650:
        return SCORE_COLORS['fair']
    elif score >= 600:
        return SCORE_COLORS['poor']
    else:
        return SCORE_COLORS['very_poor']


def get_risk_tier_color(risk_tier: str) -> str:
    """
    Get color for a given risk tier.

    Args:
        risk_tier: Risk tier ('LOW', 'MEDIUM', 'HIGH', 'VERY HIGH')

    Returns:
        Hex color code
    """
    tier_colors = {
        'LOW': COLORS['success'],
        'MEDIUM': COLORS['warning'],
        'HIGH': COLORS['danger'],
        'VERY HIGH': COLORS['danger']
    }
    return tier_colors.get(risk_tier, COLORS['neutral'])


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value * 100:.1f}%"


if __name__ == "__main__":
    """
    Example usage: Generate sample visualizations.
    """
    print("VantageFlow AI - Visualization Functions")
    print("\nAvailable functions:")
    print("  - create_score_gauge(score, risk_tier)")
    print("  - create_income_chart(monthly_data)")
    print("  - create_spending_breakdown(category_data)")
    print("  - create_cashflow_chart(monthly_data)")
    print("  - create_shap_waterfall(shap_values, base_value, prediction)")
    print("  - create_feature_importance_bar(importances)")
    print("  - create_fairness_dashboard(fairness_metrics)")
    print("\nSee docstrings for detailed examples and usage.")
