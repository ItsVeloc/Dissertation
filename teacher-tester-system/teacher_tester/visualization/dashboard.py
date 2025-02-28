"""
Interactive visualization dashboard for teacher-tester system.
"""
import os
import argparse
import sys
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from teacher_tester.data.storage import get_storage
from teacher_tester.evaluation.enhanced_metrics import EvaluationMetrics
from teacher_tester.analysis.conversation_analyzer import ConversationAnalyzer
from teacher_tester.utils.logging import setup_logger

logger = setup_logger(__name__)

def create_dashboard_data(output_dir: str = "data/dashboard") -> Dict[str, str]:
    """
    Create the data files needed for the dashboard.
    
    Args:
        output_dir: Directory to save the data files
        
    Returns:
        Dictionary mapping data type to file path
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize storage and get conversations
    storage = get_storage()
    conversation_ids = storage.list_conversations()
    
    if not conversation_ids:
        logger.warning("No conversations found. Run some conversations first.")
        return {}
    
    file_paths = {}
    
    # Create evaluation data
    evaluator = EvaluationMetrics(conversation_ids)
    evaluation_report = evaluator.generate_feedback_report()
    
    evaluation_path = os.path.join(output_dir, "evaluation_data.json")
    with open(evaluation_path, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    file_paths["evaluation"] = evaluation_path
    
    # Load all conversations
    conversations = []
    for conv_id in conversation_ids:
        conv = storage.load_conversation(conv_id)
        if conv is not None:
            conversations.append(conv)
    
    # Create conversation analysis data
    analyzer = ConversationAnalyzer(conversations)
    analysis_report = analyzer.generate_insights_report()
    
    analysis_path = os.path.join(output_dir, "conversation_analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis_report, f, indent=2)
    file_paths["analysis"] = analysis_path
    
    # Create conversation metrics CSV for easy plotting
    metrics_data = []
    for conv in conversations:
        # Basic metrics
        metrics = {
            "conversation_id": conv.id,
            "subject": conv.subject,
            "true_rating": conv.true_rating,
            "predicted_rating": conv.predicted_rating,
            "final_confidence": conv.final_confidence,
            "absolute_error": abs(conv.true_rating - conv.predicted_rating),
            "conversation_length": len([m for m in conv.messages if m.role == "teacher"]),
            "terminated_reason": conv.terminated_reason,
        }
        
        # Extract confidence trajectory
        confidence_values = [msg.confidence for msg in conv.messages 
                           if msg.role == "teacher" and msg.confidence is not None]
        for i, conf in enumerate(confidence_values):
            metrics[f"confidence_step_{i+1}"] = conf
        
        metrics_data.append(metrics)
    
    # Create dataframe and save to CSV
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        metrics_path = os.path.join(output_dir, "conversation_metrics.csv")
        df.to_csv(metrics_path, index=False)
        file_paths["metrics"] = metrics_path
    
    logger.info(f"Created dashboard data files in {output_dir}")
    return file_paths

def create_static_plots(data_dir: str = "data/dashboard", 
                       output_dir: str = "data/visualizations") -> List[str]:
    """
    Create static plots from dashboard data.
    
    Args:
        data_dir: Directory with dashboard data files
        output_dir: Directory to save the visualization files
        
    Returns:
        List of paths to generated visualization files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data files
    metrics_path = os.path.join(data_dir, "conversation_metrics.csv")
    evaluation_path = os.path.join(data_dir, "evaluation_data.json")
    analysis_path = os.path.join(data_dir, "conversation_analysis.json")
    
    # Check if files exist
    files_exist = all(os.path.exists(path) for path in [metrics_path, evaluation_path, analysis_path])
    if not files_exist:
        logger.warning("Missing dashboard data files. Run create_dashboard_data first.")
        return []
    
    plot_paths = []
    
    # Load metrics data
    df = pd.read_csv(metrics_path)
    
    # Create error vs. rating plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="true_rating", y="absolute_error", hue="subject", 
                   size="final_confidence", sizes=(20, 200), alpha=0.7)
    plt.title("Prediction Error vs. True Rating")
    plt.xlabel("True Rating")
    plt.ylabel("Absolute Error")
    plt.grid(True, alpha=0.3)
    
    error_plot_path = os.path.join(output_dir, "error_vs_rating.png")
    plt.savefig(error_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    plot_paths.append(error_plot_path)
    
   # Create confusion matrix-like heatmap
    plt.figure(figsize=(10, 8))
    rating_bins = np.arange(0, 11, 1)
    cm = np.zeros((len(rating_bins)-1, len(rating_bins)-1))
    
    # Check if we have enough data
    if len(df) > 0:
        for _, row in df.iterrows():
            # Handle NaN values
            if pd.isna(row["true_rating"]) or pd.isna(row["predicted_rating"]):
                continue
                
            true_bin = int(min(max(row["true_rating"], 0), 9.99))
            pred_bin = int(min(max(row["predicted_rating"], 0), 9.99))
            cm[true_bin][pred_bin] += 1
        
        # Normalize by row (true rating)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.zeros_like(cm)
        for i in range(cm.shape[0]):
            if row_sums[i] > 0:
                cm_norm[i] = cm[i] / row_sums[i]
        
        # Only show heatmap if we have data
        if np.sum(cm) > 0:
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="viridis",
                       xticklabels=np.arange(0, 10), yticklabels=np.arange(0, 10))
            plt.title("Prediction Distribution Heatmap")
            plt.xlabel("Predicted Rating")
            plt.ylabel("True Rating")
        else:
            plt.text(0.5, 0.5, "Not enough data for heatmap", 
                    horizontalalignment='center', fontsize=14)
    else:
        plt.text(0.5, 0.5, "No data available", 
                horizontalalignment='center', fontsize=14)
    
    # Create confidence trajectory plot
    confidence_cols = [col for col in df.columns if col.startswith("confidence_step_")]
    
    if confidence_cols:
        plt.figure(figsize=(10, 6))
        
        # Group by conversation length
        df["conversation_length"] = df["conversation_length"].astype(int)
        
        for length in sorted(df["conversation_length"].unique()):
            subset = df[df["conversation_length"] == length]
            
            # Get average confidence at each step for this length
            avg_confidence = []
            for i in range(1, length + 1):
                col = f"confidence_step_{i}"
                if col in df.columns:
                    avg_confidence.append(subset[col].mean())
                else:
                    break
            
            if avg_confidence:
                plt.plot(range(1, len(avg_confidence) + 1), avg_confidence, 
                        marker='o', label=f"{length} exchanges")
        
        plt.title("Average Confidence Trajectory by Conversation Length")
        plt.xlabel("Conversation Step")
        plt.ylabel("Average Confidence")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        trajectory_path = os.path.join(output_dir, "confidence_trajectory.png")
        plt.savefig(trajectory_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths.append(trajectory_path)
    
    # Create subject performance comparison
    if "subject" in df.columns and len(df["subject"].unique()) > 1:
        plt.figure(figsize=(12, 6))
        
        # Aggregate by subject
        subject_perf = df.groupby("subject").agg({
            "absolute_error": ["mean", "std"],
            "final_confidence": "mean",
            "conversation_length": "mean",
            "conversation_id": "count"
        }).reset_index()
        
        subject_perf.columns = ["subject", "mean_error", "std_error", 
                               "mean_confidence", "mean_length", "count"]
        
        # Sort by performance
        subject_perf.sort_values("mean_error", inplace=True)
        
        # Create bar plot (without built-in error bars)
        ax = sns.barplot(data=subject_perf, x="subject", y="mean_error", alpha=0.7)
        
        # Manually add error bars if we have multiple subjects
        if len(subject_perf) > 1:
            for i, row in enumerate(subject_perf.itertuples()):
                ax.errorbar(i, row.mean_error, yerr=row.std_error, 
                           fmt='none', color='black', capsize=5)
        
        # Add conversation count as text
        for i, row in enumerate(subject_perf.itertuples()):
            ax.text(i, 0.1, f"n={row.count}", horizontalalignment='center', 
                   size='small', color='black', weight='bold')
        
        plt.title("Error by Subject")
        plt.xlabel("Subject")
        plt.ylabel("Mean Absolute Error")
        plt.grid(True, alpha=0.3, axis='y')
        
        subject_path = os.path.join(output_dir, "subject_performance.png")
        plt.savefig(subject_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths.append(subject_path)
    
    logger.info(f"Created {len(plot_paths)} static visualization plots in {output_dir}")
    return plot_paths

def run_dashboard(data_dir: str = "data/dashboard", port: int = 8050):
    """
    Run an interactive Dash dashboard for the teacher-tester system.
    
    Args:
        data_dir: Directory with dashboard data files
        port: Port to run the dashboard on
    """
    # Check if data files exist
    metrics_path = os.path.join(data_dir, "conversation_metrics.csv")
    evaluation_path = os.path.join(data_dir, "evaluation_data.json")
    analysis_path = os.path.join(data_dir, "conversation_analysis.json")
    
    files_exist = all(os.path.exists(path) for path in [metrics_path, evaluation_path, analysis_path])
    if not files_exist:
        logger.error("Missing dashboard data files. Run create_dashboard_data first.")
        return
    
    # Load data
    df = pd.read_csv(metrics_path)
    
    with open(evaluation_path, 'r') as f:
        evaluation_data = json.load(f)
    
    with open(analysis_path, 'r') as f:
        analysis_data = json.load(f)
    
    # Initialize Dash app
    app = Dash(__name__, title="Teacher-Tester Dashboard")
    
    # Create app layout
    app.layout = html.Div([
        html.H1("Teacher-Tester System Dashboard", 
               style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px 0px'}),
        
        # Tabs for different sections
        dcc.Tabs([
            # Performance Overview Tab
            dcc.Tab(label="Performance Overview", children=[
                html.Div([
                    html.H3("System Performance Metrics", 
                           style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px 0px'}),
                    
                    # Summary metrics cards
                    html.Div([
                        html.Div([
                            html.H4("Total Conversations"),
                            html.P(f"{len(df)}", style={'font-size': '24px', 'font-weight': 'bold'})
                        ], className='metric-card'),
                        
                        html.Div([
                            html.H4("Mean Absolute Error"),
                            html.P(f"{df['absolute_error'].mean():.2f}", 
                                  style={'font-size': '24px', 'font-weight': 'bold'})
                        ], className='metric-card'),
                        
                        html.Div([
                            html.H4("Mean Confidence"),
                            html.P(f"{df['final_confidence'].mean():.2f}", 
                                  style={'font-size': '24px', 'font-weight': 'bold'})
                        ], className='metric-card'),
                        
                        html.Div([
                            html.H4("Subjects Covered"),
                            html.P(f"{df['subject'].nunique()}", 
                                  style={'font-size': '24px', 'font-weight': 'bold'})
                        ], className='metric-card'),
                    ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px 0px'}),
                    
                    # Main scatter plot
                    html.Div([
                        html.H4("Prediction Error vs. True Rating"),
                        dcc.Graph(id='error-vs-rating-plot')
                    ], style={'width': '100%', 'margin': '20px 0px'}),
                    
                    # Two column layout for additional plots
                    html.Div([
                        # Left column
                        html.Div([
                            html.H4("Subject Performance"),
                            dcc.Graph(id='subject-performance-plot')
                        ], style={'width': '48%'}),
                        
                        # Right column
                        html.Div([
                            html.H4("Rating Distribution"),
                            dcc.Graph(id='rating-distribution-plot')
                        ], style={'width': '48%'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0px'}),
                    
                    # Feedback and recommendations
                    html.Div([
                        html.H4("System Assessment and Recommendations"),
                        
                        html.Div([
                            html.H5("Strengths"),
                            html.Ul([html.Li(s) for s in evaluation_data.get("assessment", {}).get("strengths", [])])
                        ], style={'margin': '10px 0px'}),
                        
                        html.Div([
                            html.H5("Weaknesses"),
                            html.Ul([html.Li(w) for w in evaluation_data.get("assessment", {}).get("weaknesses", [])])
                        ], style={'margin': '10px 0px'}),
                        
                        html.Div([
                            html.H5("Recommendations"),
                            html.Ul([html.Li(r) for r in evaluation_data.get("assessment", {}).get("recommendations", [])])
                        ], style={'margin': '10px 0px'})
                    ], style={'margin': '20px 0px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
                    
                ], style={'padding': '20px'})
            ]),
            
            # Conversation Analysis Tab
            dcc.Tab(label="Conversation Analysis", children=[
                html.Div([
                    html.H3("Conversation Flow Analysis", 
                           style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px 0px'}),
                    
                    # Confidence trajectory plot
                    html.Div([
                        html.H4("Confidence Trajectory"),
                        dcc.Graph(id='confidence-trajectory-plot')
                    ], style={'width': '100%', 'margin': '20px 0px'}),
                    
                    # Two column layout
                    html.Div([
                        # Left column - Termination reasons
                        html.Div([
                            html.H4("Conversation Termination Reasons"),
                            dcc.Graph(id='termination-reasons-plot')
                        ], style={'width': '48%'}),
                        
                        # Right column - Accuracy by length
                        html.Div([
                            html.H4("Accuracy by Conversation Length"),
                            dcc.Graph(id='accuracy-by-length-plot')
                        ], style={'width': '48%'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0px'}),
                    
                    # Question analysis
                    html.Div([
                        html.H4("Question Type Analysis"),
                        dcc.Graph(id='question-types-plot')
                    ], style={'width': '100%', 'margin': '20px 0px'}),
                    
                    # Key insights
                    html.Div([
                        html.H4("Key Conversation Insights"),
                        html.Ul([html.Li(insight) for insight in analysis_data.get("insights", [])])
                    ], style={'margin': '20px 0px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
                    
                ], style={'padding': '20px'})
            ]),
            
            # Calibration Analysis Tab
            dcc.Tab(label="Calibration Analysis", children=[
                html.Div([
                    html.H3("Confidence Calibration Analysis", 
                           style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px 0px'}),
                    
                    # Calibration curve
                    html.Div([
                        html.H4("Confidence Calibration Curve"),
                        dcc.Graph(id='calibration-curve-plot')
                    ], style={'width': '100%', 'margin': '20px 0px'}),
                    
                    # Two column layout
                    html.Div([
                        # Left column - Calibration metrics
                        html.Div([
                            html.H4("Calibration Metrics"),
                            html.Table([
                                html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                                html.Tbody([
                                    html.Tr([
                                        html.Td("Mean Calibration Error"),
                                        html.Td(f"{evaluation_data.get('calibration_metrics', {}).get('mean_calibration_error', 0):.3f}")
                                    ]),
                                    html.Tr([
                                        html.Td("Expected Calibration Error"),
                                        html.Td(f"{evaluation_data.get('calibration_metrics', {}).get('expected_calibration_error', 0):.3f}")
                                    ]),
                                    html.Tr([
                                        html.Td("Max Calibration Error"),
                                        html.Td(f"{evaluation_data.get('calibration_metrics', {}).get('max_calibration_error', 0):.3f}")
                                    ]),
                                    html.Tr([
                                        html.Td("Overconfidence Rate"),
                                        html.Td(f"{evaluation_data.get('calibration_metrics', {}).get('overconfidence_rate', 0):.1%}")
                                    ]),
                                    html.Tr([
                                        html.Td("Underconfidence Rate"),
                                        html.Td(f"{evaluation_data.get('calibration_metrics', {}).get('underconfidence_rate', 0):.1%}")
                                    ])
                                ])
                            ], style={'width': '100%', 'borderCollapse': 'collapse'})
                        ], style={'width': '48%'}),
                        
                        # Right column - Confidence vs error scatter
                        html.Div([
                            html.H4("Confidence vs. Error"),
                            dcc.Graph(id='confidence-vs-error-plot')
                        ], style={'width': '48%'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0px'}),
                    
                    # Calibration by subject
                    html.Div([
                        html.H4("Calibration by Subject"),
                        dcc.Graph(id='calibration-by-subject-plot')
                    ], style={'width': '100%', 'margin': '20px 0px'})
                    
                ], style={'padding': '20px'})
            ])
        ])
    ], style={'maxWidth': '1200px', 'margin': '0 auto', 'fontFamily': 'Arial'})
    
    # Callbacks for interactive plots
    @app.callback(
        Output('error-vs-rating-plot', 'figure'),
        Input('error-vs-rating-plot', 'id')
    )
    def update_error_plot(_):
        fig = px.scatter(df, x="true_rating", y="absolute_error", 
                         color="subject", size="final_confidence",
                         hover_data=["conversation_id", "predicted_rating", "conversation_length"],
                         labels={"true_rating": "True Rating", 
                                "absolute_error": "Absolute Error",
                                "final_confidence": "Final Confidence"},
                         title="Prediction Error vs. True Rating")
        
        fig.update_layout(
            xaxis_title="True Rating",
            yaxis_title="Absolute Error",
            legend_title="Subject",
            template="plotly_white"
        )
        
        # Add trend line
        fig.add_trace(
            go.Scatter(
                x=[0, 10],
                y=[df["absolute_error"].mean(), df["absolute_error"].mean()],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name=f"Mean Error: {df['absolute_error'].mean():.2f}"
            )
        )
        
        return fig
    
    @app.callback(
        Output('subject-performance-plot', 'figure'),
        Input('subject-performance-plot', 'id')
    )
    def update_subject_plot(_):
        # Check if we have multiple subjects
        if len(df["subject"].unique()) <= 1:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough subjects for comparison",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Aggregate by subject
        subject_perf = df.groupby("subject").agg({
            "absolute_error": ["mean", "std"],
            "final_confidence": "mean",
            "conversation_id": "count"
        }).reset_index()
        
        subject_perf.columns = ["subject", "mean_error", "std_error", "mean_confidence", "count"]
        
        # Sort by performance
        subject_perf.sort_values("mean_error", inplace=True)
        
        # Create figure
        fig = go.Figure()
        
        # Add bar for mean error
        fig.add_trace(
            go.Bar(
                x=subject_perf["subject"],
                y=subject_perf["mean_error"],
                error_y=dict(
                    type="data",
                    array=subject_perf["std_error"],
                    visible=True
                ),
                name="Mean Error",
                marker_color="indianred"
            )
        )
        
        # Add scatter for mean confidence
        fig.add_trace(
            go.Scatter(
                x=subject_perf["subject"],
                y=subject_perf["mean_confidence"],
                mode="markers",
                name="Mean Confidence",
                marker=dict(
                    size=12,
                    color="royalblue",
                    symbol="circle"
                ),
                yaxis="y2"
            )
        )
        
        # Add text annotations for counts
        for i, row in enumerate(subject_perf.itertuples()):
            fig.add_annotation(
                x=row.subject,
                y=row.mean_error / 2,
                text=f"n={row.count}",
                showarrow=False,
                font=dict(color="white", size=10)
            )
        
        # Update layout
        fig.update_layout(
            title="Performance by Subject",
            xaxis_title="Subject",
            yaxis_title="Mean Absolute Error",
            yaxis2=dict(
                title="Mean Confidence",
                overlaying="y",
                side="right",
                range=[0, 1]
            ),
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @app.callback(
        Output('rating-distribution-plot', 'figure'),
        Input('rating-distribution-plot', 'id')
    )
    def update_rating_plot(_):
        # Create figure
        fig = go.Figure()
        
        # Add histogram for true ratings
        fig.add_trace(
            go.Histogram(
                x=df["true_rating"],
                name="True Ratings",
                opacity=0.7,
                marker_color="forestgreen",
                xbins=dict(start=0, end=10, size=1)
            )
        )
        
        # Add histogram for predicted ratings
        fig.add_trace(
            go.Histogram(
                x=df["predicted_rating"],
                name="Predicted Ratings",
                opacity=0.7,
                marker_color="coral",
                xbins=dict(start=0, end=10, size=1)
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Distribution of Ratings",
            xaxis_title="Rating",
            yaxis_title="Count",
            barmode="overlay",
            template="plotly_white"
        )
        
        return fig
    
    @app.callback(
        Output('confidence-trajectory-plot', 'figure'),
        Input('confidence-trajectory-plot', 'id')
    )
    def update_trajectory_plot(_):
        # Create figure
        fig = go.Figure()
        
        # Get confidence columns
        confidence_cols = [col for col in df.columns if col.startswith("confidence_step_")]
        
        if not confidence_cols:
            fig.add_annotation(
                text="No confidence trajectory data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Calculate average trajectory
        avg_trajectory = []
        steps = []
        
        for i in range(1, len(confidence_cols) + 1):
            col = f"confidence_step_{i}"
            if col in df.columns:
                avg = df[col].mean()
                avg_trajectory.append(avg)
                steps.append(i)
        
        # Add average trajectory
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=avg_trajectory,
                mode="lines+markers",
                name="Average Confidence",
                line=dict(color="royalblue", width=4),
                marker=dict(size=10)
            )
        )
        
        # Group by conversation length
        df["conversation_length"] = df["conversation_length"].astype(int)
        
        for length in sorted(df["conversation_length"].unique()):
            if length < 1:
                continue  # Skip invalid lengths
                
            subset = df[df["conversation_length"] == length]
            
            # Skip if not enough data points
            if len(subset) < 1:
                continue
                
            # Get average confidence at each step for this length
            avg_confidence = []
            for i in range(1, length + 1):
                col = f"confidence_step_{i}"
                if col in df.columns:
                    # Only add if we have valid data
                    if not subset[col].isna().all():
                        avg_confidence.append(subset[col].mean())
                else:
                    break
            
            if avg_confidence and len(avg_confidence) > 0:
                plt.plot(range(1, len(avg_confidence) + 1), avg_confidence, 
                        marker='o', label=f"{length} exchanges")
        
        # Update layout
        fig.update_layout(
            title="Confidence Trajectory",
            xaxis_title="Conversation Step",
            yaxis_title="Average Confidence",
            yaxis_range=[0, 1],
            template="plotly_white",
            hovermode="x unified"
        )
        
        return fig
    
    @app.callback(
        Output('termination-reasons-plot', 'figure'),
        Input('termination-reasons-plot', 'id')
    )
    def update_termination_plot(_):
        # Count termination reasons
        reason_counts = df["terminated_reason"].value_counts().reset_index()
        reason_counts.columns = ["reason", "count"]
        
        # Create figure
        fig = px.pie(
            reason_counts,
            names="reason",
            values="count",
            title="Conversation Termination Reasons",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # Update layout
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template="plotly_white")
        
        return fig
    
    @app.callback(
        Output('accuracy-by-length-plot', 'figure'),
        Input('accuracy-by-length-plot', 'id')
    )
    def update_accuracy_length_plot(_):
        # Group by conversation length
        length_perf = df.groupby("conversation_length").agg({
            "absolute_error": ["mean", "std"],
            "final_confidence": "mean",
            "conversation_id": "count"
        }).reset_index()
        
        length_perf.columns = ["length", "mean_error", "std_error", "mean_confidence", "count"]
        
        # Create figure
        fig = go.Figure()
        
        # Add bar for mean error
        fig.add_trace(
            go.Bar(
                x=length_perf["length"],
                y=length_perf["mean_error"],
                error_y=dict(
                    type="data",
                    array=length_perf["std_error"],
                    visible=True
                ),
                name="Mean Error",
                marker_color="indianred"
            )
        )
        
        # Add scatter for count
        fig.add_trace(
            go.Scatter(
                x=length_perf["length"],
                y=length_perf["count"],
                mode="markers",
                name="Count",
                marker=dict(
                    size=length_perf["count"] * 2,
                    color="darkblue",
                    opacity=0.6
                ),
                yaxis="y2"
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Accuracy by Conversation Length",
            xaxis_title="Conversation Length (exchanges)",
            yaxis_title="Mean Absolute Error",
            yaxis2=dict(
                title="Count",
                overlaying="y",
                side="right"
            ),
            template="plotly_white",
            xaxis=dict(
                tickmode="array",
                tickvals=length_perf["length"]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @app.callback(
        Output('question-types-plot', 'figure'),
        Input('question-types-plot', 'id')
    )
    def update_question_types_plot(_):
        # Get question type data
        question_data = analysis_data.get("question_analysis", {}).get("category_percentages", {})
        
        if not question_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No question analysis data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Create bar chart
        categories = list(question_data.keys())
        percentages = [question_data[cat] * 100 for cat in categories]
        
        fig = go.Figure(
            go.Bar(
                x=categories,
                y=percentages,
                marker_color="lightseagreen"
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Distribution of Question Types",
            xaxis_title="Question Type",
            yaxis_title="Percentage (%)",
            template="plotly_white"
        )
        
        return fig
    
    @app.callback(
        Output('calibration-curve-plot', 'figure'),
        Input('calibration-curve-plot', 'id')
    )
    def update_calibration_curve(_):
        # Create calibration bins
        bins = 10
        bin_size = 1.0 / bins
        
        # Function to calculate normalized error
        def norm_error(row):
            return min(row["absolute_error"] / 10.0, 1.0)
        
        # Add normalized error column
        df["normalized_error"] = df.apply(norm_error, axis=1)
        df["accuracy"] = 1.0 - df["normalized_error"]
        
        # Assign confidence bins
        df["conf_bin"] = (df["final_confidence"] // bin_size) * bin_size + bin_size / 2
        
        # Group by confidence bin
        bin_data = df.groupby("conf_bin").agg({
            "accuracy": "mean",
            "conversation_id": "count"
        }).reset_index()
        
        # Create figure
        fig = go.Figure()
        
        # Add perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="black", dash="dash"),
                name="Perfect Calibration"
            )
        )
        
        # Add calibration points
        fig.add_trace(
            go.Scatter(
                x=bin_data["conf_bin"],
                y=bin_data["accuracy"],
                mode="markers+lines",
                marker=dict(
                    size=bin_data["conversation_id"] / bin_data["conversation_id"].max() * 25 + 5,
                    color="royalblue"
                ),
                name="Actual Calibration",
                text=bin_data["conversation_id"].apply(lambda x: f"n={x}")
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Confidence Calibration Curve",
            xaxis_title="Confidence",
            yaxis_title="Accuracy",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            template="plotly_white"
        )
        
        return fig
    
    @app.callback(
        Output('confidence-vs-error-plot', 'figure'),
        Input('confidence-vs-error-plot', 'id')
    )
    def update_confidence_error_plot(_):
        # Create figure
        fig = px.scatter(
            df,
            x="final_confidence",
            y="absolute_error",
            color="subject",
            size="conversation_length",
            hover_data=["conversation_id", "true_rating", "predicted_rating"],
            labels={
                "final_confidence": "Confidence",
                "absolute_error": "Absolute Error",
                "conversation_length": "Conversation Length"
            }
        )
        
        # Update layout
        fig.update_layout(
            title="Confidence vs. Error",
            xaxis_title="Confidence",
            yaxis_title="Absolute Error",
            template="plotly_white"
        )
        
        return fig
    
    @app.callback(
        Output('calibration-by-subject-plot', 'figure'),
        Input('calibration-by-subject-plot', 'id')
    )
    def update_calibration_subject_plot(_):
        # Check if we have multiple subjects
        if len(df["subject"].unique()) <= 1:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough subjects for comparison",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Calculate calibration by subject
        df["normalized_error"] = df.apply(lambda r: min(r["absolute_error"] / 10.0, 1.0), axis=1)
        df["accuracy"] = 1.0 - df["normalized_error"]
        df["calibration_error"] = abs(df["final_confidence"] - df["accuracy"])
        
        subject_calibration = df.groupby("subject").agg({
            "calibration_error": "mean",
            "final_confidence": "mean",
            "accuracy": "mean",
            "conversation_id": "count"
        }).reset_index()
        
        # Create subplot with shared x-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Mean Calibration Error by Subject", "Confidence vs. Accuracy by Subject")
        )
        
        # Add calibration error bars
        fig.add_trace(
            go.Bar(
                x=subject_calibration["subject"],
                y=subject_calibration["calibration_error"],
                marker_color="coral",
                name="Calibration Error"
            ),
            row=1, col=1
        )
        
        # Add confidence and accuracy comparison
        fig.add_trace(
            go.Bar(
                x=subject_calibration["subject"],
                y=subject_calibration["final_confidence"],
                name="Confidence",
                marker_color="royalblue",
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=subject_calibration["subject"],
                y=subject_calibration["accuracy"],
                name="Accuracy",
                marker_color="forestgreen",
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Calibration Analysis by Subject",
            xaxis2_title="Subject",
            yaxis_title="Mean Calibration Error",
            yaxis2_title="Mean Value",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=700
        )
        
        return fig
    
    # Add CSS
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                .metric-card {
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 15px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    width: 22%;
                }
                
                .metric-card h4 {
                    margin: 0;
                    color: #6c757d;
                    font-size: 16px;
                }
                
                table {
                    border-collapse: collapse;
                    width: 100%;
                }
                
                th, td {
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                
                th {
                    background-color: #f2f2f2;
                }
                
                tr:hover {
                    background-color: #f5f5f5;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Run app
    logger.info(f"Starting dashboard server on port {port}")
    app.run_server(debug=False, port=port)

def main():
    """Run the dashboard functionality with command line arguments."""
    parser = argparse.ArgumentParser(description="Teacher-Tester visualization dashboard.")
    parser.add_argument("--data-dir", type=str, default="data/dashboard",
                      help="Directory for dashboard data files")
    parser.add_argument("--output-dir", type=str, default="data/visualizations",
                      help="Directory for static visualization outputs")
    parser.add_argument("--port", type=int, default=8050,
                      help="Port for the Dash server")
    parser.add_argument("--create-data", action="store_true",
                      help="Create dashboard data files")
    parser.add_argument("--create-plots", action="store_true",
                      help="Create static plots")
    parser.add_argument("--run-dashboard", action="store_true",
                      help="Run the interactive dashboard")
    
    args = parser.parse_args()
    
    # Create data if requested
    if args.create_data:
        create_dashboard_data(args.data_dir)
    
    # Create static plots if requested
    if args.create_plots:
        create_static_plots(args.data_dir, args.output_dir)
    
    # Run dashboard if requested
    if args.run_dashboard:
        run_dashboard(args.data_dir, args.port)
    
    # If no action specified, print help
    if not (args.create_data or args.create_plots or args.run_dashboard):
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())