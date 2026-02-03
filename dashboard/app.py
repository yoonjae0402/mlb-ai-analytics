import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import MetricsCollector, CostTracker

st.set_page_config(
    page_title="MLB Video Pipeline Dashboard",
    page_icon="⚾",
    layout="wide"
)

# Initialize collectors
metrics = MetricsCollector()
costs = CostTracker()

# Sidebar
st.sidebar.title("⚾ MLB Pipeline")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "💰 Cost Analysis", "⚡ Performance", "🎬 Recent Videos"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
days_filter = st.sidebar.slider("Days to show", 1, 30, 7)

# Main content
st.title("MLB Video Pipeline Dashboard")

if page == "📊 Overview":
    st.header("Overview")
    
    # Get recent runs
    recent_runs = metrics.get_recent_runs(days_filter)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Runs",
            len(recent_runs),
            f"Last {days_filter} days"
        )
    
    with col2:
        success_rate = metrics.get_success_rate(days_filter)
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            f"{sum(1 for r in recent_runs if r['success'])} successful"
        )
    
    with col3:
        avg_duration = metrics.get_average_duration(days_filter)
        st.metric(
            "Avg Duration",
            f"{avg_duration:.1f}s",
            "Per video"
        )
    
    with col4:
        # Calculate total cost
        start_date = (datetime.now() - timedelta(days=days_filter)).isoformat()
        end_date = datetime.now().isoformat()
        cost_summary = metrics.get_cost_summary(start_date, end_date)
        st.metric(
            "Total Cost",
            f"${cost_summary['total']:.2f}",
            f"Last {days_filter} days"
        )
    
    # Recent runs table
    st.subheader("Recent Pipeline Runs")
    
    if recent_runs:
        df = pd.DataFrame(recent_runs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['status'] = df['success'].apply(lambda x: '✅ Success' if x else '❌ Failed')
        df['total_cost'] = df['costs'].apply(lambda x: x.get('total', 0))
        
        display_df = df[['timestamp', 'team', 'date', 'status', 'duration', 'total_cost']]
        display_df.columns = ['Timestamp', 'Team', 'Game Date', 'Status', 'Duration (s)', 'Cost ($)']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No pipeline runs found in the selected time period.")

elif page == "💰 Cost Analysis":
    st.header("Cost Analysis")
    
    # Get cost data
    recent_runs = metrics.get_recent_runs(days_filter)
    
    if recent_runs:
        df = pd.DataFrame(recent_runs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date_only'] = df['timestamp'].dt.date
        df['gemini_cost'] = df['costs'].apply(lambda x: x.get('gemini', 0))
        df['audio_cost'] = df['costs'].apply(lambda x: x.get('audio', 0))
        df['total_cost'] = df['costs'].apply(lambda x: x.get('total', 0))
        
        # Daily cost chart
        st.subheader("Daily Cost Breakdown")
        
        daily_costs = df.groupby('date_only').agg({
            'gemini_cost': 'sum',
            'audio_cost': 'sum',
            'total_cost': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily_costs['date_only'],
            y=daily_costs['gemini_cost'],
            name='Gemini (Free)',
            marker_color='#10a37f'
        ))
        fig.add_trace(go.Bar(
            x=daily_costs['date_only'],
            y=daily_costs['audio_cost'],
            name='Audio (Free)',
            marker_color='#6366f1'
        ))
        
        fig.update_layout(
            barmode='stack',
            xaxis_title='Date',
            yaxis_title='Cost ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Gemini (Script)", "$0.00 (FREE)")
        
        with col2:
            st.metric("Audio (TTS)", "$0.00 (FREE)")
        
        with col3:
            st.metric("Grand Total", f"${df['total_cost'].sum():.2f}")
        
        # Cost per video
        st.subheader("Cost Per Video")
        avg_cost = df['total_cost'].mean()
        st.metric("Average Cost Per Video", f"${avg_cost:.4f}")
        
        # Projection
        st.success("🎉 All costs are FREE with Google One AI Premium (Gemini) and local Qwen3-TTS!")
        
    else:
        st.info("No cost data available.")

elif page == "⚡ Performance":
    st.header("Performance Metrics")
    
    recent_runs = metrics.get_recent_runs(days_filter)
    
    if recent_runs:
        df = pd.DataFrame(recent_runs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Duration over time
        st.subheader("Processing Time Trend")
        
        fig = px.line(
            df,
            x='timestamp',
            y='duration',
            markers=True,
            title='Pipeline Duration Over Time'
        )
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Duration (seconds)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Min Duration", f"{df['duration'].min():.1f}s")
        
        with col2:
            st.metric("Avg Duration", f"{df['duration'].mean():.1f}s")
        
        with col3:
            st.metric("Max Duration", f"{df['duration'].max():.1f}s")
        
        # Success/Failure breakdown
        st.subheader("Success vs Failure")
        
        success_counts = df['success'].value_counts()
        fig = px.pie(
            values=success_counts.values,
            names=['Success' if x else 'Failed' for x in success_counts.index],
            color_discrete_sequence=['#10b981', '#ef4444']
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No performance data available.")

elif page == "🎬 Recent Videos":
    st.header("Recent Videos")
    
    recent_runs = metrics.get_recent_runs(days_filter)
    successful_runs = [r for r in recent_runs if r['success']]
    
    if successful_runs:
        st.write(f"Showing {len(successful_runs)} successful video generations")
        
        for run in successful_runs[:10]:  # Show last 10
            with st.expander(f"📹 {run['team']} - {run['date']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Generated:** {run['timestamp']}")
                    st.write(f"**Duration:** {run['duration']:.1f}s")
                    st.write(f"**Cost:** ${run['costs'].get('total', 0):.4f}")
                
                with col2:
                    # Try to find video file
                    video_path = Path(f"outputs/videos/{run['date']}_{run['team']}.mp4")
                    if video_path.exists():
                        st.success("✅ Video file exists")
                        if st.button(f"View {run['team']}", key=run['timestamp']):
                            st.video(str(video_path))
                    else:
                        st.warning("⚠️ Video file not found")
    else:
        st.info("No successful video generations found.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**MLB Video Pipeline Dashboard**

Monitor your automated video generation pipeline.

- Track costs
- View performance
- Review recent videos
""")
