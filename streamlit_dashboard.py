import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import websockets
import json
import time
from datetime import datetime, timedelta
import threading
import queue

# Page configuration
st.set_page_config(
    page_title="Air Quality Streaming Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-connected { background-color: #28a745; }
    .status-disconnected { background-color: #dc3545; }
    .data-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class WebSocketClient:
    def __init__(self):
        self.websocket = None
        self.connected = False
        self.data_queue = queue.Queue()
        self.metrics_history = []
        self.measurements_history = []
        self.last_update = None
        self.connection_thread = None
        
    def connect_sync(self):
        """Synchronous connection method for Streamlit"""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Connect to WebSocket
            self.websocket = loop.run_until_complete(
                websockets.connect('ws://localhost:8765', timeout=10)
            )
            self.connected = True
            
            # Start receiving data in background
            self.start_receiving(loop)
            return True
            
        except Exception as e:
            st.error(f" Connection failed: {str(e)}")
            self.connected = False
            return False
    
    def start_receiving(self, loop):
        """Start receiving data in background"""
        def receive_loop():
            try:
                while self.connected and self.websocket:
                    try:
                        # Receive data with timeout
                        data = asyncio.wait_for(
                            self.websocket.recv(), 
                            timeout=1.0
                        )
                        if data:
                            parsed_data = json.loads(data)
                            self.data_queue.put(parsed_data)
                            self.last_update = datetime.now()
                            
                            # Store measurements
                            if 'measurements' in parsed_data:
                                self.measurements_history.extend(parsed_data['measurements'])
                                # Keep only last 1000 measurements
                                if len(self.measurements_history) > 1000:
                                    self.measurements_history = self.measurements_history[-1000:]
                            
                            # Store metrics
                            if 'model_metrics' in parsed_data and parsed_data['model_metrics']:
                                self.metrics_history.append({
                                    'timestamp': parsed_data['timestamp'],
                                    **parsed_data['model_metrics']
                                })
                                # Keep only last 500 metrics
                                if len(self.metrics_history) > 500:
                                    self.metrics_history = self.metrics_history[-500:]
                                    
                    except asyncio.TimeoutError:
                        continue  # No data, continue
                    except websockets.exceptions.ConnectionClosed:
                        self.connected = False
                        break
                    except Exception as e:
                        print(f"Error receiving data: {e}")
                        break
                        
            except Exception as e:
                print(f"Receive loop error: {e}")
                self.connected = False
        
        # Start receiving in background thread
        self.connection_thread = threading.Thread(target=receive_loop, daemon=True)
        self.connection_thread.start()
    
    def get_latest_data(self):
        """Get latest data from queue"""
        data = []
        while not self.data_queue.empty():
            try:
                data.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return data
    
    def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            try:
                self.websocket.close()
            except:
                pass
        self.connected = False

def create_air_quality_chart(measurements):
    """Create air quality parameter charts"""
    if not measurements:
        return go.Figure()
    
    # Convert to DataFrame
    df = pd.DataFrame(measurements)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create subplots for different parameters
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['PM2.5', 'PM10', 'NO2', 'O3', 'SO2', 'CO'],
        vertical_spacing=0.1
    )
    
    parameters = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, param in enumerate(parameters):
        param_data = df[df['parameter'] == param]
        if not param_data.empty:
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=param_data['timestamp'],
                    y=param_data['value'],
                    mode='lines+markers',
                    name=param.upper(),
                    line=dict(color=colors[i]),
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=600,
        title_text="Real-time Air Quality Parameters",
        showlegend=False
    )
    
    return fig

def create_model_performance_chart(metrics_history):
    """Create model performance visualization"""
    if not metrics_history:
        return go.Figure()
    
    df = pd.DataFrame(metrics_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create subplots for different metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Model Accuracy', 'Memory Usage', 'EdgeFrame Sparsity', 'Data Retention'],
        vertical_spacing=0.15
    )
    
    # Accuracy
    if 'accuracy' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['accuracy'],
                mode='lines+markers',
                name='Accuracy (%)',
                line=dict(color='#1f77b4')
            ),
            row=1, col=1
        )
    
    # Memory Usage
    if 'memory_usage_mb' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['memory_usage_mb'],
                mode='lines+markers',
                name='Memory (MB)',
                line=dict(color='#ff7f0e')
            ),
            row=1, col=2
        )
    
    # Sparsity Ratio
    if 'sparsity_ratio' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['sparsity_ratio'],
                mode='lines+markers',
                name='Sparsity Ratio',
                line=dict(color='#2ca02c')
            ),
            row=2, col=1
        )
    
    # Retained Size
    if 'retained_size' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['retained_size'],
                mode='lines+markers',
                name='Retained Samples',
                line=dict(color='#d62728')
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=600,
        title_text="Model Performance Metrics Over Time",
        showlegend=True
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header"> Air Quality Streaming Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    st.sidebar.markdown("### Connection Status")
    
    # Initialize WebSocket client
    if 'ws_client' not in st.session_state:
        st.session_state.ws_client = WebSocketClient()
    
    # Connection controls
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("🔌 Connect"):
        if not st.session_state.ws_client.connected:
            success = st.session_state.ws_client.connect_sync()
            if success:
                st.success("Connected to streaming server!")
                st.rerun()
    
    if col2.button("Disconnect"):
        if st.session_state.ws_client.connected:
            st.session_state.ws_client.disconnect()
            st.success(" Disconnected from streaming server!")
            st.rerun()
    
    # Status indicator
    status_color = "status-connected" if st.session_state.ws_client.connected else "status-disconnected"
    status_text = "Connected" if st.session_state.ws_client.connected else "Disconnected"
    st.sidebar.markdown(f'<span class="status-indicator {status_color}"></span>{status_text}', unsafe_allow_html=True)
    
    # Connection info
    if st.session_state.ws_client.connected:
        st.sidebar.success(" WebSocket: ws://localhost:8765")
        if st.session_state.ws_client.last_update:
            st.sidebar.info(f" Last Update: {st.session_state.ws_client.last_update.strftime('%H:%M:%S')}")
    
    # Sidebar metrics
    st.sidebar.markdown("### Real-time Metrics")
    
    if st.session_state.ws_client.metrics_history:
        latest_metrics = st.session_state.ws_client.metrics_history[-1]
        
        st.sidebar.metric("Accuracy", f"{latest_metrics.get('accuracy', 0):.2f}%")
        st.sidebar.metric("Memory Usage", f"{latest_metrics.get('memory_usage_mb', 0):.2f} MB")
        st.sidebar.metric("Sparsity Ratio", f"{latest_metrics.get('sparsity_ratio', 0):.3f}")
        st.sidebar.metric("Retained Samples", latest_metrics.get('retained_size', 0))
        st.sidebar.metric("Forest Size", latest_metrics.get('forest_size', 0))
    
    # Main content
    if not st.session_state.ws_client.connected:
        st.warning(" Please connect to the streaming server to view real-time data.")
        st.info(" Make sure the streaming server is running on port 8765")
        
        # Show connection instructions
        st.markdown("""
        ### 🔧 Connection Instructions:
        1. **Start the streaming server**: `python streaming_server.py`
        2. **Click the Connect button** in the sidebar
        3. **Wait for real-time data** to start flowing
        """)
        return
    
    # Get latest data
    latest_data = st.session_state.ws_client.get_latest_data()
    
    # Show live data status
    if latest_data:
        st.success(f"Received {len(latest_data)} new data packets!")
    
    # Create tabs for different visualization
    tab1, tab2, tab3, tab4 = st.tabs([" Air Quality", " Model Performance", "EdgeFrame", " Data Summary"])
    
    with tab1:
        st.header("Real-time Air Quality Monitoring")
        
        if st.session_state.ws_client.measurements_history:
            fig = create_air_quality_chart(st.session_state.ws_client.measurements_history)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display latest measurements
            st.subheader("Latest Measurements")
            latest_measurements = st.session_state.ws_client.measurements_history[-20:]  # Last 20
            df_latest = pd.DataFrame(latest_measurements)
            if not df_latest.empty:
                df_latest['timestamp'] = pd.to_datetime(df_latest['timestamp'])
                st.dataframe(df_latest.sort_values('timestamp', ascending=False))
                
                # Show data statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Measurements", len(st.session_state.ws_client.measurements_history))
                with col2:
                    st.metric("Parameters", df_latest['parameter'].nunique())
                with col3:
                    st.metric("Locations", df_latest['location'].nunique())
        else:
            st.info("⏳ Waiting for air quality data...")
            st.markdown("""
            **No data received yet.** This could mean:
            - The streaming server is not sending data
            - OpenAQ API is not accessible
            - Connection issues between server and dashboard
            """)
    
    with tab2:
        st.header("Model Performance Monitoring")
        
        if st.session_state.ws_client.metrics_history:
            fig = create_model_performance_chart(st.session_state.ws_client.metrics_history)
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance summary
            st.subheader("Performance Summary")
            metrics_df = pd.DataFrame(st.session_state.ws_client.metrics_history)
            if not metrics_df.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Accuracy", f"{metrics_df['accuracy'].mean():.2f}%")
                with col2:
                    st.metric("Peak Memory", f"{metrics_df['memory_usage_mb'].max():.2f} MB")
                with col3:
                    st.metric("Avg Sparsity", f"{metrics_df['sparsity_ratio'].mean():.3f}")
        else:
            st.info("⏳ Waiting for model metrics...")
    
    with tab3:
        st.header("EdgeFrame Structure Analysis")
        
        if st.session_state.ws_client.metrics_history:
            # EdgeFrame statistics
            st.subheader("EdgeFrame Statistics")
            latest_metrics = st.session_state.ws_client.metrics_history[-1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodes", latest_metrics.get('edgeframe_nodes', 0))
            with col2:
                st.metric("Edges", latest_metrics.get('edgeframe_edges', 0))
            with col3:
                st.metric("Sparsity", f"{latest_metrics.get('sparsity_ratio', 0):.3f}")
        else:
            st.info("⏳ Waiting for EdgeFrame metrics...")
    
    with tab4:
        st.header("Data Summary & Analytics")
        
        if st.session_state.ws_client.measurements_history:
            # Data statistics
            df_summary = pd.DataFrame(st.session_state.ws_client.measurements_history)
            df_summary['timestamp'] = pd.to_datetime(df_summary['timestamp'])
            
            st.subheader("Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Measurements", len(df_summary))
                st.metric("Unique Locations", df_summary['location'].nunique())
            
            with col2:
                st.metric("Parameters Monitored", df_summary['parameter'].nunique())
                st.metric("Data Points Today", len(df_summary[df_summary['timestamp'].dt.date == datetime.now().date()]))
            
            with col3:
                st.metric("Start Time", df_summary['timestamp'].min().strftime('%H:%M'))
                st.metric("Latest Update", df_summary['timestamp'].max().strftime('%H:%M'))
            
            # Parameter distribution
            st.subheader("Parameter Distribution")
            param_counts = df_summary['parameter'].value_counts()
            fig = px.bar(x=param_counts.index, y=param_counts.values, 
                        title="Measurements by Parameter")
            st.plotly_chart(fig, use_container_width=True)
            
            # Location distribution
            st.subheader("Location Distribution")
            location_counts = df_summary['location'].value_counts().head(10)
            fig = px.bar(x=location_counts.index, y=location_counts.values,
                        title="Top 10 Locations")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(" Waiting for data to generate summary...")
    
    # Auto-refresh every 3 seconds when connected
    if st.session_state.ws_client.connected:
        time.sleep(3)
        st.rerun()

if __name__ == "__main__":
    main()
