import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pymysql
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Bird Species Analysis Dashboard",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2E8B57;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 4px solid #2E8B57;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e8 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #2E8B57;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .ecosystem-forest {
        background: linear-gradient(135deg, #228B22 0%, #32CD32 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .ecosystem-grassland {
        background: linear-gradient(135deg, #DAA520 0%, #FFD700 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        font-weight: bold;
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid #17a2b8;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stSelectbox > label {
        font-size: 16px !important;
        font-weight: bold !important;
        color: #2E8B57 !important;
    }
    .stMetric > label {
        font-size: 18px !important;
        font-weight: bold !important;
    }
    .stTab > button {
        font-size: 16px !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# Database connection function
@st.cache_resource
def init_connection():
    """Initialize database connection"""
    
    # Database credentials
    host = "localhost"
    port = "3306"
    database = "david"
    username = "root"
    password = "Melmir@123"
    
    # URL encode password to handle special characters
    encoded_password = quote_plus(password)
    
    try:
        # Create connection
        engine = create_engine(
            f'mysql+pymysql://{username}:{encoded_password}@{host}:{port}/{database}'
        )
        
        # Test connection with proper SQLAlchemy syntax
        with engine.connect() as conn:
            test_query = text("SELECT 1")
            conn.execute(test_query)
        
        return engine
        
    except Exception as e:
        st.error(f"❌ Database Connection Failed: {str(e)}")
        st.markdown("""
        <div class="warning-box">
        <strong>🔧 Troubleshooting Steps:</strong><br>
        1. Ensure MySQL server is running<br>
        2. Verify database 'david' exists<br>
        3. Check if tables were created (run Jupyter notebook first)<br>
        4. Verify credentials: localhost, root, Melmir@123<br>
        5. Install packages: pip install pymysql sqlalchemy
        </div>
        """, unsafe_allow_html=True)
        return None

# Data loading functions with caching
@st.cache_data
def load_observations_data(_engine):
    """Load main observations data from SQL"""
    query = "SELECT * FROM bird_observations"
    return pd.read_sql_query(query, _engine)

@st.cache_data  
def load_species_summary(_engine):
    """Load species summary data from SQL"""
    query = "SELECT * FROM species_summary"
    return pd.read_sql_query(query, _engine)

@st.cache_data
def load_temporal_analysis(_engine):
    """Load temporal analysis data from SQL"""
    query = "SELECT * FROM temporal_analysis"
    return pd.read_sql_query(query, _engine)

@st.cache_data
def load_admin_summary(_engine):
    """Load administrative unit summary from SQL"""
    query = "SELECT * FROM admin_unit_summary"
    return pd.read_sql_query(query, _engine)

@st.cache_data
def load_conservation_priority(_engine):
    """Load conservation priority data from SQL"""
    try:
        query = "SELECT * FROM conservation_priority"
        return pd.read_sql_query(query, _engine)
    except:
        return pd.DataFrame()

# Advanced filtering function
def apply_advanced_filters(data, filters):
    """Apply multiple filters to the data"""
    filtered_data = data.copy()
    
    for column, values in filters.items():
        if values and 'All' not in values:
            if column == 'Temperature_Range':
                if 'Cold (<60°F)' in values:
                    filtered_data = filtered_data[filtered_data['Temperature'] < 60]
                elif 'Moderate (60-75°F)' in values:
                    filtered_data = filtered_data[(filtered_data['Temperature'] >= 60) & (filtered_data['Temperature'] <= 75)]
                elif 'Warm (>75°F)' in values:
                    filtered_data = filtered_data[filtered_data['Temperature'] > 75]
            else:
                filtered_data = filtered_data[filtered_data[column].isin(values)]
    
    return filtered_data

# Visualization functions
def create_ecosystem_comparison(species_summary):
    """Create comprehensive ecosystem comparison"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Species Diversity', 'Total Observations', 'Average Temperature', 'Average Humidity'),
        specs=[[{"type": "bar"}, {"type": "bar"}], 
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Group by ecosystem
    eco_summary = species_summary.groupby('Ecosystem').agg({
        'Scientific_Name': 'nunique',
        'Observation_Count': 'sum',
        'Avg_Temperature': 'mean',
        'Avg_Humidity': 'mean'
    }).reset_index()
    
    colors = ['#2E8B57', '#DAA520']
    
    # Species diversity
    fig.add_trace(
        go.Bar(x=eco_summary['Ecosystem'], y=eco_summary['Scientific_Name'],
               name='Species', marker_color=colors,
               text=eco_summary['Scientific_Name'], textposition='outside'),
        row=1, col=1
    )
    
    # Total observations
    fig.add_trace(
        go.Bar(x=eco_summary['Ecosystem'], y=eco_summary['Observation_Count'],
               name='Observations', marker_color=colors,
               text=eco_summary['Observation_Count'], textposition='outside',
               showlegend=False),
        row=1, col=2
    )
    
    # Average temperature
    fig.add_trace(
        go.Bar(x=eco_summary['Ecosystem'], y=eco_summary['Avg_Temperature'],
               name='Temperature', marker_color=colors,
               text=[f"{temp:.1f}°F" for temp in eco_summary['Avg_Temperature']], 
               textposition='outside', showlegend=False),
        row=2, col=1
    )
    
    # Average humidity
    fig.add_trace(
        go.Bar(x=eco_summary['Ecosystem'], y=eco_summary['Avg_Humidity'],
               name='Humidity', marker_color=colors,
               text=[f"{hum:.1f}%" for hum in eco_summary['Avg_Humidity']], 
               textposition='outside', showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Comprehensive Ecosystem Comparison",
        showlegend=False,
        height=600,
        title_font_size=20
    )
    
    return fig

def create_species_diversity_sunburst(species_summary):
    """Create sunburst chart for species diversity"""
    # Prepare data for sunburst
    sunburst_data = []
    
    for ecosystem in species_summary['Ecosystem'].unique():
        eco_data = species_summary[species_summary['Ecosystem'] == ecosystem]
        total_obs = eco_data['Observation_Count'].sum()
        
        # Add ecosystem level
        sunburst_data.append({
            'ids': ecosystem,
            'labels': ecosystem,
            'parents': '',
            'values': total_obs
        })
        
        # Add top 10 species for each ecosystem
        top_species = eco_data.nlargest(10, 'Observation_Count')
        for _, species in top_species.iterrows():
            sunburst_data.append({
                'ids': f"{ecosystem}_{species['Common_Name']}",
                'labels': species['Common_Name'],
                'parents': ecosystem,
                'values': species['Observation_Count']
            })
    
    df_sunburst = pd.DataFrame(sunburst_data)
    
    fig = go.Figure(go.Sunburst(
        ids=df_sunburst['ids'],
        labels=df_sunburst['labels'],
        parents=df_sunburst['parents'],
        values=df_sunburst['values'],
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Observations: %{value}<extra></extra>',
        maxdepth=2
    ))
    
    fig.update_layout(
        title="Species Diversity Breakdown by Ecosystem",
        title_font_size=18,
        height=600
    )
    
    return fig

def create_temporal_heatmap(temporal_data):
    """Create temporal heatmap"""
    # Pivot data for heatmap
    heatmap_data = temporal_data.pivot_table(
        values='Observation_Count', 
        index='Month', 
        columns='Ecosystem', 
        aggfunc='sum'
    ).fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=[f"Month {int(m)}" for m in heatmap_data.index],
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='<b>%{x}</b><br>%{y}<br>Observations: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Monthly Observation Patterns by Ecosystem',
        title_font_size=18,
        height=500
    )
    
    return fig

def create_conservation_donut(observations):
    """Create donut chart for conservation status"""
    conservation_data = observations.groupby(['Ecosystem', 'PIF_Watchlist_Status']).size().reset_index(name='Count')
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=['Forest Conservation Status', 'Grassland Conservation Status']
    )
    
    ecosystems = ['Forest', 'Grassland']
    colors = [['#90EE90', '#DC143C'], ['#98FB98', '#FF6347']]
    
    for i, ecosystem in enumerate(ecosystems):
        eco_data = conservation_data[conservation_data['Ecosystem'] == ecosystem]
        
        fig.add_trace(
            go.Pie(
                labels=['Not on Watchlist', 'On Watchlist'],
                values=eco_data['Count'],
                hole=0.4,
                marker_colors=colors[i],
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title_text="Conservation Status Distribution",
        title_font_size=18,
        height=500
    )
    
    return fig

def create_environmental_3d_scatter(observations):
    """Create 3D scatter plot for environmental analysis"""
    # Sample data for performance
    if len(observations) > 2000:
        sample_data = observations.sample(2000)
    else:
        sample_data = observations
    
    sample_data = sample_data.dropna(subset=['Temperature', 'Humidity', 'Scientific_Name'])
    
    fig = px.scatter_3d(
        sample_data,
        x='Temperature',
        y='Humidity', 
        z='Month',
        color='Ecosystem',
        size='Year',
        hover_data=['Common_Name', 'Season'],
        title='3D Environmental Analysis: Temperature, Humidity, and Temporal Patterns',
        labels={'Month': 'Month of Year'},
        color_discrete_map={'Forest': '#2E8B57', 'Grassland': '#DAA520'}
    )
    
    fig.update_layout(height=600)
    return fig

def create_species_abundance_treemap(species_summary, ecosystem=None):
    """Create treemap for species abundance"""
    if ecosystem and ecosystem != "All":
        data = species_summary[species_summary['Ecosystem'] == ecosystem].head(20)
        title = f"Top 20 Species Abundance - {ecosystem}"
    else:
        data = species_summary.nlargest(30, 'Observation_Count')
        title = "Top 30 Species Abundance - All Ecosystems"
    
    fig = px.treemap(
        data,
        path=['Ecosystem', 'Common_Name'],
        values='Observation_Count',
        title=title,
        color='Avg_Temperature',
        color_continuous_scale='RdYlBu_r',
        hover_data=['Scientific_Name', 'Avg_Humidity']
    )
    
    fig.update_layout(height=600)
    return fig

# Main application
def main():
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">🦅 Bird Species Observation Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('''
    <div style="text-align: center; font-size: 1.3rem; color: #666; margin-bottom: 2rem;">
        <strong>Comprehensive Forest vs Grassland Ecosystem Analysis</strong><br>
        <em>National Park Service Bird Monitoring Program Data</em>
    </div>
    ''', unsafe_allow_html=True)
    
    # Initialize database connection
    engine = init_connection()
    
    if engine is None:
        st.stop()
    
    # Display connection status
    st.sidebar.markdown('''
    <div class="success-box">
        ✅ <strong>Database Connected</strong><br>
        📊 MySQL Database: david<br>
        🔗 Host: localhost
    </div>
    ''', unsafe_allow_html=True)
    
    # Load data from SQL
    with st.spinner("🔄 Loading comprehensive data from database..."):
        try:
            observations = load_observations_data(engine)
            species_summary = load_species_summary(engine)
            temporal_analysis = load_temporal_analysis(engine)
            admin_summary = load_admin_summary(engine)
            conservation_priority = load_conservation_priority(engine)
            
            st.sidebar.success(f"✅ Loaded {len(observations):,} observations successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("Ensure you've run the Jupyter notebook to create all required tables.")
            st.stop()
    
    # Enhanced Sidebar Filters
    st.sidebar.header("🔍 Advanced Filters & Controls")
    
    # Primary filters
    st.sidebar.subheader("🌍 Primary Filters")
    ecosystem_filter = st.sidebar.selectbox(
        "🏞️ Select Ecosystem",
        ["All", "Forest", "Grassland"],
        help="Filter data by ecosystem type"
    )
    
    years = sorted(observations['Year'].dropna().unique())
    year_filter = st.sidebar.selectbox(
        "📅 Select Year", 
        ["All"] + [str(int(year)) for year in years],
        help="Filter by observation year"
    )
    
    seasons = observations['Season'].dropna().unique()
    season_filter = st.sidebar.selectbox(
        "🌤️ Select Season",
        ["All"] + list(seasons),
        help="Filter by season"
    )
    
    # Advanced filters
    st.sidebar.subheader("🔬 Advanced Filters")
    
    # Administrative unit filter
    admin_units = observations['Admin_Unit_Code'].dropna().unique()
    admin_filter = st.sidebar.multiselect(
        "🏛️ Administrative Units",
        options=admin_units,
        default=[],
        help="Select specific administrative units"
    )
    
    # Species filter
    species_list = observations['Common_Name'].dropna().unique()
    species_filter = st.sidebar.multiselect(
        "🐦 Specific Species",
        options=sorted(species_list)[:50],  # Show top 50 for performance
        default=[],
        help="Select specific species to analyze"
    )
    
    # Environmental filters
    st.sidebar.subheader("🌡️ Environmental Filters")
    
    temp_range = st.sidebar.slider(
        "Temperature Range (°F)",
        min_value=float(observations['Temperature'].min()),
        max_value=float(observations['Temperature'].max()),
        value=(float(observations['Temperature'].min()), float(observations['Temperature'].max())),
        help="Filter by temperature range"
    )
    
    humidity_range = st.sidebar.slider(
        "Humidity Range (%)",
        min_value=float(observations['Humidity'].min()),
        max_value=float(observations['Humidity'].max()),
        value=(float(observations['Humidity'].min()), float(observations['Humidity'].max())),
        help="Filter by humidity range"
    )
    
    # Apply filters
    filtered_obs = observations.copy()
    
    if ecosystem_filter != "All":
        filtered_obs = filtered_obs[filtered_obs['Ecosystem'] == ecosystem_filter]
    
    if year_filter != "All":
        filtered_obs = filtered_obs[filtered_obs['Year'] == int(year_filter)]
    
    if season_filter != "All":
        filtered_obs = filtered_obs[filtered_obs['Season'] == season_filter]
    
    if admin_filter:
        filtered_obs = filtered_obs[filtered_obs['Admin_Unit_Code'].isin(admin_filter)]
    
    if species_filter:
        filtered_obs = filtered_obs[filtered_obs['Common_Name'].isin(species_filter)]
    
    # Apply environmental filters
    filtered_obs = filtered_obs[
        (filtered_obs['Temperature'] >= temp_range[0]) & 
        (filtered_obs['Temperature'] <= temp_range[1]) &
        (filtered_obs['Humidity'] >= humidity_range[0]) & 
        (filtered_obs['Humidity'] <= humidity_range[1])
    ]
    
    # Enhanced Overview Metrics
    st.header("📊 Comprehensive Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🔢 Total Observations", 
            f"{len(filtered_obs):,}",
            delta=f"{len(filtered_obs) - len(observations):,}" if len(filtered_obs) != len(observations) else None
        )
    
    with col2:
        unique_species = filtered_obs['Scientific_Name'].nunique()
        st.metric(
            "🐦 Unique Species", 
            unique_species,
            delta=f"{unique_species - observations['Scientific_Name'].nunique()}" if len(filtered_obs) != len(observations) else None
        )
    
    with col3:
        forest_count = len(filtered_obs[filtered_obs['Ecosystem'] == 'Forest'])
        st.metric("🌳 Forest Observations", f"{forest_count:,}")
    
    with col4:
        grassland_count = len(filtered_obs[filtered_obs['Ecosystem'] == 'Grassland'])
        st.metric("🌾 Grassland Observations", f"{grassland_count:,}")
    
    with col5:
        avg_temp = filtered_obs['Temperature'].mean()
        st.metric("🌡️ Avg Temperature", f"{avg_temp:.1f}°F")
    
    # Additional metrics row
    col6, col7, col8, col9, col10 = st.columns(5)
    
    with col6:
        avg_humidity = filtered_obs['Humidity'].mean()
        st.metric("💧 Avg Humidity", f"{avg_humidity:.1f}%")
    
    with col7:
        admin_units_count = filtered_obs['Admin_Unit_Code'].nunique()
        st.metric("🏛️ Admin Units", admin_units_count)
    
    with col8:
        watchlist_count = (filtered_obs['PIF_Watchlist_Status'] == True).sum()
        st.metric("⚠️ Watchlist Species", f"{watchlist_count:,}")
    
    with col9:
        date_range = filtered_obs['Year'].max() - filtered_obs['Year'].min() + 1 if len(filtered_obs) > 0 else 0
        st.metric("📅 Year Span", f"{date_range} years")
    
    with col10:
        diversity_index = unique_species / len(filtered_obs) * 1000 if len(filtered_obs) > 0 else 0
        st.metric("📈 Diversity Index", f"{diversity_index:.2f}")
    
    # Main content tabs with enhanced features
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🌳 Ecosystem Analysis", 
        "🦅 Species Deep Dive", 
        "📅 Temporal Patterns", 
        "🌡️ Environmental Analysis",
        "🔍 Conservation Status",
        "🏛️ Geographic Analysis",
        "📊 Advanced Analytics"
    ])
    
    with tab1:
        st.header("🌳 Comprehensive Ecosystem Analysis")
        
        # Filter species summary based on current filters
        filtered_species = species_summary.copy()
        if ecosystem_filter != "All":
            filtered_species = filtered_species[filtered_species['Ecosystem'] == ecosystem_filter]
        
        if len(filtered_species) > 0:
            # Main comparison chart
            ecosystem_fig = create_ecosystem_comparison(filtered_species)
            st.plotly_chart(ecosystem_fig, use_container_width=True)
            
            # Sunburst chart for diversity
            col1, col2 = st.columns(2)
            
            with col1:
                sunburst_fig = create_species_diversity_sunburst(filtered_species)
                st.plotly_chart(sunburst_fig, use_container_width=True)
            
            with col2:
                # Ecosystem statistics
                st.subheader("📈 Ecosystem Statistics")
                
                for ecosystem in ['Forest', 'Grassland']:
                    if ecosystem_filter == "All" or ecosystem_filter == ecosystem:
                        eco_data = filtered_species[filtered_species['Ecosystem'] == ecosystem]
                        if len(eco_data) > 0:
                            if ecosystem == 'Forest':
                                st.markdown('<div class="ecosystem-forest">🌳 Forest Ecosystem</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="ecosystem-grassland">🌾 Grassland Ecosystem</div>', unsafe_allow_html=True)
                            
                            st.write(f"• **Species Count:** {eco_data['Scientific_Name'].nunique()}")
                            st.write(f"• **Total Observations:** {eco_data['Observation_Count'].sum():,}")
                            st.write(f"• **Average Temperature:** {eco_data['Avg_Temperature'].mean():.1f}°F")
                            st.write(f"• **Average Humidity:** {eco_data['Avg_Humidity'].mean():.1f}%")
                            st.write(f"• **Top Species:** {eco_data.loc[eco_data['Observation_Count'].idxmax(), 'Common_Name']}")
                            st.write("---")
    
    with tab2:
        st.header("🦅 Species Deep Dive Analysis")
        
        # Species abundance treemap
        treemap_fig = create_species_abundance_treemap(species_summary, ecosystem_filter)
        st.plotly_chart(treemap_fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Species by Ecosystem")
            
            # Top species charts
            if ecosystem_filter in ["All", "Forest"]:
                forest_species = species_summary[species_summary['Ecosystem'] == 'Forest'].head(15)
                fig = px.bar(
                    forest_species,
                    x='Observation_Count',
                    y='Common_Name',
                    orientation='h',
                    title='Top 15 Forest Species',
                    color='Avg_Temperature',
                    color_continuous_scale='Greens',
                    hover_data=['Scientific_Name', 'Avg_Humidity']
                )
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if ecosystem_filter in ["All", "Grassland"]:
                grassland_species = species_summary[species_summary['Ecosystem'] == 'Grassland'].head(15)
                fig = px.bar(
                    grassland_species,
                    x='Observation_Count',
                    y='Common_Name',
                    orientation='h',
                    title='Top 15 Grassland Species',
                    color='Avg_Temperature',
                    color_continuous_scale='YlOrBr',
                    hover_data=['Scientific_Name', 'Avg_Humidity']
                )
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed species table
        st.subheader("📋 Detailed Species Analysis")
        
        # Add search functionality
        search_species = st.text_input("🔍 Search for specific species:", placeholder="Enter species name...")
        
        display_species = species_summary.copy()
        if ecosystem_filter != "All":
            display_species = display_species[display_species['Ecosystem'] == ecosystem_filter]
        
        if search_species:
            display_species = display_species[
                display_species['Common_Name'].str.contains(search_species, case=False, na=False) |
                display_species['Scientific_Name'].str.contains(search_species, case=False, na=False)
            ]
        
        # Add sorting options
        sort_by = st.selectbox(
            "Sort by:",
            ["Observation_Count", "Avg_Temperature", "Avg_Humidity", "Common_Name"],
            index=0
        )
        
        sort_order = st.radio("Sort order:", ["Descending", "Ascending"], horizontal=True)
        ascending = sort_order == "Ascending"
        
        display_species = display_species.sort_values(sort_by, ascending=ascending)
        
        st.dataframe(
            display_species[['Common_Name', 'Scientific_Name', 'Ecosystem', 'Observation_Count', 
                           'Avg_Temperature', 'Avg_Humidity', 'PIF_Watchlist_Status']].round(2),
            use_container_width=True,
            height=400
        )
    
    with tab3:
        st.header("📅 Temporal Patterns Analysis")
        
        # Temporal heatmap
        if len(temporal_analysis) > 0:
            heatmap_fig = create_temporal_heatmap(temporal_analysis)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Multiple temporal charts
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Monthly Observations', 'Seasonal Distribution', 
                              'Yearly Trends', 'Species Diversity Over Time'),
                specs=[[{"type": "scatter"}, {"type": "pie"}], 
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Process temporal data
            monthly_data = temporal_analysis.groupby(['Month', 'Ecosystem'])['Observation_Count'].sum().reset_index()
            seasonal_data = temporal_analysis.groupby('Season')['Observation_Count'].sum().reset_index()
            yearly_data = temporal_analysis.groupby(['Year', 'Ecosystem'])['Observation_Count'].sum().reset_index()
            diversity_data = temporal_analysis.groupby(['Year', 'Ecosystem'])['Species_Count'].mean().reset_index()
            
            # Monthly observations
            for ecosystem in monthly_data['Ecosystem'].unique():
                eco_data = monthly_data[monthly_data['Ecosystem'] == ecosystem]
                color = '#2E8B57' if ecosystem == 'Forest' else '#DAA520'
                fig.add_trace(
                    go.Scatter(x=eco_data['Month'], y=eco_data['Observation_Count'],
                             mode='lines+markers', name=f'{ecosystem} Monthly',
                             line=dict(color=color, width=3), marker=dict(size=8)),
                    row=1, col=1
                )
            
            # Seasonal pie chart
            fig.add_trace(
                go.Pie(labels=seasonal_data['Season'], values=seasonal_data['Observation_Count'],
                       name="Seasonal Distribution"),
                row=1, col=2
            )
            
            # Yearly trends
            for ecosystem in yearly_data['Ecosystem'].unique():
                eco_data = yearly_data[yearly_data['Ecosystem'] == ecosystem]
                color = '#2E8B57' if ecosystem == 'Forest' else '#DAA520'
                fig.add_trace(
                    go.Scatter(x=eco_data['Year'], y=eco_data['Observation_Count'],
                             mode='lines+markers', name=f'{ecosystem} Yearly',
                             line=dict(color=color, width=3), marker=dict(size=8),
                             showlegend=False),
                    row=2, col=1
                )
            
            # Species diversity over time
            for ecosystem in diversity_data['Ecosystem'].unique():
                eco_data = diversity_data[diversity_data['Ecosystem'] == ecosystem]
                color = '#2E8B57' if ecosystem == 'Forest' else '#DAA520'
                fig.add_trace(
                    go.Scatter(x=eco_data['Year'], y=eco_data['Species_Count'],
                             mode='lines+markers', name=f'{ecosystem} Diversity',
                             line=dict(color=color, width=3, dash='dash'), 
                             marker=dict(size=8), showlegend=False),
                    row=2, col=2
                )
            
            fig.update_layout(title_text="Comprehensive Temporal Analysis", height=700)
            st.plotly_chart(fig, use_container_width=True)
            
            # Temporal insights
            st.subheader("🔍 Temporal Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                peak_month = monthly_data.groupby('Month')['Observation_Count'].sum().idxmax()
                st.metric("📈 Peak Observation Month", f"Month {int(peak_month)}")
            
            with col2:
                peak_season = seasonal_data.loc[seasonal_data['Observation_Count'].idxmax(), 'Season']
                st.metric("🌤️ Peak Season", peak_season)
            
            with col3:
                year_growth = yearly_data.groupby('Year')['Observation_Count'].sum().pct_change().mean() * 100
                st.metric("📊 Avg Yearly Growth", f"{year_growth:.1f}%")
    
    with tab4:
        st.header("🌡️ Environmental Analysis")
        
        # 3D environmental scatter plot
        env_3d_fig = create_environmental_3d_scatter(filtered_obs)
        st.plotly_chart(env_3d_fig, use_container_width=True)
        
        # Environmental distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature distribution with overlays
            fig = go.Figure()
            
            for ecosystem in filtered_obs['Ecosystem'].unique():
                eco_data = filtered_obs[filtered_obs['Ecosystem'] == ecosystem]['Temperature'].dropna()
                color = '#2E8B57' if ecosystem == 'Forest' else '#DAA520'
                
                fig.add_trace(go.Histogram(
                    x=eco_data,
                    name=ecosystem,
                    opacity=0.7,
                    nbinsx=30,
                    marker_color=color
                ))
            
            fig.update_layout(
                title='Temperature Distribution by Ecosystem',
                xaxis_title='Temperature (°F)',
                yaxis_title='Frequency',
                barmode='overlay',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Humidity distribution
            fig = go.Figure()
            
            for ecosystem in filtered_obs['Ecosystem'].unique():
                eco_data = filtered_obs[filtered_obs['Ecosystem'] == ecosystem]['Humidity'].dropna()
                color = '#2E8B57' if ecosystem == 'Forest' else '#DAA520'
                
                fig.add_trace(go.Histogram(
                    x=eco_data,
                    name=ecosystem,
                    opacity=0.7,
                    nbinsx=30,
                    marker_color=color
                ))
            
            fig.update_layout(
                title='Humidity Distribution by Ecosystem',
                xaxis_title='Humidity (%)',
                yaxis_title='Frequency',
                barmode='overlay',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Environmental correlations
        st.subheader("🔗 Environmental Correlations")
        
        # Create correlation matrix
        env_data = filtered_obs[['Temperature', 'Humidity', 'Month', 'Year']].dropna()
        if len(env_data) > 0:
            correlation_matrix = env_data.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(3),
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Environmental Variable Correlations',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Environmental statistics by ecosystem
        st.subheader("📊 Environmental Statistics")
        
        env_stats = filtered_obs.groupby('Ecosystem')[['Temperature', 'Humidity']].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        
        st.dataframe(env_stats, use_container_width=True)
    
    with tab5:
        st.header("🔍 Conservation Status Analysis")
        
        # Conservation donut charts
        conservation_fig = create_conservation_donut(filtered_obs)
        st.plotly_chart(conservation_fig, use_container_width=True)
        
        # Conservation priority species
        if len(conservation_priority) > 0:
            st.subheader("⚠️ Priority Species for Conservation")
            
            # Enhanced conservation table
            priority_display = conservation_priority.copy()
            priority_display['Risk_Level'] = priority_display.apply(
                lambda row: 'High' if row['PIF_Watchlist_Status'] and row['Regional_Stewardship_Status'] 
                else 'Medium' if row['PIF_Watchlist_Status'] or row['Regional_Stewardship_Status'] 
                else 'Low', axis=1
            )
            
            st.dataframe(
                priority_display[['Common_Name', 'Scientific_Name', 'Ecosystems', 'Total_Observations',
                               'Risk_Level', 'PIF_Watchlist_Status', 'Regional_Stewardship_Status']],
                use_container_width=True
            )
        
        # Conservation summary metrics
        st.subheader("📈 Conservation Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            watchlist_species = filtered_obs[filtered_obs['PIF_Watchlist_Status'] == True]['Scientific_Name'].nunique()
            st.metric("🚨 Watchlist Species", watchlist_species)
        
        with col2:
            stewardship_species = filtered_obs[filtered_obs['Regional_Stewardship_Status'] == True]['Scientific_Name'].nunique()
            st.metric("🛡️ Stewardship Species", stewardship_species)
        
        with col3:
            watchlist_obs = (filtered_obs['PIF_Watchlist_Status'] == True).sum()
            st.metric("📊 Watchlist Observations", f"{watchlist_obs:,}")
        
        with col4:
            conservation_percentage = (watchlist_obs / len(filtered_obs) * 100) if len(filtered_obs) > 0 else 0
            st.metric("📈 Conservation %", f"{conservation_percentage:.1f}%")
        
        # Conservation trends over time
        if len(filtered_obs) > 0:
            conservation_trends = filtered_obs.groupby(['Year', 'Ecosystem']).agg({
                'PIF_Watchlist_Status': lambda x: (x == True).sum(),
                'Regional_Stewardship_Status': lambda x: (x == True).sum(),
                'Scientific_Name': 'count'
            }).reset_index()
            
            fig = px.line(
                conservation_trends,
                x='Year',
                y='PIF_Watchlist_Status',
                color='Ecosystem',
                title='Conservation Species Trends Over Time',
                labels={'PIF_Watchlist_Status': 'Watchlist Observations'},
                color_discrete_map={'Forest': '#2E8B57', 'Grassland': '#DAA520'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.header("🏛️ Geographic Analysis")
        
        # Administrative unit analysis
        if len(admin_summary) > 0:
            # Admin unit performance
            fig = px.scatter(
                admin_summary,
                x='Observation_Count',
                y='Species_Count',
                color='Ecosystem',
                size='Avg_Temperature',
                hover_data=['Admin_Unit_Code', 'Avg_Humidity'],
                title='Administrative Unit Performance: Observations vs Species Diversity',
                color_discrete_map={'Forest': '#2E8B57', 'Grassland': '#DAA520'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Top performing admin units
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top Admin Units by Species Diversity")
                top_diversity = admin_summary.nlargest(10, 'Species_Count')[['Admin_Unit_Code', 'Ecosystem', 'Species_Count', 'Observation_Count']]
                st.dataframe(top_diversity, use_container_width=True)
            
            with col2:
                st.subheader("📊 Top Admin Units by Observations")
                top_observations = admin_summary.nlargest(10, 'Observation_Count')[['Admin_Unit_Code', 'Ecosystem', 'Observation_Count', 'Species_Count']]
                st.dataframe(top_observations, use_container_width=True)
            
            # Geographic distribution
            fig = px.bar(
                admin_summary,
                x='Admin_Unit_Code',
                y='Observation_Count',
                color='Ecosystem',
                title='Observation Distribution Across Administrative Units',
                color_discrete_map={'Forest': '#2E8B57', 'Grassland': '#DAA520'}
            )
            
            fig.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab7:
        st.header("📊 Advanced Analytics & Insights")
        
        # Advanced metrics and calculations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧮 Statistical Analysis")
            
            # Calculate diversity indices
            if len(filtered_obs) > 0:
                # Shannon diversity index calculation
                species_counts = filtered_obs['Scientific_Name'].value_counts()
                total_observations = len(filtered_obs)
                shannon_diversity = -sum((count/total_observations) * np.log(count/total_observations) 
                                       for count in species_counts)
                
                # Simpson's diversity index
                simpson_diversity = 1 - sum((count/total_observations)**2 for count in species_counts)
                
                # Evenness index
                num_species = len(species_counts)
                evenness = shannon_diversity / np.log(num_species) if num_species > 1 else 0
                
                st.metric("🧬 Shannon Diversity Index", f"{shannon_diversity:.3f}")
                st.metric("🎯 Simpson's Diversity Index", f"{simpson_diversity:.3f}")
                st.metric("⚖️ Evenness Index", f"{evenness:.3f}")
                st.metric("🔢 Total Species Richness", num_species)
        
        with col2:
            st.subheader("📈 Trend Analysis")
            
            if len(filtered_obs) > 0:
                # Calculate observation trends
                yearly_observations = filtered_obs.groupby('Year').size()
                if len(yearly_observations) > 1:
                    trend_slope = np.polyfit(yearly_observations.index, yearly_observations.values, 1)[0]
                    trend_direction = "📈 Increasing" if trend_slope > 0 else "📉 Decreasing"
                    st.metric("📊 Observation Trend", trend_direction)
                    st.metric("📈 Annual Change Rate", f"{trend_slope:.1f} obs/year")
                
                # Peak activity analysis
                monthly_activity = filtered_obs.groupby('Month').size()
                peak_month = monthly_activity.idxmax()
                peak_activity = monthly_activity.max()
                
                st.metric("🏃 Peak Activity Month", f"Month {int(peak_month)}")
                st.metric("🎯 Peak Activity Count", f"{peak_activity:,}")
        
        # Advanced visualizations
        st.subheader("🔬 Advanced Visualizations")
        
        # Species accumulation curve
        if len(filtered_obs) > 100:  # Only show if enough data
            sample_sizes = range(50, min(len(filtered_obs), 1000), 50)
            species_accumulation = []
            
            for size in sample_sizes:
                sample_data = filtered_obs.sample(size)
                species_count = sample_data['Scientific_Name'].nunique()
                species_accumulation.append({'Sample_Size': size, 'Species_Count': species_count})
            
            accumulation_df = pd.DataFrame(species_accumulation)
            
            fig = px.line(
                accumulation_df,
                x='Sample_Size',
                y='Species_Count',
                title='Species Accumulation Curve',
                labels={'Sample_Size': 'Sample Size', 'Species_Count': 'Cumulative Species Count'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Observation intensity analysis
        st.subheader("🔥 Observation Intensity Analysis")
        
        # Calculate observations per species
        species_intensity = filtered_obs['Scientific_Name'].value_counts().reset_index()
        species_intensity.columns = ['Species', 'Observation_Count']
        species_intensity['Log_Count'] = np.log10(species_intensity['Observation_Count'])
        
        fig = px.histogram(
            species_intensity,
            x='Log_Count',
            title='Distribution of Observation Intensity (Log Scale)',
            labels={'Log_Count': 'Log10(Observation Count)', 'count': 'Number of Species'},
            nbins=20
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data quality assessment
        st.subheader("✅ Data Quality Assessment")
        
        quality_metrics = {
            'Total Records': len(filtered_obs),
            'Complete Temperature Records': filtered_obs['Temperature'].notna().sum(),
            'Complete Humidity Records': filtered_obs['Humidity'].notna().sum(),
            'Complete Species IDs': filtered_obs['Scientific_Name'].notna().sum(),
            'Complete Date Records': filtered_obs['Date'].notna().sum(),
            'Data Completeness': f"{(filtered_obs.notna().sum().sum() / (len(filtered_obs) * len(filtered_obs.columns))) * 100:.1f}%"
        }
        
        quality_df = pd.DataFrame(list(quality_metrics.items()), columns=['Metric', 'Value'])
        st.dataframe(quality_df, use_container_width=True)
    
    # Enhanced Footer with additional information
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 1rem; padding: 2rem;'>
        <h4>🦅 Bird Species Observation Analysis Dashboard</h4>
        <strong>Database:</strong> MySQL 'david' (localhost) | 
        <strong>Total Records:</strong> {len(observations):,} | 
        <strong>Filtered Records:</strong> {len(filtered_obs):,}<br>
        <strong>Species Count:</strong> {filtered_obs['Scientific_Name'].nunique()} | 
        <strong>Ecosystems:</strong> Forest & Grassland | 
        <strong>Time Period:</strong> {int(observations['Year'].min())} - {int(observations['Year'].max())}<br><br>
        
        <em>Data Source: National Park Service Bird Monitoring Program</em><br>
        <em>Analysis Framework: Python, Streamlit, Plotly, MySQL</em><br><br>
        
        <strong>💡 Dashboard Features:</strong> Advanced Filtering, Real-time Analytics, Interactive Visualizations, 
        Conservation Insights, Temporal Analysis, Environmental Correlations, Geographic Distribution
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()