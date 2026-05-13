import streamlit as st
import pandas as pd
import numpy as np
import itertools, collections
import seaborn as sns
import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm.auto import tqdm # For progress_apply
import torch, requests, io, warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message="Accessing `__path__`", module="transformers")

st.set_page_config(page_icon="🎶", page_title="United Kingdom Music Market Dashboard Analysis", layout="wide")
st.sidebar.image("static/banner.png")
    
st.header("Welcome to the **United Kingdom's Music Market Dashboard!**")
st.write("This dashboard will provide insights into various aspects of the UK music market.")

def assign_rank_group(position):
    if 1 <= position <= 10:
        return 'Top 10'
    elif 11 <= position <= 50:
        return 'Top 11-50'
    else:
        return 'Other'

@st.cache_data
def validate_and_preprocess(df):
    # Standardize artist names
    df['artist'] = df['artist'].str.lower().str.strip()

    # Split multi-artist collaborations
    df['artist'] = df['artist'].astype(str).apply(lambda x: [a.strip() for a in x.split('&')])
    df = df.explode('artist')

    # Track collaborations
    track_collaborations = df.groupby(['date', 'song', 'position']).agg(
        num_artists=('artist', 'nunique')
    ).reset_index()
    track_collaborations['is_collaboration'] = track_collaborations['num_artists'] > 1

    df_merged = pd.merge(
        df,
        track_collaborations[['date', 'song', 'position', 'is_collaboration']],
        on=['date', 'song', 'position'],
        how='left'
    )

    # Rank group
    df_merged['rank_group'] = df_merged['position'].apply(assign_rank_group)

    # Convert date
    df_merged['date'] = pd.to_datetime(df_merged['date'], dayfirst=True)

    return df_merged

# --- Sidebar Upload Option ---
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file (optional)",
    type=["csv"]
)

# Required schema
required_columns = [
    'date', 'position', 'song', 'artist', 'popularity',
    'duration_ms', 'album_type', 'total_tracks',
    'is_explicit', 'album_cover_url'
]

# --- Decide which dataset to use ---
if uploaded_file is not None:
    df_user = pd.read_csv(uploaded_file)

    # Validate schema
    if all(col in df_user.columns for col in required_columns):
        st.success("Custom CSV uploaded successfully!")
        df_merged = validate_and_preprocess(df_user)
    else:
        # Show error and required schema
        st.error("Uploaded CSV does not satisfy analytical requirements.")
        with st.expander("Required Columns"):
            st.write(required_columns)
        # Fallback to default dataset
        df = pd.read_csv('Atlantic_United_Kingdom.csv')
        df_merged = validate_and_preprocess(df)
else:
    # Default dataset
    df = pd.read_csv('Atlantic_United_Kingdom.csv')
    df_merged = validate_and_preprocess(df)

st.write("Data loaded and preprocessed successfully.")

# --- KPI Calculations ---

# 1. Calculate total appearances per artist
total_appearances_per_artist = df_merged['artist'].value_counts()

# 2. Calculate Artist Concentration Index
top_5_artists_appearances = total_appearances_per_artist.head(5).sum()
total_all_artists_appearances = total_appearances_per_artist.sum()
artist_concentration_index = (top_5_artists_appearances / total_all_artists_appearances) * 100

# 3. Calculate Diversity Score
diversity_score = df_merged['artist'].nunique() / len(df_merged)

# 4. Calculate Content Variety Index
content_variety_index = df_merged['song'].nunique() / len(df_merged)

# 5. Recreate track_collaborations (needed for avg artists per track and collaboration frequency by rank)
# Ensure assign_rank_group is available (it's defined in load_and_preprocess_data, but needs to be accessible here)
track_collaborations = df_merged.groupby(['date', 'song', 'position']).agg(
    num_artists=('artist', 'nunique')
).reset_index()
track_collaborations['is_collaboration'] = track_collaborations['num_artists'] > 1
track_collaborations['rank_group'] = track_collaborations['position'].apply(assign_rank_group)

# 6. Calculate average artists per track entry
average_artists_per_track = track_collaborations['num_artists'].mean()

# 7. Calculate collaboration frequency by rank group
collaboration_frequency_by_rank = track_collaborations.groupby('rank_group')['is_collaboration'].mean() * 100

# 8. Calculate explicitness percentages
explicitness_counts = df_merged['is_explicit'].value_counts()
total_tracks = explicitness_counts.sum()
explicitness_percentage = (explicitness_counts / total_tracks) * 100

# 9. Calculate percentage of explicit tracks by rank group
explicit_percentage_by_rank = df_merged.groupby('rank_group')['is_explicit'].mean() * 100

# 10. Calculate album type distribution
album_type_counts = df_merged['album_type'].value_counts()
total_album_types = album_type_counts.sum()
album_type_percentage = (album_type_counts / total_album_types) * 100

# 11. Convert duration from milliseconds to minutes
df_merged['duration_min'] = df_merged['duration_ms'] / 60000

# 12. Categorize tracks into 'short-form' and 'long-form'
df_merged['duration_category'] = df_merged['duration_min'].apply(lambda x: 'short-form' if x < 3.5 else 'long-form')

# 13. Calculate duration category percentages
duration_counts = df_merged['duration_category'].value_counts()
total_tracks_duration = duration_counts.sum()
duration_percentage = (duration_counts / total_tracks_duration) * 100

st.write("KPIs calculated successfully.")

st.sidebar.header('Filter Options')

# --- Date Range Selector ---
with st.spinner("⏳ Loading date range filter..."):
    min_date = df_merged['date'].min().date()
    max_date = df_merged['date'].max().date()
    date_range = st.sidebar.date_input(
        'Select Date Range',
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

mask = (
    (df_merged['date'].dt.date >= start_date) &
    (df_merged['date'].dt.date <= end_date)
)

# --- Artist Filter ---
with st.spinner("⏳ Loading artist filter..."):
    all_artists = sorted(df_merged['artist'].unique())
    selected_artists = st.sidebar.multiselect(
        'Filter by Artist',
        options=all_artists,
        default=all_artists
    )
    if selected_artists:
        mask &= df_merged['artist'].isin(selected_artists)

# --- Track Type Filter ---
with st.spinner("⏳ Loading track type filter..."):
    collaboration_choice = st.sidebar.radio(
        'Track Type',
        ('All Tracks', 'Solo Tracks', 'Collaborative Tracks')
    )
    if collaboration_choice == 'Solo Tracks':
        mask &= df_merged['is_collaboration'] == False
    elif collaboration_choice == 'Collaborative Tracks':
        mask &= df_merged['is_collaboration'] == True

# --- Album Type Filter ---
with st.spinner("⏳ Loading album type filter..."):
    all_album_types = sorted(df_merged['album_type'].unique())
    selected_album_types = st.sidebar.multiselect(
        'Filter by Album Type',
        options=all_album_types,
        default=all_album_types
    )
    if selected_album_types:
        mask &= df_merged['album_type'].isin(selected_album_types)

# --- Duration Interval Filter ---
with st.spinner("⏳ Loading duration filter..."):
    duration_range = st.sidebar.slider(
        'Filter by Duration Interval (minutes)',
        min_value=0, max_value=10,
        value=(0, 10)
    )
    mask &= (df_merged['duration_min'] >= duration_range[0]) & (df_merged['duration_min'] <= duration_range[1])

# --- Popularity Filter ---
with st.spinner("⏳ Loading popularity filter..."):
    pop_min, pop_max = int(df_merged['popularity'].min()), int(df_merged['popularity'].max())
    selected_popularity = st.sidebar.slider(
        'Filter by Popularity',
        min_value=0, max_value=100,
        value=(pop_min, pop_max)
    )
    mask &= (df_merged['popularity'] >= selected_popularity[0]) & (df_merged['popularity'] <= selected_popularity[1])

# --- Apply mask once ---
filtered_df = df_merged.loc[mask].copy()

# --- Default values ---
default_date_range = (min_date, max_date)
default_artists = all_artists
default_collaboration_choice = 'All Tracks'
default_album_types = all_album_types
default_duration_range = (0, 10)
default_popularity_range = (pop_min, pop_max)

# --- Check differences ---
is_date_range_different = (date_range != default_date_range)
is_artist_filter_different = (set(selected_artists) != set(default_artists))
is_track_type_different = (collaboration_choice != default_collaboration_choice)
is_album_type_different = (set(selected_album_types) != set(default_album_types))
is_duration_filter_different = (duration_range != default_duration_range)
is_popularity_filter_different = (selected_popularity != default_popularity_range)

# --- Combine into one flag ---
is_any_filter_different = (
    is_date_range_different
    or is_artist_filter_different
    or is_track_type_different
    or is_album_type_different
    or is_duration_filter_different
    or is_popularity_filter_different
)

tab1, tab2 = st.tabs(["UK Music Market Structural Analysis", 
                    "Recommendational Analysis for UK's Music Listeners"])

with tab1:
    st.title('United Kingdom Music Market Structural Analysis')
    # Tabs for each subheader
    tabs = st.tabs([
        "Filtered Dataset Preview",
        "🎤 Artist Dominance & Diversity",
        "🤝 Collaboration Structures",
        "🔞 Content Explicitness",
        "💿 Album Structure & Release Strategy",
        "⏱️ Track Duration & Format Preferences",
        "📸 Snapshots and Trends",
        "📊 Key Insights and Findings",
        "📌 Strategic Insights"
    ])
    with tabs[0]:
        st.write("### Filtered Dataset Preview")
        st.write(df_merged.head())
        st.markdown('---')
    with tabs[1]:
        st.subheader('Artist Dominance Leaderboard')
        # Calculate total appearances for each artist in the filtered data
        total_appearances_per_artist_filtered = filtered_df['artist'].value_counts()
        # Select the top 10 artists from this calculation
        top_10_artists_filtered = total_appearances_per_artist_filtered.head(10)
    
        # Create a Streamlit bar chart
        if not top_10_artists_filtered.empty:
            st.bar_chart(top_10_artists_filtered)
            # Wrap extra info in expander
            with st.expander("ℹ️ More Information"):
                st.info('The bar chart above shows the total appearance count for the top 10 artists based on the selected filters.')
        else:
            st.warning('No data available for the selected filters to display top artists.')
        
        st.markdown('---')
    
    with tabs[2]:
        st.subheader('Artist Collaboration Network')
    
        if collaboration_choice == 'Solo Tracks':
            st.warning("The artist collaboration network is displayed only when 'All Tracks' or 'Collaborative Tracks' are selected. Please adjust the 'Track Type' filter to view the network.")
        elif filtered_df[filtered_df['is_collaboration'] == True].empty:
            st.error("No collaborative tracks found for the selected filters to build a network.")
        else:
            collaborative_tracks_filtered = filtered_df[filtered_df['is_collaboration'] == True].copy()
    
            if not collaborative_tracks_filtered.empty:
                # Deduplicate by unique artist sets
                unique_collaborations = {}
                for _, group in collaborative_tracks_filtered.groupby(['date', 'song', 'position']):
                    artists_in_collaboration = tuple(sorted(group['artist'].unique()))
                    if artists_in_collaboration not in unique_collaborations:
                        unique_collaborations[artists_in_collaboration] = set()
                    unique_collaborations[artists_in_collaboration].add(group['song'].iloc[0])
    
                # Build graph from unique collaborations
                G = nx.Graph()
                for artist_group, songs in unique_collaborations.items():
                    # Add nodes
                    G.add_nodes_from(artist_group)
                    # Add edges between all pairs in this group
                    for artist1, artist2 in itertools.combinations(artist_group, 2):
                        if G.has_edge(artist1, artist2):
                            G[artist1][artist2]['weight'] += 1
                        else:
                            G.add_edge(artist1, artist2, weight=1)
    
                # Visualization
                fig, ax = plt.subplots(figsize=(15, 10))
                pos = nx.spring_layout(G, k=0.15, iterations=50)
    
                nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue', alpha=0.9, ax=ax)
    
                if G.number_of_edges() > 0:
                    edge_list = list(G.edges())
                    weights = [d['weight'] for _, _, d in G.edges(data=True)]
                    max_weight = max(weights) if weights else 1
                    nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=[w / max_weight * 5 for w in weights], alpha=0.7, edge_color='gray', ax=ax)
                else:
                    st.warning("No collaboration edges to display for the current filter.")
    
                nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
                ax.set_title('Artist Collaboration Network (Unique Collaborations)', size=20)
                ax.axis('off')
                st.pyplot(fig)
    
                with st.expander("ℹ️ More Information"):
                    for artist_group, songs in unique_collaborations.items():
                        st.write(f"**{' , '.join(artist_group)}** collaborated on: {', '.join(sorted(songs))}")
                    st.info('This network graph visualizes unique artist collaborations. Nodes represent artists, edges represent collaborations, and edge thickness indicates frequency across unique collaborations.')
    
                # --- 3D Interactive Visualization ---
                st.markdown('---')
                st.subheader("Interactive 3D Collaboration Network")
                
                # Use the same collaboration_counts_filtered for 3D
                pos_3d = nx.spring_layout(G, dim=3, k=0.15, iterations=50, seed=42)
                
                node_x, node_y, node_z, node_text, node_size = [], [], [], [], []
                for node in G.nodes():
                    x, y, z = pos_3d[node]
                    node_x.append(x); node_y.append(y); node_z.append(z)
                    
                    degree_for_sizing_color = G.degree[node]
                    collaborator_details = []
                    for neighbor in G.neighbors(node):
                        weight = G.get_edge_data(node, neighbor)['weight']
                        collaborator_details.append(f" - {neighbor.title()}: {weight} times")
                    
                    hover_info = f"Artist: {node.title()}<br>Total Collaborations: {len(collaborator_details)}"
                    if collaborator_details:
                        hover_info += "<br>Collaborated with:<br>" + "<br>".join(collaborator_details)
                    
                    node_text.append(hover_info)
                    node_size.append(degree_for_sizing_color * 2 + 5)
                
                node_degrees = [G.degree[node] for node in G.nodes()]
                node_trace = go.Scatter3d(
                    x=node_x, y=node_y, z=node_z, mode='markers', hoverinfo='text', text=node_text,
                    marker=dict(
                        showscale=True, colorscale='Viridis', reversescale=True, color=node_degrees, size=node_size,
                        colorbar=dict(
                            thickness=15,
                            title=dict(
                                text='Number of Collaborations (Degree)',
                                side='right'
                            ),
                            xanchor='left',
                            tickmode='linear',   # ✅ force linear ticks
                            dtick=1              # ✅ step size of 1 → integers only
                        ),
                        line_width=2
                    )
                )
                
                edg_traces = []
                edge_weights = [d['weight'] for _, _, d in G.edges(data=True)]
                max_weight = max(edge_weights) if edge_weights else 1
                scaled_edge_widths = [(w / max_weight) * 5 + 1 for w in edge_weights]
                
                for idx, edge in enumerate(G.edges(data=True)):
                    x0, y0, z0 = pos_3d[edge[0]]
                    x1, y1, z1 = pos_3d[edge[1]]
                    edg_traces.append(go.Scatter3d(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        z=[z0, z1, None],
                        mode='lines',
                        line=dict(width=scaled_edge_widths[idx], color='rgba(128,128,128,0.7)'),
                        hoverinfo='none'
                    ))
                
                fig3d = go.Figure(
                    data=edg_traces + [node_trace],
                    layout=go.Layout(
                        title=dict(
                            text='Interactive 3D Artist Collaboration Network',
                            font=dict(size=20), y=0.95, x=0.5,
                            xanchor='center', yanchor='top'
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Collaborations weighted by frequency",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002
                        )],
                        scene=dict(
                            xaxis=dict(showbackground=False, showticklabels=False, zeroline=False, title=''),
                            yaxis=dict(showbackground=False, showticklabels=False, zeroline=False, title=''),
                            zaxis=dict(showbackground=False, showticklabels=False, zeroline=False, title=''),
                            camera=dict( up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=1.25) )
                        )
                    )
                )
                
                st.plotly_chart(fig3d, width='stretch')
            
            else:
                st.warning("No collaborative tracks found for the selected filters to build a network.")
                
        st.markdown('---')
        
    with tabs[3]: 
        st.subheader('Content Explicitness Analysis')
        
        # Calculate total count of explicit and non-explicit tracks in filtered data
        explicitness_counts_filtered = filtered_df['is_explicit'].value_counts()
        # Calculate percentage of explicit and non-explicit tracks
        total_tracks_filtered = explicitness_counts_filtered.sum()
        explicitness_percentage_filtered = (explicitness_counts_filtered / total_tracks_filtered) * 100
        
        colA, colB = st.columns(2)
        with colA:
            with st.expander(f"**Overall Content Explicitness (Filtered Data):**"):
                st.info(f"- Explicit Tracks: {explicitness_counts_filtered.get(True, 0)} ({explicitness_percentage_filtered.get(True, 0):.2f}%) ")
                st.info(f"- Non-Explicit Tracks: {explicitness_counts_filtered.get(False, 0)} ({explicitness_percentage_filtered.get(False, 0):.2f}%) ")
        
        # Prepare data for pie chart
        pie_data_filtered = explicitness_counts_filtered
        pie_labels_filtered = ['Non-Explicit' if label == False else 'Explicit' for label in pie_data_filtered.index]
        # Create the pie chart
        fig_pie, ax_pie = plt.subplots(figsize=(9, 8))
        ax_pie.pie(pie_data_filtered, labels=pie_labels_filtered, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
        ax_pie.set_title('Overall Distribution of Explicit vs. Non-Explicit Content (Filtered)')
        ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        
        with colA:
            st.pyplot(fig_pie)
            with st.expander("ℹ️ More Information"):
                st.info('This pie chart shows the overall proportion of explicit and non-explicit tracks in the dataset based on the current filters.')
        
        # Calculate the percentage of explicit tracks for each rank_group in the filtered data
        explicit_percentage_by_rank_filtered = filtered_df.groupby('rank_group')['is_explicit'].mean() * 100
        
        with colB:
            with st.expander("**Percentage of Explicit Tracks by Rank Group (Filtered Data):**"): 
                if not explicit_percentage_by_rank_filtered.empty:
                    for rank_group, percentage in explicit_percentage_by_rank_filtered.items():
                        st.info(f"- {rank_group}: {percentage:.2f}%")
                else:
                    st.warning("No data available for explicit track percentage by rank group for the selected filters.")
        
        # Create the bar chart for explicit content percentage by rank group
        if not explicit_percentage_by_rank_filtered.empty:
            fig_bar, ax_bar = plt.subplots(figsize=(9, 8))
            sns.barplot(x=explicit_percentage_by_rank_filtered.index, y=explicit_percentage_by_rank_filtered.values, hue=explicit_percentage_by_rank_filtered.index, palette='viridis', legend=False, ax=ax_bar)
            ax_bar.set_title('Percentage of Explicit Content by Rank Group (Filtered)')
            ax_bar.set_xlabel('Rank Group')
            ax_bar.set_ylabel('Percentage of Explicit Tracks (%)')
            ax_bar.set_ylim(0, 100) # Ensure y-axis goes from 0 to 100
            plt.tight_layout()
            with colB:
                st.pyplot(fig_bar)
                with st.expander("ℹ️ More Information"):
                    st.info('This bar chart compares the percentage of explicit tracks within different rank groups (Top 10 vs. Top 11-50) based on the current filters.')
        else:
            st.warning("Cannot display explicit content percentage by rank group bar chart as no data is available for the selected filters.")

        explicit_top10 = explicit_percentage_by_rank_filtered.get('Top 10', 0.0)
        explicit_11_50 = explicit_percentage_by_rank_filtered.get('Top 11-50', 0.0)
        explicit_other = explicit_percentage_by_rank_filtered.get('Other', None)
        explicit_difference = explicit_top10 - explicit_11_50
        
        st.write("#### **🎭 Cultural Sensitivity Insights for UK Listeners:**")
        
        if total_tracks_filtered == 0:
            with st.container():
                st.warning("No explicitness data is available for the current filter selection, so insights cannot be generated at this time.")
        else:
            # Main container for insights
            with st.container():
                # Headline metrics in columns
                col1, col2 = st.columns(2)
                col1.metric("Top 10 Explicit %", f"{explicit_top10:.2f}%")
                col2.metric("Top 11-50 Explicit %", f"{explicit_11_50:.2f}%")
        
                # Conditional narrative cards
                if explicit_top10 > explicit_11_50:
                    st.info(f"Explicit content is more prevalent in the **Top 10** by **{explicit_difference:.2f} percentage points**.")
                    st.markdown("➡️ This suggests UK listeners may favor authentic, unfiltered/matured lyrical themes & expressions in top-ranked entries.")
                elif explicit_top10 < explicit_11_50:
                    st.success(f"Explicit content is more prevalent in ranks **11-50** by **{abs(explicit_difference):.2f} percentage points**.")
                    st.markdown("➡️ Lower explicitness in the Top 10 indicates mainstream appeal prioritizing accessibility over edgy content.")
                else:
                    st.warning("Explicitness rates are very similar between Top 10 and Top 11-50 tracks.")
                    st.markdown("➡️ This reflects stable cultural acceptance across chart tiers.")
        
                # Expanders for deeper insights
                with st.expander("📊 Broader Cultural Context"):
                    st.markdown("- Insights are based on the current filters and may shift as the dataset changes.")
                    st.markdown("- Genre, artist profile, and release strategy all affect how explicit content performs in the UK market.")
        
                with st.expander("🎶 Granular Analysis Suggestions"):
                    st.markdown("- Examine explicitness by genre or artist cohort.")
                    st.markdown("- Some musical styles and fanbases are more accepting of explicit material than others.")
        
                if explicit_other is not None:
                    with st.expander("🌍 Beyond the Top 50"):
                        st.metric("Outside Top 50 Explicit %", f"{explicit_other:.2f}%")
                        st.markdown("- Higher explicitness in lower ranks may signal niche markets or emerging trends in UK listener demographics.")
        
        st.markdown("---")
    with tabs[4]:
        st.subheader('Album Type Distribution')
        
        # Calculate total counts for each unique value in the `album_type` column for the filtered data
        album_type_counts_filtered = filtered_df['album_type'].value_counts()
        
        # Calculate the percentage of each `album_type` for the filtered data
        total_album_types_filtered = album_type_counts_filtered.sum()
        album_type_percentage_filtered = (album_type_counts_filtered / total_album_types_filtered) * 100
        
        colA, colB = st.columns(2)
        with colA:
            with st.expander("**Total counts of each album type (Filtered Data):**"):
                st.write(album_type_counts_filtered)
            
        # Create the horizontal bar chart for album type distribution
        if not album_type_counts_filtered.empty:
            fig_album_type, ax_album_type = plt.subplots(figsize=(12, 11), constrained_layout=True)
    
            sns.barplot(
                y=album_type_counts_filtered.index,   # categories on Y-axis
                x=album_type_counts_filtered.values,  # counts on X-axis
                hue=album_type_counts_filtered.index, legend=False,
                palette='viridis',
                ax=ax_album_type
            )
            ax_album_type.set_title('Release Format Dominance in the UK Market: Distribution of Album Types (Filtered)')
            ax_album_type.set_xlabel('Number of Tracks')
            ax_album_type.set_ylabel('Album Type')
    
            # Render and close cleanly
            with colA:
                st.pyplot(fig_album_type)
            plt.close(fig_album_type)
        else:
            st.warning("No album type data available for the selected filters to display the chart.")
        
        with colB:
            with st.expander("**Percentage of each album type (Filtered Data):**"):
                st.write(album_type_percentage_filtered)
            
        # Create the bar chart for album type distribution (percentage form)
        if not album_type_counts_filtered.empty:
            # Convert counts to percentages
            album_type_percent = album_type_counts_filtered / album_type_counts_filtered.sum() * 100
        
            # Use constrained_layout=True to avoid tight_layout warnings
            fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
            
            wedges, texts, autotexts = ax.pie( # type: ignore
                album_type_percent,
                labels=album_type_percent.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette('pastel'),
                wedgeprops=dict(width=0.4)  # creates donut effect
            )
            ax.set_title('Album Type Distribution (Filtered)')
            with colB:
                st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No album type data available for the selected filters to display the chart.")
        
        with st.expander("ℹ️ More Information"):
            st.info('This bar chart displays the distribution of different album types (single, album, compilation) within the filtered dataset in numerical and percentage(%) format, indicating the prevalence of each release format.')
        
        st.markdown('---')
    with tabs[5]:
        st.subheader('Track Duration Insights')
        
        # Calculate duration_min and duration_category for the filtered_df
        filtered_df['duration_min'] = filtered_df['duration_ms'] / 60000
        filtered_df['duration_category'] = filtered_df['duration_min'].apply(lambda x: 'short-form' if x < 3.5 else 'long-form')
        
        # Calculate total count of tracks for each duration category in filtered data
        duration_counts_filtered = filtered_df['duration_category'].value_counts()
        
        # Calculate percentage of tracks for each duration category in filtered data
        total_tracks_duration_filtered = duration_counts_filtered.sum()
        duration_percentage_filtered = (duration_counts_filtered / total_tracks_duration_filtered) * 100
        
        colA, colB = st.columns(2)
        with colA:
            with st.expander("**Duration Category Counts (Filtered Data):**"):
                st.dataframe(duration_counts_filtered)
        with colB:
            with st.expander("**Duration Category Percentages (Filtered Data):**"):
                st.dataframe(duration_percentage_filtered)
        
        # Prepare data for pie chart
        pie_data_duration = duration_counts_filtered
        # Ensure labels are a plain sequence of strings (matplotlib expects a Sequence[str], not a pandas Index)
        pie_labels_duration = [str(label) for label in pie_data_duration.index]
        
        with colA:
            if not filtered_df.empty:
                fig_countplot, ax_countplot = plt.subplots(figsize=(6, 6), constrained_layout=True)
                sns.countplot(
                    data=filtered_df,
                    x='duration_category', hue='duration_category', legend=False,
                    palette='pastel',
                    ax=ax_countplot
                )
                ax_countplot.set_title('Count Plot of Duration Categories (Filtered)')
                ax_countplot.set_xlabel('Duration Category')
                ax_countplot.set_ylabel('Number of Tracks')
                st.pyplot(fig_countplot)
                plt.close(fig_countplot)
    
                with st.expander("ℹ️ More Information"):
                    st.info("This count plot shows the absolute number of short-form and long-form tracks in the filtered dataset.")
            else:
                st.warning("No data available to display count plot.")
        
        with colB:
            # Create the pie chart
            if not pie_data_duration.empty:
                # Use constrained_layout=True to avoid tight_layout warnings
                fig_duration_pie, ax_duration_pie = plt.subplots(figsize=(8, 8), constrained_layout=True)
                ax_duration_pie.pie( pie_data_duration, labels=pie_labels_duration, autopct='%1.1f%%', startangle=90, 
                                    colors=sns.color_palette('pastel')
                )
                ax_duration_pie.set_title('Overall Distribution of Track Duration Categories (Filtered)')
                ax_duration_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            
                st.pyplot(fig_duration_pie)
                plt.close(fig_duration_pie)
            
                with st.expander("ℹ️ More Information"):
                    st.info('This pie chart shows the overall proportion of short-form and long-form tracks in the filtered dataset.')
            else:
                st.warning("No track duration data available for the selected filters to display the pie chart.")
        
        # Create 'popularity buckets' for the filtered_df
        if not filtered_df.empty and 'popularity' in filtered_df.columns:
            filtered_df['popularity_bucket'] = pd.qcut( filtered_df['popularity'], q=4, 
                                                        labels=['Q1 (Least Popular)', 'Q2', 'Q3', 'Q4 (Most Popular)'], duplicates='drop'
            )
            
            # Group by popularity_bucket and duration_category, and count tracks
            duration_popularity_distribution_filtered = (
                filtered_df.groupby(['popularity_bucket', 'duration_category'], observed=False)
                .size()
                .unstack(fill_value=0)
            )
        
            with st.expander("**Distribution of Track Duration Categories by Popularity Bucket (Filtered Data):**"):
                st.dataframe(duration_popularity_distribution_filtered)
        
            # Plotting the distribution
            if not duration_popularity_distribution_filtered.empty:
                fig_pop_duration, ax_pop_duration = plt.subplots(figsize=(12, 7), constrained_layout=True)
                duration_popularity_distribution_filtered.plot(
                    kind='bar', stacked=True, ax=ax_pop_duration, colormap='viridis'
                )
                ax_pop_duration.set_title('Track Duration Distribution Across Popularity Buckets (Filtered)')
                ax_pop_duration.set_xlabel('Popularity Bucket')
                ax_pop_duration.set_ylabel('Number of Tracks')
                plt.xticks(rotation=45, ha='right')
        
                st.pyplot(fig_pop_duration)
                plt.close(fig_pop_duration)
        
                with st.expander("ℹ️ More Information"):
                    st.info('This stacked bar chart illustrates how short-form and long-form tracks are distributed across different popularity levels (quartiles).')
            else:
                st.warning("No data available to visualize track duration distribution across popularity buckets for the selected filters.")
        else:
            st.warning("Cannot analyze track duration across popularity buckets. Ensure 'popularity' column is available and data is not empty.")
        
        # --- Create 'popularity buckets' and duration ranges ---
        if not filtered_df.empty and 'popularity' in filtered_df.columns and 'duration_min' in filtered_df.columns:
            try:
                filtered_df['popularity_bucket'] = pd.qcut(filtered_df['popularity'], q=4, labels=['Q1 (Least Popular)', 'Q2', 'Q3', 'Q4 (Most Popular)'], duplicates='drop')
            except ValueError:
                filtered_df['popularity_bucket'] = pd.qcut(filtered_df['popularity'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
            
            # Define duration bins (in minutes) and labels, using the updated bins from cell ac888dd6
            duration_bins = [0, 2, 4, 6, 8, 10]
            duration_bin_labels = ['0-2 min', '2-4 min', '4-6 min', '6-8 min', '8-10 min']
            filtered_df['duration_range'] = pd.cut(filtered_df['duration_min'], bins=duration_bins, labels=duration_bin_labels, right=False, include_lowest=True)
            df_merged['duration_range'] = pd.cut(df_merged['duration_min'], bins=duration_bins, labels=duration_bin_labels, right=False, include_lowest=True)   
            duration_range_popularity_distribution = filtered_df.groupby(['popularity_bucket', 'duration_range'], observed=False).size().unstack(fill_value=0)
            
            # --- Display Table: Distribution of Track Duration Ranges by Popularity Bucket ---
            with st.expander("**Distribution of Track Duration Ranges by Popularity Bucket:**"):
                st.dataframe(duration_range_popularity_distribution)
            
            # --- Display Box Plot: Track Duration Distribution by Popularity Bucket ---
            fig_boxplot, ax_boxplot = plt.subplots(figsize=(12, 7))
            sns.boxplot(x='popularity_bucket', y='duration_min', data=filtered_df, palette='viridis', hue='popularity_bucket', legend=False, ax=ax_boxplot)
            ax_boxplot.set_title('Track Duration Distribution by Popularity Bucket (Box Plot)')
            ax_boxplot.set_xlabel('Popularity Bucket')
            ax_boxplot.set_ylabel('Duration (Minutes)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_boxplot)
            
            with st.expander("ℹ️ More Information"):
                st.info('This box plot visualizes the median, quartiles, and outliers of track duration for each popularity bucket. It helps to visualize the typical range and spread of track lengths within each popularity quartile.') 
        else:
            st.warning("Cannot analyze track duration across popularity buckets. Ensure 'popularity' and 'duration_min' columns are available and data is not empty.")
        
        st.write("### **🎧 Insights into UK Listener Preference Indicators**")
        
        overall_short_form_pct = duration_percentage.get('short-form', 0.0)
        overall_long_form_pct = duration_percentage.get('long-form', 0.0)
        
        filtered_short_form_pct = duration_percentage_filtered.get('short-form', 0.0)
        filtered_long_form_pct = duration_percentage_filtered.get('long-form', 0.0)
        
        short_form_delta = filtered_short_form_pct - overall_short_form_pct
        long_form_delta = filtered_long_form_pct - overall_long_form_pct
        
        if total_tracks_duration_filtered == 0:
            with st.container():
                st.warning("No duration data is available for the current filter selection, so insights cannot be generated at this time.")
        else:
            with st.container():
                # Headline metrics in columns
                col1, col2 = st.columns(2)
                col1.metric("Filtered Short-form %", f"{filtered_short_form_pct:.2f}%")
                col2.metric("Filtered Long-form %", f"{filtered_long_form_pct:.2f}%")
        
                # Date range comparison
                full_min_date = df_merged['date'].min().date()
                full_max_date = df_merged['date'].max().date()
                is_date_range_different = (start_date != full_min_date) or (end_date != full_max_date)
        
                if is_date_range_different:
                    col3, col4 = st.columns(2)
                    col3.metric("Overall Short-form %", f"{overall_short_form_pct:.2f}%")
                    col4.metric("Overall Long-form %", f"{overall_long_form_pct:.2f}%")
        
                    # Conditional narrative cards
                    if short_form_delta > 0:
                        st.info(f"Short-form bias stronger by **{short_form_delta:.2f} percentage points** in the filtered selection.")
                        st.markdown("➡️ Indicates evolving listener preferences toward concise content, possibly due to digital consumption habits.")
                    elif short_form_delta < 0:
                        st.success(f"Longer tracks favored by **{abs(short_form_delta):.2f} percentage points** in the filtered selection.")
                        st.markdown("➡️ Suggests niche appeal for immersive, long-form listening experiences.")
                    else:
                        st.warning("Short-form share matches the overall dataset.")
                        st.markdown("➡️ Duration preferences remain stable across the timeframe.")
        
                    # Dominance check
                    if filtered_short_form_pct >= filtered_long_form_pct:
                        st.info("Short-form tracks remain dominant in the filtered view.")
                    else:
                        st.success("Long-form tracks are more prominent in the filtered view, reflecting niche listener preferences.")
        
                    # Average duration comparison
                    avg_duration_overall = df_merged['duration_min'].mean()
                    avg_duration_filtered = filtered_df['duration_min'].mean() if not filtered_df.empty else 0.0
                    duration_delta = avg_duration_filtered - avg_duration_overall
                    
                    col5, col6 = st.columns(2)
                    col5.metric("Avg Duration (Filtered)", f"{avg_duration_filtered:.2f} min")
                    col6.metric("Avg Duration (Overall)", f"{avg_duration_overall:.2f} min")
        
                    if duration_delta > 0:
                        st.info(f"Filtered selection is longer by **{duration_delta:.2f} minutes** on average.")
                        st.markdown("➡️ Reflects a trend toward elaborate musical compositions.")
                    elif duration_delta < 0:
                        st.success(f"Filtered selection is shorter by **{abs(duration_delta):.2f} minutes** on average.")
                        st.markdown("➡️ Reinforces short-form dominance, likely driven by streaming algorithms.")
                    else:
                        st.warning("Average track duration matches the overall dataset.")
                        st.markdown("➡️ Indicates stable duration preferences across UK listeners.")
        
                    with st.expander("📊 Contextual Notes"):
                        st.markdown("- These comparisons make insights dynamic, showing whether the current filter view leans more short-form or long-form than the UK baseline.")
                        st.markdown("- Genre, release strategy, and artist profile all influence duration trends.")
        
                else:
                    # Baseline case
                    if filtered_short_form_pct >= filtered_long_form_pct:
                        st.info("Short-form tracks dominate the current selection, aligning with UK listener preferences.")
                    else:
                        st.success("Long-form tracks are more prominent, indicating immersive listening appeal.")
        
                    avg_duration_filtered = filtered_df['duration_min'].mean() if not filtered_df.empty else 0.0
                    st.metric("Avg Duration (Selection)", f"{avg_duration_filtered:.2f} min")
                    st.markdown("➡️ Since the full date range is selected, this reflects the baseline UK market duration preferences.")

        #Calculations for snapshots and insights
        # 1. Artist Dominance and Diversity Metrics
        filtered_total_appearances = filtered_df['artist'].value_counts()
        filtered_top_5_appearances = filtered_total_appearances.head(5).sum()
        filtered_total_all_appearances = filtered_total_appearances.sum()
        filtered_artist_concentration_index = (filtered_top_5_appearances / filtered_total_all_appearances) * 100 if filtered_total_all_appearances else 0
        filtered_diversity_score = (filtered_df['artist'].nunique() / len(filtered_df)) if len(filtered_df) else 0
        filtered_content_variety_index = (filtered_df['song'].nunique() / len(filtered_df)) if len(filtered_df) else 0
        # Get top artist for filtered data
        filtered_top_artist = filtered_df['artist'].value_counts().index[0] if not filtered_df.empty and len(filtered_df['artist'].value_counts()) > 0 else 'Unknown'
        filtered_top_artist_count = filtered_df['artist'].value_counts().iloc[0] if not filtered_df.empty and len(filtered_df['artist'].value_counts()) > 0 else 0
        # Get top artist for full dataset
        full_top_artist = df_merged['artist'].value_counts().index[0] if not df_merged.empty and len(df_merged['artist'].value_counts()) > 0 else 'Unknown'
        full_top_artist_count = df_merged['artist'].value_counts().iloc[0] if not df_merged.empty and len(df_merged['artist'].value_counts()) > 0 else 0
        
        # 2. Artist Collaboration Metrics
        filtered_track_collaborations = filtered_df.groupby(['date', 'song', 'position']).agg(num_artists=('artist', 'nunique')).reset_index()
        filtered_track_collaborations['is_collaboration'] = filtered_track_collaborations['num_artists'] > 1
        filtered_track_collaborations['rank_group'] = filtered_track_collaborations['position'].apply(assign_rank_group)
        filtered_collaboration_frequency_by_rank = filtered_track_collaborations.groupby('rank_group')['is_collaboration'].mean() * 100
        # Get highest collaboration pair for filtered data
        if not filtered_df.empty:
            filtered_collab_pairs = []
            for _, group in filtered_df.groupby(['date', 'song', 'position']):
                if group['is_collaboration'].iloc[0]:
                    artists = sorted(group['artist'].unique())
                    for i in range(len(artists)-1):
                        filtered_collab_pairs.append(tuple(sorted([artists[i], artists[i+1]])))
            filtered_collab_counter = collections.Counter(filtered_collab_pairs)
            filtered_highest_collab = filtered_collab_counter.most_common(1)[0] if filtered_collab_counter else (None, 0)
        else:
            filtered_highest_collab = (None, 0)
        
        # Get highest collaboration pair for full dataset
        if not df_merged.empty:
            full_collab_pairs = []
            for _, group in df_merged.groupby(['date', 'song', 'position']):
                if group['is_collaboration'].iloc[0]:
                    artists = sorted(group['artist'].unique())
                    for i in range(len(artists)-1):
                        full_collab_pairs.append(tuple(sorted([artists[i], artists[i+1]])))
            full_collab_counter = collections.Counter(full_collab_pairs)
            full_highest_collab = full_collab_counter.most_common(1)[0] if full_collab_counter else (None, 0)
        else:
            full_highest_collab = (None, 0)
            
        # Highest collaboration network (max number of artists per track in filtered_df)
        if not filtered_df.empty:
            # Group by song/date/position and count unique artists
            collab_networks = (
                filtered_df.groupby(['date', 'song', 'position'])
                .agg(num_artists=('artist', 'nunique'))
                .reset_index()
            )
        
            # Find the max number of artists
            max_artists = collab_networks['num_artists'].max()
        
            # Get all tracks with that max number of artists
            highest_network_tracks = collab_networks[collab_networks['num_artists'] == max_artists]
        
            # Build mapping: artist group → unique songs
            network_groups = {}
            for _, row in highest_network_tracks.iterrows():
                artists = tuple(sorted(filtered_df[
                    (filtered_df['date'] == row['date']) &
                    (filtered_df['song'] == row['song']) &
                    (filtered_df['position'] == row['position'])
                ]['artist'].unique()))
                network_groups.setdefault(artists, set()).add(row['song'])
        else:
            max_artists, network_groups = (0, {})
            
        # For the full dataset, we can calculate the same network metrics
        if not df_merged.empty:
            collab_networks_full = (
                df_merged.groupby(['date', 'song', 'position'])
                .agg(num_artists=('artist', 'nunique'))
                .reset_index()
            )
            max_artists_full = collab_networks_full['num_artists'].max()
            highest_network_tracks_full = collab_networks_full[collab_networks_full['num_artists'] == max_artists_full]
            network_groups_full = {}
            for _, row in highest_network_tracks_full.iterrows():
                artists = tuple(sorted(df_merged[
                    (df_merged['date'] == row['date']) &
                    (df_merged['song'] == row['song']) &
                    (df_merged['position'] == row['position'])
                ]['artist'].unique()))
                network_groups_full.setdefault(artists, set()).add(row['song'])
        else:
            max_artists_full, network_groups_full = (0, {})
        
        filtered_explicitness_counts = filtered_df['is_explicit'].value_counts()
        filtered_explicitness_percentage = (filtered_explicitness_counts / filtered_explicitness_counts.sum()) * 100 if filtered_explicitness_counts.sum() else pd.Series({True: 0.0, False: 0.0})
        filtered_explicit_percentage_by_rank = filtered_df.groupby('rank_group')['is_explicit'].mean() * 100
        filtered_album_type_percentage = (filtered_df['album_type'].value_counts() / filtered_df['album_type'].value_counts().sum() * 100) if not filtered_df['album_type'].empty else pd.Series(dtype=float)
        
        # 5. Track Duration (Define duration_range for the full dataset as well)
        df_merged['duration_range'] = pd.cut(df_merged['duration_min'], bins=duration_bins, labels=duration_bin_labels, right=False, include_lowest=True)  
            
        # Get most popular duration interval for filtered data
        if not filtered_df.empty and 'duration_range' in filtered_df.columns:
            filtered_duration_range_counts = filtered_df['duration_range'].value_counts()
            filtered_most_popular_duration = filtered_duration_range_counts.index[0] if len(filtered_duration_range_counts) > 0 else 'Unknown'
        else:
            filtered_most_popular_duration = 'Unknown'
        
        # Get most popular duration interval for full dataset
        if 'duration_range' in df_merged.columns:
            full_duration_range_counts = df_merged['duration_range'].value_counts()
            full_most_popular_duration = full_duration_range_counts.index[0] if len(full_duration_range_counts) > 0 else 'Unknown'
        else:
            full_most_popular_duration = 'Unknown'
            
        present_avg_duration = filtered_df['duration_min'].mean() if not filtered_df.empty else 0
        overall_avg_duration = df_merged['duration_min'].mean()
        
        st.markdown('---')

    with tabs[6]:
        st.markdown("## 🎯 Market Snapshot")
    
        # --- CSS for metric styling with hover + bouncing animation ---
        st.markdown("""
            <style>
            /* Base metric card */
            div[data-testid="stMetric"] {
                background-color: rgba(255,255,255,0.05);
                border-radius: 12px;
                padding: 1rem;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);
                transition: all 0.3s ease;
                text-align: left;
                overflow: hidden;
                position: relative;
            }
        
            /* Hover effect */
            div[data-testid="stMetric"]:hover {
                box-shadow: 0 6px 14px rgba(0,0,0,0.25);
                transform: translateY(-4px);
                background-color: #FFD700;
            }
        
            /* Metric label */
            div[data-testid="stMetric"] > label {
                font-size: 0.9rem;
                font-weight: 600;
                color: #cccccc;
            }
        
            /* Metric value with bouncing animation */
            div[data-testid="stMetric"] > div {
                font-size: 1.2rem;
                font-weight: 700;
                margin-top: 0.1rem;
                display: inline-block;
                white-space: nowrap;
                animation: bounceXY 6s ease-in-out infinite alternate;
            }
        
            /* Delta styling with bounce */
            div[data-testid="stMetricDelta"] {
                font-size: 0.9rem;
                font-weight: 600;
                animation: bounceXY 9s ease-in-out infinite alternate;
            }
        
            /* Bouncing animation across both axes */
            @keyframes bounceXY {
                0%   { transform: translate(0%, 0%); }
                33%  { transform: translate(-120%, 0%); }
                67%  { transform: translate(0%, 120%); }
                100% { transform: translate(0%, 0%); }
            }
            </style>
        """, unsafe_allow_html=True)
        
        # --- First Row of KPI Cards ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<div class='card'><h4>🎤 Artist Concentration</h4></div>", unsafe_allow_html=True)
            st.metric("Index", f"{filtered_artist_concentration_index:.2f}%",
                    f"{filtered_artist_concentration_index - artist_concentration_index:.2f}%" if is_any_filter_different else None)
    
        with col2:
            st.markdown("<div class='card'><h4>🌐 Diversity Score</h4></div>", unsafe_allow_html=True)
            st.metric("Score", f"{filtered_diversity_score:.4f}",
                    f"{filtered_diversity_score - diversity_score:.4f}" if is_any_filter_different else None)
    
        with col3:
            st.markdown("<div class='card'><h4>🔞 Explicit Content</h4></div>", unsafe_allow_html=True)
            st.metric("Percentage", f"{filtered_explicitness_percentage.get(True, 0):.2f}%",
                    f"{filtered_explicitness_percentage.get(True, 0) - explicitness_percentage.get(True, 0):.2f}%" if is_any_filter_different else None)
    
        with col4:
            st.markdown("<div class='card'><h4>⏱️ Avg Duration</h4></div>", unsafe_allow_html=True)
            st.metric("Minutes", f"{present_avg_duration:.2f}",
                    f"{present_avg_duration - overall_avg_duration:.2f}" if is_any_filter_different else None)
    
        # --- Second Row of KPI Cards ---
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.markdown("<div class='card'><h4>🤝 Avg Artists/Track</h4></div>", unsafe_allow_html=True)
            st.metric("Avg Artists per Track", f"{filtered_track_collaborations['num_artists'].mean():.2f}",
                    f"{filtered_track_collaborations['num_artists'].mean() - average_artists_per_track:.2f}" if is_any_filter_different else None)
    
        with col6:
            st.markdown("<div class='card'><h4>📊 Top 10 Collabs</h4></div>", unsafe_allow_html=True)
            st.metric("Top 10 Collaboration %", f"{filtered_collaboration_frequency_by_rank.get('Top 10', 0):.2f}%",
                    f"{filtered_collaboration_frequency_by_rank.get('Top 10', 0) - collaboration_frequency_by_rank.get('Top 10', 0):.2f}%" if is_any_filter_different else None)
    
        with col7:
            st.markdown("<div class='card'><h4>💿 Singles Share</h4></div>", unsafe_allow_html=True)
            st.metric("Singles Share %", f"{filtered_album_type_percentage.get('single', 0):.2f}%",
                    f"{filtered_album_type_percentage.get('single', 0) - album_type_percentage.get('single', 0):.2f}%" if is_any_filter_different else None)
    
        with col8:
            st.markdown("<div class='card'><h4>📀 Albums Share</h4></div>", unsafe_allow_html=True)
            st.metric("Albums Share %", f"{filtered_album_type_percentage.get('album', 0):.2f}%",
                    f"{filtered_album_type_percentage.get('album', 0) - album_type_percentage.get('album', 0):.2f}%" if is_any_filter_different else None)
    
        st.divider()
    
        # --- Strategic Insights Snapshot ---
        st.markdown("## 📌 Strategic Insights Snapshot")
    
        col9, col10, col11 = st.columns(3)
        with col9:
            st.markdown("<div class='card'><h4>🎯 Market Structure</h4></div>", unsafe_allow_html=True)
            if is_any_filter_different:
                if filtered_artist_concentration_index > artist_concentration_index:
                    st.metric("Structure", "Hit-driven Dominance", "↑ Concentration")
                elif filtered_artist_concentration_index < artist_concentration_index:
                    st.metric("Structure", "Emerging Diversity", "↓ Concentration")
                else:
                    st.metric("Structure", "Stable Dynamics", "→ No Change")
            else:
                st.metric("Structure", "Baseline UK Market")
    
        with col10:
            st.markdown("<div class='card'><h4>🤝 Collaboration Trends</h4></div>", unsafe_allow_html=True)
            if is_any_filter_different:
                if filtered_collaboration_frequency_by_rank.get('Top 10',0) >= collaboration_frequency_by_rank.get('Top 10',0):
                    st.metric("Trend", "Strong Collabs", "↑")
                else:
                    st.metric("Trend", "Varied Profiles", "↓")
            else:
                st.metric("Trend", "Baseline Collabs")
    
        with col11:
            st.markdown("<div class='card'><h4>⚡ Format Preference</h4></div>", unsafe_allow_html=True)
            if is_any_filter_different:
                if duration_percentage_filtered.get('short-form',0) >= duration_percentage.get('short-form',0):
                    st.metric("Preference", "Short-form Trend", "↑")
                else:
                    st.metric("Preference", "Long-form Prominence", "↑")
            else:
                st.metric("Preference", "Balanced Format")
    
        st.divider()
    
        
    with tabs[7]:
        st.subheader("📊 Key Insights and Findings")
        
        with st.expander("#### Comprehensive UK Music Market Analysis"):
            st.info("This dashboard analyzes the UK music market by comparing the current filter view against the full dataset baseline to show how the selected subset differs from the overall market.")
        
        # 1. Artist Dominance and Diversity
        with st.container():
            st.markdown("### 🎤 Artist Dominance & Diversity")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Concentration Index", 
                        f"{filtered_artist_concentration_index:.2f}%", 
                        f"{filtered_artist_concentration_index - artist_concentration_index:.2f}%" if is_any_filter_different else None)
            with col2:
                st.metric("Diversity Score", 
                        f"{filtered_diversity_score:.4f}", 
                        f"{filtered_diversity_score - diversity_score:.4f}" if is_any_filter_different else None)
            with col3:
                st.metric("Variety Index", 
                        f"{filtered_content_variety_index:.4f}", 
                        f"{filtered_content_variety_index - content_variety_index:.4f}" if is_any_filter_different else None)
            colA, colB = st.columns(2)
            with colA:
                st.info(f"**Top Artist:** {filtered_top_artist} ({filtered_top_artist_count} appearances)")
            if is_any_filter_different:
                with colB:
                    st.info(f"**Market Reference:** {full_top_artist} ({full_top_artist_count} appearances)")
                if filtered_artist_concentration_index > artist_concentration_index:
                    st.info("Selection is more concentrated → Hit-driven dominance by fewer artists.")
                elif filtered_artist_concentration_index < artist_concentration_index:
                    st.success("Selection is less concentrated → More diverse artist pool gaining traction.")
                else:
                    st.warning("Selection matches baseline concentration → Stable dynamics.")
                if filtered_diversity_score > diversity_score:
                    st.success("Higher diversity score → More unique artists relative to total entries.")
                elif filtered_diversity_score < diversity_score:
                    st.warning("Lower diversity score → Fewer unique artists relative to total entries.")
                else:            
                    st.info("Diversity score unchanged → Similar balance of unique artists.")
                    
                if filtered_content_variety_index > content_variety_index:
                    st.success("Higher variety index → More unique songs relative to total entries.")
                elif filtered_content_variety_index < content_variety_index:
                    st.warning("Lower variety index → Fewer unique songs relative to total entries.")
                else:
                    st.info("Content variety index unchanged → Similar balance of unique songs.")
            else:
                st.markdown("*Artists show balanced dominance with diversity maintained across the market.*")
        
        # 2. Collaboration Structures
        with st.container():
            st.markdown("### 🤝 Collaboration Structures")
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Avg Artists/Track", 
                        f"{filtered_track_collaborations['num_artists'].mean():.2f}", 
                        f"{filtered_track_collaborations['num_artists'].mean() - average_artists_per_track:.2f}" if is_any_filter_different else None)
            with col5:
                st.metric("Top 10 Collab Share", 
                        f"{filtered_collaboration_frequency_by_rank.get('Top 10', 0):.2f}%", 
                        f"{filtered_collaboration_frequency_by_rank.get('Top 10', 0) - collaboration_frequency_by_rank.get('Top 10', 0):.2f}%" if is_any_filter_different else None)
            with col6:
                st.metric("Top 11-50 Collab Share", 
                        f"{filtered_collaboration_frequency_by_rank.get('Top 11-50', 0):.2f}%", 
                        f"{filtered_collaboration_frequency_by_rank.get('Top 11-50', 0) - collaboration_frequency_by_rank.get('Top 11-50', 0):.2f}%" if is_any_filter_different else None)
            if filtered_highest_collab[0]:
                colA, colB = st.columns(2)
                with colA:
                    st.info(f"**Highest Pair:** {filtered_highest_collab[0][0]} & {filtered_highest_collab[0][1]} ({filtered_highest_collab[1]} collaborations)")
                if is_any_filter_different:
                    with colB:
                        st.info(f"[**Market Reference:** {full_highest_collab[0][0]} & {full_highest_collab[0][1]} ({full_highest_collab[1]} collaborations)]") # type: ignore
                    if filtered_highest_collab[1] > full_highest_collab[1]:
                        st.success("This pair is more dominant in the current selection → Strong collaborative synergy.")
                    elif filtered_highest_collab[1] < full_highest_collab[1]:
                        st.warning("This pair is less dominant in the current selection → More distributed collaborations.")
                    else:
                        st.info("This pair has the same collaboration count as the full dataset → Stable collaboration dynamics.")
                        
            if max_artists > 0:
                colA, colB = st.columns(2)
                with colA:
                    st.success(f"Largest Network: {max_artists} artists together")
                    for artists, songs in network_groups.items():
                        with st.expander(f"👥 {', '.join(artists)}"):
                            st.write(f"**Contribution:** {', '.join(sorted(songs))}")
                if is_any_filter_different:
                    with colB:
                        st.info(f"**Market Reference:** {max_artists_full} artists together")
                        for artists, songs in network_groups_full.items():
                            with st.expander(f"👥 {', '.join(artists)}"):
                                st.write(f"**Contribution:** {', '.join(sorted(songs))}")
                    if (max_artists < max_artists_full):
                        st.warning("Largest collaboration network is smaller in selection → More focused collaborations.")
                    else:
                        st.success("Largest collaboration network is as large or larger in selection → Strong collaborative networks persist.")
                        
            if not is_any_filter_different:
                st.markdown("*Collaboration networks remain strong, sustaining chart success across tiers.*")
        
        # 3. Content Explicitness
        with st.container():
            st.markdown("### 🔞 Content Explicitness")
            col7, col8, col9, col9a = st.columns(4)
            with col7:
                st.metric("Explicit Share", 
                        f"{filtered_explicitness_percentage.get(True, 0):.2f}%", 
                        f"{filtered_explicitness_percentage.get(True, 0) - explicitness_percentage.get(True, 0):.2f}%" if is_any_filter_different else None)
            with col8:
                st.metric("Non-Explicit Share", 
                        f"{filtered_explicitness_percentage.get(False, 0):.2f}%", 
                        f"{filtered_explicitness_percentage.get(False, 0) - explicitness_percentage.get(False, 0):.2f}%" if is_any_filter_different else None)
            with col9:
                st.metric("Top 10 Explicit Share", 
                        f"{filtered_explicit_percentage_by_rank.get('Top 10', 0):.2f}%", 
                        f"{filtered_explicit_percentage_by_rank.get('Top 10', 0) - explicit_percentage_by_rank.get('Top 10', 0):.2f}%" if is_any_filter_different else None)
            with col9a:
                st.metric("Top 11-50 Explicit Share", 
                        f"{filtered_explicit_percentage_by_rank.get('Top 11-50', 0):.2f}%", 
                        f"{filtered_explicit_percentage_by_rank.get('Top 11-50', 0) - explicit_percentage_by_rank.get('Top 11-50', 0):.2f}%" if is_any_filter_different else None)
            if not is_any_filter_different:
                st.info("*Explicit content maintains a significant of natural presence, especially in top-tier tracks, reflecting mainstream trends.*")
                st.success("Lower explicit share → More family-friendly content.")
            else:
                if filtered_explicitness_percentage.get(True, 0) > explicitness_percentage.get(True, 0):
                    st.info("Higher explicit share in selection → Edgier content trend.")
                elif filtered_explicitness_percentage.get(True, 0) < explicitness_percentage.get(True, 0):
                    st.success("Lower explicit share in selection → More family-friendly content.")
                else:
                    st.warning("Explicit share matches baseline → Stable content preferences, appealing to both mainstream and niche audiences in different timelines.")
        
        # 4. Album Structure
        with st.container():
            st.markdown("### 💿 Album Structure & Release Strategy")
            col10, col11 = st.columns(2)
            with col10:
                st.metric("Singles Share", 
                        f"{filtered_album_type_percentage.get('single', 0):.2f}%", 
                        f"{filtered_album_type_percentage.get('single', 0) - album_type_percentage.get('single', 0):.2f}%" if is_any_filter_different else None)
            with col11:
                st.metric("Albums Share", 
                        f"{filtered_album_type_percentage.get('album', 0):.2f}%", 
                        f"{filtered_album_type_percentage.get('album', 0) - album_type_percentage.get('album', 0):.2f}%" if is_any_filter_different else None)
            if is_any_filter_different:
                if filtered_album_type_percentage.get('single', 0) > album_type_percentage.get('single', 0):
                    st.info("Higher singles share in selection → Focus on quick-hit releases.")
                elif filtered_album_type_percentage.get('single', 0) < album_type_percentage.get('single', 0):
                    st.success("Higher albums share in selection → Emphasis on deeper engagement.")
                else:
                    st.warning("Singles and albums share matches baseline → Balanced release strategies catering to both quick hits and sustained engagement.")
            else:
                st.info("*Singles dominate quick-hit releases, while albums sustain deeper engagement.*")
                st.success("Higher singles share in selection → Focus on quick-hit releases.")
        
        # 5. Track Duration
        with st.container():
            st.markdown("### ⏱️ Track Duration & Format")
            col12, col13, col14 = st.columns(3)
            with col12:
                st.metric("Short-form Type", 
                        f"{duration_percentage_filtered.get('short-form', 0):.2f}%", 
                        f"{duration_percentage_filtered.get('short-form', 0) - duration_percentage.get('short-form', 0):.2f}%" if is_any_filter_different else None)
            with col13:
                st.metric("Long-form Type", 
                        f"{duration_percentage_filtered.get('long-form', 0):.2f}%", 
                        f"{duration_percentage_filtered.get('long-form', 0) - duration_percentage.get('long-form', 0):.2f}%" if is_any_filter_different else None)
            with col14:
                st.metric("Avg Duration", 
                        f"{present_avg_duration:.2f} min", 
                        f"{present_avg_duration - overall_avg_duration:.2f} min" if is_any_filter_different else None)
            colA, colB = st.columns(2)
            with colA:
                st.info(f"**Most Popular Interval:** {filtered_most_popular_duration}")
            if is_any_filter_different:
                with colB:
                    st.info(f"**Market Reference MPI:** {full_most_popular_duration}")
                if present_avg_duration > overall_avg_duration:
                    st.info("Longer tracks → In-depth listening experiences.")
                elif present_avg_duration < overall_avg_duration:
                    st.success("Shorter tracks → Fast-paced consumption trend.")
                else:
                    st.warning("Stable duration → Preferences unchanged.")
            else:
                st.info("Baseline UK duration preferences shown.")
        
        st.markdown('---')
    with tabs[8]:
        # Strategic Insights
        st.markdown("### 📌 Strategic Insights")
        
        if is_any_filter_different:
            st.markdown("**Filtered vs Full Dataset Insights:**")
            colA, colB, colC, colD, colE = st.columns(5)
        
            with colA:
                st.markdown("🎤 **Artists**")
                if filtered_artist_concentration_index > artist_concentration_index:
                    st.info("Hit-driven dominance → Fewer artists capture attention.")
                elif filtered_artist_concentration_index < artist_concentration_index:
                    st.success("Emerging diversity → More artists gaining traction.")
                else:
                    st.warning("Stable dynamics → Market concentration unchanged.")
        
            with colB:
                st.markdown("🤝 **Collaboration**")
                if filtered_collaboration_frequency_by_rank.get('Top 10',0) >= collaboration_frequency_by_rank.get('Top 10',0):
                    st.success("Strong collabs → Top artists maintain networking strength.")
                else:
                    st.info("Varied profiles → Mid-tier collaborations more visible.")
        
            with colC:
                st.markdown("🔞 **Explicitness**")
                if filtered_explicitness_percentage.get(True,0) > explicitness_percentage.get(True,0):
                    st.info("Higher explicit share → Audience leaning toward bold content.")
                else:
                    st.success("Lower explicit share → Broader mainstream appeal.")
        
            with colD:
                st.markdown("💿 **Release Strategy**")
                if filtered_album_type_percentage.get('single',0) > album_type_percentage.get('single',0):
                    st.info("Singles dominate → Quick-hit strategy stronger in selection.")
                else:
                    st.success("Albums stronger → Longer engagement formats preferred.")
        
            with colE:
                st.markdown("⏱️ **Duration**")
                if present_avg_duration > overall_avg_duration:
                    st.info("Longer tracks → In-depth listening experiences.")
                elif present_avg_duration < overall_avg_duration:
                    st.success("Shorter tracks → Fast-paced consumption trend.")
                else:
                    st.warning("Stable duration → Preferences unchanged.")
        
            st.markdown("*This filtered view highlights shifts in artist dominance, collaboration strength, and format preferences compared to the full dataset.*")
        
        else:
            st.markdown("**Full Market Dynamics (Complete Dataset):**")
            colF, colG, colH, colI, colJ = st.columns(5)
        
            with colF:
                st.markdown("🎤 **Artists**")
                st.info("Balanced dominance → Hit-makers coexist with diverse emerging talent.")
        
            with colG:
                st.markdown("🤝 **Collaboration**")
                st.success("Strong networks → Partnerships sustain chart success across tiers.")
        
            with colH:
                st.markdown("🔞 **Explicitness**")
                st.warning("Balanced explicit vs non-explicit → Platforms cater to both mainstream and niche audiences.")
        
            with colI:
                st.markdown("💿 **Release Strategy**")
                st.info("Dual-track approach → Singles drive quick hits, albums maintain depth.")
        
            with colJ:
                st.markdown("⏱️ **Duration**")
                st.success("Stable track lengths → Consistent listener preferences shaped by streaming algorithms.")
        
            st.markdown("*Together, these dynamics suggest a mature, stable UK market where innovation happens within predictable frameworks — ideal for benchmarking and long-term planning.*")
        
        st.markdown("---")

# --- 1. Initial Data Loading and Preprocessing ---
print("Dashboard created successfully with basic Streamlit structure. Please wait for some time (about 3-4 mins) to get the advanced Streamlit structure with recommendations & conclusion.")
print("--- Starting Data Preparation and Model Training For Recommendational Analysis ---")

# Re-initialize df from the original CSV and perform initial preprocessing
df = filtered_df.copy() # Use the already filtered data for consistency with the dashboard filters
df['artist'] = df['artist'].str.lower().str.strip()
df['artist'] = df['artist'].astype(str).apply(lambda x: [a.strip() for a in x.split('&')])
df = df.explode('artist')

# Re-create track_collaborations DataFrame
track_collaborations = df.groupby(['date', 'song', 'position']).agg(
    num_artists=('artist', 'nunique')
).reset_index()
track_collaborations['is_collaboration'] = track_collaborations['num_artists'] > 1

track_collaborations['rank_group'] = track_collaborations['position'].apply(assign_rank_group)

# Re-create df_merged with all necessary columns
df_merged = pd.merge(df, track_collaborations[['date', 'song', 'position', 'is_collaboration', 'num_artists', 'rank_group']],
                on=['date', 'song', 'position'], how='left')

# Add duration_min
df_merged['duration_min'] = df_merged['duration_ms'] / 60000

# Add duration_category
df_merged['duration_category'] = df_merged['duration_min'].apply(lambda x: 'short-form' if x < 3.5 else 'long-form')

# Add popularity_bucket
df_merged['popularity_bucket'] = pd.qcut(df_merged['popularity'], q=4, labels=['Q1 (Least Popular)', 'Q2', 'Q3', 'Q4 (Most Popular)'], duplicates='drop')

print("df_merged and its derived columns have been successfully re-created.")

# --- 2. Feature Engineering ---
# Convert 'date' to datetime objects
df_merged['date'] = pd.to_datetime(df_merged['date'], dayfirst=True)

# Extract day of the week (0=Monday, 6=Sunday) and month
df_merged['day_of_week'] = df_merged['date'].dt.dayofweek
df_merged['month'] = df_merged['date'].dt.month

# Create interaction feature: duration_x_num_artists
df_merged['duration_x_num_artists'] = df_merged['duration_min'] * df_merged['num_artists']

# Create explicit_duration interaction feature
df_merged['explicit_duration'] = df_merged['is_explicit'] * df_merged['duration_min']

print("Engineered features 'day_of_week', 'month', 'duration_x_num_artists', 'explicit_duration' created.")

# --- 3.0 Predictive Modeling for Chart Success (No Engineering Features) ---
# Model 1: Logistic Regression
# Define the target variable: Chart Success (Top 10 vs. not Top 10)
df_merged['chart_success'] = (df_merged['position'] <= 10).astype(int)

# Select features for Logistic Regression
features_lr = ['duration_min', 'num_artists', 'is_explicit']
categorical_features_lr = ['album_type', 'duration_category']

df_temp_lr = df_merged[features_lr + categorical_features_lr].copy()
X_lr = pd.get_dummies(df_temp_lr, columns=categorical_features_lr, drop_first=True)
y_lr = df_merged['chart_success']

X_train_no_eng, X_test_no_eng, y_train_no_eng, y_test_no_eng = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42, stratify=y_lr)

model_lr = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
model_lr.fit(X_train_no_eng, y_train_no_eng)
y_pred_lr = model_lr.predict(X_test_no_eng)
lr_accuracy_no_eng = accuracy_score(y_test_no_eng, y_pred_lr)
metrics_df = pd.DataFrame(classification_report(y_test_no_eng, y_pred_lr, output_dict=True)).transpose()
metrics_df = metrics_df.drop(labels=['accuracy', 'macro avg', 'weighted avg'])
metrics_df.rename(index={'0': 'Class 0 (Not Top 10)', '1': 'Class 1 (Top 10)'}, inplace=True)
print("Logistic Regression (Without Engineering Features) model trained and metrics calculated.")

# --- Module 2: Linear Regression ---
linear_model_no_eng = LinearRegression()
linear_model_no_eng.fit(X_train_no_eng, y_train_no_eng)
linear_y_pred_no_eng = linear_model_no_eng.predict(X_test_no_eng)
Lr_accuracy_no_eng = accuracy_score(y_test_no_eng, (linear_y_pred_no_eng > 0.5).astype(int))
Lr_metrics_df_no_eng = pd.DataFrame(classification_report(y_test_no_eng, (linear_y_pred_no_eng > 0.5).astype(int), zero_division=0, output_dict=True)).transpose()
Lr_metrics_df_no_eng = Lr_metrics_df_no_eng.drop(labels=['accuracy', 'macro avg', 'weighted avg'])
Lr_metrics_df_no_eng.rename(index={'0': 'Class 0 (Not Top 10)', '1': 'Class 1 (Top 10)'}, inplace=True)

# --- Model 3: Random Forest (No Engineered Features) ---
# Use the same feature set as Logistic Regression for comparison without engineered features
rf_model_no_eng = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model_no_eng.fit(X_train_no_eng, y_train_no_eng)
rf_y_pred_no_eng = rf_model_no_eng.predict(X_test_no_eng)
rf_accuracy_no_eng = accuracy_score(y_test_no_eng, rf_y_pred_no_eng)
rf_metrics_df_no_eng = pd.DataFrame(classification_report(y_test_no_eng, rf_y_pred_no_eng, output_dict=True)).transpose()
rf_metrics_df_no_eng = rf_metrics_df_no_eng.drop(labels=['accuracy', 'macro avg', 'weighted avg'])
rf_metrics_df_no_eng.rename(index={'0': 'Class 0 (Not Top 10)', '1': 'Class 1 (Top 10)'}, inplace=True)
print("Random Forest model (no engineered features) trained and metrics calculated.")

# --- Model 4: XGBoost (No Engineered Features) ---
xgb_model_no_eng = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', 
    scale_pos_weight=(len(y_train_no_eng) - y_train_no_eng.sum()) / y_train_no_eng.sum()) # Handle class imbalance
xgb_model_no_eng.fit(X_train_no_eng, y_train_no_eng)
xgb_y_pred_no_eng = xgb_model_no_eng.predict(X_test_no_eng)
xgb_accuracy_no_eng = accuracy_score(y_test_no_eng, xgb_y_pred_no_eng)
xgb_metrics_df_no_eng = pd.DataFrame(classification_report(y_test_no_eng, xgb_y_pred_no_eng, output_dict=True)).transpose()
xgb_metrics_df_no_eng = xgb_metrics_df_no_eng.drop(labels=['accuracy', 'macro avg', 'weighted avg'])
xgb_metrics_df_no_eng.rename(index={'0': 'Class 0 (Not Top 10)', '1': 'Class 1 (Top 10)'}, inplace=True)

# --- 3.1 Predictive Modeling For Chart Success (With Engineered Features) ---
features_engineered_rf = [
    'duration_min', 'num_artists', 'is_explicit',
    'day_of_week', 'month', 'duration_x_num_artists', 'explicit_duration'
]
categorical_features_rf_eng = ['album_type', 'duration_category']

df_temp_engineered_rf = df_merged[features_engineered_rf + categorical_features_rf_eng].copy()
X_engineered_rf = pd.get_dummies(df_temp_engineered_rf, columns=categorical_features_rf_eng, drop_first=True)
y_engineered_rf = df_merged['chart_success']

X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(X_engineered_rf, y_engineered_rf, test_size=0.2, random_state=42, stratify=y_engineered_rf)

# --- Model 1: Logistic Regression ---
lr_model_eng = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
lr_model_eng.fit(X_train_eng, y_train_eng)
lr_y_pred_eng = lr_model_eng.predict(X_test_eng)
# Calculate accuracy
lr_accuracy_eng = accuracy_score(y_test_eng, lr_y_pred_eng)
lr_metrics_eng_df = pd.DataFrame(classification_report(y_test_eng, lr_y_pred_eng, output_dict=True)).transpose()
lr_metrics_eng_df = lr_metrics_eng_df.drop(labels=['accuracy', 'macro avg', 'weighted avg'])
lr_metrics_eng_df.rename(index={'0': 'Class 0 (Not Top 10)', '1': 'Class 1 (Top 10)'}, inplace=True)

# --- Model 2: Linear Regression ---
Lr_model_eng = LinearRegression()
Lr_model_eng.fit(X_train_eng, y_train_eng)
Lr_y_pred_eng = Lr_model_eng.predict(X_test_eng)
Lr_accuracy_eng = accuracy_score(y_test_eng, (Lr_y_pred_eng> 0.5).astype(int))
Lr_metrics_eng_df = pd.DataFrame(classification_report(y_test_eng, (Lr_y_pred_eng > 0.5).astype(int), zero_division=0, output_dict=True)).transpose()
Lr_metrics_eng_df = Lr_metrics_eng_df.drop(labels=['accuracy', 'macro avg', 'weighted avg'])
Lr_metrics_eng_df.rename(index={'0': 'Class 0 (Not Top 10)', '1': 'Class 1 (Top 10)'}, inplace=True)

# --- Model 3: Random Forest ---
rf_model_eng = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model_eng.fit(X_train_eng, y_train_eng)
rf_y_pred_eng = rf_model_eng.predict(X_test_eng)
rf_accuracy_eng = accuracy_score(y_test_eng, rf_y_pred_eng)
rf_metrics_eng_df = pd.DataFrame(classification_report(y_test_eng, rf_y_pred_eng, output_dict=True)).transpose()
rf_metrics_eng_df = rf_metrics_eng_df.drop(labels=['accuracy', 'macro avg', 'weighted avg'])
rf_metrics_eng_df.rename(index={'0': 'Class 0 (Not Top 10)', '1': 'Class 1 (Top 10)'}, inplace=True)
print("Random Forest model (with engineered features) trained and metrics calculated.")

# --- Model 4: XGBoost ---
xgb_model_eng = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', 
    scale_pos_weight=(len(y_train_eng) - y_train_eng.sum()) / y_train_eng.sum()) # Handle class imbalance
xgb_model_eng.fit(X_train_eng, y_train_eng)
xgb_y_pred_eng = xgb_model_eng.predict(X_test_eng)
xgb_accuracy_eng = accuracy_score(y_test_eng, xgb_y_pred_eng)
xgb_metrics_df_eng = pd.DataFrame(classification_report(y_test_eng, xgb_y_pred_eng, output_dict=True)).transpose()
xgb_metrics_df_eng = xgb_metrics_df_eng.drop(labels=['accuracy', 'macro avg', 'weighted avg'])
xgb_metrics_df_eng.rename(index={'0': 'Class 0 (Not Top 10)', '1': 'Class 1 (Top 10)'}, inplace=True)

# --- 3.3 Model Comparison DataFrames for All Models ---

def build_comparison_df(metrics_no_eng_df, metrics_eng_df, model_name):
    comparison_data = []
    metrics_no_eng_dict = metrics_no_eng_df.to_dict('index')
    metrics_eng_dict = metrics_eng_df.to_dict('index')

    for class_name_key in metrics_no_eng_dict:
        metrics = metrics_no_eng_dict[class_name_key]
        comparison_data.append({'Model': f'{model_name} (No Features)', 'Class': class_name_key, 'Metric': 'Precision', 'Score': metrics['precision']})
        comparison_data.append({'Model': f'{model_name} (No Features)', 'Class': class_name_key, 'Metric': 'Recall', 'Score': metrics['recall']})
        comparison_data.append({'Model': f'{model_name} (No Features)', 'Class': class_name_key, 'Metric': 'F1-score', 'Score': metrics['f1-score']})

    for class_name_key in metrics_eng_dict:
        metrics = metrics_eng_dict[class_name_key]
        comparison_data.append({'Model': f'{model_name} (Engineered Features)', 'Class': class_name_key, 'Metric': 'Precision', 'Score': metrics['precision']})
        comparison_data.append({'Model': f'{model_name} (Engineered Features)', 'Class': class_name_key, 'Metric': 'Recall', 'Score': metrics['recall']})
        comparison_data.append({'Model': f'{model_name} (Engineered Features)', 'Class': class_name_key, 'Metric': 'F1-score', 'Score': metrics['f1-score']})

    return pd.DataFrame(comparison_data)

# Build comparison DataFrames for each model
lr_comparison_df = build_comparison_df(metrics_df, lr_metrics_eng_df, "Logistic Regression")
linear_comparison_df = build_comparison_df(Lr_metrics_df_no_eng, Lr_metrics_eng_df, "Linear Regression")
rf_comparison_df = build_comparison_df(rf_metrics_df_no_eng, rf_metrics_eng_df, "Random Forest")
xgb_comparison_df = build_comparison_df(xgb_metrics_df_no_eng, xgb_metrics_df_eng, "XGBoost")

# Combine all into one master comparison DataFrame
all_models_comparison_df = pd.concat([lr_comparison_df, linear_comparison_df, rf_comparison_df, xgb_comparison_df], ignore_index=True)

print("All model comparison dataframe created.")

# Create comparison_df for plotting RF performance with/without engineered features
comparison_data = []

rf_metrics_no_eng_dict = rf_metrics_df_no_eng.to_dict('index')
rf_metrics_with_eng_dict = rf_metrics_eng_df.to_dict('index')

for class_name_key in rf_metrics_no_eng_dict:
    metrics = rf_metrics_no_eng_dict[class_name_key]
    comparison_data.append({'Model': 'RF (No Features)', 'Class': class_name_key, 'Metric': 'Precision', 'Score': metrics['precision']})
    comparison_data.append({'Model': 'RF (No Features)', 'Class': class_name_key, 'Metric': 'Recall', 'Score': metrics['recall']})
    comparison_data.append({'Model': 'RF (No Features)', 'Class': class_name_key, 'Metric': 'F1-score', 'Score': metrics['f1-score']})

for class_name_key in rf_metrics_with_eng_dict:
    metrics = rf_metrics_with_eng_dict[class_name_key]
    comparison_data.append({'Model': 'RF (Engineered Features)', 'Class': class_name_key, 'Metric': 'Precision', 'Score': metrics['precision']})
    comparison_data.append({'Model': 'RF (Engineered Features)', 'Class': class_name_key, 'Metric': 'Recall', 'Score': metrics['recall']})
    comparison_data.append({'Model': 'RF (Engineered Features)', 'Class': class_name_key, 'Metric': 'F1-score', 'Score': metrics['f1-score']})

comparison_df = pd.DataFrame(comparison_data)

# --- 3.4 Model Accuracy Comparison DataFrames to find best model ---
# Create a DataFrame for the summary table
accuracy_summary_df = pd.DataFrame({
    'Model': [
        'Logistic Regression (No Features)', 'Logistic Regression (With Engineered Features)', 'Random Forest (No Features)', 
        'Random Forest (With Engineered Features)', 'Linear Regression (With Engineered Features)', 'Linear Regression (No Engineered Features)',
        'XGBoost (With Engineered Features)', 'XGBoost (Without Engineered Features)'
    ],
    'Accuracy': [
        lr_accuracy_no_eng, lr_accuracy_eng, rf_accuracy_no_eng, rf_accuracy_eng, Lr_accuracy_eng, Lr_accuracy_no_eng, xgb_accuracy_eng, xgb_accuracy_no_eng
    ]
})
print("Model comparison dataframes created.")

# --- 4. Time Series Data (`unique_artists_per_day`) (from Section II, needed for dashboard) ---
unique_artists_per_day = filtered_df.groupby('date')['artist'].nunique()
print("Unique artists per day calculated for Time Series Analysis.")

# --- 5. Genre Prediction Function and Application (from Section XV) ---
# Conceptual definition of major genres
major_genres = ['Pop', 'Rock', 'Hip-Hop/Rap', 'Jazz', 'Country',
                'Classical', 'Dance', 'R&B/soul', 'Electronic/EDM', 'Folk',
                'Metal', 'Blues', 'Reggae', 'Instrumental', 'Indie',
                'Gospel', 'Punk', 'Latin', 'Afrobeats', 'World Music']
# Load CLIP model from openai
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def predict_genre_from_image_ai_conceptual(image_url):
    """
    Predict genre from album cover URL using CLIP zero-shot classification.
    Always returns one of the major_genres.
    """
    if pd.isna(image_url) or not isinstance(image_url, str) or not image_url.startswith('http'):
        return 'Unknown'
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Compare image against all genre prompts
        # inputs = processor(text=major_genres, images=image, return_tensors="pt", padding=True)
        text_inputs = processor.tokenizer(
            major_genres,
            padding=True,
            return_tensors="pt"
        )
        image_inputs = processor.image_processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            **text_inputs,
            **image_inputs
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        predicted_idx = probs.argmax(-1).item()
        return major_genres[predicted_idx]
    except Exception:
        return "Unknown"

# --- Initialize or reuse genre mapping in Streamlit session state ---
if "genre_mapping" not in st.session_state:
    st.session_state["genre_mapping"] = {}

@st.cache_data(show_spinner=False)
def build_genre_mapping(unique_urls, existing_mapping):
    """
    Predict genres only for new album cover URLs not already in mapping.
    Shows a Streamlit progress bar with percentage in the UI,
    and a tqdm progress bar in the console logs.
    """
    mapping = dict(existing_mapping)
    progress_bar = st.progress(0)
    progress_text_success = st.empty()
    progress_text_info = st.empty()
    total = len(unique_urls)

    # tqdm progress bar in console
    for i, url in enumerate(tqdm(unique_urls, desc="Genre prediction progress", unit="track")):
        if url not in mapping:  # only predict if not already cached
            mapping[url] = predict_genre_from_image_ai_conceptual(url)

        percent_complete = int(((i + 1) / total) * 100)

        with tab2:
            # Update Streamlit UI
            progress_text_success.success(f"✅ In Progress: {percent_complete}%")
            progress_bar.progress((i + 1) / total)
            progress_text_info.info("⏳ Wait for about 2–3 minutes while we prepare your recommendational analysis tab.")


    # Clear progress bar and text after completion
    progress_bar.empty()
    progress_text_success.empty()
    progress_text_info.empty()

    # Optional: show completion message in UI
    print("✅ Genre prediction complete!")

    return mapping

# --- Build or update mapping based on current filters ---
unique_album_covers = df_merged['album_cover_url'].dropna().unique().tolist()
st.session_state["genre_mapping"] = build_genre_mapping( unique_album_covers, st.session_state["genre_mapping"]) # type: ignore

# Apply to full dataset
df_merged['genre'] = df_merged['album_cover_url'].map(st.session_state["genre_mapping"])
print("Conceptual AI-driven genre prediction function defined.\nGenre Prediction is in progress and may take some time (2-3 minutes)")

# --- Genre Filter (20 major genres) ---
with st.spinner("⏳ Loading genre filter... this may take a few minutes"):
    selected_genres = st.sidebar.multiselect(
        'Filter by Genre',
        options=major_genres,
        default=major_genres
    )

if selected_genres and 'genre' in df_merged.columns:
    mask &= df_merged['genre'].isin(selected_genres)

# --- When creating filtered_df, inherit the genre column directly using mask ---
filtered_df = df_merged.loc[mask].copy()
print("Genre prediction applied to your filtered dataset.")

is_genre_filter_different = (set(selected_genres) != set(major_genres))
is_any_filter_different = (
    is_date_range_different or is_artist_filter_different or is_track_type_different or is_album_type_different 
    or is_duration_filter_different or is_popularity_filter_different or is_genre_filter_different
)

# --- 5.1 Genre-Specific Analysis DataFrames (from Section XVI) ---
genre_popularity_stats = filtered_df.groupby('genre')['popularity'].agg(['mean', 'median', 'std']).sort_values(by='mean', ascending=False)
genre_explicitness_percentage = filtered_df.groupby('genre')['is_explicit'].mean() * 100
genre_duration_stats = filtered_df.groupby('genre')['duration_min'].agg(['mean', 'median', 'std']).sort_values(by='mean', ascending=False)
print("✅ Genre-specific statistics calculated.")

print("--- All required dataframes and variables are now prepared. ---")

with tab2:
    # --- Dashboard Title and Introduction ---
    st.title("Recommendational Analysis Dashboard For UK Music Market Listeners")
    st.markdown("""
    This dashboard presents key insights and recommendations from the UK's Music Market Analysis,
    leveraging our data validation, descriptive analysis, and predictive modeling capabilities.
    """)
    # Tabs for each subheader
    tabs = st.tabs([
        "📈 Predictive Modelling",
        "🛠️ Feature Engineering",
        "⏳ Time Series",
        "🎶 Genre Analysis",
        "📊 Multivariate Analysis",
        "📌 Conclusion"
    ])
    with tabs[0]:
        st.subheader("**Predictive Modeling of Chart Success**")
        st.markdown("""
        Explore how different models perform in predicting Top 10 chart success.
        Use the filter below to select a model and view its performance metrics.
        """)
    
        # --- Model Selection Filter ---
        model_choice = st.selectbox(
            "Select a Model to View Metrics",
            ["Logistic Regression", "Linear Regression", "Random Forest", "XGBoost"]
        )
    
        # --- Helper function to plot heatmap ---
        def plot_heatmap(metrics_df, title):
            fig, ax = plt.subplots(figsize=(6,4))
            sns.heatmap(metrics_df.drop(columns=['support'], errors='ignore'), 
                        annot=True, cmap="viridis", fmt=".2f", ax=ax)
            ax.set_title(title)
            st.pyplot(fig)
            plt.close(fig)
    
        # --- Dictionary mapping model names to their metric DataFrames ---
        model_metrics = {
            "Logistic Regression": (metrics_df, lr_metrics_eng_df, lr_comparison_df),
            "Linear Regression": (Lr_metrics_df_no_eng, Lr_metrics_eng_df, linear_comparison_df),
            "Random Forest": (rf_metrics_df_no_eng, rf_metrics_eng_df, rf_comparison_df),
            "XGBoost": (xgb_metrics_df_no_eng, xgb_metrics_df_eng, xgb_comparison_df)
        }
    
        # Get the correct DataFrames
        metrics_no_eng, metrics_eng, comparison_df_model = model_metrics[model_choice]
    
        # --- Display Classification Report (Heatmaps + Tables) ---
        colA, colB = st.columns(2)
        st.write(f"**{model_choice} Model Performance Comparison:**")
        with colA:
            with st.expander(f"{model_choice} Performance (Without Engineered Features):"):
                plot_heatmap(metrics_no_eng, f"{model_choice} (No Features)")
                st.dataframe(metrics_no_eng.drop(columns=['support'], errors='ignore'))
        with colB:
            with st.expander(f"{model_choice} Performance (With Engineered Features):"):
                plot_heatmap(metrics_eng, f"{model_choice} (Engineered Features)")
                st.dataframe(metrics_eng.drop(columns=['support'], errors='ignore'))
    
        # --- Display Comparison Plot (Chart) ---
        if comparison_df_model is not None:
            g = sns.catplot(
                x='Metric', y='Score', hue='Model', col='Class',
                data=comparison_df_model, kind='bar', palette='viridis',
                errorbar=None, col_wrap=2, height=6, aspect=1.2
            )
            g.fig.suptitle(f'{model_choice} Performance Comparison: With vs. Without Engineered Features', y=1.08, fontsize=18)
            g.set_axis_labels('Metric', 'Score')
            g.set_titles('Class: {col_name}')
            # Force legend creation
            g.add_legend()
            # Reposition legend: center top, just below title
            legend = g.fig.legends[0]   # first legend object in the figure
            legend.set_bbox_to_anchor((0.5, 1.02))   # center horizontally, slightly above plots
            legend.set_loc("upper center")  
            plt.ylim(0, 1)
            plt.tight_layout(rect=(0, 0, 1, 0.95))
            st.pyplot(g.fig)
        else:
            st.warning("Comparison DataFrame not found. Please ensure the model comparison section was run.")
    
        # --- Fixed Accuracy Comparison Table ---
        if 'accuracy_summary_df' in locals():
            # Summary table of model accuracies
            with st.expander("📊 Model Accuracy Comparison (All Models)"):
                st.dataframe(accuracy_summary_df)
        
            # Add the bar chart for model accuracies
            fig_accuracy_comp, ax_accuracy_comp = plt.subplots(figsize=(10, 6), constrained_layout=True)
            sns.barplot(x='Model', y='Accuracy', data=accuracy_summary_df, palette='viridis', hue='Model', legend=False, ax=ax_accuracy_comp)
            ax_accuracy_comp.set_title('Comparison of Model Accuracies')
            ax_accuracy_comp.set_xlabel('Model')
            ax_accuracy_comp.set_ylabel('Accuracy Score')
            ax_accuracy_comp.set_ylim(0, 1)  # Accuracy scores range from 0 to 1
            plt.xticks(rotation=45, ha='right')
        
            st.pyplot(fig_accuracy_comp)
            plt.close(fig_accuracy_comp)
        else:
            st.warning("`accuracy_summary_df` not found. Please ensure the model accuracy summary section was run.")
    
        # --- Your Plotly figure construction code ---
        metric_types_to_plot = ['F1-score', 'Precision', 'Recall']
        metric_color_map = {'F1-score': 'yellow', 'Precision': 'red', 'Recall': 'green'}
        
        # Map models and classes to numeric indices for plotting
        model_names_unique = all_models_comparison_df['Model'].unique()
        class_labels = all_models_comparison_df['Class'].unique()
        
        model_map = {m: i for i, m in enumerate(model_names_unique)}
        class_map = {c: i for i, c in enumerate(class_labels)}
        
        # Add index columns
        plot_df = all_models_comparison_df.copy()
        plot_df['Model_Idx'] = plot_df['Model'].map(model_map)
        plot_df['Class_Idx'] = plot_df['Class'].map(class_map)
        plot_df['Metric Type'] = plot_df['Metric']  # rename for clarity
    
        fig = go.Figure()
        surface_traces_list = []
        metric_matrices = []
    
        for i, metric_type_to_plot in enumerate(metric_types_to_plot):
            metric_plot_df = plot_df[plot_df['Metric Type'] == metric_type_to_plot].copy()
            metric_matrix = metric_plot_df.pivot_table(index='Model_Idx', columns='Class_Idx', values='Score')
            metric_matrix = metric_matrix.reindex(index=np.arange(len(model_names_unique)), columns=np.arange(len(class_labels)))
            metric_matrices.append(metric_matrix)
    
            X_model_indices = metric_matrix.index.values
            Y_class_indices = metric_matrix.columns.values
            Z_scores = metric_matrix.values.T
            X, Y = np.meshgrid(X_model_indices, Y_class_indices)
    
            current_color = metric_color_map[metric_type_to_plot]
            surface_trace = go.Surface(
                x=X, y=Y, z=Z_scores,
                colorscale=[[0, current_color], [1, current_color]],
                showscale=False, opacity=0.8, name=metric_type_to_plot,
                hovertemplate='<b>Model:</b> %{x}<br><b>Class:</b> %{y}<br><b>' + metric_type_to_plot + ':</b> %{z:.2f}<extra></extra>'
            )
            surface_traces_list.append(surface_trace)
    
        # Wall trace
        wall_x = np.array([X_model_indices.min(), X_model_indices.max()])
        wall_z = np.array([0, 1])
        wall_X_grid, wall_Z_grid = np.meshgrid(wall_x, wall_z)
        wall_Y_grid = np.full_like(wall_X_grid, 0.5, dtype=float)
        wall_trace = go.Surface(
            x=wall_X_grid, y=wall_Y_grid, z=wall_Z_grid,
            colorscale=[[0, 'grey'], [1, 'white']], opacity=0.3,
            showscale=False, name='Class Separator Wall'
        )
    
        # Intersection lines
        intersection_line_traces_list = []
        for i, metric_type_to_plot in enumerate(metric_types_to_plot):
            metric_matrix = metric_matrices[i]
            line_X_coords = X_model_indices
            line_Y_coords = np.full_like(line_X_coords, 0.5, dtype=float)
            line_Z_intersection_for_metric = (metric_matrix.iloc[:, 0] + metric_matrix.iloc[:, 1]) / 2
    
            intersection_line_trace = go.Scatter3d(
                x=line_X_coords, y=line_Y_coords, z=line_Z_intersection_for_metric,
                mode='lines', line=dict(color='black', width=3, dash='dot'),
                name=f"{metric_type_to_plot} Intersection Line",
                hovertemplate='<b>Model:</b> %{x}<br><b>Class Separator Y:</b> 0.5<br><b>' + metric_type_to_plot + ' Intersection:</b> %{z:.2f}<extra></extra>'
            )
            intersection_line_traces_list.append(intersection_line_trace)
    
        # Add traces
        for trace in surface_traces_list:
            fig.add_trace(trace)
        fig.add_trace(wall_trace)
        for trace in intersection_line_traces_list:
            fig.add_trace(trace)
    
        # Dropdown buttons
        total_traces = len(list(fig.data))
        buttons = []
        visible_all = [True] * total_traces
        buttons.append(dict(label="All Metrics", method="update",
                            args=[{"visible": visible_all}, {"scene.zaxis.title.text": "Score", "title.text": "Interactive Combined 3D Surface Plot (All Metrics)"}]))
        for i, metric_name in enumerate(metric_types_to_plot):
            visible_args = [False] * total_traces
            visible_args[i] = True
            visible_args[3] = True
            visible_args[4+i] = True
            buttons.append(dict(label=metric_name, method="update",
                                args=[{"visible": visible_args}, {"scene.zaxis.title.text": metric_name, "title.text": f"Interactive Combined 3D Surface Plot ({metric_name})"}]))
    
        # Layout
        fig.update_layout(
            title={'text': 'Interactive Combined 3D Surface Plot of Model Performance', 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
            scene=dict(
                xaxis_title='Model',
                yaxis_title='Class',
                zaxis_title='Score',
                xaxis=dict(tickvals=X_model_indices, ticktext=[model_names_unique[i] for i in X_model_indices]),
                yaxis=dict(tickvals=Y_class_indices, ticktext=class_labels[Y_class_indices]),
                zaxis=dict(range=[0, 1])
            ),
            autosize=False, width=1200, height=800,
            margin=dict(l=65, r=50, b=65, t=90),
            updatemenus=[dict(type="dropdown", direction="down", x=0.0, y=1.15, showactive=True, buttons=buttons)]
        )
    
        # --- Streamlit rendering ---
        st.plotly_chart(fig, width='stretch')
    
        st.markdown('---')
    with tabs[1]:
        # --- Section 2: Feature Engineering Visualizations (Relevant to Chart Success) ---
        st.subheader("Feature Engineering Visualizations for Chart Success")
        st.markdown("""
        Visualizations of engineered features provide insights into their relationship with chart success.
        """)
        
        st.write("**Chart Success by Day of the Week:**")
        fig_day_of_week, ax_day_of_week = plt.subplots(figsize=(10, 6), constrained_layout=True)
        sns.countplot(x='day_of_week', hue='chart_success', data=df_merged, palette='viridis', ax=ax_day_of_week)
        ax_day_of_week.set_title('Chart Success by Day of the Week')
        ax_day_of_week.set_xlabel('Day of Week (0=Monday, 6=Sunday)')
        ax_day_of_week.set_ylabel('Number of Tracks')
        ax_day_of_week.legend(title='Chart Success (0=No, 1=Yes)')
        ax_day_of_week.set_xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        st.pyplot(fig_day_of_week)
        plt.close(fig_day_of_week)
        
        with st.expander("ℹ️ More Information"):
            st.info("- Chart success varies by day, with certain weekdays showing higher Top 10 entries, indicating optimal release timing in the UK market.")
            st.info("- Weekly patterns may reflect audience engagement cycles and promotional strategies in UK music consumption.")
        
        st.write("**Interaction of Duration and Number of Artists vs. Popularity:**")
        fig_duration_x_artists, ax_duration_x_artists = plt.subplots(figsize=(12, 7), constrained_layout=True)
        sns.scatterplot(x='duration_x_num_artists', y='popularity', hue='chart_success', data=df_merged, palette='coolwarm', alpha=0.6, ax=ax_duration_x_artists)
        ax_duration_x_artists.set_title('Interaction of Duration and Number of Artists vs. Popularity by Chart Success')
        ax_duration_x_artists.set_xlabel('Duration (min) * Number of Artists')
        ax_duration_x_artists.set_ylabel('Popularity')
        ax_duration_x_artists.legend(title='Chart Success (0=No, 1=Yes)')
        
        st.pyplot(fig_duration_x_artists)
        plt.close(fig_duration_x_artists)
        
        with st.expander("ℹ️ More Information"):
            st.info("- Longer durations with more artists correlate with higher popularity, suggesting collaborative extended tracks appeal to UK listeners.")
            st.info("- Interaction effects highlight how production complexity influences chart performance in the UK market.")
        
        st.write("**Distribution of Explicit Track Duration by Chart Success:**")
        fig_explicit_duration, ax_explicit_duration = plt.subplots(figsize=(8, 6), constrained_layout=True)
        sns.violinplot(x='chart_success', y='explicit_duration', data=df_merged, palette='Set2', hue='chart_success', legend=False, ax=ax_explicit_duration)
        ax_explicit_duration.set_title('Distribution of Explicit Track Duration by Chart Success')
        ax_explicit_duration.set_xlabel('Chart Success (0=No, 1=Yes)')
        ax_explicit_duration.set_ylabel('Explicit Duration (minutes)')
        ax_explicit_duration.set_xticks(ticks=[0, 1], labels=['Not Top 10', 'Top 10'])
        
        st.pyplot(fig_explicit_duration)
        plt.close(fig_explicit_duration)
        
        with st.expander("ℹ️ More Information"):
            st.info("- Explicit tracks in Top 10 tend to be shorter, balancing maturity with concise delivery for UK audience preferences.")
            st.info("- Duration-explicitness interplay reveals content strategy nuances in achieving UK chart success.")
        
        st.markdown('---')
    with tabs[2]:
        # --- Recommendation 3: Time Series Analysis of Trends ---
        st.subheader("**Time Series Analysis of Trends**")
        st.markdown("""
        Analyzing trends over time can reveal seasonality, shifts in artist dominance, or changes in content preferences.
        Here's a look at the number of unique artists appearing in the Top 50 chart each day.
        """)
        
        if 'unique_artists_per_day' in locals():
            st.write("**Daily Unique Artists in Top 50:**")
        
            # Use constrained_layout=True to avoid tight_layout warnings
            fig_unique_artists, ax_unique_artists = plt.subplots(figsize=(12, 6), constrained_layout=True)
            unique_artists_per_day_df = unique_artists_per_day.reset_index()
            # Assuming the date format is 'DD-MM-YYYY' based on previous processing of df['date']
            unique_artists_per_day_df['date'] = pd.to_datetime(unique_artists_per_day_df['date'], dayfirst=True)
            unique_artists_per_day_df = unique_artists_per_day_df.sort_values('date')  # Sort by date for correct line plot
            sns.lineplot(x='date', y='artist', data=unique_artists_per_day_df, ax=ax_unique_artists)
            ax_unique_artists.set_title('Number of Unique Artists in Top 50 Per Day')
            ax_unique_artists.set_xlabel('Date')
            ax_unique_artists.set_ylabel('Number of Unique Artists')
            plt.xticks(rotation=45)
            st.pyplot(fig_unique_artists)
            plt.close(fig_unique_artists)
        else:
            st.warning("`unique_artists_per_day` data not found. Please ensure the artist dominance analysis section was run.")
        
        with st.expander("ℹ️ More Information"):
            st.info("- Unique artist counts fluctuate over time, revealing periods of high diversity versus concentration in UK Top 50 charts.")
            st.info("- Temporal trends may indicate market saturation, new entries, or seasonal influences on UK music landscape.")
        
        st.markdown('---')
    with tabs[3]:
        # --- Recommendation 4: Genre-Specific Analysis ---
        st.subheader("Genre-Specific Analysis (Conceptual)")
        st.markdown("""
        While genre prediction is currently conceptual (using CLIPModel & CLIPProcessor from OpenAI) and sensitive in nature (for prediction of genre (based on album_cover_url column as a proxy for images)),
        we can explore how different genres might relate to popularity, explicitness, and duration.
        """)
        
        if 'genre_popularity_stats' in locals() and not df_merged.empty:
            st.write("**Genre vs. Popularity:**")
            fig_genre_pop, ax_genre_pop = plt.subplots(figsize=(14, 7), constrained_layout=True)
            sns.boxplot(
                x='genre', y='popularity', data=df_merged,
                palette='coolwarm', hue='genre', legend=False,
                order=genre_popularity_stats.index, ax=ax_genre_pop
            )
            ax_genre_pop.set_title('Popularity Distribution by Genre')
            ax_genre_pop.set_xlabel('Genre')
            ax_genre_pop.set_ylabel('Popularity Score')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_genre_pop)
            plt.close(fig_genre_pop)
        
            if not genre_popularity_stats.empty:
                top_genre = genre_popularity_stats.index[0]
                bottom_genre = genre_popularity_stats.index[-1]
                top_pop = genre_popularity_stats['mean'].iloc[0]
                bottom_pop = genre_popularity_stats['mean'].iloc[-1]
                with st.expander("ℹ️ More Information"):
                    st.info(f"- **{top_genre}** leads in **popularity** with an average of **{top_pop:.1f}**, while **{bottom_genre}** ranks lowest at **{bottom_pop:.1f}**, showing clear **genre-based listener appeal** differences in the current filtered UK market.")
                    st.info("- Popularity variations by genre may stem from cultural trends, marketing strategies, or demographic preferences in the UK.")
        else:
            st.warning("`genre_popularity_stats` or `df_merged` not found. Please ensure the genre analysis section was run.")
        
        if 'genre_explicitness_percentage' in locals() and not df_merged.empty:
            st.write("**Genre vs. Explicitness:**")
            fig_genre_exp, ax_genre_exp = plt.subplots(figsize=(14, 7), constrained_layout=True)
            sns.barplot(
                x=genre_explicitness_percentage.index,
                y=genre_explicitness_percentage.values,
                palette='viridis', hue=genre_explicitness_percentage.index,
                legend=False,
                order=genre_explicitness_percentage.sort_values(ascending=False).index,
                ax=ax_genre_exp
            )
            ax_genre_exp.set_title('Percentage of Explicit Content by Genre')
            ax_genre_exp.set_xlabel('Genre')
            ax_genre_exp.set_ylabel('Percentage Explicit (%)')
            ax_genre_exp.set_ylim(0, 100)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_genre_exp)
            plt.close(fig_genre_exp)
        
            if not genre_explicitness_percentage.empty:
                most_explicit_genre = genre_explicitness_percentage.idxmax()
                least_explicit_genre = genre_explicitness_percentage.idxmin()
                most_explicit_pct = genre_explicitness_percentage.max()
                least_explicit_pct = genre_explicitness_percentage.min()
                with st.expander("ℹ️ More Information"):
                    st.info(f"- **{most_explicit_genre}** shows the highest **content maturity** at **{most_explicit_pct:.1f}%** explicit tracks, while **{least_explicit_genre}** is most **family-friendly** at **{least_explicit_pct:.1f}%**, reflecting distinct **audience expectations** across genres in the current filtered dataset.")
                    st.info("- Explicitness differences highlight genre-specific cultural norms and target audience demographics in UK music.")
        else:
            st.warning("`genre_explicitness_percentage` or `df_merged` not found. Please ensure the genre analysis section was run.")
        
        if 'genre_duration_stats' in locals() and not df_merged.empty:
            st.write("**Genre vs. Duration:**")
            fig_genre_dur, ax_genre_dur = plt.subplots(figsize=(14, 7), constrained_layout=True)
            sns.boxplot(
                x='genre', y='duration_min', data=df_merged,
                palette='plasma', hue='genre', legend=False,
                order=genre_duration_stats.index, ax=ax_genre_dur
            )
            ax_genre_dur.set_title('Track Duration Distribution by Genre')
            ax_genre_dur.set_xlabel('Genre')
            ax_genre_dur.set_ylabel('Duration (minutes)')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_genre_dur)
            plt.close(fig_genre_dur)
        
            if not genre_duration_stats.empty:
                longest_genre = genre_duration_stats.index[0]
                shortest_genre = genre_duration_stats.index[-1]
                longest_dur = genre_duration_stats['mean'].iloc[0]
                shortest_dur = genre_duration_stats['mean'].iloc[-1]
                with st.expander("ℹ️ More Information"):
                    st.info(f"- **{longest_genre}** dominates with longer **track lengths** (avg **{longest_dur:.2f}** min), while **{shortest_genre}** favors concise content (avg **{shortest_dur:.2f}** min), highlighting **genre-specific listening patterns** and production conventions in the UK market.")
                    st.info("- Duration preferences by genre may reflect traditional formats, audience attention spans, or production styles in UK music culture.")
        else:
            st.warning("`genre_duration_stats` or `df_merged` not found. Please ensure the genre analysis section was run.")
        
        # Define a list of major genres and their definitions
        genre_definitions = {
            "Pop": "Mainstream, catchy, and melodic songs designed for mass appeal.",
            "Rock": "Guitar-driven music with strong rhythms, often associated with rebellion and energy.",
            "Hip-Hop/Rap": "Rhythm-focused music featuring spoken rhymes, storytelling, and social commentary.",
            "Jazz": "Improvisational music with complex harmonies and swing rhythms.",
            "Country": "Storytelling songs rooted in rural life, often using acoustic instruments.",
            "Classical": "Structured compositions from orchestral traditions across historical periods.",
            "Dance": "Upbeat, rhythmic tracks designed for clubs and parties.",
            "R&B/Soul": "Smooth, emotive music blending rhythm and blues with soulful vocals.",
            "Electronic/EDM": "Synthesizer-driven music with heavy beats and drops.",
            "Folk": "Acoustic, traditional storytelling music rooted in cultural heritage.",
            "Metal": "Aggressive, loud music with distorted guitars and powerful drumming.",
            "Blues": "Emotional, soulful music built on 12-bar progressions.",
            "Reggae": "Jamaican-origin music with offbeat rhythms and relaxed grooves.",
            "Instrumental": "Music without vocals, focusing purely on instruments and melodies.",
            "Indie": "Independent, often experimental music outside mainstream labels.",
            "Gospel": "Christian religious music emphasizing vocal harmonies and worship.",
            "Punk": "Fast, raw rock music with anti-establishment themes.",
            "Latin": "Music rooted in Latin American rhythms and styles.",
            "Afrobeats": "Contemporary African pop blending traditional rhythms with modern influences.",
            "World Music": "Traditional and contemporary music from diverse cultures worldwide."
        }
        
        # Convert dictionary to DataFrame
        df_genres = pd.DataFrame(list(genre_definitions.items()), columns=["Genre", "Definition"])
        # Display in Streamlit
        with st.expander("ℹ️ General Knowledge Regarding The Context"):
            st.subheader("🎵 Major Song Genres and Definitions")
            st.table(df_genres)
        
        st.markdown('---')
    with tabs[4]:
        # --- Recommendation 5: Multivariate Analysis ---
        st.subheader("Multivariate Analysis (3D Scatter Plot)")
        st.markdown("""
        This 3D scatter plot visualizes the interplay between track duration, number of artists, and popularity,
        with points colored by chart success and distinguished by duration category (short-form vs. long-form).
        """)
        
        if not df_merged.empty and all(col in df_merged.columns for col in ['duration_min', 'num_artists', 'popularity', 'chart_success', 'duration_category']):
            # Use constrained_layout=True to avoid tight_layout warnings
            fig_3d_scatter = plt.figure(figsize=(14, 12), constrained_layout=True)
            ax_3d_scatter = fig_3d_scatter.add_subplot(111, projection='3d')
        
            # Plot the scatter points, using mapped markers
            for chart_success_val, chart_success_color in {0: 'blue', 1: 'red'}.items():
                for duration_cat_name, duration_cat_marker in {'short-form': 'o', 'long-form': '^'}.items():
                    subset = df_merged[
                        (df_merged['chart_success'] == chart_success_val) &
                        (df_merged['duration_category'] == duration_cat_name)
                    ]
                    ax_3d_scatter.scatter(
                        subset['duration_min'], subset['num_artists'], subset['popularity'], c=chart_success_color, 
                        marker=duration_cat_marker, alpha=0.6, s=15,
                        label=f'Chart Success: {chart_success_val}, Duration: {duration_cat_name}' if chart_success_val == 0 else "_nolegend_"  # Only add legend once for each type
                    )
        
            ax_3d_scatter.set_xlabel('Duration (minutes)')
            ax_3d_scatter.set_ylabel('Number of Artists')
            ax_3d_scatter.set_zlabel('Popularity')
            ax_3d_scatter.set_title('3D Scatter Plot: Duration, Artists, Popularity by Chart Success & Duration Category')
        
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Short-form', markerfacecolor='gray', markersize=10),
                Line2D([0], [0], marker='^', color='w', label='Long-form', markerfacecolor='gray', markersize=10),
                mpatches.Patch(color='blue', label='Not Top 10'), mpatches.Patch(color='red', label='Top 10')
            ]
            ax_3d_scatter.legend(handles=legend_elements, title='Legend')
        
            # Render and close cleanly
            st.pyplot(fig_3d_scatter)
            plt.close(fig_3d_scatter)
        
            with st.expander("ℹ️ More Information"):
                st.info("- Top 10 tracks cluster with moderate durations and collaborations, while non-Top 10 spread across longer solos, showing success patterns in UK charts.")
                st.info("- Multivariate clusters indicate optimal combinations of duration, artists, and popularity for UK market penetration.")
        else:
            st.warning("Required data for 3D scatter plot not found. Please ensure the multivariate analysis section was run.")
        
        st.markdown('---')
    with tabs[5]:
        st.subheader("📌 Conclusion")
        
        if len(filtered_df) == 0:
            with st.container():
                st.warning("Please reload the website and wait for the OpenAI Model to predict and calculate Genre Specific Statistics for the entire date-range filter option. Then select any required date-range to see the conclusion.")
        else:
            with st.container():
                st.info("This project provides both structural and cultural intelligence into the UK music market by comparing the current filter view with the full dataset baseline. Recommendations balance the selected subset with the overall UK market context.")
        
                if is_any_filter_different:
                    # Headline metrics in columns
                    col1, col2, col3 = st.columns(3)
                    concentration_trend = 'more concentrated' if filtered_artist_concentration_index > artist_concentration_index else 'less concentrated' if filtered_artist_concentration_index < artist_concentration_index else 'similarly concentrated'
                    explicit_trend = 'higher' if filtered_explicitness_percentage.get(True, 0) > explicitness_percentage.get(True, 0) else 'lower' if filtered_explicitness_percentage.get(True, 0) < explicitness_percentage.get(True, 0) else 'the same'
                    duration_trend = 'shorter' if filtered_short_form_pct >= overall_short_form_pct else 'longer'
        
                    col1.metric("Artist Concentration", f"{filtered_artist_concentration_index:.2f}%", f"{filtered_artist_concentration_index - artist_concentration_index:.2f}% vs baseline")
                    col2.metric("Explicit Content %", f"{filtered_explicitness_percentage.get(True, 0):.2f}%", f"{filtered_explicitness_percentage.get(True, 0) - explicitness_percentage.get(True, 0):.2f}% vs baseline")
                    col3.metric("Avg Duration (min)", f"{filtered_df['duration_min'].mean():.2f}", f"{filtered_df['duration_min'].mean() - df_merged['duration_min'].mean():.2f} vs baseline")
        
                    # Narrative cards
                    st.write(f"- The present filtered view is **{concentration_trend}** than the entire dataset.")
                    st.write(f"- Content explicitness is **{explicit_trend}** than the full dataset baseline.")
                    st.write(f"- The selected subset remains **{duration_trend}** in average track length compared to the overall UK market.")
        
                    # Genre composition by popularity
                    with st.expander("🎶 Genre Popularity Insights"):
                        overall_genre_popularity = df_merged.groupby('genre')['popularity'].mean().sort_values(ascending=False)
                        filtered_genre_popularity = filtered_df.groupby('genre')['popularity'].mean().sort_values(ascending=False)
                        overall_top3 = overall_genre_popularity.head(3).index.tolist()
                        filtered_top3 = filtered_genre_popularity.head(3).index.tolist()
        
                        if overall_top3 and filtered_top3:
                            st.write(f"- Overall Top 3 Genres: **{', '.join(overall_top3)}**")
                            st.write(f"- Filtered Top 3 Genres: **{', '.join(filtered_top3)}**")
                            if overall_top3 == filtered_top3:
                                st.success("Subset mirrors full dataset popularity trends → stable listener preferences.")
                            else:
                                st.info("Subset shows a different popularity mix → highlights emerging trends or niche preferences.")
        
                    # Predictive modeling
                    if 'rf_accuracy_eng' in locals() and 'rf_accuracy' in locals():
                        better_model = 'with engineered features' if rf_accuracy_eng >= rf_accuracy_no_eng else 'without engineered features'
                        with st.expander("📊 Predictive Modeling"):
                            st.write(f"- Random Forest model **{better_model}** performs stronger for the current UK market slice.")
                            st.markdown("- Reinforces the value of feature engineering in chart success forecasting.")
        
                    st.info("Together, these dynamic conclusions highlight whether the current filter view reflects a representative market slice or a distinctive subsegment with unique preferences.")
        
                else:
                    # Baseline case
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Artist Concentration", f"{filtered_artist_concentration_index:.2f}%")
                    col2.metric("Explicit Content %", f"{filtered_explicitness_percentage.get(True, 0):.2f}%")
                    col3.metric("Avg Duration (min)", f"{filtered_df['duration_min'].mean():.2f}")
        
                    overall_genre_popularity = df_merged.groupby('genre')['popularity'].mean().sort_values(ascending=False)
                    overall_top3 = overall_genre_popularity.head(3).index.tolist()
                    if overall_top3:
                        st.write(f"- Top 3 Genres by popularity: **{', '.join(overall_top3)}**, representing the most appealing genres in the UK music market.")
        
                    st.info("These baseline metrics provide a comprehensive view of the UK music market dynamics.")
        
            st.success("✅ The dashboard is useful for Atlantic Recording Corporation to identify UK listener preference indicators, collaboration strengths, and content composition trends in real time.")
        
            st.markdown("### 🎶 Memorable Ending")
            st.markdown("""
            In the rhythm of data and the melody of insights, this dashboard transforms analytics into **market intelligence**.  
            Just as every chart-topping track finds its perfect beat, every strategy here finds its **perfect note** —  
            guiding the UK music industry toward harmony between artistry and audience.
            """)
            
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: grey; font-size: 14px;'>"
            "🔖 Dashboard created by <b>Prathamesh Bhurke</b>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")
        #st.markdown(
        #    """
        #    <div style='text-align: center; color: grey; font-size: 14px;'>
        #        🔖 Dashboard created by <b>Prathamesh Bhurke</b><br>
        #        <a href="https://github.com/Prathamesh666/United-Kingdom-Music-Market-Structure-Artist-Diversity-Content-Localization-Analysis" target="_blank">
        #            📂 GitHub Repository
        #        </a>
        #    </div>
        #    """,
        #    unsafe_allow_html=True
        #)