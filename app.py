import streamlit as st
import pandas as pd
import numpy as np
import itertools, collections
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from model_functions import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm # For progress_apply
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message="Accessing `__path__`", module="transformers")

ga_tag = """
    <meta name="google-site-verification" content="8qhJewqcfQuP-HpMtrPOHyc72ENL1xOzBI_THkMVHKo" />
    <!-- Global site tag (gtag.js) - Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-H7VP3CYEPB"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-H7VP3CYEPB');
    </script>
    
    <!-- Google Tag Manager -->
    <script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
    new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
    j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
    'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
    })(window,document,'script','dataLayer','GTM-M9J42J44');</script>
    <!-- End Google Tag Manager -->
    
    <!-- Google Tag Manager (noscript) -->
    <noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-M9J42J44"
    height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
    <!-- End Google Tag Manager (noscript) -->
    """
st.html(ga_tag)
st.set_page_config(page_icon="🎶", page_title="United Kingdom Music Market Dashboard Analysis", layout="wide", menu_items={ 'About': "Gain a **comprehensive, interactive view** of the UK music industry through dynamic visuals, insightful analytics, and actionable recommendations."})
st.logo("static/banner.png")
st.sidebar.image("static/banner.png")
st.header("🎵 Welcome to the United Kingdom's Music Market Dashboard!")
st.markdown(
    """
    <meta name="google-site-verification" content="8qhJewqcfQuP-HpMtrPOHyc72ENL1xOzBI_THkMVHKo" />
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-H7VP3CYEPB"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
    
        gtag('config', 'G-H7VP3CYEPB');
    </script>
    """, unsafe_allow_html=True
)
st.markdown("""
Gain a **comprehensive, interactive view** of the UK music industry through dynamic visuals, insightful analytics, and actionable recommendations.  
This dashboard uncovers patterns in **artist diversity, genre preferences, chart success, and listener behaviors**, helping you explore the forces shaping the Top tracks.  

Designed for analysts, enthusiasts, and professionals alike, it provides a **data‑driven lens** into the evolving UK music landscape.
""")

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

st.caption("Data loaded and preprocessed successfully.")

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

st.caption("KPIs calculated successfully.")

st.sidebar.header('Filter Options')

# --- Date Range Selector ---
with st.spinner("⏳ Loading date range filter..."):
    min_date = df_merged['date'].min().date()
    max_date = df_merged['date'].max().date()
    date_range = st.sidebar.date_input(
        'Date Range',
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
        'Artist',
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
        'Album Type',
        options=all_album_types,
        default=all_album_types
    )
    if selected_album_types:
        mask &= df_merged['album_type'].isin(selected_album_types)

# --- Duration Interval Filter ---
with st.spinner("⏳ Loading duration filter..."):
    duration_range = st.sidebar.slider(
        'Duration Interval (minutes) Of Songs',
        min_value=0, max_value=10,
        value=(0, 10)
    )
    mask &= (df_merged['duration_min'] >= duration_range[0]) & (df_merged['duration_min'] <= duration_range[1])

# --- Popularity Filter ---
with st.spinner("⏳ Loading popularity filter..."):
    pop_min, pop_max = int(df_merged['popularity'].min()), int(df_merged['popularity'].max())
    selected_popularity = st.sidebar.slider(
        'Popularity',
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

# --- 4. Time Series Data (`unique_artists_per_day`) (from Section II, needed for dashboard) ---
unique_artists_per_day = filtered_df.groupby('date')['artist'].nunique()
try:
    # --- Save baseline unique artists per day once ---
    if "baseline_unique_artists_per_day" not in st.session_state:
        st.session_state["baseline_unique_artists_per_day"] = (
            df_merged.groupby('date')['artist'].nunique()
        )
    print("Unique artists per day calculated for Time Series Analysis.")
except:
    message()

# --- 5. Genre Prediction Function and Application (from Section XV) ---
# Conceptual definition of major genres
from huggingface_hub import login
# Try Streamlit secrets first
hf_token = st.secrets.get("HF_TOKEN")

if hf_token:
    try:
        login(token=hf_token)
    except Exception as e:
        st.warning(f"⚠️ Hugging Face token found but login failed: {e}")
else:
    st.warning("⚠️ Hugging Face token not found. Please set HF_TOKEN in secrets.toml or .env")

try:
    # --- Initialize or reuse genre mapping in Streamlit session state ---
    if "genre_mapping" not in st.session_state:
        st.session_state["genre_mapping"] = {}
except:
    message()

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
        # Update Streamlit UI
        progress_text_success.success(f"✅ Genre Prediction In Progress: {percent_complete}%")
        progress_bar.progress((i + 1) / total)
        progress_text_info.info("⏳ Wait for about 3–4 minutes while we prepare your analysis tabs.")

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
filtered_df = df_merged.loc[mask].copy()

# --- Genre Filter (20 major genres) ---
with st.spinner("⏳ Loading genre filter... this may take a few minutes"):
    available_genres = sorted(filtered_df['genre'].dropna().unique().tolist())
    default_genres = sorted(set(major_genres).intersection(available_genres))
    
    selected_genres = st.sidebar.multiselect(
        'Genre',
        options=available_genres,
        default=default_genres
    )

if selected_genres and 'genre' in df_merged.columns:
    mask &= df_merged['genre'].isin(selected_genres)

# --- When creating filtered_df, inherit the genre column directly using mask ---
filtered_df = df_merged.loc[mask].copy()
print("Genre prediction applied to your filtered dataset.")

is_genre_filter_different = (set(selected_genres) != set(default_genres))
is_any_filter_different = (
    is_date_range_different or is_artist_filter_different or is_track_type_different or is_album_type_different 
    or is_duration_filter_different or is_popularity_filter_different or is_genre_filter_different
)

# Deduplicate songs
unique_songs = (
    filtered_df
    .assign(
        song_norm = filtered_df["song"].str.strip().str.lower(),
        artist_norm = filtered_df["artist"].str.strip().str.lower()
    )
    .groupby("song_norm")
    .agg({
        "song": lambda x: x.iloc[0].title(),
        "artist": lambda x: ", ".join(sorted(set(x.str.title()))),
        "album_cover_url": "first"  # keep one representative cover
    })
    .reset_index(drop=True)
)

try:
    if not is_any_filter_different:
        st.session_state["baseline_total"] = len(unique_songs)
        baseline_total = st.session_state["baseline_total"]
    else:
        current_total = len(unique_songs)
        delta = current_total - st.session_state["baseline_total"]
        
        if delta >= 0:
            delta_str = f"⬆️ +{delta}"
        else:
            delta_str = f"⬇️ {delta}"  
except:
    message()
# --- Sidebar caption with arrows ---
with st.sidebar:
    if is_any_filter_different:
        st.caption(f"🎵 Total Songs: {current_total} ({delta_str})")
    else:
        st.caption(f"🎵 Total Songs: {baseline_total}")

# --- Genre-Specific Analysis DataFrames ---
genre_popularity_stats = filtered_df.groupby('genre')['popularity'].agg(['mean', 'median', 'std']).sort_values(by=["mean", "median", "std"], ascending=[False, False, False])
genre_explicitness_percentage = filtered_df.groupby('genre')['is_explicit'].mean() * 100
genre_duration_stats = filtered_df.groupby('genre')['duration_min'].agg(['mean', 'median', 'std']).sort_values(by=["mean", "median", "std"], ascending=[False, False, False])
print("✅ Genre-specific statistics calculated.")

print("--- All required dataframes and variables are now prepared. ---")

tab1, tab2, tab3 = st.tabs([
    "🏛️ UK Music Market Structural Analysis", "📊 Recommendational Analysis for UK's Music Listeners", "🎧 Music Streaming for All Music Listeners"])

with tab3:
    # Header row with symbol + title
    col1, col2 = st.columns([1,9])
    with col1:
        st.image("static/Livestream_symbol.png", width=96)
    with col2:
        st.header("🎵 United Kingdom's Music Streaming")
        st.balloons()

    # Search bar
    query = st.text_input("🔍 Search for a song or artist")
    if query:
        try:
            filtered = unique_songs[
                unique_songs["song"].str.contains(query, case=False) |
                unique_songs["artist"].str.contains(query, case=False)
            ]
        except:
            st.warning("Song is not from the United Kingdom's dataset")
    else:
        filtered = unique_songs

    # Dropdown
    choices = filtered.sort_values(by="song").apply(lambda r: f"{r['song']} — {r['artist']}", axis=1)
    st.success(f"Total Songs: {len(choices)}")
    selected = st.selectbox("🎶 Choose a song to play:", choices)   
    row = filtered[filtered.apply(lambda r: f"{r['song']} — {r['artist']}", axis=1) == selected].iloc[0]
    
    song, artist = selected.split(" — ")

    # Two-column layout for Spotify + YouTube
    col_audio, col_video = st.columns([1,2.25])
    with col_audio:
        st.markdown(f"🔗 [Album Cover Link]({row['album_cover_url']})")
        st.image( row['album_cover_url'], caption=f"{row['song']} — {row['artist']}", width='stretch' )
    with col_video:
        api_key = st.secrets.get("Api_Key")
        if api_key:
            video_id = get_youtube_video_id(song, artist, api_key)
            if video_id and video_id != "None":
                st.video(f"https://www.youtube.com/watch?v={video_id}")
            else:
                st.warning("No official video found at instance. Try again later!")
        else:
            st.warning("YouTube API key not configured.")

    # Spotify setup
    try:
        token = get_spotify_token_cached()
        headers = {"Authorization": f"Bearer {token}"}
        result = search_spotify_track(song, artist, headers)
            
        if result is not None:
            track_id, preview_url = result
            with col_audio:
                if track_id:
                    spotify_embed = f"""
                    <iframe src="https://open.spotify.com/embed/track/{track_id}" 
                    width="100%" height="100%" frameborder="0" allowtransparency="true" 
                    allow="encrypted-media"></iframe>
                    """
                    st.iframe(spotify_embed)
                elif preview_url:
                    st.audio(preview_url, format="audio/mp3")
    except:
        st.warning("Try again later: Bad Luck")

    st.divider()
    st.subheader("📀 Create a Spotify Playlist")
    with st.expander("📝 Steps to Create Your Playlist", expanded=True):
        st.markdown(
        """
        <div style="text-align: center;">
        
            ⚡ Lucky User Rule ⚡  
            - Only the first few lucky users each day can create a playlist.  
            - Spotify allows max 690 searches per day in a playlist.  
            - If the rate limit is hit, playlist creation is disabled until tomorrow.  
        
            Before Login 🔑: You are on 'https://um-unitedkingdommusicmarketanalysisdashboard.streamlit.app/'
            1. Select filters → date range, album types, popularity, duration.
            2. Decide if the playlist should be collaborative.
            3. Review your selections.

            During Login 🔐: You are on 'https://accounts.spotify.com/authorize' 
            - Accept the required <b>Spotify scopes/permissions</b> shown in the login window 
            (e.g. playlist‑modify‑public, playlist‑modify‑private).
            - Without granting these, playlist creation will not work.

            After Login ✅: You are redirected to 'https://um-unitedkingdommusicmarketanalysisdashboard.streamlit.app/?code=Your_Auth_Code'
            4. Click the button below to generate your personalized playlist.
            5. Wait for sometime & open it directly in Spotify and enjoy your mix!
        </div>
        """,
        unsafe_allow_html=True
        )
        
    if "playlist_disabled" not in st.session_state:
        st.session_state["playlist_disabled"] = False
    
    # Lucky user logic
    if not st.session_state["playlist_disabled"]:
        if len(choices) <=690:
            create_playlist_from_dataframe(unique_songs, start_date, end_date, collaboration_choice, selected_album_types, duration_range,
                                selected_popularity, is_any_filter_different)
        else:
            st.error("🚫 Cannot create playlist: Spotify only allows up to 690 songs per day. Try adjusting your sidebar filters to select your own 'Filtered Market' with songs equal to or less than 690.")
    else:
        st.warning("⚠️ Bad luck! Today's lucky users have already been selected. Please try again tomorrow...")
    
    st.subheader("🎬 UK Music Market Dashboard - YouTube Playlist")
    st.info("Youtube Playlist Is only allowed for Test Users. To become One, please inform me at this mail: prathamesh.b20104546@kccollege.edu.in")
    import json
    import os
    from googleapiclient.errors import HttpError
    
    filters = {"start_date": start_date, "end_date": end_date, "collaboration_choice": collaboration_choice, "selected_album_types": selected_album_types,
            "duration_range": duration_range, "selected_popularity": selected_popularity, "is_any_filter_different": is_any_filter_different}
    try:
        if st.button("🎵 Create YouTube Playlist"):
            
            # OAuth clients (4 files)
            client_files = [
                ".streamlit/client_secret_3.json",
                ".streamlit/client_secret_4.json",
                ".streamlit/client_secret_5.json",
                ".streamlit/client_secret_2.json"
            ]
            
            # API keys (14 keys from st.secrets)
            api_keys = []
            for i in range(1, 10):  # 1 through 9
                key_name = f"Key_{i}"
                if key_name in st.secrets:
                    api_keys.append(st.secrets[key_name])
            print(f"Loaded {len(api_keys)} API keys from Streamlit secrets: ",api_keys)
            
            songs_per_client = 200   # inserts per OAuth client
            songs_per_key = 100       # searches per API key
            num_clients = len(client_files)
            num_keys = len(api_keys)
            
            # Load cached IDs if they exist
            vid_ids_per_client = {}
            for i in range(num_clients):
                key_name = f"Vid_ids_client_{i+1}"
                filename = f"{key_name}.json"
                if os.path.exists(filename):
                    with open(filename, "r") as f:
                        vid_ids_per_client[key_name] = json.load(f)
                else:
                    vid_ids_per_client[key_name] = []
            
            skipped_songs = []
            progress = st.progress(0)
            # Authenticate all clients once at the start
            youtube_clients = []
            ports = [8502, 8503, 8504, 8081]  # one free port per client
            for secret_file, port in zip(client_files, ports):
                youtube_clients.append(get_youtube_client(secret_file, port))
            playlist_id = create_playlist(youtube_clients[0], "🎬 Atlantic United Kingdom Music Videos", filters)    #"PLZ08N3lwKEoc" "PLQJFzcDXdSCc" 610 & 608
            
            for i, choice in enumerate(choices, start=0):
                # Split "Song — Artist"
                song, artist = choice.split(" — ")
                
                # --- Rotation logic ---
                client_index = (i // songs_per_client) % num_clients
                key_index = (i // songs_per_key) % num_keys
            
                if client_index >= num_clients:
                    st.error("🚫 Not enough OAuth clients to process all songs.")
                    break
                if key_index >= num_keys:
                    st.error("🚫 Not enough API keys to search all songs.")
                    break
            
                youtube = youtube_clients[client_index] 
                api_key = api_keys[key_index]
                key_name = f"Vid_ids_client_{client_index+1}"
            
                # Skip if already cached
                if vid_ids_per_client[key_name] and len(vid_ids_per_client[key_name]) >= songs_per_client:
                    continue
            
                try:
                    vid = get_youtube_video_id(song, artist, api_key)
                except Exception as e:
                    st.toast(f"Video not found for {song} — {artist}. Skipping...")
                    skipped_songs.append({"song": song, "artist": artist})
            
                if vid and vid != "None":
                    try:
                        # ✅ Check for duplicates before adding
                        if vid not in vid_ids_per_client[key_name]:
                            add_video(youtube, playlist_id, vid)
                            vid_ids_per_client[key_name].append(vid)
                        else:
                            st.toast(f"Duplicate video ID {vid} for {song} — {artist}. Skipping...")
                    except HttpError as e:
                        st.toast(f"Insert quota error for client {client_index+1}: {e}. Switching to next client...")
                        client_index += 1
                        if client_index >= num_clients:
                            st.error("🚫 All clients exhausted.")
                            break
                        continue
    
                # Auto-save after each insert
                filename = f"{key_name}.json"
                with open(filename, "w") as f:
                    json.dump(vid_ids_per_client[key_name], f, indent=2)
            
                progress.progress(int(i+1)/len(choices))
                progress.text(f"Adding video {i+1}/{len(choices)}: {song} — {artist}")
            
            progress.empty()
            st.success(f"Playlist updated! 🎉 View it here: https://www.youtube.com/playlist?list={playlist_id}")
            # after the loop finishes
            if skipped_songs:
                with open("skipped_songs.json", "w") as f:
                    json.dump(skipped_songs, f, indent=2)
            
            # Manual save button
            if st.button("💾 Save Video IDs per Client"):
                for key_name, ids in vid_ids_per_client.items():
                    if ids:
                        filename = f"{key_name}.json"
                        with open(filename, "w") as f:
                            json.dump(ids, f, indent=2)
                        st.success(f"Saved {len(ids)} IDs to {filename}")
    except:
        st.warning("⚠️ You are not registered as a 'Test User'")
    # Banner
    st.divider()
    st.image("static/Livestream_banner.png")
    st.markdown(
        """
        <style>
        [data-testid="stImage"] img {
            width: 100% !important;
            height: auto !important;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

with tab1:
    st.balloons()
    st.title('United Kingdom Music Market Structural Analysis')
    # Tabs for each subheader
    tabs = st.tabs([
        "🔎 Filtered Dataset Preview",
        "👑 Artist Dominance & Diversity",
        "🤝 Collaboration Structures",
        "🔞 Content Explicitness",
        "💿 Album Structure & Release Strategy",
        "⏱️ Track Duration & Format Preferences",
        "📸 Snapshots and Trends",
        "📊 Key Insights and Findings",
        "📌 Strategic Insights"
    ])
    with tabs[0]:
        st.subheader("📋 Filtered Dataset Preview")
        st.dataframe(filtered_df)
        
        st.markdown('---')
    with tabs[1]:
        st.subheader('🎤 Artist Dominance Leaderboard')
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
        st.subheader('🕸️ Artist Collaboration Network')
    
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
    
                # Build graph
                G = nx.Graph()
                for artist_group, songs in unique_collaborations.items():
                    G.add_nodes_from(artist_group)
                    for artist1, artist2 in itertools.combinations(artist_group, 2):
                        if G.has_edge(artist1, artist2):
                            G[artist1][artist2]['weight'] += 1
                        else:
                            G.add_edge(artist1, artist2, weight=1)
    
                # Layout positions
                pos = nx.spring_layout(G, k=0.15, iterations=50, seed=40)
    
                # Edge traces (one per edge so width can vary)
                edge_traces = []
                max_weight = max([d['weight'] for _, _, d in G.edges(data=True)]) if G.number_of_edges() > 0 else 1
    
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    w = edge[2]['weight']
                    edge_trace = go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        line=dict(width=(w/max_weight)*5, color='gray'),
                        mode='lines',
                        hoverinfo='text',
                        text=f"{edge[0]} ↔ {edge[1]} ({w} collaborations)"
                    )
                    edge_traces.append(edge_trace)
    
                # Node data
                node_x, node_y, node_text = [], [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
    
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition="top center",
                    hoverinfo='text',
                    marker=dict(
                        size=20,
                        color='skyblue',
                        line=dict(width=2, color='darkblue')
                    )
                )
    
                # Plotly figure
                fig = go.Figure(data=edge_traces + [node_trace],
                                layout=go.Layout(
                                    title=dict(
                                        text='Artist Collaboration Network',
                                        font=dict(size=20), x=0.5, y=0.95, xanchor='center', yanchor='top'),
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=20, l=5, r=5, t=40),
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                                ))        
                fig.update_layout(autosize=True)            
    
                st.plotly_chart(fig, width='stretch')
    
                with st.expander("ℹ️ More Information"):
                    st.info(f"Total unique collaborations found: **{len(unique_collaborations)}**")
                    collab_list = sorted(unique_collaborations.items(), key=lambda x: ', '.join(x[0]))
                    for idx, (artist_group, songs) in enumerate(collab_list, start=1):
                        # Style artists in green
                        artists_html = ', '.join([f"<span style='color:green'>{artist}</span>" for artist in sorted(artist_group)])
                        # Style songs in gold
                        songs_html = ', '.join([f"<span style='color:gold'>{song}</span>" for song in sorted(songs)])
                        
                        # Display with Streamlit markdown
                        st.markdown(
                            f"{idx}. <b>{artists_html} <span style='color:red'>collaborated on</span>: {songs_html}</b>",
                            unsafe_allow_html=True
                        )
                    
                    st.info("This interactive Plotly graph visualizes unique artist collaborations. Nodes represent artists, edges represent collaborations, and edge thickness indicates frequency across unique collaborations.")
            
                # --- 3D Interactive Visualization ---
                st.markdown('---')
                st.subheader("🌐 3D Collaboration World")
                
                # Use the same collaboration_counts_filtered for 3D
                pos_3d = nx.spring_layout(G, dim=3, k=0.15, iterations=50, seed=40)
                
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
                fig3d.update_layout(autosize=True)
                
                st.plotly_chart(fig3d, width='stretch')
            
            else:
                st.warning("No collaborative tracks found for the selected filters to build a network.")
                
        st.markdown('---')

    with tabs[3]: 
        st.subheader('☢️ Content Explicitness Analysis')
        
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
        
        # Pie chart using Plotly
        pie_data = pd.DataFrame({
            "Content Type": ['Explicit', 'Non-Explicit'],
            "Tracks": [explicitness_counts_filtered.get(True, 0), explicitness_counts_filtered.get(False, 0)]
        })
        fig_pie = px.pie(
            pie_data,
            names="Content Type",
            values="Tracks",
            color="Content Type",
            color_discrete_map={"Explicit": "orange", "Non-Explicit": "skyblue"},
            title="Overall Distribution of Explicit vs. Non-Explicit Content (Filtered)"
        )
        fig_pie.update_layout(autosize=True, title=dict( text="Overall Distribution of Explicit vs. Non-Explicit Content (Filtered)", x=0.5, xanchor="center",  y=0.95, yanchor="top"))
        with colA:
            st.plotly_chart(fig_pie,width='stretch')
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
            bar_data = explicit_percentage_by_rank_filtered.reset_index()
            bar_data.columns = ["Rank Group", "Explicit %"]
            fig_bar = px.bar(
                bar_data,
                x="Rank Group",
                y="Explicit %",
                color="Rank Group",
                text="Explicit %",
                color_discrete_sequence=px.colors.sequential.Viridis_r,
                title="Percentage of Explicit Content by Rank Group (Filtered)"
            )
            fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_bar.update_yaxes(range=[0, 100], title="Percentage of Explicit Tracks (%)")
            fig_bar.update_layout(title=dict( text="Percentage of Explicit Content by Rank Group (Filtered)", x=0.5, xanchor="center",  y=0.95, yanchor="top"), autosize=True)
            
            with colB:
                st.plotly_chart(fig_bar, width='stretch')
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
                    st.info("- Insights are based on the current filters and may shift as the dataset changes.")
                    st.info("- Genre, artist profile, and release strategy all affect how explicit content performs in the UK market.")
        
                with st.expander("🎶 Granular Analysis Suggestions"):
                    st.info("- Examine explicitness by genre or artist cohort.")
                    st.info("- Some musical styles and fanbases are more accepting of explicit material than others.")
        
                if explicit_other is not None:
                    with st.expander("🌍 Beyond the Top 50"):
                        st.metric("Outside Top 50 Explicit %", f"{explicit_other:.2f}%")
                        st.info("- Higher explicitness in lower ranks may signal niche markets or emerging trends in UK listener demographics.")
        
        st.markdown("---")
    with tabs[4]:
        st.subheader('📀 Album Type Distribution')
        
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
            bar_data = album_type_counts_filtered.reset_index()
            bar_data.columns = ["Album Type", "Count"]
    
            fig_album_type = px.bar(
                bar_data,
                x="Count",
                y="Album Type",
                orientation="h",
                color="Album Type",
                color_discrete_sequence=px.colors.sequential.Viridis_r,
                title="Release Format Dominance in the UK Market: Distribution of Album Types (Filtered)"
            )
            fig_album_type.update_layout(
                title=dict(text="Release Format Dominance in the UK Market: Distribution of Album Types (Filtered)", x=0.5, xanchor="center", y=0.95, yanchor="top", font=dict(size=12, family="Black")),
                xaxis_title="Number of Tracks",
                yaxis_title="Album Type", legend=dict(font=dict(size=12, family="Black")), autosize=True
            )
            fig_pie.update_traces(textfont=dict(size=12, family="Black"))
            with colA:
                st.plotly_chart(fig_album_type, width='stretch')
        else:
            st.warning("No album type data available for the selected filters to display the chart.")
        
        with colB:
            with st.expander("**Percentage of each album type (Filtered Data):**"):
                st.write(album_type_percentage_filtered)
            
        # Create the Donut chart for album type distribution (percentage form)
        if not album_type_counts_filtered.empty:
            album_type_percent = album_type_counts_filtered / album_type_counts_filtered.sum() * 100
            pie_data = album_type_percent.reset_index()
            pie_data.columns = ["Album Type", "Percentage"]
    
            fig_pie = px.pie(
                pie_data,
                names="Album Type",
                values="Percentage",
                hole=0.4,  # donut effect
                color="Album Type",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Album Type Distribution (Filtered)"
            )
            
            # Make all text bold
            fig_pie.update_layout(
                title=dict( text="Album Type Distribution (Filtered)", x=0.5, xanchor="center", y=0.95, yanchor="top", font=dict(size=14, family="Arial Black")  # Bold title
                ), legend=dict(font=dict(size=12, family="Black")), autosize=True)
            fig_pie.update_traces(textfont=dict(size=12, family="Black"))
            with colB:
                st.plotly_chart(fig_pie, width='stretch')
        else:
            st.warning("No album type data available for the selected filters to display the chart.")
        
        with st.expander("ℹ️ More Information"):
            st.info('These charts displays the distribution of different album types (single, album, compilation) within the filtered dataset in numerical and percentage(%) format, indicating the prevalence of each release format.')
        
        st.markdown('---')
    with tabs[5]:
        st.subheader('🎼 Track Duration Insights')
    
        # Calculate duration_min and duration_category
        filtered_df['duration_min'] = filtered_df['duration_ms'] / 60000
        filtered_df['duration_category'] = filtered_df['duration_min'].apply(lambda x: 'short-form' if x < 3.5 else 'long-form')
    
        # Counts and percentages
        duration_counts_filtered = filtered_df['duration_category'].value_counts()
        total_tracks_duration_filtered = duration_counts_filtered.sum()
        duration_percentage_filtered = (duration_counts_filtered / total_tracks_duration_filtered) * 100
    
        colA, colB = st.columns(2)
        with colA:
            with st.expander("**Duration Category Counts (Filtered Data):**"):
                st.dataframe(duration_counts_filtered)
        with colB:
            with st.expander("**Duration Category Percentages (Filtered Data):**"):
                st.dataframe(duration_percentage_filtered)
    
        # Count plot (bar chart)
        if not filtered_df.empty:
            bar_data = duration_counts_filtered.reset_index()
            bar_data.columns = ["Duration Category", "Tracks"]
            
            fig_countplot = px.bar(
                bar_data,
                x="Duration Category",
                y="Tracks",
                color="Duration Category",
                color_discrete_sequence=px.colors.qualitative.G10,
                title="Count Plot of Duration Categories (Filtered)"
            )
            fig_countplot.update_layout(
                xaxis_title="Duration Category",
                yaxis_title="Number of Tracks",
                title=dict(x=0.5, xanchor="center", font=dict(size=16)), autosize=True
            )
            with colA:
                st.plotly_chart(fig_countplot, width='stretch')
                with st.expander("ℹ️ More Information"):
                    st.info("This count plot shows the absolute number of short-form and long-form tracks in the filtered dataset.")
        else:
            st.warning("No data available to display count plot.")
    
        # Pie chart
        if not duration_counts_filtered.empty:
            pie_data = duration_counts_filtered.reset_index()
            pie_data.columns = ["Duration Category", "Tracks"]
    
            fig_duration_pie = px.pie(
                pie_data,
                names="Duration Category",
                values="Tracks",
                color="Duration Category",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Overall Distribution of Track Duration Categories (Filtered)"
            )
            fig_duration_pie.update_layout(
                title=dict(x=0.5, xanchor="center", font=dict(size=16)), autosize=True
            )
            with colB:
                st.plotly_chart(fig_duration_pie, width='stretch')
                with st.expander("ℹ️ More Information"):
                    st.info("This pie chart shows the overall proportion of short-form and long-form tracks in the filtered dataset.")
        else:
            st.warning("No track duration data available for the selected filters to display the pie chart.")
    
        # Popularity buckets
        if not filtered_df.empty and 'popularity' in filtered_df.columns:
            filtered_df['popularity_bucket'] = pd.qcut(
                filtered_df['popularity'], q=4,
                labels=['Q1 (Least Popular)', 'Q2', 'Q3', 'Q4 (Most Popular)'],
                duplicates='drop'
            )
    
            duration_popularity_distribution_filtered = (
                filtered_df.groupby(['popularity_bucket', 'duration_category'], observed=False)
                .size()
                .unstack(fill_value=0)
            )
    
            with st.expander("**Distribution of Track Duration Categories by Popularity Bucket (Filtered Data):**"):
                st.dataframe(duration_popularity_distribution_filtered)
    
            if not duration_popularity_distribution_filtered.empty:
                bar_data = duration_popularity_distribution_filtered.reset_index().melt(
                    id_vars="popularity_bucket", var_name="Duration Category", value_name="Tracks"
                )
                bar_data.rename(columns={"popularity_bucket": "Popularity Bucket"}, inplace=True)
                
                fig_pop_duration = px.bar(
                    bar_data,
                    x="Popularity Bucket",
                    y="Tracks",
                    color="Duration Category",
                    barmode="stack",
                    color_discrete_sequence=px.colors.sequential.Aggrnyl_r,
                    title="Track Duration Distribution Across Popularity Buckets (Filtered)"
                )
                fig_pop_duration.update_layout(
                    xaxis_title="Popularity Bucket",
                    yaxis_title="Number of Tracks",
                    title=dict(x=0.5, xanchor="center", font=dict(size=16)), autosize=True
                )
                st.plotly_chart(fig_pop_duration, width='stretch')
    
                with st.expander("ℹ️ More Information"):
                    st.info("This stacked bar chart illustrates how short-form and long-form tracks are distributed across different popularity levels (quartiles).")
        else:
            st.warning("Cannot analyze track duration across popularity buckets. Ensure 'popularity' column is available and data is not empty.")
    
        # Duration ranges + box plot
        if not filtered_df.empty and 'popularity' in filtered_df.columns and 'duration_min' in filtered_df.columns:
            duration_bins = [0, 2, 4, 6, 8, 10]
            duration_bin_labels = ['0-2 min', '2-4 min', '4-6 min', '6-8 min', '8-10 min']
            filtered_df['duration_range'] = pd.cut(
                filtered_df['duration_min'], bins=duration_bins,
                labels=duration_bin_labels, right=False, include_lowest=True
            )
            df_merged['duration_range'] = pd.cut(
                df_merged['duration_min'], bins=duration_bins,
                labels=duration_bin_labels, right=False, include_lowest=True
            )
    
            duration_range_popularity_distribution = (
                filtered_df.groupby(['popularity_bucket', 'duration_range'], observed=False)
                .size()
                .unstack(fill_value=0)
            )
    
            with st.expander("**Distribution of Track Duration Ranges by Popularity Bucket:**"):
                st.dataframe(duration_range_popularity_distribution)
    
            fig_boxplot = px.box(
                filtered_df,
                x="popularity_bucket",
                y="duration_min",
                color="popularity_bucket",
                color_discrete_sequence=px.colors.sequential.Agsunset_r,
                category_orders={"popularity_bucket": ["Q1 (Least Popular)", "Q2", "Q3", "Q4 (Most Popular)"]},  
                title="Track Duration Distribution by Popularity Bucket (Box Plot)"
            )
            fig_boxplot.update_layout(
                xaxis_title="Popularity Bucket",
                yaxis_title="Duration (Minutes)",
                title=dict(x=0.5, xanchor="center", font=dict(size=16)), autosize=True
            )
            st.plotly_chart(fig_boxplot, width='stretch')
    
            with st.expander("ℹ️ More Information"):
                st.info("This box plot visualizes the median, quartiles, and outliers of track duration for each popularity bucket. It helps to visualize the typical range and spread of track lengths within each popularity quartile.")
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
                        st.info("- These comparisons make insights dynamic, showing whether the current filter view leans more short-form or long-form than the UK baseline.")
                        st.info("- Genre, release strategy, and artist profile all influence duration trends.")
        
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
            .stAlert {
                text-align: center;
            }
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
                67%  { transform: translate(120%, 0%); }
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
        if is_any_filter_different:
            st.toast("🔍 Peek Into Enhanced View Of Snapshot With Trends")
        else:
            st.toast("🖼️ View The Scenery In Gallery Of Metric Cards")
        
    with tabs[7]:
        st.subheader("📊 Key Insights and Findings")
        
        with st.expander("Comprehensive UK Music Market Analysis"):
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
                        st.info(f"**Market Reference:** {full_highest_collab[0][0]} & {full_highest_collab[0][1]} ({full_highest_collab[1]} collaborations)") # type: ignore
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
        if is_any_filter_different:
            st.toast("🔀 Explore The Comparisons Of Structural Analysis For Filtered V/S Baseline Version ")
        else:
            st.toast("📊 Findings & Insights Are Ready For Baseline Version")
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
        
            st.success("*This filtered view highlights shifts in artist dominance, collaboration strength, and format preferences compared to the full dataset.*")
        
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
        
            st.success("*Together, these dynamics suggest a mature, stable UK market where innovation happens within predictable frameworks — ideal for benchmarking and long-term planning.*")
        
        st.markdown("---")
        if is_any_filter_different:
            st.toast("Morals Of Structural Analysis Are Affected....")
        else:
            st.toast("Morals Of Structural Analysis Are Waiting....")

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

try: 
    X_lr = pd.get_dummies(df_temp_lr, columns=categorical_features_lr, drop_first=True)
    y_lr = df_merged['chart_success']
    
    X_train_no_eng, X_test_no_eng, y_train_no_eng, y_test_no_eng = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42, stratify=y_lr)
    
    model_lr = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
    model_lr.fit(X_train_no_eng, y_train_no_eng)
    y_pred_lr = model_lr.predict(X_test_no_eng)
    lr_accuracy_no_eng, metrics_df = get_metrics(y_test_no_eng, y_pred_lr, "Logistic Regression (No Features)")
    
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
    rf_accuracy_no_eng, rf_metrics_df_no_eng = get_metrics(y_test_no_eng, rf_y_pred_no_eng, "Random Forest (No Features)")
    
    # --- Model 4: XGBoost (No Engineered Features) ---
    xgb_model_no_eng = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', 
        scale_pos_weight=(len(y_train_no_eng) - y_train_no_eng.sum()) / y_train_no_eng.sum()) # Handle class imbalance
    xgb_model_no_eng.fit(X_train_no_eng, y_train_no_eng)
    xgb_y_pred_no_eng = xgb_model_no_eng.predict(X_test_no_eng)
    xgb_accuracy_no_eng, xgb_metrics_df_no_eng = get_metrics(y_test_no_eng, xgb_y_pred_no_eng, "XGBoost (No Features)")
    
    # --- Model 5: K-Means Clustering ---
    kmeans_no_eng = KMeans(n_clusters=2, random_state=42)
    kmeans_no_eng.fit(X_train_no_eng)
    kmeans_labels_no_eng = kmeans_no_eng.predict(X_test_no_eng)
    kmeans_accuracy_no_eng, kmeans_metrics_df_no_eng = get_metrics(y_test_no_eng, kmeans_labels_no_eng, "KMeans (No Features)")
    
    # --- Model 6: SVM ---
    svm_no_eng = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm_no_eng.fit(X_train_no_eng, y_train_no_eng)
    svm_y_pred_no_eng = svm_no_eng.predict(X_test_no_eng)
    svm_accuracy_no_eng, svm_metrics_df_no_eng = get_metrics(y_test_no_eng, svm_y_pred_no_eng, "SVM (No Features)")
    
    # --- Model 7: Gradient Boosting ---
    gb_no_eng = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_no_eng.fit(X_train_no_eng, y_train_no_eng)
    gb_y_pred_no_eng = gb_no_eng.predict(X_test_no_eng)
    gb_accuracy_no_eng, gb_metrics_df_no_eng = get_metrics(y_test_no_eng, gb_y_pred_no_eng, "Gradient Boosting (No Features)")
    
    # --- Model 8: Gaussian Mixture ---
    gmm_no_eng = GaussianMixture(n_components=2, random_state=42)
    gmm_no_eng.fit(X_train_no_eng)
    gmm_labels_no_eng = gmm_no_eng.predict(X_test_no_eng)
    gmm_accuracy_no_eng, gmm_metrics_df_no_eng = get_metrics(y_test_no_eng, gmm_labels_no_eng, "Gaussian Mixture (No Features)")
    
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
    lr_accuracy_eng, lr_metrics_eng_df = get_metrics(y_test_eng, lr_y_pred_eng, "Logistic Regression (Engineered Features)")
    
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
    rf_accuracy_eng, rf_metrics_eng_df = get_metrics(y_test_eng, rf_y_pred_eng, "Random Forest (Engineered Features)")
    
    # --- Model 4: XGBoost ---
    xgb_model_eng = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', 
        scale_pos_weight=(len(y_train_eng) - y_train_eng.sum()) / y_train_eng.sum()) # Handle class imbalance
    xgb_model_eng.fit(X_train_eng, y_train_eng)
    xgb_y_pred_eng = xgb_model_eng.predict(X_test_eng)
    xgb_accuracy_eng, xgb_metrics_eng_df = get_metrics(y_test_eng, xgb_y_pred_eng, "XGBoost (Engineered Features)")
    
    # --- Model 5: KMeans --- 
    kmeans_eng = KMeans(n_clusters=2, random_state=42)
    kmeans_eng.fit(X_train_eng)
    kmeans_labels_eng = kmeans_eng.predict(X_test_eng)
    kmeans_accuracy_eng, kmeans_metrics_df_eng = get_metrics(y_test_eng, kmeans_labels_eng, "KMeans (Engineered Features)")
    
    # --- Model 6: SVM --- 
    svm_eng = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm_eng.fit(X_train_eng, y_train_eng)
    svm_y_pred_eng = svm_eng.predict(X_test_eng)
    svm_accuracy_eng, svm_metrics_df_eng = get_metrics(y_test_eng, svm_y_pred_eng, "SVM (Engineered Features)")
    
    # --- Model 7: Gradient Boosting --- 
    gb_eng = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_eng.fit(X_train_eng, y_train_eng)
    gb_y_pred_eng = gb_eng.predict(X_test_eng)
    gb_accuracy_eng, gb_metrics_df_eng = get_metrics(y_test_eng, gb_y_pred_eng, "Gradient Boosting (Engineered Features)")
    
    # --- Model 8: Gaussian Mixture --- 
    gmm_eng = GaussianMixture(n_components=2, random_state=42)
    gmm_eng.fit(X_train_eng)
    gmm_labels_eng = gmm_eng.predict(X_test_eng)
    gmm_accuracy_eng, gmm_metrics_df_eng = get_metrics(y_test_eng, gmm_labels_eng, "Gaussian Mixture (Engineered Features)")
    
    try:
        # Build comparison DataFrames for each model
        lr_comparison_df = build_comparison_df(metrics_df, lr_metrics_eng_df, "Logistic Regression")
        linear_comparison_df = build_comparison_df(Lr_metrics_df_no_eng, Lr_metrics_eng_df, "Linear Regression")
        rf_comparison_df = build_comparison_df(rf_metrics_df_no_eng, rf_metrics_eng_df, "Random Forest")
        xgb_comparison_df = build_comparison_df(xgb_metrics_df_no_eng, xgb_metrics_eng_df, "XGBoost")
        kmeans_comparison_df = build_comparison_df(kmeans_metrics_df_no_eng, kmeans_metrics_df_eng, "KMeans")
        svm_comparison_df = build_comparison_df(svm_metrics_df_no_eng, svm_metrics_df_eng, "SVM")
        gb_comparison_df = build_comparison_df(gb_metrics_df_no_eng, gb_metrics_df_eng, "Gradient Boosting")
        gmm_comparison_df = build_comparison_df(gmm_metrics_df_no_eng, gmm_metrics_df_eng, "Gaussian Mixture")
        
        # Combine all into one master comparison DataFrame
        all_models_comparison_df = pd.concat([lr_comparison_df, linear_comparison_df, rf_comparison_df, xgb_comparison_df,
                                            kmeans_comparison_df, svm_comparison_df, gb_comparison_df, gmm_comparison_df], ignore_index=True)
        
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
                'XGBoost (With Engineered Features)', 'XGBoost (Without Engineered Features)', 'KMeans (No Features)', 'KMeans (Engineered Features)',
                'SVM (No Features)', 'SVM (Engineered Features)', 'Gradient Boosting (No Features)', 'Gradient Boosting (Engineered Features)',
                'Gaussian Mixture (No Features)', 'Gaussian Mixture (Engineered Features)'
            ],
            'Accuracy': [
                lr_accuracy_no_eng, lr_accuracy_eng, rf_accuracy_no_eng, rf_accuracy_eng, Lr_accuracy_eng, Lr_accuracy_no_eng, xgb_accuracy_eng, xgb_accuracy_no_eng,
                kmeans_accuracy_no_eng, kmeans_accuracy_eng, svm_accuracy_no_eng, svm_accuracy_eng, gb_accuracy_no_eng, gb_accuracy_eng, gmm_accuracy_no_eng, gmm_accuracy_eng
            ]
        })
        if "accuracy_summary_df_original" not in st.session_state:
            if not is_any_filter_different:
                st.session_state["accuracy_summary_df_original"] = accuracy_summary_df
    except:
        message()

except Exception as e: 
    print(f"Error during model training and comparison: {e}")
    st.warning("Not enough data for models to perform testing. Please adjust your filters to include more data.")
filtered_df = df_merged.copy()

with tab2:
    st.balloons()
    # --- Dashboard Title and Introduction ---
    st.title("Recommendational Analysis Dashboard For UK's Music Listeners")
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
        st.subheader("🤖 Predictive Modeling of Chart Success")
        st.markdown("""
        Explore how different models perform in predicting Top 10 chart success.
        Use the filter below to select a model and view its performance metrics.
        """)
    
        # --- Model Selection Filter ---
        warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")
        model_choice = st.selectbox(
            "Select a Model to View Metrics",
            ["🧩 Logistic Regression", "📈 Linear Regression", "🌳 Random Forest 🌲", "🚀 XGBoost",
            "🔄 KMeans Clustering", "⚔️ Support Vector Machine (SVM)", "🔥 Gradient Boosting", "🎭 Gaussian Mixture"]
        )
    
        # --- Dictionary mapping model names to their metric DataFrames ---
        model_metrics = {
            "🧩 Logistic Regression": (metrics_df, lr_metrics_eng_df, lr_comparison_df),
            "📈 Linear Regression": (Lr_metrics_df_no_eng, Lr_metrics_eng_df, linear_comparison_df),
            "🌳 Random Forest 🌲": (rf_metrics_df_no_eng, rf_metrics_eng_df, rf_comparison_df),
            "🚀 XGBoost": (xgb_metrics_df_no_eng, xgb_metrics_eng_df, xgb_comparison_df),
            "🔄 KMeans Clustering": (kmeans_metrics_df_no_eng, kmeans_metrics_df_eng, kmeans_comparison_df),
            "⚔️ Support Vector Machine (SVM)": (svm_metrics_df_no_eng, svm_metrics_df_eng, svm_comparison_df),
            "🔥 Gradient Boosting": (gb_metrics_df_no_eng, gb_metrics_df_eng, gb_comparison_df),
            "🎭 Gaussian Mixture": (gmm_metrics_df_no_eng, gmm_metrics_df_eng, gmm_comparison_df)
        }
        # Get the correct DataFrames
        metrics_no_eng, metrics_eng, comparison_df_model = model_metrics[model_choice]
        # --- Display Classification Report (Heatmaps + Tables) ---
        colA, colB = st.columns(2)
    
        with colA:
            with st.expander(f"{model_choice} Performance (Without Engineered Features):"):
                # Interactive heatmap for metrics_no_eng
                df_no_eng = metrics_no_eng.drop(columns=['support'], errors='ignore')
                fig_no_eng = px.imshow(
                    df_no_eng.values,
                    x=df_no_eng.columns,
                    y=df_no_eng.index,
                    color_continuous_scale="Viridis",
                    labels=dict(x="Metric", y="Class", color="Score"),
                    title=f"{model_choice} (No Features)"
                )
                fig_no_eng.update_layout(
                    margin=dict(t=80, b=40),
                    title=dict(text=f"{model_choice} (No Features)", x=0.5, xanchor="center", y=0.95, yanchor="top"), autosize=True
                )
                st.plotly_chart(fig_no_eng, width='stretch')
                st.dataframe(df_no_eng)
        
        with colB:
            with st.expander(f"{model_choice} Performance (With Engineered Features):"):
                # Interactive heatmap for metrics_eng
                df_eng = metrics_eng.drop(columns=['support'], errors='ignore')
                fig_eng = px.imshow(
                    df_eng.values,
                    x=df_eng.columns,
                    y=df_eng.index,
                    color_continuous_scale="Viridis",
                    labels=dict(x="Metric", y="Class", color="Score"),
                    title=f"{model_choice} (Engineered Features)"
                )
                fig_eng.update_layout(
                    margin=dict(t=80, b=40),
                    title=dict(text=f"{model_choice} (Engineered Features)", x=0.5, xanchor="center", y=0.95, yanchor="top"), autosize=True
                )
                st.plotly_chart(fig_eng, width='stretch')
                st.dataframe(df_eng)
    
        # --- Display Comparison Plot (Chart) ---
        if comparison_df_model is not None:
            fig_comp = px.bar(
                comparison_df_model,
                x="Metric", y="Score", color="Model",
                facet_col="Class", facet_col_wrap=2,
                barmode="group",
                color_discrete_sequence=px.colors.sequential.Viridis_r
            )
            
            fig_comp.update_layout(
                title=f"{model_choice} Performance Comparison: With vs. Without Engineered Features",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,   # place legend just below title
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(t=120, b=80, l=50, r=50),  # extra space for title + axis labels
                title_pad=dict(t=60),
                yaxis=dict(range=[0, 1], title="Score"),
                xaxis=dict(title="Metric", title_standoff=20), autosize=True
            )
            
            # Custom facet titles for clarity
            fig_comp.for_each_annotation(lambda a: a.update(
                text="Class 0 (Not Top 10)" if "Class=0" in a.text else
                    "Class 1 (Top 10)" if "Class=1" in a.text else a.text
            ))
        
            st.plotly_chart(fig_comp, width='stretch')
            if is_any_filter_different:
                st.toast(f"🔀 {model_choice}'s Visualizations Are Ready")
            else:
                st.toast("📌 Logistic Regression & Other 7 Model Predictions Are Lined In Queue")
            
        else:
            st.warning("Comparison DataFrame not found. Please ensure the model comparison section was run.")
        
        # --- Accuracy Comparison Table + Interactive Bar Chart ---
        if 'accuracy_summary_df' in locals():
            with st.expander("📊 **Model Accuracy Comparison (All Models):**"):
                st.dataframe(accuracy_summary_df)
    
            fig_acc = px.bar(
                accuracy_summary_df,
                x="Model", y="Accuracy", color="Model",
                text="Accuracy", color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig_acc.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_acc.update_layout(title="Comparison of Model Accuracies", yaxis=dict(range=[0,1]), autosize=True)
            st.plotly_chart(fig_acc, width='stretch')
    
            best_model_row = accuracy_summary_df.loc[accuracy_summary_df['Accuracy'].idxmax()]
            best_model_name = best_model_row['Model']
            best_model_accuracy = best_model_row['Accuracy']
            st.success(f"🏆 Best Model: **{best_model_name}** with Accuracy = {best_model_accuracy:.4f}")
        else:
            st.warning("`accuracy_summary_df` not found. Please ensure the model accuracy summary section was run.")
        
        st.subheader("🎼 Rhythm of Success Plot")
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
            margin=dict(l=65, r=50, b=65, t=90),
            updatemenus=[dict(type="dropdown", direction="down", x=0.0, y=1.15, showactive=True, buttons=buttons)], autosize=True
        )
    
        # --- Streamlit rendering ---
        st.plotly_chart(fig, width='stretch')
    
        st.markdown('---')
        
    with tabs[1]:
        try:
            # --- Save baseline engineered feature distributions once ---
            if "baseline_day_of_week_counts" not in st.session_state:
                st.session_state["baseline_day_of_week_counts"] = (
                    df_merged.groupby('day_of_week')['chart_success'].value_counts()
                )
            if "baseline_duration_x_num_artists" not in st.session_state:
                st.session_state["baseline_duration_x_num_artists"] = df_merged[['duration_x_num_artists','popularity','chart_success']]
            if "baseline_explicit_duration" not in st.session_state:
                st.session_state["baseline_explicit_duration"] = df_merged[['explicit_duration','chart_success']]
        except:
            message()
            
        # --- Section 2: Feature Engineering Visualizations (Relevant to Chart Success) ---
        st.subheader("🛠️ Feature Engineering Visualizations for Chart Success")
        st.markdown("""
        Visualizations of engineered features provide insights into their relationship with chart success.
        """)
    
        # --- Chart Success by Day of the Week ---
        fig_day_of_week = px.histogram(
            filtered_df,
            x="day_of_week",
            color="chart_success",
            barmode="group",
            color_discrete_sequence=px.colors.sequential.Agsunset_r,
            labels={"day_of_week":"Day of Week (0=Mon, 6=Sun)", "chart_success":"Chart Success"}
        )
        fig_day_of_week.update_layout(
            title="Chart Success by Day of the Week",
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(7)),
                ticktext=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
            ),
            yaxis_title="Number of Tracks",
            legend_title="Chart Success (0=No, 1=Yes)", autosize=True
        )
        st.plotly_chart(fig_day_of_week, width='stretch')
    
        with st.expander("ℹ️ More Information"):
            st.info("- Chart success varies by day, with certain weekdays showing higher Top 10 entries, indicating optimal release timing in the UK market.")
            st.info("- Weekly patterns may reflect audience engagement cycles and promotional strategies in UK music consumption.")
    
        # --- Interaction of Duration and Number of Artists vs. Popularity ---
        fig_duration_x_artists = px.scatter(
            filtered_df,
            x="duration_x_num_artists",
            y="popularity",
            color="chart_success",
            color_discrete_sequence=px.colors.diverging.RdBu,
            opacity=0.6,
            labels={"duration_x_num_artists":"Duration (min) * Number of Artists", "popularity":"Popularity", "chart_success":"Chart Success"}
        )
        fig_duration_x_artists.update_layout(
            title="Interaction of Duration and Number of Artists vs. Popularity by Chart Success",
            legend_title="Chart Success (0=No, 1=Yes)", autosize=True
        )
        st.plotly_chart(fig_duration_x_artists, width='stretch')
    
        with st.expander("ℹ️ More Information"):
            st.info("- Longer durations with more artists correlate with higher popularity, suggesting collaborative extended tracks appeal to UK listeners.")
            st.info("- Interaction effects highlight how production complexity influences chart performance in the UK market.")
    
        # --- Distribution of Explicit Track Duration by Chart Success ---
        fig_explicit_duration = px.violin(
            filtered_df,
            x="chart_success",
            y="explicit_duration",
            color="chart_success",
            box=True, points="all",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"chart_success":"Chart Success", "explicit_duration":"Explicit Duration (minutes)"}
        )
        fig_explicit_duration.update_layout(
            title="Distribution of Explicit Track Duration by Chart Success",
            xaxis=dict(
                tickmode="array",
                tickvals=[0,1],
                ticktext=['Not Top 10','Top 10']
            ), autosize=True
        )
        st.plotly_chart(fig_explicit_duration, width='stretch')
    
        with st.expander("ℹ️ More Information"):
            st.info("- Explicit tracks in Top 10 tend to be shorter, balancing maturity with concise delivery for UK audience preferences.")
            st.info("- Duration-explicitness interplay reveals content strategy nuances in achieving UK chart success.")
    
        st.markdown('---')
    with tabs[2]:
        # --- Recommendation 3: Time Series Analysis of Trends ---
        st.subheader("🕰️ Time Series Analysis of Trends")
        st.markdown("""
        Analyzing trends over time can reveal seasonality, shifts in artist dominance, or changes in content preferences.
        Here's a look at the number of unique artists appearing in the Top 50 chart each day.
        """)
    
        if 'unique_artists_per_day' in locals():
            st.write("### Daily Unique Artists in Top 50:")
    
            # Prepare dataframe
            unique_artists_per_day_df = unique_artists_per_day.reset_index()
            unique_artists_per_day_df['date'] = pd.to_datetime(
                unique_artists_per_day_df['date'], dayfirst=True
            )
            unique_artists_per_day_df = unique_artists_per_day_df.sort_values('date')
    
            # Interactive Plotly line chart
            fig_unique_artists = px.line(
                unique_artists_per_day_df,
                x="date",
                y="artist",
                markers=True,
                labels={"date": "Date", "artist": "Number of Unique Artists"},
                title="Number of Unique Artists in Top 50 Per Day"
            )
            fig_unique_artists.update_layout(
                xaxis=dict(title="Date", tickangle=0),
                yaxis=dict(title="Number of Unique Artists"), autosize=True
            )
    
            st.plotly_chart(fig_unique_artists, width='stretch')
    
        else:
            st.warning("`unique_artists_per_day` data not found. Please ensure the artist dominance analysis section was run.")
    
        with st.expander("ℹ️ More Information"):
            st.info("- Unique artist counts fluctuate over time, revealing periods of high diversity versus concentration in UK Top 50 charts.")
            st.info("- Temporal trends may indicate market saturation, new entries, or seasonal influences on UK music landscape.")
    
        st.markdown('---')
        
        if is_any_filter_different:
            st.toast("🎶 Deep Dive Into Feature Engineering Of The Music Market")
        else:
            st.toast("✨ Look The Engineering Features Of The Music Market")
        
    with tabs[3]:
        # --- Recommendation 4: Genre-Specific Analysis ---
        st.subheader("📚 Genre-Specific Analysis (Conceptual)")
        st.markdown("""
        While genre prediction is currently conceptual (using CLIPModel & CLIPProcessor from OpenAI) and sensitive in nature,
        we can explore how different genres might relate to popularity, explicitness, and duration.
        """)
    
        # --- Genre vs. Popularity ---
        if 'genre_popularity_stats' in locals() and not filtered_df.empty:
            st.write("### Genre vs. Popularity:")
            fig_genre_pop = px.box(
                filtered_df,
                x="genre", y="popularity", color="genre",
                category_orders={"genre": sorted(genre_popularity_stats.index.tolist())},
                labels={"genre":"Genre","popularity":"Popularity Score"},
                title="Popularity Distribution by Genre"
            )
            fig_genre_pop.update_layout(xaxis=dict(tickangle=45), autosize=True)
            st.plotly_chart(fig_genre_pop, width='stretch')
    
            if not genre_popularity_stats.empty:
                # Sort by mean popularity
                sorted_stats = genre_popularity_stats.sort_values(by="median", ascending=False)
            
                # Top and bottom genres
                top_genre = sorted_stats.index[0]
                bottom_genre = sorted_stats.index[-1]
                top_pop = sorted_stats["median"].iloc[0]
                bottom_pop = sorted_stats["median"].iloc[-1]
                
                with st.expander("ℹ️ More Information"):
                    st.info(f"- **{top_genre}** leads in popularity (median {top_pop:.1f}), while **{bottom_genre}** ranks lowest (median {bottom_pop:.1f}).")
                    st.info("- Popularity variations by genre may stem from cultural trends, marketing strategies, or demographic preferences in the UK.")
            
        else:
            st.warning("`genre_popularity_stats` or `df_merged` not found. Please ensure the genre analysis section was run.")
    
        # --- Genre vs. Explicitness ---
        if 'genre_explicitness_percentage' in locals() and not filtered_df.empty:
            st.write("### Genre vs. Explicitness:")
            fig_genre_exp = px.bar(
                x=genre_explicitness_percentage.index,
                y=genre_explicitness_percentage.values,
                color=genre_explicitness_percentage.index,
                labels={"x":"Genre","y":"Percentage Explicit (%)"},
                title="Percentage of Explicit Content by Genre"
            )
            fig_genre_exp.update_layout(xaxis=dict(tickangle=45), yaxis=dict(range=[0,100]), autosize=True)
            st.plotly_chart(fig_genre_exp, width='stretch')
    
            if not genre_explicitness_percentage.empty:
                # Sort by explicitness percentage
                sorted_explicitness = genre_explicitness_percentage.sort_values(ascending=False)
            
                # Top and bottom genres
                most_explicit_genre = sorted_explicitness.index[0]
                least_explicit_genre = sorted_explicitness.index[-1]
                most_explicit_pct = sorted_explicitness.iloc[0]
                least_explicit_pct = sorted_explicitness.iloc[-1]
                
                with st.expander("ℹ️ More Information"):
                    st.info(f"- **{most_explicit_genre}** has the highest explicitness ({most_explicit_pct:.1f}%), while **{least_explicit_genre}** is lowest ({least_explicit_pct:.1f}%).")
                    st.info("- Explicitness differences highlight genre-specific cultural norms and target audience demographics in UK music.")
                
        else:
            st.warning("`genre_explicitness_percentage` or `df_merged` not found. Please ensure the genre analysis section was run.")
    
        # --- Genre vs. Duration ---
        if 'genre_duration_stats' in locals() and not filtered_df.empty:
            st.write("### Genre vs. Duration:")
            fig_genre_dur = px.box(
                filtered_df,
                x="genre", y="duration_min", color="genre",
                category_orders={"genre": sorted(genre_duration_stats.index.tolist())},
                labels={"genre":"Genre","duration_min":"Duration (minutes)"},
                title="Track Duration Distribution by Genre"
            )
            fig_genre_dur.update_layout(xaxis=dict(tickangle=45), autosize=True)
            st.plotly_chart(fig_genre_dur, width='stretch')
    
            if not genre_duration_stats.empty:
                # Sort by mean duration
                sorted_duration = genre_duration_stats.sort_values(by="median", ascending=False)
            
                # Longest and shortest genres
                longest_genre = sorted_duration.index[0]
                shortest_genre = sorted_duration.index[-1]
                longest_dur = sorted_duration["median"].iloc[0]
                shortest_dur = sorted_duration["median"].iloc[-1]
                
                with st.expander("ℹ️ More Information"):
                    st.info(f"- **{longest_genre}** has longer tracks (median {longest_dur:.2f} min), while **{shortest_genre}** favors shorter tracks (median {shortest_dur:.2f} min).")
                    st.info("- Duration preferences by genre may reflect traditional formats, audience attention spans, or production styles in UK music culture.")
            
        else:
            st.warning("`genre_duration_stats` or `df_merged` not found. Please ensure the genre analysis section was run.")
    
        # --- 3D Multivariate Analysis ---
        st.subheader("🎶 Genre Multivariate Analysis (3D)")
        genre_popularity_median = filtered_df.groupby('genre')['popularity'].median()
        genre_duration_median = filtered_df.groupby('genre')['duration_min'].median()
        genre_explicitness_percentage = filtered_df.groupby('genre')['is_explicit'].mean() * 100
    
        genre_3d_df = pd.DataFrame({
            'Genre': genre_popularity_median.index,
            'Median Popularity': genre_popularity_median.values,
            'Median Duration (min)': genre_duration_median.values,
            'Explicit Content (%)': genre_explicitness_percentage.values
        }).fillna(0)
    
        fig3d = px.scatter_3d(
            genre_3d_df,
            x="Median Popularity", y="Median Duration (min)", z="Explicit Content (%)",
            color="Genre", hover_name="Genre",
            hover_data={"Median Popularity":":.2f","Median Duration (min)":":.2f","Explicit Content (%)":":.2f"},
            title="<b>3D Multivariate Analysis: Genre Characteristics</b>",
            height=700
        )
        fig3d.update_layout(
            scene=dict(
                xaxis_title="Median Popularity (0-100)",
                yaxis_title="Median Duration (minutes)",
                zaxis_title="Explicit Content (%)"
            ),
            margin=dict(l=0,r=0,b=0,t=50),
            legend_title_text="Genres", autosize=True
        )
        st.plotly_chart(fig3d, width='stretch')
    
        with st.expander("📊 Insights from 3D Genre Analysis"):
            st.info("- Genres with higher popularity tend to cluster with moderate durations.")
            st.info("- Explicit content percentage varies widely across genres, showing cultural differences.")
            st.info("- The 3D view helps identify optimal combinations of duration, popularity, and explicitness for UK market penetration.")
    
        with st.expander("📋 Aggregated Genre Metrics Table"):
            st.dataframe(genre_3d_df.style.format({
                "Mean Popularity":"{:.2f}",
                "Mean Duration (min)":"{:.2f}",
                "Explicit Content (%)":"{:.2f}"
            }))
    
        # --- Genre Definitions ---
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
        df_genres = pd.DataFrame(list(genre_definitions.items()), columns=["Genre","Definition"])
        with st.expander("ℹ️ General Knowledge Regarding The Context"):
            st.subheader("🎵 Major Song Genres and Definitions")
            st.table(df_genres)
    
        st.markdown("---")
        
        if is_any_filter_different:
            st.toast("🔍 Deep Dive Into Genre Comparisons — Filtered V/S Baseline")
        else:
            st.toast("💡 Genre Visualizations Are Ready For Baseline View")
        
        
    with tabs[4]:
        # --- Recommendation 5: Multivariate Analysis ---
        st.subheader("🧮 Multivariate Analysis (3D Scatter Plot)")
        st.markdown("""
        This 3D scatter plot visualizes the interplay between track duration, number of artists, and popularity,
        with points colored by chart success and distinguished by duration category (short-form vs. long-form).
        """)
    
        if not filtered_df.empty and all(col in filtered_df.columns for col in ['duration_min','num_artists','popularity','chart_success','duration_category']):
            # Interactive Plotly 3D scatter
            fig_3d_scatter = px.scatter_3d(
                filtered_df,
                x="duration_min",
                y="num_artists",
                z="popularity",
                symbol="chart_success",
                color="duration_category",   # short-form vs long-form
                opacity=0.6,
                size_max=10,
                labels={
                    "duration_min":"Duration (minutes)",
                    "num_artists":"Number of Artists",
                    "popularity":"Popularity",
                    "chart_success":"Chart Success"
                },
                title="3D Scatter Plot: Duration, Artists, Popularity by Chart Success & Duration Category",
                #color_discrete_sequence=px.colors.sequential.algae_r,
                color_discrete_map={
                    "short-form": "gold",    # custom color for short-form
                    "long-form": "royalblue" # custom color for long-form
                }
            )
    
            # Customize layout
            fig_3d_scatter.update_layout(
                legend_title_text="⏳ Duration Category & 🏆 Chart Success", autosize=True,
                scene=dict(
                    xaxis_title="⏱️ Duration (minutes)",
                    yaxis_title="👥 Number of Artists",
                    zaxis_title="🔥 Popularity"
                ),
                title=dict(
                    text="✨ 3D Scatter Plot: Duration, Artists, Popularity 🎶",
                    x=0.5,
                    xanchor="center",
                    yanchor="top"
                ),
                margin=dict(l=50, r=150, b=50, t=100),  # extra right margin for legend/colorbar
                legend=dict(
                    x=1.05,   # push legend outside to the right
                    y=0.5,
                    xanchor="left",
                    yanchor="top"
                ),
                coloraxis=dict(
                    colorbar=dict(
                        x=1.25,   # push colorbar further right so it doesn’t overlap legend
                        y=0.5,
                        xanchor="left",
                        yanchor="middle",
                        len=0.7   # shrink bar length if needed
                    )
                )
            )
            
            st.plotly_chart(fig_3d_scatter, width='stretch')
            
            with st.expander("ℹ️ More Information"):
                st.info("🏆 Top 10 tracks cluster with moderate durations and collaborations.")
                st.success("📈 Multivariate clusters indicate optimal combinations for UK market penetration.")
                
        else:
            st.warning("Required data for 3D scatter plot not found. Please ensure the multivariate analysis section was run.")
    
        st.markdown('---')
        
        if is_any_filter_different:
            st.toast("🔗 Deep Dive Into All-In-One Relationships")
        else:
            st.toast("🧠 All-In-One Plot Ready For Baseline View")
        
    with tabs[5]:
        st.subheader("📌 Conclusion")
        st.markdown("""
        This section synthesizes insights from predictive modeling, feature engineering,
        time series, genre analysis, and multivariate analysis to provide actionable strategies
        for the UK music market.
        """)
    
        # --- Best Model Summary ---
        st.markdown("#### ⭐ Best Model Insights")
        if 'accuracy_summary_df' in locals():
            if is_any_filter_different:
                # Compare baseline vs filtered model accuracies
                overall_best_row = st.session_state["accuracy_summary_df_original"].loc[
                    st.session_state["accuracy_summary_df_original"]['Accuracy'].idxmax()
                ]
                st.success(f"🏆 Overall Best Model: **{overall_best_row['Model']}** with Accuracy = {overall_best_row['Accuracy']:.4f}")
                st.info(f"Filtered Best Model: **{best_model_name}** with Accuracy = {best_model_accuracy:.4f}")
                if overall_best_row['Model'] == best_model_name:
                    st.success("Filtered subset mirrors baseline → same best model.")
                else:
                    st.warning("Filtered subset highlights a different best model → potential shift in predictive performance.")
            else:
                st.success(f"🏆 Best Performing Model (Baseline): **{best_model_name}** with Accuracy = {best_model_accuracy:.4f}")
        else:
            st.warning("Model accuracy summary not found. Please ensure predictive modeling section was run.")
            
        st.markdown("#### 📊 Feature Engineering Comparison Insights")
        if is_any_filter_different:
            # --- Day of Week Comparison ---
            with st.expander("📅 Chart Success by Day of Week (Baseline vs Filtered)"):
                st.write("**Baseline:**")
                st.dataframe(st.session_state["baseline_day_of_week_counts"].unstack(fill_value=0))
                st.write("**Filtered:**")
                st.dataframe(filtered_df.groupby('day_of_week')['chart_success'].value_counts().unstack(fill_value=0)) 
        
            # --- Duration × Num Artists Comparison ---
            with st.expander("🎶 Duration × Num Artists vs Popularity (Baseline vs Filtered)"):
                st.write("**Baseline Sample:**")
                st.dataframe(st.session_state["baseline_duration_x_num_artists"].head(10))
                st.write("**Filtered Sample:**")
                st.dataframe(filtered_df[['duration_x_num_artists','popularity','chart_success']].head(10))
                
        
            # --- Explicit Duration Comparison ---
            with st.expander("⚡ Explicit Duration by Chart Success (Baseline vs Filtered)"):
                st.write("**Baseline:**")
                st.dataframe(st.session_state["baseline_explicit_duration"].groupby('chart_success')['explicit_duration'].mean())
                st.write("**Filtered:**")
                st.dataframe(filtered_df.groupby('chart_success')['explicit_duration'].mean())
        else:
            # --- Baseline Only ---
            with st.expander("📅 Chart Success by Day of Week (Baseline)"):
                st.dataframe(st.session_state["baseline_day_of_week_counts"].unstack(fill_value=0))
        
            with st.expander("🎶 Duration × Num Artists vs Popularity (Baseline)"):
                st.dataframe(st.session_state["baseline_duration_x_num_artists"].head(10))
        
            with st.expander("⚡ Explicit Duration by Chart Success (Baseline)"):
                st.dataframe(st.session_state["baseline_explicit_duration"].groupby('chart_success')['explicit_duration'].mean())
        
        # --- Time Series Insight ---
        if 'unique_artists_per_day' in locals():
            st.markdown("#### 📈 Time Series Insights")
        
            if is_any_filter_different:
                overall_unique_artists = st.session_state["baseline_unique_artists_per_day"]
                filtered_unique_artists = filtered_df.groupby('date')['artist'].nunique()
        
                st.info(
                    f"Overall Top 3 Days with Highest Artist Diversity: "
                    f"{', '.join(overall_unique_artists.sort_values(ascending=False).head(3).index.astype(str))}"
                )
                st.warning(
                    f"Overall Bottom 3 Days with Lowest Artist Diversity: "
                    f"{', '.join(overall_unique_artists.sort_values(ascending=True).head(3).index.astype(str))}"
                )
        
                st.info(
                    f"Filtered Top 3 Days with Highest Artist Diversity: "
                    f"{', '.join(filtered_unique_artists.sort_values(ascending=False).head(3).index.astype(str))}"
                )
                st.warning(
                    f"Filtered Bottom 3 Days with Lowest Artist Diversity: "
                    f"{', '.join(filtered_unique_artists.sort_values(ascending=True).head(3).index.astype(str))}"
                )
            else:
                overall_unique_artists = st.session_state["baseline_unique_artists_per_day"]
        
                st.info(
                    f"Top 3 Days with Highest Artist Diversity: "
                    f"{', '.join(overall_unique_artists.sort_values(ascending=False).head(3).index.astype(str))}"
                )
                st.warning(
                    f"Bottom 3 Days with Lowest Artist Diversity: "
                    f"{', '.join(overall_unique_artists.sort_values(ascending=True).head(3).index.astype(str))}"
                )
        
        # --- Multivariate Insight ---
        def get_top_correlations(df, cols, n=3):
            """
            Compute top correlations among selected columns in tidy format.
            Removes self-correlations and duplicate pairs (e.g., num-popularity vs popularity-num).
            Returns top n strongest relationships.
            """
            corr = df[cols].corr()
            corr_df = (
                corr.unstack()
                .reset_index()
                .rename(columns={'level_0': 'Variable 1', 'level_1': 'Variable 2', 0: 'Correlation'})
                .query("`Variable 1` != `Variable 2`")   # remove self-correlations
            )
        
            # Ensure consistent ordering of variable pairs to avoid duplicates
            corr_df["Pair"] = corr_df.apply(
                lambda row: tuple(sorted([row["Variable 1"], row["Variable 2"]])), axis=1
            )
            corr_df = corr_df.drop_duplicates(subset="Pair")
        
            # Sort by absolute correlation strength
            corr_df = corr_df.sort_values(by="Correlation", ascending=False).head(n)
        
            return corr_df[["Variable 1", "Variable 2", "Correlation"]]
        
        # --- Multivariate Insight ---
        if not filtered_df.empty and all(col in filtered_df.columns for col in ['duration_min','num_artists','popularity','chart_success']):
            st.markdown("#### 🔮 Multivariate Insights")
        
            if is_any_filter_different:
                overall_corr_df = get_top_correlations(df_merged, ['duration_min','num_artists','popularity'])
                filtered_corr_df = get_top_correlations(filtered_df, ['duration_min','num_artists','popularity'])
        
                st.info("Overall Correlation Matrix (Top 3 strongest relationships):")
                st.dataframe(overall_corr_df)
        
                st.info("Filtered Correlation Matrix (Top 3 strongest relationships):")
                st.dataframe(filtered_corr_df)
        
                st.success("Filtered vs baseline correlations highlight how collaboration and duration interact differently under subset conditions.")
            else:
                overall_corr_df = get_top_correlations(df_merged, ['duration_min','num_artists','popularity'])
        
                st.info("Baseline Correlation Matrix (Top 3 strongest relationships):")
                st.dataframe(overall_corr_df)
        
        # --- Save baseline genre metrics once ---
        if "baseline_genre_popularity_stats" not in st.session_state:
            st.session_state["baseline_genre_popularity_stats"] = (
                df_merged.groupby('genre')['popularity']
                .agg(['mean', 'median', 'std'])
                .sort_values(by=["mean", "median", "std"], ascending=[False, False, False])
            )
        
        if "baseline_genre_explicitness_stats" not in st.session_state:
            st.session_state["baseline_genre_explicitness_stats"] = (
                df_merged.groupby('genre')['is_explicit']
                .agg(['mean', 'median', 'std'])
                .sort_values(by=["mean", "median", "std"], ascending=[False, False, False]) * 100
            )
        
        if "baseline_genre_duration_stats" not in st.session_state:
            st.session_state["baseline_genre_duration_stats"] = (
                df_merged.groupby('genre')['duration_min']
                .agg(['mean', 'median', 'std'])
                .sort_values(by=["mean", "median", "std"], ascending=[False, False, False])
            )
        
        # --- Genre Insights ---
        if len(filtered_df) == 0:
            st.warning("Please reload the website and wait for the OpenAI Model to predict and calculate Genre Specific Statistics for the baseline metrics. Then select any required custom filter option to see the conclusion.")
        else:
            st.markdown("#### 🎵 Genre Insights")
            
            # --- Genre Popularity ---
            with st.expander("🎶 **Genre Popularity Insights**"):
                overall_pop_stats = st.session_state["baseline_genre_popularity_stats"]
                filtered_pop_stats = (
                    filtered_df.groupby("genre")["popularity"]
                    .agg(["mean", "median", "std"])
                    .sort_values(by=["mean", "median", "std"], ascending=[False, False, False])
                )
                summary_df = pd.DataFrame({
                    "Metric": ["Mean Popularity", "Median Popularity", " Popularity Variability (Std Dev)"],
                    "Overall Top 3": [
                        ", ".join(overall_pop_stats.sort_values(by='mean', ascending=False).head(3).index),
                        ", ".join(overall_pop_stats.sort_values(by='median', ascending=False).head(3).index),
                        ", ".join(overall_pop_stats.sort_values(by='std', ascending=False).head(3).index)
                    ],
                    "Overall Bottom 3": [
                        ", ".join(overall_pop_stats.sort_values(by='mean').head(3).index),
                        ", ".join(overall_pop_stats.sort_values(by='median').head(3).index),
                        ", ".join(overall_pop_stats.sort_values(by='std').head(3).index)
                    ],
                    "Filtered Top 3": [
                        ", ".join(filtered_pop_stats.sort_values(by='mean', ascending=False).head(3).index) if is_any_filter_different else "-",
                        ", ".join(filtered_pop_stats.sort_values(by='median', ascending=False).head(3).index) if is_any_filter_different else "-",
                        ", ".join(filtered_pop_stats.sort_values(by='std', ascending=False).head(3).index) if is_any_filter_different else "-"
                    ],
                    "Filtered Bottom 3": [
                        ", ".join(filtered_pop_stats.sort_values(by='mean').head(3).index) if is_any_filter_different else "-",
                        ", ".join(filtered_pop_stats.sort_values(by='median').head(3).index) if is_any_filter_different else "-",
                        ", ".join(filtered_pop_stats.sort_values(by='std').head(3).index) if is_any_filter_different else "-"
                    ]
                })
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"Overall Top 3 Genres (Mean): **{', '.join(overall_pop_stats.sort_values(by='mean', ascending=False).head(3).index)}**")
                    st.warning(f"Overall Bottom 3 Genres (Mean): **{', '.join(overall_pop_stats.sort_values(by='mean').head(3).index)}**")
                    if is_any_filter_different:
                        st.success(f"Filtered Top 3 Genres (Mean): **{', '.join(filtered_pop_stats.sort_values(by='mean', ascending=False).head(3).index)}**")
                        st.warning(f"Filtered Bottom 3 Genres (Mean): **{', '.join(filtered_pop_stats.sort_values(by='mean').head(3).index)}**")
                with col2:
                    st.info(f"Overall Top 3 Genres (Median): **{', '.join(overall_pop_stats.sort_values(by='median', ascending=False).head(3).index)}**")
                    st.warning(f"Overall Bottom 3 Genres (Median): **{', '.join(overall_pop_stats.sort_values(by='median').head(3).index)}**")
                    if is_any_filter_different:
                        st.success(f"Filtered Top 3 Genres (Median): **{', '.join(filtered_pop_stats.sort_values(by='median', ascending=False).head(3).index)}**")
                        st.warning(f"Filtered Bottom 3 Genres (Median): **{', '.join(filtered_pop_stats.sort_values(by='median').head(3).index)}**")
                with col3:
                    st.info(f"Overall Top 3 Genres (Std Dev): **{', '.join(overall_pop_stats.sort_values(by='std', ascending=False).head(3).index)}**")
                    st.warning(f"Overall Bottom 3 Genres (Std Dev): **{', '.join(overall_pop_stats.sort_values(by='std').head(3).index)}**")
                    if is_any_filter_different:
                        st.success(f"Filtered Top 3 Genres (Std Dev): **{', '.join(filtered_pop_stats.sort_values(by='std', ascending=False).head(3).index)}**")
                        st.warning(f"Filtered Bottom 3 Genres (Std Dev): **{', '.join(filtered_pop_stats.sort_values(by='std').head(3).index)}**")
            
                with st.expander("Table Form"):
                    st.dataframe(summary_df, width='stretch', hide_index=True)
            
            # --- Genre Explicitness ---
            with st.expander("⚡ **Genre Explicitness Insights**"):
                overall_exp_stats = st.session_state["baseline_genre_explicitness_stats"]
                filtered_exp_stats = (
                    filtered_df.groupby("genre")["is_explicit"]
                    .agg(["mean", "median", "std"])
                    .sort_values(by=["mean", "median", "std"], ascending=[False, False, False]) * 100
                )
                summary_exp_df = pd.DataFrame({
                    "Metric": ["Mean Explicitness (%)", "Median Explicitness (%)", "Explicitness Variability (Std Dev of %)"],
                    "Overall Top 3": [
                        ", ".join(overall_exp_stats.sort_values(by='mean', ascending=False).head(3).index),
                        ", ".join(overall_exp_stats.sort_values(by='median', ascending=False).head(3).index),
                        ", ".join(overall_exp_stats.sort_values(by='std', ascending=False).head(3).index)
                    ],
                    "Overall Bottom 3": [
                        ", ".join(overall_exp_stats.sort_values(by='mean').head(3).index),
                        ", ".join(overall_exp_stats.sort_values(by='median').head(3).index),
                        ", ".join(overall_exp_stats.sort_values(by='std').head(3).index)
                    ],
                    "Filtered Top 3": [
                        ", ".join(filtered_exp_stats.sort_values(by='mean', ascending=False).head(3).index) if is_any_filter_different else "-",
                        ", ".join(filtered_exp_stats.sort_values(by='median', ascending=False).head(3).index) if is_any_filter_different else "-",
                        ", ".join(filtered_exp_stats.sort_values(by='std', ascending=False).head(3).index) if is_any_filter_different else "-"
                    ],
                    "Filtered Bottom 3": [
                        ", ".join(filtered_exp_stats.sort_values(by='mean').head(3).index) if is_any_filter_different else "-",
                        ", ".join(filtered_exp_stats.sort_values(by='median').head(3).index) if is_any_filter_different else "-",
                        ", ".join(filtered_exp_stats.sort_values(by='std').head(3).index) if is_any_filter_different else "-"
                    ]
                })
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"Overall Top 3 Genres (Mean Explicitness): {', '.join(overall_exp_stats.sort_values(by='mean', ascending=False).head(3).index)}")
                    st.warning(f"Overall Bottom 3 Genres (Mean Explicitness): {', '.join(overall_exp_stats.sort_values(by='mean').head(3).index)}")
                    if is_any_filter_different:
                        st.success(f"Filtered Top 3 Genres (Mean Explicitness): {', '.join(filtered_exp_stats.sort_values(by='mean', ascending=False).head(3).index)}")
                        st.warning(f"Filtered Bottom 3 Genres (Mean Explicitness): {', '.join(filtered_exp_stats.sort_values(by='mean').head(3).index)}")
                with col2:
                    st.info(f"Overall Top 3 Genres (Median Explicitness): {', '.join(overall_exp_stats.sort_values(by='median', ascending=False).head(3).index)}")
                    st.warning(f"Overall Bottom 3 Genres (Median Explicitness): {', '.join(overall_exp_stats.sort_values(by='median').head(3).index)}")
                    if is_any_filter_different:
                        st.success(f"Filtered Top 3 Genres (Median Explicitness): {', '.join(filtered_exp_stats.sort_values(by='median', ascending=False).head(3).index)}")
                        st.warning(f"Filtered Bottom 3 Genres (Median Explicitness): {', '.join(filtered_exp_stats.sort_values(by='median').head(3).index)}")
                with col3:
                    st.info(f"Overall Top 3 Genres (Std Dev Explicitness): {', '.join(overall_exp_stats.sort_values(by='std', ascending=False).head(3).index)}")
                    st.warning(f"Overall Bottom 3 Genres (Std Dev Explicitness): {', '.join(overall_exp_stats.sort_values(by='std').head(3).index)}")
                    if is_any_filter_different:
                        st.success(f"Filtered Top 3 Genres (Std Dev Explicitness): {', '.join(filtered_exp_stats.sort_values(by='std', ascending=False).head(3).index)}")
                        st.warning(f"Filtered Bottom 3 Genres (Std Dev Explicitness): {', '.join(filtered_exp_stats.sort_values(by='std').head(3).index)}")
            
                with st.expander("Table Form"):
                    st.dataframe(summary_exp_df, width='stretch', hide_index=True)
                
            # --- Genre Duration --- 
            with st.expander("⏳ **Genre Duration Insights**"):
                overall_dur_stats = st.session_state["baseline_genre_duration_stats"]
                filtered_dur_stats = (
                    filtered_df.groupby("genre")["duration_min"]
                    .agg(["mean", "median", "std"])
                    .sort_values(by=["mean", "median", "std"], ascending=[False, False, False])
                )
                summary_dur_df = pd.DataFrame({
                    "Metric": ["Mean Duration (m)", "Median Duration (m)", "Duration Variability (Std Dev in minutes)"],
                    "Overall Top 3": [
                        ", ".join(overall_dur_stats.sort_values(by='mean', ascending=False).head(3).index),
                        ", ".join(overall_dur_stats.sort_values(by='median', ascending=False).head(3).index),
                        ", ".join(overall_dur_stats.sort_values(by='std', ascending=False).head(3).index)
                    ],
                    "Overall Bottom 3": [
                        ", ".join(overall_dur_stats.sort_values(by='mean').head(3).index),
                        ", ".join(overall_dur_stats.sort_values(by='median').head(3).index),
                        ", ".join(overall_dur_stats.sort_values(by='std').head(3).index)
                    ],
                    "Filtered Top 3": [
                        ", ".join(filtered_dur_stats.sort_values(by='mean', ascending=False).head(3).index) if is_any_filter_different else "-",
                        ", ".join(filtered_dur_stats.sort_values(by='median', ascending=False).head(3).index) if is_any_filter_different else "-",
                        ", ".join(filtered_dur_stats.sort_values(by='std', ascending=False).head(3).index) if is_any_filter_different else "-"
                    ],
                    "Filtered Bottom 3": [
                        ", ".join(filtered_dur_stats.sort_values(by='mean').head(3).index) if is_any_filter_different else "-",
                        ", ".join(filtered_dur_stats.sort_values(by='median').head(3).index) if is_any_filter_different else "-",
                        ", ".join(filtered_dur_stats.sort_values(by='std').head(3).index) if is_any_filter_different else "-"
                    ]
                })
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"Overall Top 3 Genres (Mean Duration): {', '.join(overall_dur_stats.sort_values(by='mean', ascending=False).head(3).index)}")
                    st.warning(f"Overall Bottom 3 Genres (Mean Duration): {', '.join(overall_dur_stats.sort_values(by='mean').head(3).index)}")
                    if is_any_filter_different:
                        st.success(f"Filtered Top 3 Genres (Mean Duration): {', '.join(filtered_dur_stats.sort_values(by='mean', ascending=False).head(3).index)}")
                        st.warning(f"Filtered Bottom 3 Genres (Mean Duration): {', '.join(filtered_dur_stats.sort_values(by='mean').head(3).index)}")
                with col2:
                    st.info(f"Overall Top 3 Genres (Median Duration): {', '.join(overall_dur_stats.sort_values(by='median', ascending=False).head(3).index)}")
                    st.warning(f"Overall Bottom 3 Genres (Median Duration): {', '.join(overall_dur_stats.sort_values(by='median').head(3).index)}")
                    if is_any_filter_different:
                        st.success(f"Filtered Top 3 Genres (Median Duration): {', '.join(filtered_dur_stats.sort_values(by='median', ascending=False).head(3).index)}")
                        st.warning(f"Filtered Bottom 3 Genres (Median Duration): {', '.join(filtered_dur_stats.sort_values(by='median').head(3).index)}")
                with col3:
                    st.info(f"Overall Top 3 Genres (Std Dev Duration): {', '.join(overall_dur_stats.sort_values(by='std', ascending=False).head(3).index)}")
                    st.warning(f"Overall Bottom 3 Genres (Std Dev Duration): {', '.join(overall_dur_stats.sort_values(by='std').head(3).index)}")
                    if is_any_filter_different:
                        st.success(f"Filtered Top 3 Genres (Std Dev Duration): {', '.join(filtered_dur_stats.sort_values(by='std', ascending=False).head(3).index)}")
                        st.warning(f"Filtered Bottom 3 Genres (Std Dev Duration): {', '.join(filtered_dur_stats.sort_values(by='std').head(3).index)}")
                with st.expander("Table Form"):
                    st.dataframe(summary_dur_df, width='stretch', hide_index=True)
            st.info("This project provides both structural and cultural intelligence into the UK music market by comparing the current filter view with the full dataset baseline. Recommendations balance the selected subset with the overall UK market context.")
            st.success("✅ The dashboard is useful for Atlantic Recording Corporation to identify UK listener preference indicators, collaboration strengths, and content composition trends in real time.")
            
        st.divider()
        if is_any_filter_different:
            st.toast("🎤 Final Note — Filtered Insights Concluded With Harmony")
        else:
            st.toast("🏁 Key Takeaways Sealed - Baseline Conclusion Ready - Good Luck For Further Journey")
        
        st.markdown(
            """
            <div style='text-align: center; color: grey; font-size: 14px;'>
                🎵 In the rhythm of data, the melody takes flight,  
                <br>this dashboard turns numbers into <b>insightful light</b>.  
                <br>Like chart-topping tracks that find their beat,  
                <br>lyrics here make the UK's music listeners complete.  
                <br>
                <br>From artistry’s spark to audience’s embrace,  
                <br>we harmonize trends in a balanced space.  
                <br>So let the metrics sing, let the <b>visuals rhyme</b>,  
                <br>guiding the industry in <b>intangibility of time</b>. 🎶<br><hr>
                🔖 Created & Powered by <b>Prathamesh Bhurke</b><br>
                <a href="https://github.com/Prathamesh666/United-Kingdom-Music-Market-Structure-Artist-Diversity-Content-Localization-Analysis./" target="_blank">
                    📂 GitHub Repository
                </a><br>
                <a href="https://prathamesh666.github.io/United-Kingdom-Music-Market-Structure-Artist-Diversity-Content-Localization-Analysis./Research%20Paper.html" target="_blank">
                    📑 Research Paper
                </a>
                
            </div>
            """,
            unsafe_allow_html=True
        )
