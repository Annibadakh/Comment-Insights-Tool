import csv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import streamlit as st
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Replace with your YouTube Data API key
DEVELOPER_KEY = st.secrets["YOUTUBE_API_KEY"]
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# Create a YouTube API client
try:
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
except Exception as e:
    st.error(f"Error initializing YouTube API client: {e}")
    youtube = None


def get_channel_id(video_id: str) -> str:
    """Retrieve the channel ID for a given video ID."""
    try:
        response = youtube.videos().list(part='snippet', id=video_id).execute()
        return response['items'][0]['snippet']['channelId']
    except Exception as e:
        st.error(f"Error fetching channel ID: {e}")
        return None


def save_video_comments_to_csv(video_id: str) -> str:
    """Retrieve and save video comments to a CSV file."""
    comments = []
    try:
        results = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText'
        ).execute()

        while results:
            for item in results['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                username = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                comments.append([username, comment])

            # Check for next page
            if 'nextPageToken' in results:
                nextPage = results['nextPageToken']
                results = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    textFormat='plainText',
                    pageToken=nextPage
                ).execute()
            else:
                break

        # Save comments to CSV
        filename = f"{video_id}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Username', 'Comment'])
            writer.writerows(comments)

        return filename

    except HttpError as e:
        st.error(f"Error fetching comments: {e}")
        return None


def get_video_stats(video_id: str) -> dict:
    """Fetch video statistics such as view count, like count, etc."""
    try:
        response = youtube.videos().list(part='statistics', id=video_id).execute()
        return response['items'][0]['statistics']
    except HttpError as e:
        st.error(f"Error fetching video stats: {e}")
        return None


def get_channel_info(channel_id: str) -> dict:
    """Fetch channel information such as title, subscriber count, and description."""
    try:
        response = youtube.channels().list(
            part='snippet,statistics,brandingSettings',
            id=channel_id
        ).execute()

        channel_info = {
            'channel_title': response['items'][0]['snippet']['title'],
            'video_count': response['items'][0]['statistics']['videoCount'],
            'channel_logo_url': response['items'][0]['snippet']['thumbnails']['high']['url'],
            'channel_created_date': response['items'][0]['snippet']['publishedAt'],
            'subscriber_count': response['items'][0]['statistics']['subscriberCount'],
            'channel_description': response['items'][0]['snippet']['description']
        }

        return channel_info

    except HttpError as e:
        st.error(f"Error fetching channel info: {e}")
        return None


# Example usage in Streamlit
if __name__ == "__main__":
    st.title("YouTube Data Fetcher")

    # Input for video ID
    video_id_input = st.text_input("Enter YouTube Video ID:", placeholder="e.g., dQw4w9WgXcQ")
    if st.button("Fetch Data") and video_id_input:
        # Fetch channel ID
        channel_id = get_channel_id(video_id_input)
        if channel_id:
            st.write(f"Channel ID: {channel_id}")

        # Fetch video stats
        video_stats = get_video_stats(video_id_input)
        if video_stats:
            st.write("Video Stats:", video_stats)

        # Fetch comments and save to CSV
        csv_file = save_video_comments_to_csv(video_id_input)
        if csv_file:
            st.success(f"Comments saved to {csv_file}")

        # Fetch channel info
        if channel_id:
            channel_info = get_channel_info(channel_id)
            if channel_info:
                st.write("Channel Info:", channel_info)
