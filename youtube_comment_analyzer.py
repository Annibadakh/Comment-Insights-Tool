import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
from tqdm import tqdm

# Load environment variables
load_dotenv()

class YouTubeCommentAnalyzer:
    def __init__(self, api_key=None):
        """Initialize the YouTube API client."""
        if api_key is None:
            api_key = os.getenv("YOUTUBE_API_KEY")
            if not api_key:
                raise ValueError("YouTube API key is required. Set YOUTUBE_API_KEY environment variable or pass as parameter.")
        
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        
    def extract_video_id(self, url):
        """Extract the video ID from a YouTube URL."""
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
        if video_id_match:
            return video_id_match.group(1)
        return None
    
    def get_video_details(self, video_id):
        """Get basic information about a YouTube video."""
        try:
            response = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            ).execute()
            
            if not response['items']:
                return None
            
            video_data = response['items'][0]
            return {
                'title': video_data['snippet']['title'],
                'channel': video_data['snippet']['channelTitle'],
                'published_at': video_data['snippet']['publishedAt'],
                'view_count': int(video_data['statistics'].get('viewCount', 0)),
                'like_count': int(video_data['statistics'].get('likeCount', 0)),
                'comment_count': int(video_data['statistics'].get('commentCount', 0)),
                'duration': video_data['contentDetails']['duration']
            }
        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            return None
    
    def get_comments(self, video_id, max_comments=100):
        """Get comments from a YouTube video."""
        comments = []
        try:
            # Get the top-level comments
            next_page_token = None
            comment_count = 0
            
            with tqdm(total=min(max_comments, 100), desc="Fetching comments") as pbar:
                while comment_count < max_comments:
                    response = self.youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        maxResults=min(100, max_comments - comment_count),
                        pageToken=next_page_token,
                        textFormat='plainText'
                    ).execute()
                    
                    for item in response['items']:
                        comment = item['snippet']['topLevelComment']['snippet']
                        comments.append({
                            'author': comment['authorDisplayName'],
                            'text': comment['textDisplay'],
                            'published_at': comment['publishedAt'],
                            'like_count': comment['likeCount'],
                            'reply_count': item['snippet']['totalReplyCount']
                        })
                        comment_count += 1
                        pbar.update(1)
                    
                    # Check if there are more comments
                    next_page_token = response.get('nextPageToken')
                    if not next_page_token or comment_count >= max_comments:
                        break
            
            return comments
        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            return []
    
    def analyze_sentiment(self, comments):
        """Analyze the sentiment of comments using TextBlob."""
        for comment in tqdm(comments, desc="Analyzing sentiment"):
            analysis = TextBlob(comment['text'])
            comment['polarity'] = analysis.sentiment.polarity
            comment['subjectivity'] = analysis.sentiment.subjectivity
            
            # Categorize sentiment
            if comment['polarity'] > 0.1:
                comment['sentiment'] = 'positive'
            elif comment['polarity'] < -0.1:
                comment['sentiment'] = 'negative'
            else:
                comment['sentiment'] = 'neutral'
        
        return comments
    
    def extract_common_words(self, comments, min_length=4, max_words=100, stopwords=None):
        """Extract most common words from comments."""
        if stopwords is None:
            # Default English stopwords
            stopwords = {'the', 'and', 'is', 'of', 'to', 'in', 'that', 'it', 'with', 'for', 'on', 'at', 'this', 'was', 'are', 'be', 'as', 'but', 'or', 'have', 'from', 'by', 'not', 'what', 'all', 'were', 'when', 'we', 'there', 'been', 'one', 'will', 'would', 'who', 'you', 'your', 'they', 'their', 'has', 'had', 'how', 'up', 'his', 'her', 'an', 'my', 'so', 'if', 'out', 'about', 'me', 'no', 'more', 'do', 'can'}
        
        all_words = []
        for comment in comments:
            words = re.findall(r'\b[a-zA-Z]{%d,}\b' % min_length, comment['text'].lower())
            words = [word for word in words if word not in stopwords]
            all_words.extend(words)
        
        return Counter(all_words).most_common(max_words)
    
    def generate_report(self, url, max_comments=100):
        """Generate a full analysis report for a YouTube video."""
        video_id = self.extract_video_id(url)
        if not video_id:
            return {"error": "Invalid YouTube URL"}
        
        print(f"Analyzing video ID: {video_id}")
        
        # Get video details
        details = self.get_video_details(video_id)
        if not details:
            return {"error": "Could not retrieve video details"}
        
        # Get comments
        comments = self.get_comments(video_id, max_comments)
        if not comments:
            return {"error": "No comments found or comments are disabled"}
        
        # Analyze sentiment
        analyzed_comments = self.analyze_sentiment(comments)
        
        # Extract common words
        common_words = self.extract_common_words(analyzed_comments)
        
        # Create dataframe
        df = pd.DataFrame(analyzed_comments)
        
        # Generate insights
        sentiment_counts = df['sentiment'].value_counts()
        avg_polarity = df['polarity'].mean()
        avg_subjectivity = df['subjectivity'].mean()
        popular_comments = df.sort_values('like_count', ascending=False).head(5)
        
        # Compile results
        results = {
            "video_details": details,
            "comment_stats": {
                "total_analyzed": len(analyzed_comments),
                "sentiment_distribution": sentiment_counts.to_dict(),
                "avg_polarity": avg_polarity,
                "avg_subjectivity": avg_subjectivity
            },
            "common_words": dict(common_words),
            "top_comments": popular_comments.to_dict('records'),
            "all_comments": df
        }
        
        return results
    
    def visualize_insights(self, results):
        """Visualize the insights from the comment analysis."""
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        # Set up the figure
        plt.figure(figsize=(15, 12))
        
        # 1. Video details
        plt.subplot(3, 2, 1)
        details = results["video_details"]
        plt.axis('off')
        info_text = (
            f"Title: {details['title']}\n"
            f"Channel: {details['channel']}\n"
            f"Views: {details['view_count']:,}\n"
            f"Likes: {details['like_count']:,}\n"
            f"Comments: {details['comment_count']:,}\n"
        )
        plt.text(0.1, 0.5, info_text, fontsize=12)
        plt.title("Video Information", fontsize=14)
        
        # 2. Sentiment distribution pie chart
        plt.subplot(3, 2, 2)
        sentiment_dist = results["comment_stats"]["sentiment_distribution"]
        labels = list(sentiment_dist.keys())
        sizes = list(sentiment_dist.values())
        colors = ['green', 'gray', 'red']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title("Comment Sentiment Distribution", fontsize=14)
        
        # 3. Word cloud
        plt.subplot(3, 2, 3)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(results["common_words"])
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Most Common Words in Comments", fontsize=14)
        
        # 4. Sentiment over comment likes
        plt.subplot(3, 2, 4)
        df = results["all_comments"]
        sentiment_order = ['positive', 'neutral', 'negative']
        sns.boxplot(x='sentiment', y='like_count', data=df, order=sentiment_order)
        plt.title("Likes Distribution by Sentiment", fontsize=14)
        
        # 5. Polarity vs. Subjectivity scatter plot
        plt.subplot(3, 2, 5)
        plt.scatter(df['polarity'], df['subjectivity'], alpha=0.5)
        plt.xlim(-1, 1)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Polarity (Negative to Positive)')
        plt.ylabel('Subjectivity (Factual to Personal)')
        plt.title("Comment Sentiment Analysis", fontsize=14)
        
        # 6. Top 10 most frequent words
        plt.subplot(3, 2, 6)
        top_words = dict(list(results["common_words"].items())[:10])
        words = list(top_words.keys())
        freqs = list(top_words.values())
        y_pos = np.arange(len(words))
        plt.barh(y_pos, freqs)
        plt.yticks(y_pos, words)
        plt.xlabel('Frequency')
        plt.title("Top 10 Words", fontsize=14)
        
        plt.tight_layout()
        plt.savefig('youtube_analysis_results.png')
        plt.show()
        
        # Print additional insights
        print("\n--- COMMENT ANALYSIS SUMMARY ---")
        print(f"Total comments analyzed: {results['comment_stats']['total_analyzed']}")
        print(f"Average sentiment polarity: {results['comment_stats']['avg_polarity']:.2f} (-1 negative to +1 positive)")
        print(f"Average subjectivity: {results['comment_stats']['avg_subjectivity']:.2f} (0 factual to 1 personal opinion)")
        
        sentiment_dist = results["comment_stats"]["sentiment_distribution"]
        total = sum(sentiment_dist.values())
        print("\nSentiment breakdown:")
        for sentiment, count in sentiment_dist.items():
            percentage = (count / total) * 100
            print(f"  {sentiment.capitalize()}: {count} comments ({percentage:.1f}%)")
        
        print("\nTop 5 most liked comments:")
        for i, comment in enumerate(results["top_comments"][:5], 1):
            print(f"\n{i}. Likes: {comment['like_count']} | Sentiment: {comment['sentiment']}")
            print(f"   \"{comment['text'][:100]}{'...' if len(comment['text']) > 100 else ''}\"")

def main():
    # Load API key from environment variable
    api_key = os.getenv("YOUTUBE_API_KEY")
    
    if not api_key:
        print("Please set your YouTube API key:")
        api_key = input().strip()
    
    analyzer = YouTubeCommentAnalyzer(api_key)
    
    # Get YouTube URL from user
    print("\nEnter a YouTube video URL:")
    url = input().strip()
    
    # Get number of comments to analyze
    print("How many comments would you like to analyze? (default: 100)")
    try:
        max_comments = int(input().strip() or "100")
    except ValueError:
        max_comments = 100
    
    # Generate and visualize the report
    print("\nGenerating analysis...")
    results = analyzer.generate_report(url, max_comments)
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        analyzer.visualize_insights(results)
        
        # Save results to CSV
        print("\nSaving comment data to 'comment_analysis.csv'...")
        results["all_comments"].to_csv('comment_analysis.csv', index=False)
        print("Analysis complete!")

if __name__ == "__main__":
    main()