import praw
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple, Optional
import re
import torch
import torchaudio
import time
import gc
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from chatterbox.tts import ChatterboxTTS
import threading
from functools import lru_cache
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pydub import AudioSegment
import random
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, TextClip, CompositeVideoClip, ImageClip
import multiprocessing as mp
import logging
import uuid
import shutil # <-- ADD THIS IMPORT
import whisper
from tqdm import tqdm
import PIL.Image
import traceback
from difflib import SequenceMatcher
import cv2
import subprocess #<- added updated version TYLER
import json #<- added TYLER
import glob #<- ADDED TYLER
import shutil #< 
import os
# ADD THESE IMPORTS AT THE TOP OF YOUR SCRIPT
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
# ADD THIS WITH YOUR OTHER DATETIME IMPORTS

MAX_WORKERS_CTA = min(os.cpu_count() or 1, 4)
LOG_ERRORS_CTA = True

AUDIO_PROMPT = "util/0602-vocals.mp3"

# Overlay sizing configuration
OVERLAY_WIDTH_PERCENT = 1.2  # Percentage of video width (0.8 = 80%)
OVERLAY_HEIGHT_PERCENT = 0.5  # Percentage of video height (0.4 = 40%)
OVERLAY_POSITION = 'center'   # Position: 'center', 'top', 'bottom', ('left', 'top'), etc.
# Fade out configuration
FADE_DURATION = .2  # Duration of fade out in seconds (1.0 = 1 second fade)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Configure logging to reduce verbose output
logging.getLogger("moviepy").setLevel(logging.WARNING)
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PASTE THE WORKER FUNCTIONS HERE (AT THE TOP LEVEL)
def process_audio_cleaning_worker(audio_path):
    """
    Worker 1: This function is used ONLY for the AudioCleaner class.
    It takes an audio file, trims the silence, and saves it.
    """
    MODEL_NAME = "base"
    try:
        # These imports are needed within the worker for multiprocessing
        import os
        import whisper
        import soundfile as sf

        model = whisper.load_model(MODEL_NAME, device="cpu")
        result = model.transcribe(audio_path, word_timestamps=True, verbose=False)

        if not result.get("segments"):
            return f"Skipped (no speech): {audio_path}"

        last_word_end = result["segments"][-1].get('end', 0)
        audio_data, sr = sf.read(audio_path)
        padding_seconds = 0.2
        trim_samples = int((last_word_end + padding_seconds) * sr)
        trimmed_audio = audio_data[:min(trim_samples, len(audio_data))]
        sf.write(audio_path, trimmed_audio, sr)

        return f"Cleaned: {audio_path}"
    except Exception as e:
        return f"Error cleaning {os.path.basename(audio_path)}: {e}"


def process_video_creation_worker(audio_path_and_vid_id):
    """
    Worker 2: This function is used ONLY for the VideoCreator class.
    It takes an audio file and creates a complete video for it.
    """
    audio_path, vid_id = audio_path_and_vid_id
    # These imports are needed within the worker for multiprocessing
    import os
    import gc
    
    worker_id = os.getpid()
    creator = None
    try:
        # Note: This assumes the VideoCreator class and its dependencies
        # are also defined globally or imported correctly. Given your
        # current structure, you should move ALL classes outside of run_reddit.
        creator = VideoCreator(vid_id=vid_id, worker_id=worker_id)
        return creator.process_single_audio_file(audio_path)
    except Exception as e:
        print(f"[Worker {worker_id}] CRITICAL ERROR: {e}")
        return False, f"CRITICAL error in worker for {os.path.basename(audio_path)}"
    finally:
        gc.collect()

# In the class definition for RedditConfig
class RedditConfig:
    def __init__(self, vid_id_from_gui): # Change here
        self.client_id = 'OM3KAoU5NuKhbxgD5llgCA'
        self.client_secret = 'ab8fhgzyPxltmWEkQGewGQH9anfmeg'
        self.user_agent = 'Story2Vid by /u/roughpowerhouse'
        self.words_per_second = 2.5
        self.subreddits = ['nosleep']
        self.vid_id = vid_id_from_gui # And change here


class DatabaseManagerStoryGrabber:
    """Handles database operations for storing Reddit stories."""
    
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
    
    def create_tables(self, subreddit_name: str):
        """Create tables for storing short and long stories."""
        # --- MODIFIED: Added "UNIQUE" constraint to the title column ---
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {subreddit_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE,  -- <-- ADD "UNIQUE" HERE
                body TEXT
            )
        ''')
        
        # --- MODIFIED: Also add it to the _long table ---
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {subreddit_name}_long (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE,  -- <-- AND ADD "UNIQUE" HERE
                body TEXT
            )
        ''')
    
    def insert_stories(self, subreddit_name: str, short_stories: List[Dict], long_stories: List[Dict]):
        """Insert stories into appropriate tables."""
        for story in short_stories:
            # --- MODIFIED: Changed "INSERT INTO" to "INSERT OR IGNORE INTO" ---
            self.cursor.execute(f'''
                INSERT OR IGNORE INTO {subreddit_name} (title, body) VALUES (?, ?)
            ''', (story['title'], story['body']))
        
        for story in long_stories:
            # --- MODIFIED: Also change it here for the _long table ---
            self.cursor.execute(f'''
                INSERT OR IGNORE INTO {subreddit_name}_long (title, body) VALUES (?, ?)
            ''', (story['title'], story['body']))
    
    def commit_and_close(self):
        """Commit changes and close database connection."""
        if self.conn:
            self.conn.commit()
            self.conn.close()
class StoryProcessorRedditStory:
    """Processes and categorizes Reddit stories."""
    
    def __init__(self, words_per_second: float):
        self.words_per_second = words_per_second
    
    def process_post(self, post) -> Dict[str, Any]:
        """Process a single Reddit post and return story data."""
        word_count = len(post.selftext.split())
        read_time_seconds = word_count / self.words_per_second
        score = post.score + post.num_comments
        
        return {
            'title': post.title,
            'body': post.selftext,
            'score': score,
            'read_time': read_time_seconds,
            'post_time': datetime.utcfromtimestamp(post.created_utc)
        }
    
    def categorize_stories(self, stories: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """Categorize stories into short and long based on reading time."""
        short_stories = []
        long_stories = []
        
        for story in stories:
            story_data = {'title': story['title'], 'body': story['body'], 'score': story['score']}
            if story['read_time'] >= 60:
                long_stories.append(story_data)
            else:
                short_stories.append(story_data)
        
        return short_stories, long_stories
    
    def get_top_stories(self, stories: List[Dict], limit: int = 1 ) -> List[Dict]:
        """Sort stories by score and return top performers."""
        return sorted(stories, key=lambda x: x['score'], reverse=True)[:limit]
class RedditScraper:
    """Main class for scraping Reddit stories."""
    
    def __init__(self, config: RedditConfig):
        self.config = config
        self.reddit = praw.Reddit(
            client_id=config.client_id,
            client_secret=config.client_secret,
            user_agent=config.user_agent
        )
        self.db_manager = DatabaseManagerStoryGrabber(f'{config.vid_id}_reddit_nosleep.db')
        self.story_processor = StoryProcessorRedditStory(config.words_per_second)
    
    def scrape_subreddit(self, subreddit_name: str):
        """Scrape stories from a specific subreddit."""
        print(f"Scraping stories from r/{subreddit_name}...")
        
        stories = []
        cutoff = datetime.utcnow() - timedelta(days=1)
        
        for post in self.reddit.subreddit(subreddit_name).new(limit=10):
            if not post.stickied and len(post.selftext.strip()) > 0:
                story_data = self.story_processor.process_post(post)
                
                if story_data['post_time'] > cutoff:
                    stories.append(story_data)

        # --- NEW: FILTER OUT STORIES WITH LINKS BEFORE CATEGORIZING ---
        print(f"Scraped {len(stories)} initial stories. Now filtering out posts with links...")
        text_cleaner = TextCleaner() # Use the existing TextCleaner class
        
        stories_without_links = [
            story for story in stories 
            if not text_cleaner.has_links(story['title']) and not text_cleaner.has_links(story['body'])
        ]
        
        print(f"Removed {len(stories) - len(stories_without_links)} stories with links. {len(stories_without_links)} stories remaining.")
        
        # Use the filtered list for all subsequent operations
        stories = stories_without_links
        # --- END OF NEW CODE ---

        # Categorize stories (now using the link-free list)
        short_stories, long_stories = self.story_processor.categorize_stories(stories)
        
        # Get top stories (now from the link-free list)
        top_short = self.story_processor.get_top_stories(short_stories)
        top_long = self.story_processor.get_top_stories(long_stories)
        
        # Save to database
        self.db_manager.create_tables(subreddit_name)
        self.db_manager.insert_stories(subreddit_name, top_short, top_long)
        
        print(f"Saved {len(top_short)} short and {len(top_long)} long stories from r/{subreddit_name}.")
    
    def run(self):
        """Main execution method."""
        self.db_manager.connect()
        
        try:
            for subreddit_name in self.config.subreddits:
                self.scrape_subreddit(subreddit_name)
            
            print(f"Finished scraping. Stories saved to '{self.db_manager.db_name}'.")
        
        finally:
            self.db_manager.commit_and_close()

class DatabaseConfig:
    """Configuration class for database cleaning settings."""
    
    def __init__(self, vid_id: str):  # No longer has a default value
        self.vid_id = vid_id
        self.db_name = f'{vid_id}_reddit_nosleep.db'
        self.subreddits = ['nosleep']
    
    def get_expected_tables(self) -> List[str]:
        """Get list of expected table names."""
        tables = []
        for subreddit in self.subreddits:
            tables.extend([subreddit, f"{subreddit}_long"])
        return tables
    
class TextCleaner:
    """Handles text cleaning operations."""
    
    @staticmethod
    def has_links(text: str) -> bool:
        """Check if text contains markdown links or link-like patterns."""
        if not text:
            return False
        
        # Patterns to match various link formats:
        patterns = [
            r'\[([^\]]*)\]\([^\)]*\)',  # Standard markdown: [text](url)
            r'\[([^\]]*)\]\([^)]*$',    # Incomplete markdown: [text](url... (missing closing paren)
            r'\[([^\]]*)\]\(https?://[^\s\)]*',  # Links starting with http/https
            r'https?://[^\s\)]*',       # Raw URLs
            r'\[([^\]]*)\]\([^\s]*',    # Any [text](something pattern
            r'\[Update\s*#?\d*\]',      # [Update], [Update #1], [Update #2], etc.
            r'\[Original\s+Post\]',     # [Original Post]
            r'\[Update\]',              # [Update]
            r'\[[^\]]*Update[^\]]*\]',  # Any text containing "Update" in brackets
            r'\[[^\]]*Post[^\]]*\]',    # Any text containing "Post" in brackets
            r'\[[^\]]*Original[^\]]*\]', # Any text containing "Original" in brackets
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def has_part2(text: str) -> bool:
        """Check if text contains 'part 2' or similar patterns (case insensitive)."""
        if not text:
            return False
        # Patterns to match:
        # - part 2, part2, Part 2, etc.
        # - update #2, update 2, etc.
        # - [part 2], (part 2), etc.
        # - #2, 2/2, 2 of 2, etc.
        patterns = [
            r'part\s*2',           # part 2, part2
            r'update\s*#?\s*2',    # update #2, update 2
            r'[\[\(].*?part\s*2.*?[\]\)]',  # [part 2], (part 2)
            r'#2\b',               # #2 (word boundary to avoid matching #20, #21, etc.)
            r'\b2/2\b',            # 2/2
            r'\b2\s+of\s+2\b',     # 2 of 2
            r'[\[\(].*?update.*?2.*?[\]\)]',  # [update 2], (update #2)
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
class DatabaseManagerStoryFilter:
    """Handles database operations for cleaning."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.conn = None
        self.cursor = None
    
    def connect(self) -> bool:
        """Establish database connection."""
        if not os.path.exists(self.config.db_name):
            print(f"Database '{self.config.db_name}' not found!")
            return False
        
        self.conn = sqlite3.connect(self.config.db_name)
        self.cursor = self.conn.cursor()
        return True
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def commit(self):
        """Commit changes to database."""
        if self.conn:
            self.conn.commit()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in database."""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return bool(self.cursor.fetchone())
    
    def get_posts_to_delete(self, table_name: str) -> List[Tuple]:
        """Get posts that should be deleted (contain links or 'part 2')."""
        self.cursor.execute(f"SELECT id, title, body FROM {table_name}")
        all_posts = self.cursor.fetchall()
        
        posts_to_delete = []
        text_cleaner = TextCleaner()
        
        for post_id, title, body in all_posts:
            # Check if title or body contains links or 'part 2'
            if (text_cleaner.has_links(title) or 
                text_cleaner.has_links(body or "") or
                text_cleaner.has_part2(title) or 
                text_cleaner.has_part2(body or "")):
                posts_to_delete.append((post_id, title, body))
        
        return posts_to_delete
    
    def delete_posts_by_ids(self, table_name: str, post_ids: List[int]) -> int:
        """Delete posts by their IDs. Returns count of deleted posts."""
        if not post_ids:
            return 0
        
        placeholders = ','.join(['?' for _ in post_ids])
        self.cursor.execute(f"DELETE FROM {table_name} WHERE id IN ({placeholders})", post_ids)
        return len(post_ids)
    
    def get_all_posts_ordered(self, table_name: str) -> List[Tuple]:
        """Get all posts from table ordered by ID."""
        self.cursor.execute(f"SELECT * FROM {table_name} ORDER BY id")
        return self.cursor.fetchall()
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """Get column names for a table."""
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = self.cursor.fetchall()
        return [column[1] for column in columns]  # column[1] is the column name
    
    def recreate_table_with_reindexed_data(self, table_name: str, data: List[Tuple], columns: List[str]):
        """Recreate table with reindexed data."""
        # Get the original table schema
        self.cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        create_sql = self.cursor.fetchone()[0]
        
        # Create temporary table
        temp_table = f"{table_name}_temp"
        temp_create_sql = create_sql.replace(f"CREATE TABLE {table_name}", f"CREATE TABLE {temp_table}")
        
        # Drop temp table if it exists
        self.cursor.execute(f"DROP TABLE IF EXISTS {temp_table}")
        
        # Create temp table
        self.cursor.execute(temp_create_sql)
        
        # Insert reindexed data
        placeholders = ','.join(['?' for _ in columns])
        insert_sql = f"INSERT INTO {temp_table} ({','.join(columns)}) VALUES ({placeholders})"
        
        reindexed_data = []
        for new_id, row in enumerate(data, 1):
            # Replace the old ID with new sequential ID
            new_row = list(row)
            new_row[0] = new_id  # Assuming ID is the first column
            reindexed_data.append(tuple(new_row))
        
        self.cursor.executemany(insert_sql, reindexed_data)
        
        # Drop original table and rename temp table
        self.cursor.execute(f"DROP TABLE {table_name}")
        self.cursor.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}")

class DatabaseCleaner:
    """Main class for cleaning the Reddit stories database."""
    def __init__(self, vid_id: str): # REMOVED: Default value ` = VID_ID`
        self.config = DatabaseConfig(vid_id)
        self.db_manager = DatabaseManagerStoryFilter(self.config)
        self.text_cleaner = TextCleaner()
    
    def clean_table(self, table_name: str) -> int:
        """Clean a single table. Returns deleted_count."""
        print(f"\nProcessing table: {table_name}")
        
        if not self.db_manager.table_exists(table_name):
            print(f"  Table {table_name} does not exist, skipping...")
            return 0
        
        try:
            # Get initial count
            initial_posts = self.db_manager.get_all_posts_ordered(table_name)
            initial_count = len(initial_posts)
            
            # Get posts to delete (those with links or 'part 2')
            posts_to_delete = self.db_manager.get_posts_to_delete(table_name)
            
            if posts_to_delete:
                # Extract IDs for deletion
                post_ids_to_delete = [post[0] for post in posts_to_delete]
                
                # Show what we're deleting
                print(f"  Found {len(posts_to_delete)} posts to delete:")
                for i, (post_id, title, body) in enumerate(posts_to_delete[:1]):  # Show first 5
                    reason = []
                    if self.text_cleaner.has_links(title) or self.text_cleaner.has_links(body or ""):
                        reason.append("contains links")
                    if self.text_cleaner.has_part2(title) or self.text_cleaner.has_part2(body or ""):
                        reason.append("contains part 2 indicator")
                    
                    print(f"    - ID {post_id}: {title[:80]}... ({', '.join(reason)})")
                
                if len(posts_to_delete) > 1:
                    print(f"    ... and {len(posts_to_delete) - 1} more")
                
                # Delete the posts
                deleted_count = self.db_manager.delete_posts_by_ids(table_name, post_ids_to_delete)
                print(f"  Deleted {deleted_count} posts")
                
                # Reindex the table
                print(f"  Reindexing table...")
                remaining_posts = self.db_manager.get_all_posts_ordered(table_name)
                columns = self.db_manager.get_table_columns(table_name)
                
                if remaining_posts:
                    self.db_manager.recreate_table_with_reindexed_data(table_name, remaining_posts, columns)
                    final_count = len(remaining_posts)
                    print(f"  Reindexed table: {initial_count} → {final_count} posts (sequential IDs 1-{final_count})")
                else:
                    print(f"  Table is now empty after deletions")
                
                return deleted_count
            else:
                print(f"  No posts found with links or part 2 indicators - nothing to delete")
                return 0
            
        except sqlite3.OperationalError as e:
            print(f"  Error processing table {table_name}: {e}")
            return 0
    
    def clean_database(self):
        """Clean the entire database."""
        if not self.db_manager.connect():
            return
        
        try:
            expected_tables = self.config.get_expected_tables()
            total_deleted_posts = 0
            
            for table_name in expected_tables:
                deleted = self.clean_table(table_name)
                total_deleted_posts += deleted
            
            self.db_manager.commit()
            
            print(f"\n--- Cleanup Summary ---")
            print(f"Total posts deleted: {total_deleted_posts}")
            print(f"Deletion criteria: posts containing links or part 2 indicators (update #2, #2, 2/2, etc.)")
            print(f"All tables have been reindexed with sequential IDs")
            print(f"Database '{self.config.db_name}' has been cleaned!")
            
        finally:
            self.db_manager.close()
    
    def preview_table_changes(self, table_name: str):
        """Preview changes for a single table."""
        print(f"\nTable: {table_name}")
        
        if not self.db_manager.table_exists(table_name):
            print(f"  Table does not exist, skipping...")
            return
        
        try:
            # Get total count
            all_posts = self.db_manager.get_all_posts_ordered(table_name)
            total_count = len(all_posts)
            
            # Get posts that would be deleted
            posts_to_delete = self.db_manager.get_posts_to_delete(table_name)
            
            if posts_to_delete:
                print(f"  Posts to delete ({len(posts_to_delete)}):")
                for i, (post_id, title, body) in enumerate(posts_to_delete[:3]):  # Show first 3
                    reason = []
                    if self.text_cleaner.has_links(title) or self.text_cleaner.has_links(body or ""):
                        reason.append("has links")
                    if self.text_cleaner.has_part2(title) or self.text_cleaner.has_part2(body or ""):
                        reason.append("has part 2 indicator")
                    
                    print(f"    - {title[:80]}... ({', '.join(reason)})")
                
                if len(posts_to_delete) > 3:
                    print(f"    ... and {len(posts_to_delete) - 3} more")
                
                print(f"  After deletion: {total_count - len(posts_to_delete)} posts will remain (reindexed as IDs 1-{total_count - len(posts_to_delete)})")
            else:
                print(f"  No posts found with links or part 2 indicators - no changes needed")
                
        except sqlite3.OperationalError as e:
            print(f"  Error reading table {table_name}: {e}")
    
    def preview_changes(self):
        """Preview what changes would be made without actually making them."""
        if not self.db_manager.connect():
            return
        
        try:
            print("--- PREVIEW MODE ---")
            print("Will delete posts containing:")
            print("  - Links: [text](url), raw URLs, or partial/malformed links")
            print("  - Update links: [Update #2], [Update #1], [Original Post], [Update]")
            print("  - Part 2 indicators: 'part 2', 'update #2', '#2', '2/2', '2 of 2', etc.")
            print("  - Any variation in brackets/parentheses")
            print()
            
            expected_tables = self.config.get_expected_tables()
            
            for table_name in expected_tables:
                self.preview_table_changes(table_name)
                
        finally:
            self.db_manager.close()

class UserInterfaceStoryFilter:
    def __init__(self, vid_id: str): # <-- It accepts vid_id
        self.cleaner = DatabaseCleaner(vid_id)
    
    def run(self):
        """Main user interface loop."""
        print("Reddit Stories Database Cleaner - DELETE MODE")
        print("This will DELETE posts containing links or part 2 indicators anywhere in title/body")
        print()
        print("1. Preview changes")
        print("2. Clean database (DELETE posts)")
        
        choice = '2'  # Default to clean
        
        if choice == "1":
            self.cleaner.preview_changes()
        elif choice == "2":
            confirm = 'y'  # Default to yes
            if confirm.lower() == 'y':
                self.cleaner.clean_database()
            else:
                print("Operation cancelled.")
        else:
            print("Invalid choice. Please run again and select 1 or 2.")
class TextProcessingConfig:
    """Configuration for text processing including abbreviations and censorship."""
    
    def __init__(self):
        self.abbreviation_replacements = {
            # Reddit specific
            r'\bAITA\b': 'Am I the a hole',
            r'\bAITAH\b': 'Am I the a hole',
            r'\bNTA\b': 'Not the a hole',
            r'\bYTA\b': 'You are the a hole',
            r'\bESH\b': 'Everyone sucks here',
            r'\bNAH\b': 'No a holes here',
            r'\bINFO\b': 'I need more information',
            r'\bTL;DR\b': 'Too long didn\'t read',
            r'\bTLDR\b': 'Too long didn\'t read',
            r'\bOP\b': 'Original poster',
            r'\bOOP\b': 'Original poster',
            r'\bEDIT\b': 'Edit',
            r'\bETA\b': 'Edited to add',
            r'\bUPDATE\b': 'Update',
            r'\bTIL\b': 'Today I learned',
            r'\bAMA\b': 'Ask me anything',
            
            # Family/relationship abbreviations
            r'\bSO\b': 'Significant other',
            r'\bBF\b': 'Boyfriend',
            r'\bGF\b': 'Girlfriend',
            r'\bFH\b': 'Future husband',
            r'\bFW\b': 'Future wife',
            r'\bDH\b': 'Dear husband',
            r'\bDW\b': 'Dear wife',
            r'\bMIL\b': 'Mother in law',
            r'\bFIL\b': 'Father in law',
            r'\bSIL\b': 'Sister in law',
            r'\bBIL\b': 'Brother in law',
            r'\bFMIL\b': 'Future mother in law',
            r'\bFFIL\b': 'Future father in law',
            
            # Common internet abbreviations
            r'\bIMO\b': 'In my opinion',
            r'\bIMHO\b': 'In my humble opinion',
            r'\bTBH\b': 'To be honest',
            r'\bIRL\b': 'In real life',
            r'\bFWIW\b': 'For what it\'s worth',
            r'\bAFAIK\b': 'As far as I know',
            r'\bLOL\b': 'Laugh out loud',
            r'\bLMAO\b': 'Laughing my a hole off',
            r'\bLMFAO\b': 'Laughing my freaking a hole off',
            r'\bROFL\b': 'Rolling on the floor laughing',
            r'\bOMG\b': 'Oh my god',
            r'\bWTF\b': 'What the heck',
            r'\bTIA\b': 'Thanks in advance',
            r'\bJK\b': 'Just kidding',
            r'\bFYI\b': 'For your information',
            r'\bBTW\b': 'By the way',
            r'\bIDK\b': 'I don\'t know',
            r'\bIKR\b': 'I know right',
            r'\bSMH\b': 'Shaking my head',
            r'\bTBF\b': 'To be fair',
            r'\bNGL\b': 'Not gonna lie',
            r'\bIIRC\b': 'If I recall correctly',
            r'\bICYMI\b': 'In case you missed it',
            r'\bIANAL\b': 'I am not a lawyer',
            r'\bYMMV\b': 'Your mileage may vary',
            r'\bNSFW\b': 'Not safe for work',
            r'\bNSFL\b': 'Not safe for life',
            r'\bDM\b': 'Direct message',
            r'\bPM\b': 'Private message',
            r'\bFML\b': 'Frick my life',
            r'\bAF\b': 'As frick',
            r'\bRIP\b': 'Rest in peace',
            r'\bGTH\b': 'Go to heck',
        }
        
        self.censorship_replacements = {
            # Explicit words (case insensitive)
            r'\bsex\b': 'fun',
            r'\bpussy\b': 'part',
            r'\bvagina\b': 'lady part',
            r'\bsexual\b': 'friendly',
            r'\bfuck\b': 'frick',
            r'\bfucking\b': 'fricking',
            r'\bfucked\b': 'fricked',
            r'\bshit\b': 'poop',
            r'\bbitch\b': 'nuisance',
            r'\bbackend\b': 'backend',
            r'\bdamn\b': 'dang',
            r'\bhell\b': 'heck',
            r'\bcrap\b': 'crud',
            r'\bpiss\b': 'p word',
            r'\bcock\b': 'johnson',
            r'\bdick\b': 'johnson',
            r'\bballs\b': 'parts',
            r'\btits\b': 'chest',
            r'\bboobs\b': 'chest',
            r'\bjizz\b': 'stuff',
            r'\bcum\b': 'stuff',
            
            # Harm-related words
            r'\bkill\b': 'eliminate',
            r'\bdead\b': 'gone',
            r'\bdie\b': 'pass away',
            r'\bsuicide\b': 'self harm',
            r'\bmurder\b': 'harm',
            r'\brape\b': 'attack',
            r'\bmolest\b': 'harm',
            r'\bincest\b': 'family romance',
            r'\bpedo\b': 'predator',
            r'\bpedophile\b': 'predator',
            
            # Slurs and offensive terms
            r'\bnigger\b': 'brother',
            r'\bnigga\b': 'brotha',
            r'\bfag\b': 'nuisance',
            r'\bfaggot\b': 'nuisance',
            r'\bkike\b': 'heeb',
            r'\bretard\b': 'slow',
            r'\btranny\b': 'trans person',
            r'\bchink\b': 'person',
            r'\bgook\b': 'person',
            r'\bspic\b': 'person',
            r'\bwetback\b': 'person',
            r'\bjap\b': 'person',
            r'\bgypsy\b': 'traveler',
            r'\bcracker\b': 'white',
            r'\bhonky\b': 'person',
            r'\bcoon\b': 'black',
            r'\btwink\b': 'femboy',
            r'\bshemale\b': 'trans woman',
            r'\bdyke\b': 'lez',
            r'\bslut\b': 'active woman',
            r'\bwhore\b': 'active woman',
            r'\bcunt\b': 'nuisance',
        }
#TYLER entire TextProcessor Class
class TextProcessor:
    """Handles text cleaning and processing for TTS generation."""
    
    def __init__(self):
        self.config = TextProcessingConfig()
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations to full phrases."""
        if not text:
            return ""
        
        for pattern, replacement in self.config.abbreviation_replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE if pattern.lower() == pattern else 0)
        
        return text
    
    def apply_censorship(self, text: str) -> str:
        """Apply censorship to explicit content."""
        if not text:
            return ""
        
        for pattern, replacement in self.config.censorship_replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def clean_text_for_tts(self, text: str) -> str:
        """Clean and preprocess text for TTS with abbreviation expansion and censorship."""
        if not text:
            return ""
        
        # First expand abbreviations and apply censorship
        text = self.expand_abbreviations(text)
        text = self.apply_censorship(text)
        
        # --- NEW & ENHANCED CLEANING STEPS ---
        # 1. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 2. Remove any text within brackets or parentheses (often metadata like [deleted] or (edit))
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        # 3. Handle special cases for better TTS pronunciation
        text = re.sub(r'\b(\d+)\s*yo\b', r'\1 year old', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\d+)m\b', r'\1 male', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\d+)f\b', r'\1 female', text, flags=re.IGNORECASE)
        text = re.sub(r'\$(\d+)', r'\1 dollars', text)
        # 4. A more restrictive character set. Allow basic punctuation, letters, and numbers.
        # This removes emojis, special symbols, etc., which are common causes for errors.
        text = re.sub(r'[^\w\s\.,!?;:\'"]', '', text)
        
        # 5. Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # 6. Check for abnormally long words that can break tokenizers
        # A word over 45 chars is likely a typo or junk.
        words = text.split()
        cleaned_words = [word for word in words if len(word) < 45]
        text = ' '.join(cleaned_words)
        # Final check for empty or whitespace-only text after cleaning
        if not text.strip():
            return ""
            
        return text
    
    def split_text_into_chunks(self, text: str, max_words: int = 70) -> List[str]: # Reduced default slightly
        """Split text into chunks, now more robustly."""
        if not text:
            return []
        
        # Use sentence splitting for more natural breaks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) <= max_words:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # If any chunk is still too long (due to a very long sentence), split it hard.
        final_chunks = []
        for chunk in chunks:
            if len(chunk.split()) > max_words:
                words = chunk.split()
                for i in range(0, len(words), max_words):
                    final_chunks.append(' '.join(words[i:i + max_words]))
            else:
                final_chunks.append(chunk)
        return [c for c in final_chunks if c] # Ensure no empty chunks are returned
    def test_text_processing(self):
        """Test function to see how text processing works."""
        test_texts = [
            "AITA for telling my BF that his ex-GF is a bitch? TBH she's fucking crazy.",
            "My MIL is 45yo and keeps saying shit about my relationship. WTF is wrong with her?",
            "UPDATE: So my DH finally told his mother to fuck off. NTA for sure!",
            "This is some serious bullshit. The asshole needs to die, IMO.",
            "Check out this link https://example.com it's crazy. Also [deleted].",
            "Thisisaverylongwordthatwillbreakthetokenizerandcauseacudacrashhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh."
        ]
        
        print("\n=== TEXT PROCESSING TEST ===")
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}:")
            print(f"Original: {text}")
            processed = self.clean_text_for_tts(text)
            print(f"Processed: {processed}")
            if processed:
                chunks = self.split_text_into_chunks(processed)
                print(f"Chunks: {chunks}")
class DatabaseManagerTTS:
    """Handles database operations for loading Reddit stories."""
    
    def __init__(self, vid_id: str):
        self.vid_id = vid_id
        self.db_path = f"{vid_id}_reddit_nosleep.db"
    
    def database_exists(self) -> bool:
        """Check if the database file exists."""
        return os.path.exists(self.db_path)
    
    def get_all_tables(self) -> List[str]:
        """Get all table names from the database."""
        if not self.database_exists():
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]
            return tables
        finally:
            conn.close()
    
    def get_stories_from_table(self, table_name: str) -> List[Tuple[int, str, str]]:
        """Get all stories from a specific table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"SELECT id, title, body FROM {table_name}")
            return cursor.fetchall()
        finally:
            conn.close()
    
    def get_story_count(self, table_name: str) -> int:
        """Get the number of stories in a table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def list_available_stories(self):
        """List all available stories in the database."""
        if not self.database_exists():
            print(f"Database file {self.db_path} not found. Run the Reddit scraper first.")
            return
        
        tables = self.get_all_tables()
        
        print(f"\n=== AVAILABLE STORIES IN {self.db_path} ===")
        
        total_stories = 0
        for table_name in tables:
            count = self.get_story_count(table_name)
            total_stories += count
            print(f"{table_name}: {count} stories")
        
        print(f"Total stories: {total_stories}")
class StoryProcessorTTS:
    """Processes Reddit stories and prepares them for TTS generation."""
    
    def __init__(self, vid_id: str):
        self.vid_id = vid_id
        self.db_manager = DatabaseManagerTTS(vid_id)
        self.text_processor = TextProcessor()
    
    def load_reddit_stories(self) -> List[Dict[str, Any]]:
        """Load stories from Reddit database and prepare them for TTS."""
        if not self.db_manager.database_exists():
            raise FileNotFoundError(f"Database file {self.db_manager.db_path} not found. Run the Reddit scraper first.")
        
        stories_data = []
        tables = self.db_manager.get_all_tables()
        
        for table_name in tables:
            print(f"Processing table: {table_name}")
            
            stories = self.db_manager.get_stories_from_table(table_name)
            
            for story_id, title, body in stories:
                try:
                    # Clean title and body (includes abbreviation expansion and censorship)
                    clean_title = self.text_processor.clean_text_for_tts(title)
                    clean_body = self.text_processor.clean_text_for_tts(body)
                    
                    if clean_title and clean_body:
                        # Combine title and body for the full story
                        full_story = f"{clean_title}. {clean_body}"
                        
                        # Split story into 30-second chunks
                        story_chunks = self.text_processor.split_text_into_chunks(full_story)
                        
                        # Create entry for each chunk
                        for chunk_index, chunk_text in enumerate(story_chunks, 1):
                            stories_data.append({
                                'table_name': table_name,
                                'story_id': story_id,
                                'chunk_index': chunk_index,
                                'chunk_text': chunk_text,
                                'total_chunks': len(story_chunks),
                                'filename': f"{self.vid_id}_{table_name}_{story_id}_{chunk_index}.wav"
                            })
                except Exception as e:
                    print(f"Warning: Error processing story {story_id} from {table_name}: {e}")
                    continue
        
        return stories_data

class AudioGenerator:
    """Handles TTS audio generation using ChatterboxTTS."""
    
    def __init__(self, vid_id: str, audio_prompt_path: str, batch_size: int = 1):
        self.vid_id = vid_id
        self.audio_prompt_path = audio_prompt_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.main_output_dir = f"{vid_id}_Reddit_Audio"
        self.batch_size = batch_size
    
    def check_cuda_health(self):
        """Check CUDA device health and reset if needed."""
        if self.device == "cuda":
            try:
                # Force synchronization to catch any pending errors
                torch.cuda.synchronize()
                # Check memory status
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                print(f"CUDA Memory - Allocated: {memory_allocated / 1024**2:.1f}MB, Reserved: {memory_reserved / 1024**2:.1f}MB")
                return True
            except RuntimeError as e:
                print(f"CUDA error detected: {e}")
                print("Attempting to reset CUDA state...")
                torch.cuda.empty_cache()
                try:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    return True
                except:
                    print("CUDA reset failed. Switching to CPU mode.")
                    self.device = "cpu"
                    return False
        return True
    
    def initialize_model(self):
        print(f"Using device: {self.device}")
    
        if self.device == "cuda":
            # Set CUDA memory fraction to use more VRAM (75% of 8GB = ~6GB)
            torch.cuda.set_per_process_memory_fraction(0.75)
        
        # Verify audio prompt exists
        if not os.path.exists(self.audio_prompt_path):
            raise FileNotFoundError(f"Audio prompt file {self.audio_prompt_path} not found")
        
        # Check CUDA health before loading model
        if not self.check_cuda_health():
            print("CUDA health check failed, using CPU")
        
        try:
            # Load ChatterboxTTS model with explicit device specification
            print("Loading ChatterboxTTS model...")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            
            # Move model to device explicitly if needed
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            print("Model loaded successfully!")
            
            # Test model with a short sample
            print("Testing model with sample text...")
            test_text = "Hello, this is a test."
            with torch.no_grad():
                test_wav = self.model.generate(test_text, audio_prompt_path=self.audio_prompt_path)
                if test_wav is None or test_wav.numel() == 0:
                    raise RuntimeError("Model test failed - no audio generated")
            print("Model test successful!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            if self.device == "cuda":
                print("Retrying with CPU...")
                self.device = "cpu"
                torch.cuda.empty_cache()
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
            else:
                raise
    
    def create_output_directories(self, stories_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Create output directories for each table."""
        os.makedirs(self.main_output_dir, exist_ok=True)
        print(f"Main output directory: {self.main_output_dir}")
        
        table_folders = {}
        for story_chunk in stories_data:
            table_name = story_chunk['table_name']
            if table_name not in table_folders:
                table_folder = os.path.join(self.main_output_dir, table_name)
                os.makedirs(table_folder, exist_ok=True)
                table_folders[table_name] = table_folder
                print(f"Created folder: {table_folder}")
        
        return table_folders
    
    def filter_existing_files(self, stories_data: List[Dict[str, Any]], table_folders: Dict[str, str]) -> List[Dict[str, Any]]:
        """Filter out stories that already have generated audio files."""
        remaining_stories = []
        
        for story_chunk in stories_data:
            table_folder = table_folders[story_chunk['table_name']]
            output_path = os.path.join(table_folder, story_chunk['filename'])
            if not os.path.exists(output_path):
                remaining_stories.append(story_chunk)
        
        return remaining_stories
    
    # TYLER VVV - just this function
    def generate_audio_chunk(self, story_chunk: Dict[str, Any], output_path: str) -> bool:
        """
        Generate audio for a single story chunk with robust validation and error handling
        to prevent GPU crashes.
        """
        text = story_chunk.get('chunk_text', '').strip()
        # === LAYER 2: PRE-GENERATION VALIDATION ===
        # Final check on the text before sending to the model
        if not text:
            print(f"✗ Skipping empty text for: {story_chunk['filename']}")
            return False
        
        # You could add more checks here, e.g., for character length
        if len(text) > 1000: # A reasonable character limit for a chunk
            print(f"✗ Skipping overly long chunk ({len(text)} chars) for: {story_chunk['filename']}")
            return False
        try:
            # Generate audio with proper context management
            with torch.no_grad():
                wav = self.model.generate(text, audio_prompt_path=self.audio_prompt_path)
            
            # Validate model output
            if wav is None or wav.numel() == 0:
                print(f"✗ Model returned empty output for: {story_chunk['filename']}")
                return False
            
            # Save the file
            wav_cpu = wav.detach().cpu()
            sample_rate = getattr(self.model, 'sr', 22050)
            torchaudio.save(output_path, wav_cpu, sample_rate)
            
            # Final verification
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                return True
            else:
                print(f"✗ Output file invalid after saving: {story_chunk['filename']}")
                return False
            
        # === LAYER 3: THE SAFETY NET ===
        except Exception as e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"CRITICAL ERROR CAUGHT processing: {story_chunk['filename']}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            print(f"This chunk will be SKIPPED. The process will continue.")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
            # Log the problematic text to a file for later review
            with open(f"{self.vid_id}_error_log.txt", "a", encoding="utf-8") as f:
                f.write("="*50 + "\n")
                f.write(f"Timestamp: {time.ctime()}\n")
                f.write(f"File: {story_chunk['filename']}\n")
                f.write(f"Error: {e}\n")
                f.write(f"Problematic Text:\n---\n{text}\n---\n\n")
            # Attempt to clean up GPU memory to prevent cascading failures
            self.cleanup_memory()
            return False
        
    def generate_audio_batch(self, story_chunks: List[Dict[str, Any]], table_folders: Dict[str, str]) -> Tuple[int, int]:
        """Generate audio for multiple chunks in batches."""
        successful = 0
        failed = 0
        total_chunks = len(story_chunks)
        
        print(f"Processing {total_chunks} chunks in batches of {self.batch_size}")
        
        for i in range(0, total_chunks, self.batch_size):
            batch = story_chunks[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_chunks + self.batch_size - 1) // self.batch_size
            
            print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            # Process each chunk in the batch
            with torch.no_grad():
                for j, story_chunk in enumerate(batch):
                    chunk_in_batch = j + 1
                    overall_chunk = i + j + 1
                    
                    print(f"  Chunk {chunk_in_batch}/{len(batch)} (Overall: {overall_chunk}/{total_chunks})")
                    print(f"  {story_chunk['table_name']} - Story {story_chunk['story_id']} - Chunk {story_chunk['chunk_index']}")
                    
                    table_folder = table_folders[story_chunk['table_name']]
                    output_path = os.path.join(table_folder, story_chunk['filename'])
                    
                    if self.generate_audio_chunk(story_chunk, output_path):
                        successful += 1
                        print(f"  ✓ Success: {story_chunk['filename']}")
                    else:
                        failed += 1
                        print(f"  ✗ Failed: {story_chunk['filename']}")
            
            # Clean up after each batch
            print(f"  Batch {batch_num} complete. Cleaning up memory...")
            self.cleanup_memory()
            
            # Check CUDA health after each batch
            if not self.check_cuda_health():
                print("  Warning: CUDA health check failed after batch")
        
        return successful, failed
    
    def cleanup_memory(self):
        """Enhanced memory cleanup with error handling."""
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        except Exception as e:
            print(f"Warning: Memory cleanup error: {e}")
    
    def print_story_statistics(self, stories_data: List[Dict[str, Any]]):
        """Print statistics about the stories to be processed."""
        # Group by story for better reporting
        stories_by_story = {}
        for chunk in stories_data:
            story_key = f"{chunk['table_name']}_story_{chunk['story_id']}"
            if story_key not in stories_by_story:
                stories_by_story[story_key] = []
            stories_by_story[story_key].append(chunk)
        
        print(f"Total stories: {len(stories_by_story)} (split into {len(stories_data)} chunks)")
        
        # Show chunk breakdown
        for story_key, chunks in stories_by_story.items():
            if len(chunks) > 1:
                print(f"  {story_key}: {len(chunks)} chunks")
    
    def generate_all_audio(self, stories_data: List[Dict[str, Any]]):
        """Generate audio for all story chunks using batch processing."""
        if not stories_data:
            print("No stories found in database!")
            return
        
        print(f"Found {len(stories_data)} story chunks to process")
        self.print_story_statistics(stories_data)
        
        # Create directories and filter existing files
        table_folders = self.create_output_directories(stories_data)
        remaining_stories = self.filter_existing_files(stories_data, table_folders)
        
        total_chunks = len(stories_data)
        remaining_count = len(remaining_stories)
        
        print(f"Total chunks: {total_chunks}")
        print(f"Remaining to process: {remaining_count}")
        print(f"Batch size: {self.batch_size}")
        
        if remaining_count == 0:
            print("All audio files already exist!")
            return
        
        # Process chunks in batches
        print(f"Generating audio for {remaining_count} story chunks...")
        start_time = time.time()
        
        successful, failed = self.generate_audio_batch(remaining_stories, table_folders)
        
        # Final summary
        duration = time.time() - start_time
        print(f"\n=== GENERATION COMPLETE ===")
        print(f"Successfully processed: {successful}/{remaining_count} story chunks")
        print(f"Failed: {failed}/{remaining_count} story chunks")
        print(f"Batch size used: {self.batch_size}")
        if remaining_count > 0:
            print(f"Total time: {duration:.1f} seconds")
            print(f"Average time per chunk: {duration/remaining_count:.1f} seconds")
        print(f"Output structure:")
        for table_name, folder_path in table_folders.items():
            file_count = len([f for f in os.listdir(folder_path) if f.endswith('.wav')])
            print(f"  {folder_path}: {file_count} audio files")
class RedditTTSGenerator:
    """Main class that orchestrates the entire TTS generation process."""
    
    def __init__(self, vid_id: str, audio_prompt: str):
        self.vid_id = vid_id
        self.audio_prompt = audio_prompt
        self.db_manager = DatabaseManagerTTS(vid_id)
        self.story_processor = StoryProcessorTTS(vid_id)
        self.audio_generator = AudioGenerator(vid_id, audio_prompt)
        self.text_processor = TextProcessor()
    
    def list_available_stories(self):
        """List all available stories in the database."""
        self.db_manager.list_available_stories()
    
    def test_text_processing(self):
        """Test the text processing functionality."""
        self.text_processor.test_text_processing()
    
    def generate_all_audio(self):
        """Generate audio files for all Reddit stories."""
        try:
            # Initialize the TTS model
            self.audio_generator.initialize_model()
            
            # Load stories from database
            print("Loading stories from database...")
            stories_data = self.story_processor.load_reddit_stories()
            
            # Generate audio for all stories
            self.audio_generator.generate_all_audio(stories_data)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
class UserInterfaceTTS:
    """Handles user interaction for the TTS generator."""
    
    def __init__(self, vid_id: str, audio_prompt: str):
        self.generator = RedditTTSGenerator(vid_id, audio_prompt)
    
    def run(self):
        """Main user interface."""
        print("=== REDDIT STORIES TTS GENERATOR ===")
        
        # Uncomment the line below to test text processing
        # self.generator.test_text_processing()
        
        self.generator.list_available_stories()
        
        proceed = 'y'  # input(f"\nGenerate audio for all stories using '{AUDIO_PROMPT}'? (y/n): ").lower().strip()
        
        if proceed == 'y':
            self.generator.generate_all_audio()
        else:
            print("Audio generation cancelled.")
#Tyler this function and the entire AudioCleaner Class -----------------
def process_file_worker(audio_path):
    MODEL_NAME = "base" # Define model name globally for worker processes
    """
    This is the core function that will be executed by each worker process.
    It handles a single audio file from start to finish.
    """
    try:
        # 1. Each worker loads its own instance of the model.
        #    We force it to the CPU since we are parallelizing across CPU cores.
        model = whisper.load_model(MODEL_NAME, device="cpu")
        # 2. The original logic from your 'trim_after_last_word' function is now here.
        #    Note: The initial lines for loading audio and creating a mel spectrogram
        #    in your original code were redundant, as model.transcribe handles it.
        result = model.transcribe(audio_path, word_timestamps=True, verbose=False)
        if not result.get("segments"):
            return f"Skipped (no speech): {audio_path}"
        last_segment = result["segments"][-1]
        if 'words' in last_segment and last_segment["words"]:
            last_word_end = last_segment["words"][-1]["end"]
        else:
            last_word_end = last_segment["end"]
        # 3. Load the audio data with soundfile to trim and save it.
        audio_data, sr = sf.read(audio_path)
        padding_seconds = 0.2
        trim_samples = int((last_word_end + padding_seconds) * sr)
        trim_samples = min(trim_samples, len(audio_data))
        trimmed_audio = audio_data[:trim_samples]
        # 4. The worker writes the file itself.
        sf.write(audio_path, trimmed_audio, sr)
        
        return f"Cleaned: {audio_path}"
    except Exception as e:
        return f"Error processing {audio_path}: {e}"
class AudioCleaner:
    def __init__(self, base_folder):
        self.base_folder = base_folder
        # The model is no longer loaded in the main class.
    def process_all_audio_parallel(self):
        # Step 1: Gather all file paths into a list.
        audio_files = []
        for root, _, files in os.walk(self.base_folder):
            for filename in files:
                if filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    audio_files.append(os.path.join(root, filename))
        
        if not audio_files:
            print("No audio files found to process.")
            return
        print(f"Found {len(audio_files)} audio files. Starting parallel processing...")
        
        # Step 2: Create a pool of worker processes. os.cpu_count() is a good default.
        num_processes = os.cpu_count()
        print(f"Using {num_processes} CPU cores.")
        
        start_time = time.time()
        
        # The 'with' statement ensures the pool is properly closed.
        with mp.Pool(processes=num_processes) as pool:
            # Step 3: Map the worker function to the list of files and get the results.
            # This distributes the work and blocks until all files are processed.
            results = pool.map(process_file_worker, audio_files)
        
        end_time = time.time()
        print(f"\n--- Finished processing all files in {end_time - start_time:.2f} seconds ---")
        # Step 4: Print the status from each worker.
        print("\n--- Summary ---")
        for res in results:
            print(res)
# ---------------------------------------------------------------------------
class FakeRedditPostGenerator:
    def __init__(self, vid_id: str):  # <-- REMOVED: Default value `= VID_ID`
        """Initialize the Reddit post generator with configurable settings"""
        self.vid_id = vid_id
        
        # User and styling settings
        self.username = "u/RedditAITATales"
        self.profile_image_path = r"PostImages\throne.png"
        self.verified_emoji_path = r"PostImages\verified.png"
        self.output_dir_png = f"{vid_id}_fake_reddit_posts"
        
        # Award emoji paths (6 different awards)
        self.award_emojis = [
            r"PostImages\award1.png",
            r"PostImages\award2.png",
            r"PostImages\award3.png", 
            r"PostImages\award4.webp", 
            r"PostImages\award5.webp",
            r"PostImages\award5.png",
            r"PostImages\award4.png"
        ]
        
        # Social interaction emoji paths
        self.heart_emoji_path = r"PostImages\heart.png"
        self.comment_emoji_path = r"PostImages\comment.png"
        self.share_emoji_path = r"PostImages\share.png"
        
        # Image dimensions and styling
        self.canvas_width = 1000
        self.base_box_width = 700
        self.base_box_height = 200
        self.box_color = (255, 255, 255)  # White
        self.shadow_color = (0, 0, 0, 60)  # Semi-transparent black
        self.shadow_offset = (8, 8)
        self.border_radius = 15
        
        # Text settings
        self.title_font_size = 32
        self.max_chars_per_line = 40
        self.line_height = 38
        
        # Element sizes
        self.profile_size = 60
        self.verified_size = 18
        self.award_size = 20
        self.username_font_size = 18
        
        # Colors
        self.text_color = (33, 37, 41)  # Dark gray
        self.username_color = (255, 69, 0)  # Reddit orange
        self.share_text_color = (172, 170, 170)  # #acaaaa
        
        # Award colors for fallbacks
        self.award_colors = [
            (255, 215, 0),   # Gold
            (192, 192, 192), # Silver  
            (205, 127, 50),  # Bronze
            (255, 20, 147),  # Pink
            (50, 205, 50),   # Green
            (138, 43, 226)   # Purple
        ]
        
        # Censorship settings
        self.censorship_patterns = {
            # Explicit words (case insensitive)
            r'\bsex\b': 's*x',
            r'\bpussy\b': 'p*ss*',
            r'\bvagina\b': 'v*g*n*',
            r'\bsexual\b': 's*x**l',
            r'\bfuck\b': 'f**k',
            r'\bfucking\b': 'f**k*ng',
            r'\bfucked\b': 'f**k*d',
            r'\bshit\b': 'sh*t',
            r'\bbitch\b': 'b**ch',
            r'\basshole\b': '*ssh*le',
            r'\bdamn\b': 'd*mn',
            r'\bhell\b': 'h*ll',
            r'\bcrap\b': 'cr*p',
            r'\bpiss\b': 'p*ss',
            r'\bcock\b': 'c**k',
            r'\bdick\b': 'd**k',
            r'\bballs\b': 'b**ls',
            r'\btits\b': 't*ts',
            r'\bboobs\b': 'b**bs',
            r'\bjizz\b': 'j**z',
            r'\bcum\b': 'c*m',
            
            # Harm-related words
            r'\bkill\b': 'elimate',
            r'\bdead\b': 'gone',
            r'\bdie\b': 'pass away',
            r'\bsuicide\b': 'self harm',
            r'\bmurder\b': 'harm',
            r'\brape\b': 'attack',
            r'\bmolest\b': 'harm',
            r'\bincest\b': '*nc*st',
            r'\bpedo\b': 'p*d*',
            r'\bpedophile\b': 'p*d*ph*l*',
            
            # Slurs and offensive terms
            r'\bnigger\b': 'brother',
            r'\bnigga\b': 'brotha',
            r'\bfag\b': 'nuisance',
            r'\bfaggot\b': 'nuisance',
            r'\bkike\b': 'heeb',
            r'\bretard\b': 'slow',
            r'\btranny\b': 'trans person',
            r'\bchink\b': 'person',
            r'\bgook\b': 'person',
            r'\bspic\b': 'person',
            r'\bwetback\b': 'person',
            r'\bjap\b': 'person',
            r'\bgypsy\b': 'traveler',
            r'\bcracker\b': 'white',
            r'\bhonky\b': 'person',
            r'\bcoon\b': 'black',
            r'\btwink\b': 'femboy',
            r'\bshemale\b': 'trans woman',
            r'\bdyke\b': 'd*ke',
            r'\bslut\b': 'sl*t',
            r'\bwhore\b': 'wh*r*',
            r'\bcunt\b': 'c**t',
        }
    
    def censor_vowels_in_word(self, word):
        """Replace vowels with asterisks in a word"""
        vowels = 'aeiouAEIOU'
        censored = ''
        for char in word:
            if char in vowels:
                censored += '*'
            else:
                censored += char
        return censored
    
    def apply_censorship(self, text):
        """Apply censorship to text by replacing vowels with asterisks in flagged words"""
        censored_text = text
        
        # Get all censorship patterns
        censorship_patterns = list(self.censorship_patterns.keys())
        
        # For each pattern, find matches and replace vowels with asterisks
        for pattern in censorship_patterns:
            matches = re.finditer(pattern, censored_text, re.IGNORECASE)
            
            # Process matches in reverse order to avoid index shifting
            matches_list = list(matches)
            for match in reversed(matches_list):
                original_word = match.group()
                censored_word = self.censor_vowels_in_word(original_word)
                
                # Replace the original word with the censored version
                start, end = match.span()
                censored_text = censored_text[:start] + censored_word + censored_text[end:]
        
        return censored_text
    
    def create_rounded_rectangle_with_shadow(self, width, height, radius, shadow_offset, shadow_color, fill_color):
        """Create a rounded rectangle with drop shadow"""
        # Create canvas with extra space for shadow
        canvas_width = width + abs(shadow_offset[0]) + 20
        canvas_height = height + abs(shadow_offset[1]) + 20
        
        # Create transparent canvas
        canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        # Calculate positions
        shadow_x = max(0, shadow_offset[0]) + 10
        shadow_y = max(0, shadow_offset[1]) + 10
        box_x = max(0, -shadow_offset[0]) + 10
        box_y = max(0, -shadow_offset[1]) + 10
        
        # Draw shadow (rounded rectangle)
        shadow_coords = [shadow_x, shadow_y, shadow_x + width, shadow_y + height]
        draw.rounded_rectangle(shadow_coords, radius=radius, fill=shadow_color)
        
        # Blur the shadow
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=3))
        
        # Draw the main box on top
        draw = ImageDraw.Draw(canvas)
        box_coords = [box_x, box_y, box_x + width, box_y + height]
        draw.rounded_rectangle(box_coords, radius=radius, fill=fill_color)
        
        return canvas, (box_x, box_y)
    
    def load_and_resize_image(self, path, size, fallback_color=None):
        """Load an image and resize it, return None if file doesn't exist"""
        try:
            if os.path.exists(path):
                img = Image.open(path)
                # Convert to RGBA for transparency support
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                # Resize maintaining aspect ratio
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                return img
            else:
                # Create fallback colored circle if image doesn't exist
                if fallback_color:
                    fallback = Image.new('RGBA', (size, size), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(fallback)
                    draw.ellipse([0, 0, size, size], fill=fallback_color)
                    return fallback
                return None
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def create_verified_badge(self, size):
        """Create a simple verified checkmark if the PNG doesn't exist"""
        badge = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(badge)
        
        # Blue circle
        draw.ellipse([0, 0, size, size], fill=(29, 161, 242))
        
        # White checkmark
        center = size // 2
        # Simple checkmark using lines
        points = [
            (center - 6, center),
            (center - 2, center + 4),
            (center + 6, center - 4)
        ]
        
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=(255, 255, 255), width=3)
        
        return badge
    
    def create_award_emoji(self, size, color):
        """Create a simple award emoji if PNG doesn't exist"""
        emoji = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(emoji)
        
        # Simple colored circle as award
        draw.ellipse([2, 2, size-2, size-2], fill=color)
        
        return emoji
    
    def create_heart_icon(self, size):
        """Create a simple heart icon"""
        heart = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(heart)
        
        # Simple heart shape using circles and triangle
        heart_color = (255, 99, 132)  # Pink-red color
        
        # Draw two circles for top of heart
        circle_size = size // 3
        draw.ellipse([0, size//4, circle_size, size//4 + circle_size], fill=heart_color)
        draw.ellipse([size - circle_size, size//4, size, size//4 + circle_size], fill=heart_color)
        
        # Draw triangle for bottom of heart
        points = [
            (size//2, size - 2),
            (2, size//2),
            (size - 2, size//2)
        ]
        draw.polygon(points, fill=heart_color)
        
        return heart
    
    def create_comment_icon(self, size):
        """Create a simple comment bubble icon"""
        comment = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(comment)
        
        comment_color = (100, 149, 237)  # Cornflower blue
        
        # Draw rounded rectangle for speech bubble
        bubble_size = size - 4
        draw.rounded_rectangle([2, 2, bubble_size, bubble_size - 6], radius=bubble_size//4, fill=comment_color)
        
        # Draw small triangle for speech bubble tail
        points = [
            (size//4, bubble_size - 6),
            (size//4 + 4, bubble_size - 6),
            (size//4 + 2, bubble_size + 2)
        ]
        draw.polygon(points, fill=comment_color)
        
        return comment
    
    def create_share_icon(self, size):
        """Create a simple share icon"""
        share = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(share)
        
        share_color = (120, 120, 120)  # Gray color
        
        # Draw arrow pointing right
        # Arrow body (rectangle)
        body_height = size // 4
        body_width = size * 2 // 3
        body_y = (size - body_height) // 2
        draw.rectangle([2, body_y, body_width, body_y + body_height], fill=share_color)
        
        # Arrow head (triangle)
        arrow_points = [
            (body_width - 2, body_y - body_height // 2),
            (size - 2, size // 2),
            (body_width - 2, body_y + body_height + body_height // 2)
        ]
        draw.polygon(arrow_points, fill=share_color)
        
        return share
    
    def wrap_title_text(self, title, max_chars_per_line):
        """Wrap title text to fit within character limit per line"""
        words = title.split()
        lines = []
        current_line = ""
        
        for word in words:
            # Check if adding this word would exceed the character limit
            test_line = current_line + (" " if current_line else "") + word
            
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                # If current line is not empty, save it and start new line
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Single word is too long, break it
                    if len(word) > max_chars_per_line:
                        # Split the word
                        lines.append(word[:max_chars_per_line])
                        current_line = word[max_chars_per_line:]
                    else:
                        current_line = word
        
        # Add the last line if it exists
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def calculate_box_width(self, title_lines, title_font, draw):
        """Calculate the box width based on the longest title line with minimal padding"""
        if not title_lines:
            return self.base_box_width
        
        # Find the longest line in pixels
        max_width = 0
        for line in title_lines:
            bbox = draw.textbbox((0, 0), line, font=title_font)
            line_width = bbox[2] - bbox[0]
            max_width = max(max_width, line_width)
        
        # Add minimal padding: profile (60) + spacing (15) + left margin (20) + right margin (20)
        total_width = max_width + 115
        
        # Ensure minimum width for profile section
        min_width = 350
        return max(min_width, total_width)
    
    def calculate_box_height(self, title_lines):
        """Calculate the box height based on content with minimal padding"""
        # Compact spacing: subreddit (20) + profile section (30) + awards (25) + spacing (10)
        base_content_height = 85
        # Add height for title lines
        title_height = len(title_lines) * self.line_height
        # Add height for interaction buttons with minimal spacing
        buttons_height = 30
        
        return base_content_height + title_height + buttons_height + 20  # Minimal padding
    
    def create_fake_reddit_post(self, title="Sample Reddit Post Title", subreddit="reddit"):
        """Create a fake Reddit post image with title only"""
        
        # Apply censorship to the title
        censored_title = self.apply_censorship(title)
        
        # Create a temporary draw object to measure text
        temp_canvas = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_canvas)
        
        # Load title font early for measurements (bold)
        try:
            title_font = ImageFont.truetype("arialbd.ttf", self.title_font_size)  # Bold font
        except:
            try:
                title_font = ImageFont.truetype("arial.ttf", self.title_font_size)
            except:
                title_font = ImageFont.load_default()
        
        # Wrap title text (using censored title)
        title_lines = self.wrap_title_text(censored_title, self.max_chars_per_line)
        
        # Calculate dynamic box dimensions
        box_width = self.calculate_box_width(title_lines, title_font, temp_draw)
        box_height = self.calculate_box_height(title_lines)
        canvas_height = box_height + 100  # Reduced extra space
        
        # Create main canvas with transparent background
        canvas = Image.new('RGBA', (self.canvas_width, canvas_height), (0, 0, 0, 0))
        
        # Create rounded box with shadow
        post_box, box_offset = self.create_rounded_rectangle_with_shadow(
            box_width, box_height, self.border_radius, self.shadow_offset, self.shadow_color, self.box_color
        )
        
        # Paste the box onto canvas (centered)
        box_x = (self.canvas_width - post_box.width) // 2
        box_y = (canvas_height - post_box.height) // 2
        canvas.paste(post_box, (box_x, box_y), post_box)
        
        # Calculate actual content area inside the box with minimal padding
        content_x = box_x + box_offset[0] + 15  # Reduced padding
        content_y = box_y + box_offset[1] + 10  # Reduced padding
        
        # Add subreddit name at the very top
        try:
            subreddit_font = ImageFont.truetype("arial.ttf", 14)
        except:
            subreddit_font = ImageFont.load_default()
        
        draw = ImageDraw.Draw(canvas)
        subreddit_text = f"r/{subreddit}"
        draw.text((content_x, content_y), subreddit_text, fill=(120, 120, 120), font=subreddit_font)
        
        # Adjust content_y to account for subreddit text
        content_y += 20  # Reduced spacing
        
        # Load or create profile image
        profile_img = self.load_and_resize_image(self.profile_image_path, self.profile_size, (150, 150, 150))
        if profile_img is None:
            # Create default gray circle
            profile_img = Image.new('RGBA', (self.profile_size, self.profile_size), (0, 0, 0, 0))
            profile_draw = ImageDraw.Draw(profile_img)
            profile_draw.ellipse([0, 0, self.profile_size, self.profile_size], fill=(150, 150, 150))
        
        # Paste profile image
        canvas.paste(profile_img, (content_x, content_y), profile_img)
        
        # Load or create verified badge (same size as username text)
        verified_img = self.load_and_resize_image(self.verified_emoji_path, self.verified_size)
        if verified_img is None:
            verified_img = self.create_verified_badge(self.verified_size)
        
        # Try to load fonts
        try:
            username_font = ImageFont.truetype("arial.ttf", self.username_font_size)
            button_font = ImageFont.truetype("arial.ttf", 16)
        except:
            username_font = ImageFont.load_default()
            button_font = ImageFont.load_default()
        
        # Draw username with verified badge right next to it
        username_x = content_x + self.profile_size + 15
        username_y = content_y + 5
        
        draw.text((username_x, username_y), self.username, fill=self.username_color, font=username_font)
        
        # Calculate username width to position verified badge immediately next to it
        username_bbox = draw.textbbox((0, 0), self.username, font=username_font)
        username_width = username_bbox[2] - username_bbox[0]
        
        # Paste verified badge right next to username with minimal spacing
        verified_x = username_x + username_width + 5  # Reduced spacing
        verified_y = username_y + 1  # Align better with text baseline
        canvas.paste(verified_img, (verified_x, verified_y), verified_img)
        
        # Create and paste award emojis below username with reduced spacing
        award_y = content_y + 25  # Reduced spacing
        award_x = username_x
        
        for i, award_path in enumerate(self.award_emojis[:7]):  # Limit to 6 awards
            award_img = self.load_and_resize_image(award_path, self.award_size)
            if award_img is None:
                # Create colored circle as fallback
                award_img = self.create_award_emoji(self.award_size, self.award_colors[i % len(self.award_colors)])
            
            # Paste award emoji
            canvas.paste(award_img, (award_x + i * (self.award_size + 5), award_y), award_img)
        
        # Draw title (bold font, multiple lines) with reduced spacing - using censored title
        title_y = content_y + 55  # Reduced spacing
        
        for line in title_lines:
            draw.text((content_x, title_y), line, fill=self.text_color, font=title_font)
            title_y += self.line_height
        
        # Add interaction buttons below title with minimal spacing
        buttons_y = title_y + 10  # Reduced spacing
        
        # Load or create heart (like) icon - same size as username text
        heart_icon = self.load_and_resize_image(self.heart_emoji_path, self.username_font_size)
        if heart_icon is None:
            heart_icon = self.create_heart_icon(self.username_font_size)
        canvas.paste(heart_icon, (content_x, buttons_y), heart_icon)
        
        # Add "99+" text next to heart icon with minimal spacing
        heart_count_x = content_x + self.username_font_size + 5  # Reduced spacing
        heart_count_y = buttons_y + 2  # Align with icon
        draw.text((heart_count_x, heart_count_y), "99+", fill=(120, 120, 120), font=button_font)
        
        # Load or create comment icon positioned closer to heart section
        comment_x = content_x + self.username_font_size + 45  # Closer positioning
        comment_icon = self.load_and_resize_image(self.comment_emoji_path, self.username_font_size)
        if comment_icon is None:
            comment_icon = self.create_comment_icon(self.username_font_size)
        canvas.paste(comment_icon, (comment_x, buttons_y), comment_icon)
        
        # Add "99+" text next to comment icon with minimal spacing
        comment_count_x = comment_x + self.username_font_size + 5  # Reduced spacing
        comment_count_y = buttons_y + 2  # Align with icon
        draw.text((comment_count_x, comment_count_y), "99+", fill=(120, 120, 120), font=button_font)
        
        # Calculate the right edge of the content box for positioning the share icon
        content_right = box_x + box_offset[0] + box_width - 15  # Right edge with padding
        
        # Load or create share icon positioned in bottom right corner
        share_icon = self.load_and_resize_image(self.share_emoji_path, self.username_font_size)
        if share_icon is None:
            share_icon = self.create_share_icon(self.username_font_size)
        
        # Position share icon and text
        share_text = "Share"
        share_text_bbox = draw.textbbox((0, 0), share_text, font=button_font)
        share_text_width = share_text_bbox[2] - share_text_bbox[0]
        
        # Calculate positions for share icon and text (right-aligned)
        share_text_x = content_right - share_text_width
        share_icon_x = share_text_x - self.username_font_size - 5  # Icon to the left of text with small gap
        share_y = buttons_y
        
        # Paste share icon
        canvas.paste(share_icon, (share_icon_x, share_y), share_icon)
        
        # Add "Share" text next to share icon
        share_text_y = buttons_y + 2  # Align with icon
        draw.text((share_text_x, share_text_y), share_text, fill=self.share_text_color, font=button_font)
        
        return canvas
    
    def generate_posts_from_database(self):
        """Generate fake Reddit posts for all stories in the database"""
        
        db_name = f'{self.vid_id}_reddit_nosleep.db'
        if not os.path.exists(db_name):
            print(f"Database {db_name} not found!")
            print("Make sure you've run the Reddit scraper first.")
            return
        
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Get all table names (subreddits)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in database!")
            return
        
        os.makedirs(self.output_dir_png, exist_ok=True)
        total_generated = 0
        
        print(f"Found {len(tables)} table(s) in database:")
        for table_tuple in tables:
            table_name = table_tuple[0]
            print(f"  - {table_name}")
        
        print("\nGenerating fake Reddit posts with censorship (title only)...")
        print("=" * 50)
        
        # Process each table
        for table_tuple in tables:
            table_name = table_tuple[0]
            
            # Determine subreddit name and story type
            if table_name.endswith('_long'):
                subreddit_name = table_name.replace('_long', '')
                story_type = "long"
            else:
                subreddit_name = table_name
                story_type = "short"
            
            print(f"\nProcessing r/{subreddit_name} ({story_type} stories)...")
            
            # Get stories from this table
            try:
                cursor.execute(f"SELECT id, title FROM {table_name}")  # Only get title
                stories = cursor.fetchall()
                
                if not stories:
                    print(f"  No stories found in {table_name}")
                    continue
                    
                print(f"  Found {len(stories)} stories")
                
                # Generate image for each story
                for story_id, title in stories:
                    try:
                        # Create filename
                        if story_type == "long":
                            filename = f"{self.vid_id}_{subreddit_name}_long_{story_id}.png"
                        else:
                            filename = f"{self.vid_id}_{subreddit_name}_{story_id}.png"
                        
                        # Generate the fake post image (title only) with censorship
                        post_image = self.create_fake_reddit_post(title, subreddit_name)
                        
                        # Save the image
                        output_path = os.path.join(self.output_dir_png, filename)
                        post_image.save(output_path, "PNG")
                        
                        print(f"  ✓ Created: {filename}")
                        total_generated += 1
                        
                    except Exception as e:
                        print(f"  ✗ Error creating image for story {story_id}: {e}")
                        
            except Exception as e:
                print(f"  ✗ Error reading from table {table_name}: {e}")
        
        conn.close()
        
        print("\n" + "=" * 50)
        print(f"Generation complete!")
        print(f"Total images created: {total_generated}")
        print(f"Images saved in: {self.output_dir_png}/")
        print(f"Settings: {self.max_chars_per_line} chars/line, {self.title_font_size}px font")
        print(f"Censorship: Enabled ({len(self.censorship_patterns)} patterns)")
        
        if total_generated == 0:
            print("\nTroubleshooting:")
            print("1. Make sure the database file exists")
            print("2. Check that tables contain data")
            print("3. Verify file permissions")
    
    def run(self):
        """Main method to run the generator"""
        print("Fake Reddit Post Generator - Title Only with Censorship")
        print("=" * 60)
        print(f"Video ID: {self.vid_id}")
        print(f"Username: {self.username}")
        print(f"Database: {self.vid_id}_reddit_nosleep.db")
        print(f"Max chars per line: {self.max_chars_per_line}")
        print(f"Title font size: {self.title_font_size}px")
        print(f"Censorship: Enabled ({len(self.censorship_patterns)} patterns)")
        print("=" * 60)
        
        # Generate posts from database
        self.generate_posts_from_database()
class AudioCombiner:
    def __init__(self, vid_id):
        self.vid_id = vid_id
        self.base_dir = f'{vid_id}_Reddit_Audio'
    
    def combine_audio_files(self):
        """Combine audio files in each folder based on post groupings."""
        for folder in os.listdir(self.base_dir):
            folder_path = os.path.join(self.base_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            # Look for both .wav and .mp3 files
            audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3'))]
            posts = self._group_audio_files(audio_files)
            self._combine_posts(posts, folder_path)
    
    def _group_audio_files(self, audio_files):
        """Group audio files by post based on their filename structure."""
        posts = {}
        for audio_file in audio_files:
            # Get file extension and remove it
            file_ext = os.path.splitext(audio_file)[1]
            parts = os.path.splitext(audio_file)[0].split('_')
            
            if len(parts) < 4:
                continue
            vid_id = parts[0]
            subreddit = parts[1]
            
            # Handle both formats:
            # Format 1: {VID_ID}_{subreddit}_{index}_{audio_number}
            # Format 2: {VID_ID}_{subreddit}_long_{index}_{audio_number}
            if len(parts) == 4:
                # Standard format: VID_ID, subreddit, post_number, audio_number
                post_number = parts[2]
                audio_number = parts[3]
                is_long = False
            elif len(parts) == 5 and parts[2] == 'long':
                # Long format: VID_ID, subreddit, 'long', post_number, audio_number
                post_number = parts[3]
                audio_number = parts[4]
                is_long = True
            else:
                # Skip files that don't match expected formats
                print(f"Skipping file with unexpected format: {audio_file}")
                continue
            # Use is_long flag in the key to separate standard and long posts
            key = (subreddit, post_number, is_long)
            if key not in posts:
                posts[key] = []
            posts[key].append((audio_file, int(audio_number), file_ext))
        
        return posts
    
    def _combine_posts(self, posts, folder_path):
        """Combine audio files for each post and export the result."""
        for (subreddit, post_number, is_long), files in posts.items():
            combined = AudioSegment.empty()
            # Sort files by audio number to combine in order
            sorted_files = sorted(files, key=lambda x: x[1])
            for audio_file, _, file_ext in sorted_files:
                audio_path = os.path.join(folder_path, audio_file)
                
                # Load audio based on file extension
                if file_ext == '.wav':
                    audio_segment = AudioSegment.from_wav(audio_path)
                elif file_ext == '.mp3':
                    audio_segment = AudioSegment.from_mp3(audio_path)
                
                combined += audio_segment
            # Create output filename based on whether it's a long post or not
            if is_long:
                output_filename = f'{self.vid_id}_{subreddit}_long_{post_number}.mp3'
            else:
                output_filename = f'{self.vid_id}_{subreddit}_{post_number}.mp3'
            
            output_path = os.path.join(folder_path, output_filename)
            # Export as mp3 (you can change this to 'wav' if you prefer)
            combined.export(output_path, format='mp3')
            print(f'Created combined audio: {output_path}')
            
class VideoCreator:
    def __init__(self, vid_id: str, worker_id=None): # <-- REMOVED: Default value `= VID_ID`
        """Initialize VideoCreator with specified video ID"""
        self.vid_id = vid_id
        self.clips_folder = "util\clips\clip_nosleep"
        self.audio_base_folder = f"{self.vid_id}_Reddit_Audio"
        self.output_folder = f"{self.vid_id}_Final"
        
        # --- NEW: Create a unique temporary directory for this specific instance ---
        worker_suffix = f"-{worker_id}" if worker_id else f"-main-{uuid.uuid4().hex[:6]}"
        self.temp_clips_dir = Path(f"temp_clips_dir_{self.vid_id}{worker_suffix}")
        os.makedirs(self.temp_clips_dir, exist_ok=True)
        # --- END NEW ---
        
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.clip_segment_duration = 60
        self._video_files_cache = None
        self._clip_cache = {}
        self._max_cache_size = 3
        self._cache_hits = 0
        self._cache_misses = 0
    
    @lru_cache(maxsize=1)
    def get_video_files_list(self):
        """Cache the list of video files to avoid repeated directory scans"""
        video_files = [f for f in os.listdir(self.clips_folder) if f.endswith('.mp4')]
        if not video_files:
            raise FileNotFoundError(f"No MP4 files found in {self.clips_folder}")
        return video_files
        
    def cleanup_stale_temp_files(self, directory='.'):
        """Removes stale temporary audio and clip directories."""
        print("--- Running stale temporary file cleanup ---")
        stale_files_found = 0
        # Cleanup old temp audio files
        for filename in os.listdir(directory):
            if filename.startswith('temp-audio-') and filename.endswith('.m4a'):
                try:
                    os.remove(os.path.join(directory, filename))
                    print(f"🗑️ Removed stale temp file: {filename}")
                    stale_files_found += 1
                except OSError as e:
                    print(f"⚠️ Could not remove stale file {filename}: {e}")
        
        # Cleanup old temp clip directories
        for dirname in os.listdir(directory):
            if dirname.startswith('temp_clips_dir_'):
                try:
                    shutil.rmtree(os.path.join(directory, dirname))
                    print(f"🗑️ Removed stale temp clip directory: {dirname}")
                    stale_files_found += 1
                except OSError as e:
                    print(f"⚠️ Could not remove stale directory {dirname}: {e}")
        if stale_files_found == 0:
            print("No stale temp files or directories found.")
        print("--- Cleanup complete ---")
    # --- NEW: Method to get an ISOLATED copy of a clip ---
    def get_isolated_clip(self, original_clip_path):
        """Copies a clip to the instance's temp dir and returns the path to the copy."""
        try:
            # Create a unique name for the copy to avoid any conflicts
            unique_filename = f"{uuid.uuid4().hex[:8]}-{os.path.basename(original_clip_path)}"
            isolated_path = self.temp_clips_dir / unique_filename
            
            # Copy the file
            shutil.copy(original_clip_path, isolated_path)
            
            # Load the VideoFileClip from the ISOLATED copy
            clip = VideoFileClip(str(isolated_path))
            if clip.duration is None or clip.duration <= 0:
                print(f"Warning: Copied clip {isolated_path} has invalid duration. Skipping.")
                clip.close()
                return None
            return clip
        except Exception as e:
            print(f"Error copying or loading isolated clip {original_clip_path}: {e}")
            return None
    def create_video_for_audio_duration(self, target_duration):
        """
        --- FIXED ---
        Creates a video object and returns it AND the list of source clips that must be closed later.
        """
        clips_for_concatenation = []  # Clips that will be concatenated
        base_clips_to_cleanup = []    # Base clips that need cleanup but aren't concatenated
        current_duration = 0
        try:
            video_files = self.get_video_files_list()
            while current_duration < target_duration:
                random_video_file = random.choice(video_files)
                original_clip_path = os.path.join(self.clips_folder, random_video_file)
                base_clip = self.get_isolated_clip(original_clip_path)
                if base_clip is None:
                    print("Failed to get a valid isolated video clip. Cannot complete video.")
                    # Clean up everything we've created so far
                    for clip in clips_for_concatenation:
                        if clip: clip.close()
                    for clip in base_clips_to_cleanup:
                        if clip: clip.close()
                    gc.collect()
                    return None, []
                remaining_time = target_duration - current_duration
                if remaining_time <= 0:
                    base_clip.close()
                    break
                
                if base_clip.duration >= remaining_time:
                    # We need a subclip
                    clip_to_add = base_clip.subclip(0, remaining_time)
                    clips_for_concatenation.append(clip_to_add)
                    base_clips_to_cleanup.append(base_clip)  # Keep base clip for cleanup only
                    current_duration += remaining_time  # Use remaining_time, not clip duration
                else:
                    # Use the whole clip
                    clips_for_concatenation.append(base_clip)
                    current_duration += base_clip.duration
            if not clips_for_concatenation:
                print("No clips were successfully generated for the target duration.")
                return None, []
            print(f"Concatenating {len(clips_for_concatenation)} clips for target duration {target_duration:.2f}s...")
            # Concatenate only the clips intended for the final video
            final_video = concatenate_videoclips(clips_for_concatenation, method='chain')
            # Combine both lists for cleanup
            all_clips_to_cleanup = clips_for_concatenation + base_clips_to_cleanup
            print(f"✅ Created video with duration: {final_video.duration:.2f}s (target: {target_duration:.2f}s)")
            return final_video, all_clips_to_cleanup
        
        except Exception as e:
            print(f"Error in create_video_for_audio_duration: {str(e)}")
            # Clean up any clips we did manage to open
            for clip in clips_for_concatenation:
                try:
                    if clip: clip.close()
                except: pass
            for clip in base_clips_to_cleanup:
                try:
                    if clip: clip.close()
                except: pass
            return None, []
    
    # ... (get_primary_audio_files and generate_output_filename remain the same) ...
    def get_primary_audio_files(self):
        if hasattr(self, '_audio_files_cache') and self._audio_files_cache:
            return self._audio_files_cache
        audio_files = []
        for subfolder in os.listdir(self.audio_base_folder):
            subfolder_path = os.path.join(self.audio_base_folder, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            for audio_file in os.listdir(subfolder_path):
                if not (audio_file.endswith('.wav') or audio_file.endswith('.mp3')):
                    continue
                filename_without_ext = audio_file.replace('.wav', '').replace('.mp3', '')
                parts = filename_without_ext.split('_')
                valid_file = False
                if len(parts) == 3 and parts[0] == self.vid_id and parts[2].isdigit():
                    valid_file = True
                elif len(parts) == 4 and parts[0] == self.vid_id and parts[2] == 'long' and parts[3].isdigit():
                    valid_file = True
                if valid_file:
                    audio_path = os.path.join(subfolder_path, audio_file)
                    audio_files.append(audio_path)
        self._audio_files_cache = audio_files
        return audio_files
    def generate_output_filename(self, audio_path):
        audio_filename = os.path.basename(audio_path).replace('.wav', '').replace('.mp3', '')
        parts = audio_filename.split('_')
        if len(parts) >= 5 and 'long' in parts:
            long_index = parts.index('long')
            if long_index >= 2:
                new_filename = f"{parts[0]}_{parts[1]}_long_{parts[long_index + 1]}"
                output_path = os.path.join(self.output_folder, f"{new_filename}.mp4")
                return output_path
        output_path = os.path.join(self.output_folder, f"{audio_filename}.mp4")
        return output_path
    
    # process_single_audio_file remains mostly the same, just ensure cleanup is robust.
    def process_single_audio_file(self, audio_path):
        """ --- MODIFIED --- Now handles the cleanup of source clips correctly. """
        filename = os.path.basename(audio_path)
        print(f"\n🎵 Processing: {filename}")
        # Initialize all resources to None
        audio = None
        final_video = None
        clips_to_clean_up = [] # The list of source clips to close
        unique_temp_audio = f'temp-audio-{uuid.uuid4().hex[:8]}.m4a'
    
        try:
            if not os.path.exists(audio_path):
                return False, f"❌ Audio file not found: {filename}"
            output_path = self.generate_output_filename(audio_path)
            if os.path.exists(output_path):
                return True, f"⏩ Skipped {os.path.basename(output_path)} - already exists"
            # 1. Load Audio
            audio = AudioFileClip(audio_path)
            print(f"✅ Audio loaded: {audio.duration:.2f} seconds")
            # 2. Create video object AND get the list of clips to clean
            final_video, clips_to_clean_up = self.create_video_for_audio_duration(audio.duration)
            if final_video is None or final_video.duration <= 0:
                raise ValueError(f"Failed to create a valid video for {filename}.")
            print(f"✅ Video created: {final_video.duration:.2f}s")
            final_video.audio = audio
            # 3. Write the final video to disk. The source clips MUST be open during this step.
            print(f"Writing final video to: {os.path.basename(output_path)}")
            final_video.write_videofile(
                output_path,
                codec='libx264', audio_codec='aac',
                temp_audiofile=unique_temp_audio,
                remove_temp=True, verbose=False, logger=None,
                preset='ultrafast', threads=mp.cpu_count() # Use more threads for writing
            )
            print(f"✅ Successfully created: {os.path.basename(output_path)}")
            return True, f"✅ Successfully created: {os.path.basename(output_path)}"
        except Exception as e:
            print(f"❌ Exception in process_single_audio_file for {filename}: {str(e)}")
            # No need for traceback here unless debugging, it's very verbose
            return False, f"❌ Error processing {filename}: {str(e)}"
        finally:
            # --- THIS IS THE CRITICAL CLEANUP SEQUENCE ---
            # It runs regardless of success or failure.
            # 4. NOW that writing is done (or failed), close all resources.
            if audio: audio.close()
            if final_video: final_video.close()
            # Close all the source video clips
            for clip in clips_to_clean_up:
                try:
                    if clip: clip.close()
                except Exception: pass
            # Clean up the temporary copied files and directory for this worker
            self.cleanup_instance() 
            # Clean up the temp audio file
            if os.path.exists(unique_temp_audio):
                try:
                    os.remove(unique_temp_audio)
                except OSError: pass
            gc.collect()
    # --- MODIFIED: Renamed cleanup_cache to cleanup_instance ---
    def cleanup_instance(self):
        """Cleans up all resources used by this instance, including the temp dir."""
        try:
            # This will fail if clips are still open, which is a good indicator
            # that cleanup in the main logic needs to be fixed.
            shutil.rmtree(self.temp_clips_dir)
            # print(f"🧹 Successfully cleaned up temp dir: {self.temp_clips_dir}")
        except Exception as e:
            print(f"⚠️ Could not remove temp dir {self.temp_clips_dir}. It may have open files. Error: {e}")
    
    # ... (create_videos_parallel, create_videos, create_videos_sequential remain the same) ...
    def create_videos_parallel(self, max_workers=8):
        print(f"Starting parallel video creation for {self.vid_id}")
        audio_files = self.get_primary_audio_files()
        if not audio_files:
            print("No primary audio files found!")
            return
        if max_workers is None:
            max_workers = 8
        print(f"Found {len(audio_files)} audio files to process")
        print(f"Using {max_workers} parallel workers")
        successful_count = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_audio = {
                executor.submit(process_single_file_worker, audio_path, self.vid_id): audio_path 
                for audio_path in audio_files
            }
            for future in as_completed(future_to_audio):
                audio_path = future_to_audio[future]
                try:
                    success, message = future.result()
                    print(message)
                    if success:
                        successful_count += 1
                except Exception as exc:
                    print(f"Error processing {os.path.basename(audio_path)}: {exc}")
        print(f"\n🎬 Parallel video creation complete!")
        print(f"Successfully created {successful_count}/{len(audio_files)} videos")
    def create_videos(self, use_parallel=True, max_workers=8):
        if use_parallel:
            self.create_videos_parallel(max_workers)
        else:
            self.create_videos_sequential()
    def create_videos_sequential(self):
        print(f"Starting sequential video creation for {self.vid_id}")
        audio_files = self.get_primary_audio_files()
        if not audio_files:
            print("No primary audio files found!")
            return
        print(f"Found {len(audio_files)} audio files to process")
        successful_count = 0
        for i, audio_path in enumerate(audio_files, 1):
            success, message = self.process_single_audio_file(audio_path)
            print(f"[{i}/{len(audio_files)}] {message}")
            if success:
                successful_count += 1
        print(f"\n🎬 Sequential video creation complete!")
        print(f"Successfully created {successful_count}/{len(audio_files)} videos")
# --- MODIFIED WORKER FUNCTION ---
def process_single_file_worker(audio_path, vid_id):
    """ --- MODIFIED --- Simplified the finally block. """
    worker_id = os.getpid()
    creator = None
    try:
        creator = VideoCreator(vid_id=vid_id, worker_id=worker_id)
        # The main logic and all cleanup is now self-contained in process_single_audio_file
        return creator.process_single_audio_file(audio_path)
    except Exception as e:
        print(f"[Worker {worker_id}] CRITICAL ERROR: {e}")
        return False, f"CRITICAL error in worker for {os.path.basename(audio_path)}"
    finally:
        # The creator's own finally block will handle all cleanup.
        # No need to call creator.cleanup_instance() again here.
        gc.collect()
class VideoCaptioner:
    """
    A class for automatically captioning video files using Whisper AI and MoviePy.
    """
    
    def __init__(self, vid_id: str, model_size="base"): # <-- REMOVED: Default value for vid_id
        """
        Initialize the VideoCaptioner.
        
        Args:
            vid_id (str): Video ID used for folder naming and file processing.
            model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
        """
        self.vid_id = vid_id
        self.model_size = model_size
        self.input_folder = f"{vid_id}_Final"
        self.output_folder = f"{vid_id}_Captioned"
        self.device = self._setup_device()
        self.model = None
        
        # Caption styling configuration
        self.caption_config = {
            'fontsize': 80,
            'font': 'Britannic',
            'color': 'white',
            'highlight_color': 'yellow',
            'shadow_offset': 3,
            'stroke_width': 2,
            'words_per_caption': 3,
            'board_label_size': 50,
            'min_duration': 0.5,  # Minimum duration for captions
            'max_duration': 3.0,  # Maximum duration for captions
            'highlight_duration_ratio': 0.3  # Fraction of word duration for highlighting
        }
    
    def _setup_device(self):
        """Configure CUDA device and display GPU information."""
        if torch.cuda.is_available():
            device = "cuda"
            print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = "cpu"
            print("CUDA not available, falling back to CPU")
        return device
    
    def _load_model(self):
        """Load the Whisper model with CUDA support."""
        print(f"Loading Whisper model '{self.model_size}' on {self.device}...")
        self.model = whisper.load_model(self.model_size, device=self.device)
    
    def _clear_gpu_cache(self):
        """Clear GPU cache to prevent memory issues."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _print_gpu_memory(self):
        """Print current GPU memory usage."""
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Used: {gpu_memory_used:.2f} GB, Cached: {gpu_memory_cached:.2f} GB")
    
    def extract_reddit_board(self, filename):
        """
        Extract reddit board name from filename.
        
        Args:
            filename (str): Input filename to parse
            
        Returns:
            str: Reddit board name in format 'r/boardname'
        """
        name_without_ext = os.path.splitext(filename)[0]
        print(f"Debug - Processing filename: {name_without_ext}")
        
        # Try pattern with 'long': VID_ID_boardname_long_index
        long_pattern = rf"{re.escape(self.vid_id)}_(.+?)_long_(\d+)"
        long_match = re.match(long_pattern, name_without_ext)
        
        if long_match:
            reddit_board = long_match.group(1)
            print(f"Debug - Matched long pattern: board={reddit_board}")
            return f"r/{reddit_board}"
        
        # Try pattern without 'long': VID_ID_boardname_index
        short_pattern = rf"{re.escape(self.vid_id)}_(.+?)_(\d+)$"
        short_match = re.match(short_pattern, name_without_ext)
        
        if short_match:
            reddit_board = short_match.group(1)
            print(f"Debug - Matched short pattern: board={reddit_board}")
            return f"r/{reddit_board}"
        
        # Fallback
        print(f"Debug - No pattern matched, using fallback")
        return "r/unknown"
    
    def generate_output_filename(self, filename):
        """
        Generate output filename based on input format.
        
        Args:
            filename (str): Input filename
            
        Returns:
            str: Generated output filename
        """
        name_without_ext = os.path.splitext(filename)[0]
        
        # Check if it's a 'long' video: VID_ID_boardname_long_index
        long_pattern = rf"{re.escape(self.vid_id)}_(.+?)_long_(\d+)"
        long_match = re.match(long_pattern, name_without_ext)
        
        if long_match:
            reddit_board = long_match.group(1)
            index = long_match.group(2)
            result = f"final_captioned_{self.vid_id}_{reddit_board}_long_{index}.mp4"
            print(f"Debug - Generated long filename: {result}")
            return result
        
        # Check if it's a regular video: VID_ID_boardname_index
        short_pattern = rf"{re.escape(self.vid_id)}_(.+?)_(\d+)$"
        short_match = re.match(short_pattern, name_without_ext)
        
        if short_match:
            reddit_board = short_match.group(1)
            index = short_match.group(2)
            result = f"final_captioned_{self.vid_id}_{reddit_board}_{index}.mp4"
            print(f"Debug - Generated short filename: {result}")
            return result
        
        # Fallback: use original filename with prefix
        result = f"final_captioned_{filename}"
        print(f"Debug - Used fallback filename: {result}")
        return result
    
    def _group_words_sequential(self, words):
        """
        Group words into sequential caption segments without overlapping.
        
        Args:
            words (list): List of word information with timestamps
            
        Returns:
            list: List of caption groups with sequential timing
        """
        if not words:
            return []
        
        captions = []
        words_per_caption = self.caption_config['words_per_caption']
        min_duration = self.caption_config['min_duration']
        max_duration = self.caption_config['max_duration']
        
        for i in range(0, len(words), words_per_caption):
            chunk = words[i:i + words_per_caption]
            if not chunk:
                continue
            
            # Get timing from first and last word in chunk
            start_time = chunk[0]['start']
            end_time = chunk[-1]['end']
            
            # Ensure minimum duration
            duration = end_time - start_time
            if duration < min_duration:
                end_time = start_time + min_duration
                duration = min_duration
            
            # Cap maximum duration
            if duration > max_duration:
                end_time = start_time + max_duration
                duration = max_duration
            
            # Combine words into single text
            text = ' '.join([word['word'].strip() for word in chunk])
            
            captions.append({
                'text': text,
                'start': start_time,
                'end': end_time,
                'duration': duration,
                'words': chunk  # Keep word info for highlighting
            })
        
        # Ensure no overlaps by adjusting end times
        for i in range(len(captions) - 1):
            if captions[i]['end'] > captions[i + 1]['start']:
                captions[i]['end'] = captions[i + 1]['start']
                captions[i]['duration'] = captions[i]['end'] - captions[i]['start']
        
        return captions
    
    def _create_highlighted_caption_clips(self, caption_info, video_width):
        """
        Create caption clips with word-by-word highlighting that don't overlap.
        
        Args:
            caption_info (dict): Caption information including words and timing
            video_width (int): Width of the video
            
        Returns:
            list: List of TextClip objects with highlighting
        """
        clips = []
        words = caption_info['words']
        full_text = caption_info['text']
        caption_start = caption_info['start']
        caption_end = caption_info['end']
        caption_duration = caption_info['duration']
        
        # Calculate highlight timing
        highlight_duration_per_word = caption_duration / len(words)
        min_highlight_duration = 0.1  # Minimum time to show each highlight
        
        # Ensure each word gets at least minimum highlight time
        if highlight_duration_per_word < min_highlight_duration:
            highlight_duration_per_word = min_highlight_duration
        
        # Create clips for each highlight state
        current_time = caption_start
        
        for word_idx, word_info in enumerate(words):
            # Calculate timing for this highlight
            highlight_start = current_time
            highlight_end = min(highlight_start + highlight_duration_per_word, caption_end)
            highlight_duration = highlight_end - highlight_start
            
            if highlight_duration <= 0:
                break
            
            # Create clip with current word highlighted
            clip = self._create_pango_caption_clip(
                full_text, 
                highlight_duration, 
                video_width,
                word_idx
            )
            clip = clip.set_start(highlight_start)
            clips.append(clip)
            
            current_time = highlight_end
            
            # If we've reached the end of the caption, break
            if current_time >= caption_end:
                break
        
        # If there's remaining time, show the full text without highlighting
        if current_time < caption_end:
            remaining_duration = caption_end - current_time
            final_clip = self._create_pango_caption_clip(
                full_text, 
                remaining_duration, 
                video_width,
                None  # No highlighting
            )
            final_clip = final_clip.set_start(current_time)
            clips.append(final_clip)
        
        return clips
    
    def _create_pango_caption_clip(self, text, duration, video_width, highlight_word_index=None):
        """
        Create a caption clip using Pango markup with optional word highlighting.
        
        Args:
            text (str): Caption text
            duration (float): Duration of the caption
            video_width (int): Width of the video
            highlight_word_index (int, optional): Index of word to highlight in the text
            
        Returns:
            TextClip: Styled caption clip with Pango markup
        """
        try:
            # If highlighting is requested, apply markup
            if highlight_word_index is not None:
                words = text.split()
                if 0 <= highlight_word_index < len(words):
                    words[highlight_word_index] = f'<span foreground="{self.caption_config["highlight_color"]}">{words[highlight_word_index]}</span>'
                    markup_text = ' '.join(words)
                else:
                    markup_text = text
            else:
                markup_text = text
            # Create shadow clip
            shadow_offset = self.caption_config['shadow_offset']
            shadow_clip = TextClip(text,  # Use plain text for shadow
                                fontsize=self.caption_config['fontsize'], 
                                color='black', 
                                font=self.caption_config['font'],
                                method='pango', 
                                size=(video_width * 0.9, None))
            
            # Create main text clip with markup
            txt_clip = TextClip(markup_text,
                                fontsize=self.caption_config['fontsize'], 
                                color=self.caption_config['color'], 
                                font=self.caption_config['font'],
                                method='pango', 
                                size=(video_width * 0.9, None))
            # Composite shadow and text
            text_composite_clip = CompositeVideoClip([
                shadow_clip.set_position((shadow_offset, shadow_offset)),
                txt_clip.set_position((0, 0)),
            ]).set_position(('center', 'center'))
            
            return text_composite_clip.set_duration(duration)
            
        except Exception as pango_error:
            print(f"Pango method failed: {pango_error}. Falling back to stroke method.")
            # Fallback to stroke method
            txt_clip = TextClip(text,
                                fontsize=self.caption_config['fontsize'], 
                                color=self.caption_config['color'], 
                                font=self.caption_config['font'],
                                stroke_color='black',
                                stroke_width=self.caption_config['stroke_width'],
                                method='caption',
                                size=(video_width * 0.9, None))
            
            return txt_clip.set_position('center').set_duration(duration)
    
    def _create_board_label(self, reddit_board, video_duration):
        """
        Create a Reddit board label for the video.
        
        Args:
            reddit_board (str): Reddit board name
            video_duration (float): Duration of the video
            
        Returns:
            TextClip: Board label clip
        """
        board_label = TextClip(reddit_board,
                            fontsize=self.caption_config['board_label_size'], 
                            color=self.caption_config['color'], 
                            font=self.caption_config['font'],
                            stroke_color='black', 
                            stroke_width=self.caption_config['stroke_width'])
        return board_label.set_position(('left', 'top')).set_duration(video_duration)
    
    def process_single_video(self, video_file):
        """
        Process a single video file to add captions.
        
        Args:
            video_file (str): Name of the video file to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        input_path = os.path.join(self.input_folder, video_file)
        output_filename = self.generate_output_filename(video_file)
        output_path = os.path.join(self.output_folder, output_filename)
        
        print(f"\nProcessing: {video_file}")
        print(f"Output will be: {output_filename}")
        
        self._print_gpu_memory()
        
        if not os.path.exists(input_path):
            print(f"Warning: File {input_path} does not exist, skipping...")
            return False
        video = None
        final = None
        try:
            print("Transcribing audio with CUDA acceleration...")
            result = self.model.transcribe(input_path, word_timestamps=True)
            video = VideoFileClip(input_path)
            clips = [video]  # Start with the video
            reddit_board = self.extract_reddit_board(video_file)
            print(f"Detected Reddit board: {reddit_board}")
            
            # Add board label
            board_label = self._create_board_label(reddit_board, video.duration)
            clips.append(board_label)
            # Extract all words with timestamps
            words = []
            for segment in result["segments"]:
                words.extend(segment.get("words", []))
            print(f"Processing {len(words)} words into sequential caption groups...")
            
            # Group words into sequential caption segments (no overlapping)
            captions = self._group_words_sequential(words)
            
            print(f"Created {len(captions)} sequential caption segments")
            
            # Create caption clips with highlighting
            all_caption_clips = []
            for caption in tqdm(captions, desc=f"Creating highlighted captions for {video_file}"):
                highlighted_clips = self._create_highlighted_caption_clips(caption, video.w)
                all_caption_clips.extend(highlighted_clips)
            
            clips.extend(all_caption_clips)
            print(f"Total clips created: {len(clips)} (1 video + 1 board label + {len(all_caption_clips)} caption clips)")
            print("Compositing final video...")
            final = CompositeVideoClip(clips)
            
            print("Writing final video file...")
            final.write_videofile(output_path, 
                                codec='libx264', 
                                audio_codec='aac',
                                temp_audiofile='temp-audio.m4a',
                                remove_temp=True)
            
            print(f"Successfully processed: {video_file}")
            return True
            
        except Exception as e:
            print(f"An error occurred while processing {video_file}: {e}")
            return False
        
        finally:
            print("--- Running post-video cleanup ---")
            if video:
                video.close()
            if final:
                final.close()
            
            print("Triggering garbage collection...")
            gc.collect()
            
            print("Clearing GPU cache...")
            self._clear_gpu_cache()
            print("--- Cleanup finished ---")
    
    def process_all_videos(self):
        """
        Process all video files in the input folder.
        
        Returns:
            dict: Processing results with success/failure counts
        """
        # Setup: Check if input folder exists and create output folder
        if not os.path.exists(self.input_folder):
            raise FileNotFoundError(f"Input folder '{self.input_folder}' does not exist. Please check the VID_ID value and folder structure.")
        os.makedirs(self.output_folder, exist_ok=True)
        # Load the Whisper model
        if self.model is None:
            self._load_model()
        # Find video files
        try:
            video_files = [f for f in os.listdir(self.input_folder) if f.endswith(('.mp4', '.mov', '.avi'))]
        except FileNotFoundError as e:
            print(f"Error accessing folder '{self.input_folder}': {e}")
            raise
        if not video_files:
            raise FileNotFoundError(f"No video files found in {self.input_folder}")
        print(f"Found {len(video_files)} video files: {video_files}")
        # Process all videos
        results = {'successful': 0, 'failed': 0, 'total': len(video_files)}
        
        for video_idx, video_file in enumerate(tqdm(video_files, desc="Processing videos")):
            print(f"\n--- Processing video {video_idx + 1}/{len(video_files)} ---")
            
            success = self.process_single_video(video_file)
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1
        print(f"\nProcessing complete!")
        print(f"Results: {results['successful']}/{results['total']} videos processed successfully")
        
        if torch.cuda.is_available():
            self._clear_gpu_cache()
            print("Final GPU memory cleared.")
        
        return results
    
class RedditVideoProcessor:
    def __init__(self, fake_reddit_folder, captioned_folder, output_folder_p2v="output_videos"):
        self.fake_reddit_folder = fake_reddit_folder
        self.captioned_folder = captioned_folder
        self.output_folder_p2v = output_folder_p2v
        self.whisper_model = None
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder_p2v, exist_ok=True)
        
    def load_whisper_model(self, model_size="base"):
        """Load Whisper model for speech recognition"""
        logging.info(f"Loading Whisper model: {model_size}")
        self.whisper_model = whisper.load_model(model_size)
        
    def get_database_connection(self, vid_id):
        """Get database connection for a specific video ID"""
        db_path = f"{vid_id}_reddit_nosleep.db"
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
        return sqlite3.connect(db_path)
    
    def get_title_from_db(self, vid_id, reddit_forum, forum_index, is_long=False):
        """Get title from database for specific video, forum, and index"""
        try:
            conn = self.get_database_connection(vid_id)
            cursor = conn.cursor()
            # Check what tables exist in the database
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            logging.info(f"Available tables in database: {tables}")
            # For long format files, try different table naming strategies
            table_names_to_try = []
            if is_long:
                # For long files, try these table name variations:
                table_names_to_try = [
                    f"{reddit_forum}_long",  # e.g., "nosleep_long"
                    f"{reddit_forum}Long",   # e.g., "nosleepLong"
                    reddit_forum,            # fallback to regular table
                ]
            else:
                # For regular files, just use the forum name
                table_names_to_try = [reddit_forum]
            logging.info(f"Will try these table names: {table_names_to_try}")
            result = None
            successful_table = None
            for table_name in table_names_to_try:
                if table_name in tables:
                    try:
                        logging.info(f"Trying table: {table_name}")
                        query = f"SELECT title FROM {table_name} WHERE rowid = ?"
                        cursor.execute(query, (forum_index,))
                        result = cursor.fetchone()
                        if result:
                            successful_table = table_name
                            logging.info(f"✓ Found title in table: {table_name}")
                            break
                        else:
                            logging.info(f"No entry with rowid {forum_index} in table {table_name}")
                            # Check what entries exist in this table
                            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                            count_result = cursor.fetchone()
                            total_entries = count_result[0] if count_result else 0
                            logging.info(f"Table {table_name} has {total_entries} entries")
                            if total_entries > 0:
                                cursor.execute(f"SELECT rowid, title FROM {table_name} LIMIT 5")
                                sample_results = cursor.fetchall()
                                logging.info(f"Sample entries from {table_name}:")
                                for row_id, title in sample_results:
                                    logging.info(f"  {row_id}: {title[:100]}...")
                    except sqlite3.Error as e:
                        logging.warning(f"Error querying table {table_name}: {e}")
                        continue
                else:
                    logging.info(f"Table {table_name} does not exist in database")
            # If still no result, try fallback strategies
            if not result and is_long:
                logging.info("No result found for long format, trying fallback strategies...")
                # Try to find any table that might contain the data
                for table_name in tables:
                    if reddit_forum.lower() in table_name.lower():
                        try:
                            logging.info(f"Trying fallback table: {table_name}")
                            cursor.execute(f"SELECT title FROM {table_name} WHERE rowid = ?", (forum_index,))
                            result = cursor.fetchone()
                            if result:
                                successful_table = table_name
                                logging.info(f"✓ Found title in fallback table: {table_name}")
                                break
                        except sqlite3.Error as e:
                            logging.warning(f"Error querying fallback table {table_name}: {e}")
                            continue
                        
            conn.close()
            if result:
                logging.info(f"Successfully retrieved title from table: {successful_table}")
                return result[0]
            else:
                logging.warning(f"No title found for {vid_id}_{reddit_forum}_{forum_index} (is_long: {is_long})")
                logging.warning("Consider checking your database structure and table names")
                return None
        except Exception as e:
            logging.error(f"Database error: {e}")
            logging.error(traceback.format_exc())
            return None
    
    def parse_filename(self, filename):
        """Parse filename to extract video ID, reddit forum, and index with enhanced debugging"""
        try:
            # Remove file extension
            name_without_ext = os.path.splitext(filename)[0]
            logging.info(f"=== PARSING FILENAME: {filename} ===")
            logging.info(f"Name without extension: {name_without_ext}")
            
            # Handle both regular and "long" versions
            if "_long_" in name_without_ext:
                logging.info("Detected '_long_' pattern in filename")
                # Pattern: {VID_ID}_{reddit_forum}_long_{forum_index}
                parts = name_without_ext.split("_long_")
                logging.info(f"Split by '_long_': {parts}")
                
                if len(parts) == 2:
                    vid_forum_part = parts[0]
                    forum_index = int(parts[1])
                    logging.info(f"vid_forum_part: '{vid_forum_part}', forum_index: {forum_index}")
                    
                    # Split vid_forum_part to get VID_ID and reddit_forum
                    vid_forum_parts = vid_forum_part.split("_")
                    logging.info(f"vid_forum_parts: {vid_forum_parts}")
                    
                    if len(vid_forum_parts) >= 2:
                        vid_id = vid_forum_parts[0]
                        reddit_forum = "_".join(vid_forum_parts[1:])
                        logging.info(f"✓ LONG FORMAT PARSED:")
                        logging.info(f"  vid_id: '{vid_id}'")
                        logging.info(f"  reddit_forum: '{reddit_forum}'")
                        logging.info(f"  forum_index: {forum_index}")
                        logging.info(f"  is_long: True")
                        return vid_id, reddit_forum, forum_index, True
                    else:
                        logging.warning(f"Insufficient parts in vid_forum_part: {vid_forum_parts}")
                else:
                    logging.warning(f"Unexpected number of parts when splitting by '_long_': {parts}")
            else:
                logging.info("No '_long_' pattern detected, trying regular format")
                # Pattern: {VID_ID}_{reddit_forum}_{forum_index}
                parts = name_without_ext.split("_")
                logging.info(f"Split by '_': {parts}")
                
                if len(parts) >= 3:
                    vid_id = parts[0]
                    forum_index = int(parts[-1])
                    reddit_forum = "_".join(parts[1:-1])
                    logging.info(f"✓ REGULAR FORMAT PARSED:")
                    logging.info(f"  vid_id: '{vid_id}'")
                    logging.info(f"  reddit_forum: '{reddit_forum}'")
                    logging.info(f"  forum_index: {forum_index}")
                    logging.info(f"  is_long: False")
                    return vid_id, reddit_forum, forum_index, False
                else:
                    logging.warning(f"Insufficient parts for regular format: {parts}")
            
            logging.warning(f"Could not parse filename: {filename}")
            return None, None, None, False
            
        except Exception as e:
            logging.error(f"Error parsing filename {filename}: {e}")
            logging.error(traceback.format_exc())
            return None, None, None, False
    
    def find_matching_files(self):
        """Find matching PNG and MP4 files"""
        try:
            # Check if directories exist
            if not os.path.exists(self.fake_reddit_folder):
                logging.error(f"Fake reddit folder not found: {self.fake_reddit_folder}")
                return []
            
            if not os.path.exists(self.captioned_folder):
                logging.error(f"Captioned folder not found: {self.captioned_folder}")
                return []
            
            png_files = [f for f in os.listdir(self.fake_reddit_folder) if f.endswith('.png')]
            mp4_files = [f for f in os.listdir(self.captioned_folder) if f.endswith('.mp4')]
            
            logging.info(f"Found {len(png_files)} PNG files and {len(mp4_files)} MP4 files")
            logging.debug(f"PNG files: {png_files}")
            logging.debug(f"MP4 files: {mp4_files}")
            
            matches = []
            
            for png_file in png_files:
                logging.info(f"Processing PNG file: {png_file}")
                parsed_png = self.parse_filename(png_file)
                if parsed_png[0] is None:
                    logging.warning(f"Skipping PNG file with unparseable name: {png_file}")
                    continue
                    
                vid_id, reddit_forum, forum_index, is_long = parsed_png
                
                # Look for corresponding MP4 file with new naming convention
                if is_long:
                    expected_mp4 = f"final_captioned_{vid_id}_{reddit_forum}_long_{forum_index}.mp4"
                else:
                    expected_mp4 = f"final_captioned_{vid_id}_{reddit_forum}_{forum_index}.mp4"
                
                logging.info(f"Looking for matching MP4: {expected_mp4}")
                
                if expected_mp4 in mp4_files:
                    logging.info(f"Found match: {png_file} <-> {expected_mp4}")
                    matches.append({
                        'png_file': png_file,
                        'mp4_file': expected_mp4,
                        'vid_id': vid_id,
                        'reddit_forum': reddit_forum,
                        'forum_index': forum_index,
                        'is_long': is_long
                    })
                else:
                    logging.warning(f"No matching MP4 found for {png_file}. Expected: {expected_mp4}")
            
            logging.info(f"Total matches found: {len(matches)}")
            return matches
            
        except Exception as e:
            logging.error(f"Error in find_matching_files: {e}")
            logging.error(traceback.format_exc())
            return []
    
    def transcribe_video(self, video_path, max_duration=20):
        """Transcribe only the first portion of video using Whisper and return segments with timestamps"""
        try:
            if self.whisper_model is None:
                self.load_whisper_model()
            
            logging.info(f"Transcribing first {max_duration} seconds of video: {video_path}")
            
            # Create a temporary audio file with only the first max_duration seconds
            import tempfile
            from moviepy.editor import VideoFileClip
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            try:
                # Extract first max_duration seconds of audio
                video_clip = VideoFileClip(video_path)
                audio_clip = video_clip.subclip(0, min(max_duration, video_clip.duration)).audio
                
                # Write to temporary audio file
                audio_clip.write_audiofile(temp_audio_path, verbose=False, logger=None)
                
                # Clean up video clips
                audio_clip.close()
                video_clip.close()
                
                # Transcribe the shortened audio
                result = self.whisper_model.transcribe(temp_audio_path, word_timestamps=True)
                
                logging.info(f"Transcription completed for first {max_duration}s of: {video_path}")
                return result
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
        
        except Exception as e:
            logging.error(f"Error transcribing video {video_path}: {e}")
            raise
    
    def find_title_end_time(self, transcription, title_text):
        """Find when the title text ends in the transcription with improved matching"""
        try:
            if not title_text:
                logging.warning("No title text provided")
                return None
            # Clean and normalize title text
            def clean_text(text):
                # Remove extra whitespace, convert to lowercase, remove punctuation
                import string
                text = text.lower()
                text = text.translate(str.maketrans('', '', string.punctuation))
                text = ' '.join(text.split())  # Remove extra whitespace
                return text
            title_clean = clean_text(title_text)
            title_words = title_clean.split()
            logging.info(f"Original title: {title_text}")
            logging.info(f"Cleaned title: {title_clean}")
            logging.info(f"Title words to find: {title_words}")
            # Extract all words with timestamps from transcription
            all_words = []
            transcribed_text_parts = []
            for segment in transcription['segments']:
                logging.debug(f"Processing segment: {segment.get('text', '')}")
                # Add segment text for debugging
                transcribed_text_parts.append(segment.get('text', ''))
                if 'words' in segment and segment['words']:
                    for word_info in segment['words']:
                        if 'word' in word_info and 'start' in word_info and 'end' in word_info:
                            clean_word = clean_text(word_info['word'].strip())
                            if clean_word:  # Only add non-empty words
                                all_words.append({
                                    'word': clean_word,
                                    'original': word_info['word'].strip(),
                                    'start': word_info['start'],
                                    'end': word_info['end']
                                })
                else:
                    # Fallback: if no word-level timestamps, try to estimate from segment
                    segment_text = segment.get('text', '').strip()
                    if segment_text:
                        segment_words = clean_text(segment_text).split()
                        segment_start = segment.get('start', 0)
                        segment_end = segment.get('end', segment_start + 5)
                        segment_duration = segment_end - segment_start
                        if len(segment_words) > 0:
                            word_duration = segment_duration / len(segment_words)
                            for i, word in enumerate(segment_words):
                                word_start = segment_start + (i * word_duration)
                                word_end = segment_start + ((i + 1) * word_duration)
                                all_words.append({
                                    'word': word,
                                    'original': word,
                                    'start': word_start,
                                    'end': word_end
                                })
            # Log full transcribed text for debugging
            full_transcribed_text = ' '.join(transcribed_text_parts)
            logging.info(f"Full transcribed text: {full_transcribed_text}")
            logging.info(f"Cleaned transcribed text: {clean_text(full_transcribed_text)}")
            logging.info(f"Total words with timestamps: {len(all_words)}")
            if len(all_words) == 0:
                logging.warning("No words with timestamps found in transcription")
                return None
            # Log first few words for debugging
            logging.info("First 10 transcribed words:")
            for i, word_info in enumerate(all_words[:10]):
                logging.info(f"  {i}: '{word_info['word']}' (original: '{word_info['original']}') at {word_info['start']:.2f}-{word_info['end']:.2f}s")
            # Try different matching strategies
            # Strategy 1: Exact sequence matching
            logging.info("Trying exact sequence matching...")
            for i in range(len(all_words) - len(title_words) + 1):
                window_words = [w['word'] for w in all_words[i:i+len(title_words)]]
                if window_words == title_words:
                    end_time = all_words[i+len(title_words)-1]['end']
                    logging.info(f"✓ Exact match found! End time: {end_time:.2f}s")
                    logging.info(f"Matched sequence: {window_words}")
                    return end_time
            # Strategy 2: Fuzzy matching with similarity threshold
            logging.info("Trying fuzzy matching...")
            best_match_end = None
            best_similarity = 0
            best_match_details = None
            for i in range(len(all_words) - len(title_words) + 1):
                window_words = [w['word'] for w in all_words[i:i+len(title_words)]]
                window_text = ' '.join(window_words)
                similarity = SequenceMatcher(None, window_text, title_clean).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_end = all_words[i+len(title_words)-1]['end']
                    best_match_details = {
                        'window_words': window_words,
                        'window_text': window_text,
                        'start_index': i,
                        'end_index': i+len(title_words)-1,
                        'start_time': all_words[i]['start'],
                        'end_time': best_match_end
                    }
            logging.info(f"Best fuzzy match similarity: {best_similarity:.3f}")
            if best_match_details:
                logging.info(f"Best match details: {best_match_details}")
                logging.info(f"Title words: {title_words}")
                logging.info(f"Matched words: {best_match_details['window_words']}")
            # Use fuzzy match if similarity is above threshold
            if best_similarity > 0.5:  # Lowered threshold
                logging.info(f"✓ Fuzzy match accepted (similarity: {best_similarity:.3f})")
                return best_match_end
            # Strategy 3: Partial matching - look for significant portion of title
            logging.info("Trying partial matching...")
            min_words_to_match = max(1, len(title_words) // 2)  # At least half the words
            for i in range(len(all_words) - min_words_to_match + 1):
                for window_size in range(min_words_to_match, min(len(title_words) + 1, len(all_words) - i + 1)):
                    window_words = [w['word'] for w in all_words[i:i+window_size]]
                    # Count matching words
                    matches = sum(1 for word in window_words if word in title_words)
                    match_ratio = matches / len(title_words)
                    if match_ratio >= 0.5:  # At least 50% of title words found
                        end_time = all_words[i+window_size-1]['end']
                        logging.info(f"✓ Partial match found! {matches}/{len(title_words)} words matched")
                        logging.info(f"Match ratio: {match_ratio:.3f}, End time: {end_time:.2f}s")
                        logging.info(f"Matched words: {window_words}")
                        return end_time
            # Strategy 4: Look for any substantial word overlap in first 20 seconds
            logging.info("Trying early-segment word overlap...")
            for i, word_info in enumerate(all_words):
                if word_info['start'] > 20:  # Don't look beyond 20 seconds
                    break
                
                if word_info['word'] in title_words:
                    # Found a title word, look for more nearby
                    consecutive_matches = 0
                    j = i
                    while j < len(all_words) and all_words[j]['start'] < 20:
                        if all_words[j]['word'] in title_words:
                            consecutive_matches += 1
                        else:
                            break
                        j += 1
                    if consecutive_matches >= 2:  # At least 2 consecutive title words
                        end_time = all_words[j-1]['end']
                        logging.info(f"✓ Early word overlap found! {consecutive_matches} consecutive matches")
                        logging.info(f"End time: {end_time:.2f}s")
                        return end_time
            logging.warning("No reliable title match found in transcription")
            logging.info("Consider checking if the title text actually appears in the audio")
            return None
        except Exception as e:
            logging.error(f"Error finding title end time: {e}")
            logging.error(traceback.format_exc())
            return None
    
    def create_overlay_video(self, match_info):
        """Create video with PNG overlay that disappears when title ends"""
        try:
            png_path = os.path.join(self.fake_reddit_folder, match_info['png_file'])
            mp4_path = os.path.join(self.captioned_folder, match_info['mp4_file'])
            
            logging.info(f"Creating overlay video for: {match_info['mp4_file']}")
            logging.info(f"PNG path: {png_path}")
            logging.info(f"MP4 path: {mp4_path}")
            
            # Check if files exist
            if not os.path.exists(png_path):
                logging.error(f"PNG file not found: {png_path}")
                return None
            
            if not os.path.exists(mp4_path):
                logging.error(f"MP4 file not found: {mp4_path}")
                return None
            
            # Get title from database
            # Get title from database
            title = self.get_title_from_db(
                match_info['vid_id'],
                match_info['reddit_forum'],
                match_info['forum_index'],
                is_long=match_info['is_long']  # <-- Add this line
            )
            
            if not title:
                logging.error(f"Could not get title for {match_info['mp4_file']}")
                return None
            
            logging.info(f"Title: {title}")
            
            # Transcribe video to find when title ends
            transcription = self.transcribe_video(mp4_path)
            title_end_time = self.find_title_end_time(transcription, title)
            
            if title_end_time is None:
                logging.warning(f"Could not find title end time for {match_info['mp4_file']}, using default 10 seconds")
                title_end_time = 10.0
            
            # Load video first to get dimensions
            logging.info("Loading video clip...")
            video = VideoFileClip(mp4_path)
            video_width, video_height = video.size
            logging.info(f"Video dimensions: {video_width}x{video_height}")
            
            # Calculate fade start time (fade should start before title ends)
            fade_start_time = max(0, title_end_time - FADE_DURATION)
            total_overlay_duration = title_end_time
            
            # Get original image dimensions using PIL
            logging.info("Loading PNG image...")
            pil_image = PIL.Image.open(png_path)
            original_width, original_height = pil_image.size
            pil_image.close()
            logging.info(f"Original image dimensions: {original_width}x{original_height}")
            
            # Calculate target size with configurable padding
            target_width = int(video_width * OVERLAY_WIDTH_PERCENT)
            target_height = int(video_height * OVERLAY_HEIGHT_PERCENT)
            
            # Calculate scale factors for both dimensions
            width_scale = target_width / original_width
            height_scale = target_height / original_height
            
            # Use the smaller scale factor to maintain aspect ratio and fit within bounds
            scale_factor = min(width_scale, height_scale)
            
            # Calculate final dimensions
            final_width = int(original_width * scale_factor)
            final_height = int(original_height * scale_factor)
            
            # Create image clip with calculated dimensions and duration
            logging.info("Creating image clip...")
            image = ImageClip(png_path, duration=total_overlay_duration)
            image = image.resize((final_width, final_height))
            
            # Position image using configurable position
            image = image.set_position(OVERLAY_POSITION)
            
            # Add fade to transparent effect
            if FADE_DURATION > 0 and total_overlay_duration > FADE_DURATION:
                image = image.crossfadeout(FADE_DURATION)
            
            logging.info(f"Video size: {video_width}x{video_height}, Overlay size: {final_width}x{final_height}")
            logging.info(f"Overlay duration: {total_overlay_duration:.2f}s, Fade starts at: {fade_start_time:.2f}s")
            
            # Create composite video
            logging.info("Creating composite video...")
            final_video = CompositeVideoClip([video, image])
            
            # Generate output filename
            output_filename = f"final_{match_info['mp4_file']}"
            output_path = os.path.join(self.output_folder_p2v, output_filename)
            
            # Check if output already exists
            if os.path.exists(output_path):
                logging.info(f"Output file already exists, skipping: {output_path}")
                # Clean up
                video.close()
                image.close()
                final_video.close()
                return output_path
            
            # Write video
            logging.info(f"Writing output video: {output_path}")
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,  # Reduce moviepy output
                logger=None     # Disable moviepy logger
            )
            
            # Clean up
            video.close()
            image.close()
            final_video.close()
            
            logging.info(f"Successfully created: {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error creating overlay video for {match_info['mp4_file']}: {e}")
            logging.error(traceback.format_exc())
            return None
    
    def process_all_videos(self):
        """Process all matching video files"""
        try:
            matches = self.find_matching_files()
            
            if not matches:
                logging.error("No matching files found!")
                return
            
            logging.info(f"Found {len(matches)} matching file pairs")
            
            successful_count = 0
            failed_count = 0
            
            # In the process_all_videos method...
            for i, match in enumerate(matches, 1):
                logging.info(f"\n{'='*50}")
                logging.info(f"Processing {i}/{len(matches)}: {match['mp4_file']}")
                logging.info(f"Match details: {match}")
                logging.info(f"{'='*50}")
                try:
                    output_path = self.create_overlay_video(match)
                    if output_path:
                        logging.info(f"✓ Successfully processed: {match['mp4_file']}")
                        successful_count += 1
                    else:
                        logging.error(f"✗ Failed to process: {match['mp4_file']}")
                        failed_count += 1
                except Exception as e:
                    logging.error(f"✗ Error processing {match['mp4_file']}: {e}")
                    logging.error(traceback.format_exc())
                    failed_count += 1
                finally:
                    # =========== CACHE AND MEMORY CLEANUP ===========
                    logging.info("--- Starting post-video cleanup ---")
                    # 1. Clear PyTorch GPU cache if CUDA is available
                    if self.whisper_model is not None and torch.cuda.is_available():
                        logging.info("Clearing PyTorch GPU cache...")
                        torch.cuda.empty_cache()
                    # 2. Trigger Python's Garbage Collector to free up RAM
                    logging.info("Triggering garbage collection...")
                    gc.collect()
                    logging.info("--- Cleanup finished ---")
        # ================================================
            
            logging.info(f"\n{'='*50}")
            logging.info(f"PROCESSING COMPLETE")
            logging.info(f"Successful: {successful_count}")
            logging.info(f"Failed: {failed_count}")
            logging.info(f"Total: {len(matches)}")
            logging.info(f"{'='*50}")
            
        except Exception as e:
            logging.error(f"Critical error in process_all_videos: {e}")
            logging.error(traceback.format_exc())

class VideoClipProcessorCC:
    def __init__(self, vid_id, font_path, subscribe_path):
        self.vid_id = vid_id
        self.font_path = font_path
        self.subscribe_animation_path = subscribe_path
        self.folder_name = f"{vid_id}_Ready_To_Post"
        self.output_folder = os.path.join(self.folder_name, "Long_Vid_Shorts")
        
        # --- FFmpeg settings ---
        self.ffmpeg_cmd = self.find_executable('ffmpeg')
        self.ffprobe_cmd = self.find_executable('ffprobe')
        self.initial_encoder = self.select_encoder()
        self.quality_setting = '23' # Quality value for CRF (CPU) or CQ (GPU). 18-28 is a good range.
        self.preset = 'fast' # Encoding speed.
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    def find_executable(self, name):
        """Finds FFmpeg/FFprobe executable in common locations."""
        possible_paths = [
            name, f'{name}.exe',
            fr'C:\ffmpeg\bin\{name}.exe',
            fr'C:\Program Files\ffmpeg\bin\{name}.exe'
        ]
        for path in possible_paths:
            try:
                subprocess.run([path, '-version'], capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                print(f"Found {name} at: {path}")
                return path
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        return None
    def select_encoder(self):
        """Selects the best available FFmpeg encoder (GPU hardware acceleration is fastest)."""
        if not self.ffmpeg_cmd: return 'libx264'
        print("\nChecking for hardware accelerated encoders...")
        # Check for NVIDIA NVENC
        try:
            cmd = [self.ffmpeg_cmd, '-h', 'encoder=h264_nvenc']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if "NVIDIA NVENC" in result.stdout:
                print("INFO: NVIDIA NVENC encoder found. Will attempt to use for hardware acceleration.")
                return 'h264_nvenc'
        except Exception: pass
        # Check for Intel Quick Sync Video (QSV)
        try:
            cmd = [self.ffmpeg_cmd, '-h', 'encoder=h264_qsv']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if "Intel QSV" in result.stdout:
                print("INFO: Intel QSV encoder found. Will attempt to use for hardware acceleration.")
                return 'h264_qsv'
        except Exception: pass
        print("INFO: No hardware encoders found. Using CPU encoder (libx264).")
        return 'libx264'
        
    def get_video_info(self, video_path):
        """Gets video duration and checks if it's longer than 60 seconds."""
        if not self.ffprobe_cmd:
            print("ERROR: ffprobe not found. Cannot get video information.")
            return None, False
        try:
            cmd = [self.ffprobe_cmd, '-v', 'quiet', '-print_format', 'json', '-show_format', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            return duration, duration > 60
        except Exception as e:
            print(f"Error getting info for {os.path.basename(video_path)}: {e}")
            return None, False
    def _build_ffmpeg_command(self, video_path, output_path, encoder):
        """Helper function to construct the full FFmpeg command."""

        cmd = [self.ffmpeg_cmd, '-y', '-i', video_path]

        # Add subscribe animation input if it exists
        if os.path.exists(self.subscribe_animation_path):
            cmd.extend(['-i', self.subscribe_animation_path])

        # Define video and audio filters
        video_filters = [
            "[0:v]trim=duration=55,setpts=PTS-STARTPTS[trimmed_v]",
            "[trimmed_v]trim=end=50,setpts=PTS-STARTPTS[main_v]",
            "[trimmed_v]trim=start=50,setpts=PTS-STARTPTS,gblur=sigma=20[blurred_v]",
        ]

        audio_filters = [
            "[0:a]atrim=duration=55,asetpts=PTS-STARTPTS[final_a]"
        ]

        # Font settings
        font_option = ""
        if self.font_path and os.path.exists(self.font_path):
            sanitized_font_path = self.font_path.replace(':', '\\:')
            font_option = f"fontfile='{sanitized_font_path}':"

        # Text overlay filter
        text_overlay_filter = (
            f"[blurred_v]drawtext=text='Full Video on Channel':{font_option}"
            "fontsize=60:fontcolor=white:bordercolor=black:borderw=4:"
            "x=(w-text_w)/2:y=(h-text_h)/2-40"
        )

        # Combine filters depending on whether a subscribe animation exists
        if os.path.exists(self.subscribe_animation_path):
            subscribe_filters = [
                "[1:v]chromakey=green:0.1:0.2[keyed_sub]",
                "[keyed_sub]scale=900:-1[sub_scaled]"
            ]

            combined_ending_filters = [
                f"{text_overlay_filter}[text_overlay]",
                "[text_overlay][sub_scaled]overlay="
                "x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2+60[ending_v]"
            ]

            filter_parts = (
                video_filters +
                audio_filters +
                subscribe_filters +
                combined_ending_filters
            )
        else:
            combined_ending_filters = [f"{text_overlay_filter}[ending_v]"]
            filter_parts = video_filters + audio_filters + combined_ending_filters

        # Concatenate main and ending videos
        filter_parts.append("[main_v][ending_v]concat=n=2:v=1:a=0[final_v]")

        # Add filters and output mapping
        cmd.extend(['-filter_complex', ';'.join(filter_parts)])
        cmd.extend(['-map', '[final_v]', '-map', '[final_a]'])

        # Encoder configuration
        if encoder == 'libx264':
            cmd.extend(['-c:v', encoder, '-crf', self.quality_setting])
        else:
            cmd.extend(['-c:v', encoder, '-cq', self.quality_setting])

        # Audio, preset, and output settings
        cmd.extend([
            '-c:a', 'aac',
            '-preset', self.preset,
            '-pix_fmt', 'yuv420p',
            '-t', '55',
            output_path
        ])

        return cmd

    def process_video_task(self, video_info):
        """Processes a single video, with automatic fallback from GPU to CPU on error."""
        video_path, _ = video_info
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(self.output_folder, f"{base_name}_55sec_clip.mp4")
        cmd = self._build_ffmpeg_command(video_path, output_path, self.initial_encoder)
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            return True 
        except subprocess.CalledProcessError as e:
            # *** THIS IS THE FIX: Added 'incompatible client key' to the check ***
            is_gpu_driver_error = (
                'nvenc API version' in e.stderr or 
                'Driver does not support' in e.stderr or
                'incompatible client key' in e.stderr
            )
            
            if self.initial_encoder != 'libx264' and is_gpu_driver_error:
                print(f"\nWARNING: GPU processing failed for {base_name} due to an incompatible driver.")
                print("Automatically retrying with CPU encoder. For best performance, please update your GPU drivers.")
                
                fallback_cmd = self._build_ffmpeg_command(video_path, output_path, 'libx264')
                try:
                    subprocess.run(fallback_cmd, capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    return True 
                except subprocess.CalledProcessError as e2:
                    print(f"\n--- CPU Fallback FFmpeg Error for {base_name} ---")
                    print(f"Stderr: {e2.stderr}")
                    print("--- End of Error ---")
                    return False
            else:
                # The error was something else
                print(f"\n--- FFmpeg Error for {base_name} ---")
                print(f"Stderr: {e.stderr}")
                print("--- End of Error ---")
                return False
    def process_all_videos(self):
        """Finds all qualifying videos and processes them in parallel."""
        if not self.ffmpeg_cmd or not self.ffprobe_cmd:
            print("\nERROR: FFmpeg/FFprobe not found!")
            return
        all_mp4_files = glob.glob(os.path.join(self.folder_name, "*.mp4"))
        if not all_mp4_files:
            print("No MP4 files found to process.")
            return
        videos_to_process = []
        print("\nChecking video durations...")
        for video_path in tqdm(all_mp4_files, desc="Scanning files"):
            duration, is_long_enough = self.get_video_info(video_path)
            if is_long_enough:
                videos_to_process.append((video_path, duration))
        if not videos_to_process:
            print("\nNo videos longer than 60 seconds were found.")
            return
        
        print(f"\nFound {len(videos_to_process)} video(s) to process.")
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(self.process_video_task, videos_to_process), total=len(videos_to_process), desc="Processing videos"))
        
        processed_count = sum(1 for r in results if r)
        print(f"\nProcessing complete! Successfully processed {processed_count} out of {len(videos_to_process)} videos.")
        print(f"Output files saved in: {self.output_folder}")

class VideoClipProcessorCTA:
    def __init__(self, vid_id, subscribe_path):
        self.vid_id = vid_id
        self.subscribe_animation_path = subscribe_path
        self.folder_name = f"{vid_id}_Ready_To_Post"
        self.output_folder = os.path.join(self.folder_name, "Long_Vid_With_CTA")
        os.makedirs(self.output_folder, exist_ok=True)
        self.ffmpeg_cmd = self.find_executable('ffmpeg')
        self.ffprobe_cmd = self.find_executable('ffprobe')
        print(f"DEBUG: ffmpeg_cmd={self.ffmpeg_cmd}, ffprobe_cmd={self.ffprobe_cmd}")
        self.encoder = 'libx264'
        self.quality_setting = '23'
        self.preset = 'fast'
    def find_executable(self, name):
        candidates = [name, f"{name}.exe", rf"C:\ffmpeg\bin\{name}.exe"]
        for p in candidates:
            try:
                subprocess.run([p, '-version'], capture_output=True, check=True)
                return p
            except Exception:
                continue
        return None
    def get_video_info(self, path):
        if not self.ffprobe_cmd:
            return 0.0, 30.0
        try:
            proc = subprocess.run([
                self.ffprobe_cmd,
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams', path
            ], capture_output=True, text=True, check=True)
            data = json.loads(proc.stdout)
            for s in data.get('streams', []):
                if s.get('codec_type') == 'video':
                    dur = float(s.get('duration') or 0.0)
                    num, den = (s.get('r_frame_rate') or '30/1').split('/')
                    fps = float(num) / float(den)
                    return dur, fps
        except Exception as e:
            print(f"WARNING: ffprobe failed on {path}: {e}")
        return 0.0, 30.0
    def _build_ffmpeg_command(self, in_path, out_path, sf, ef, fps):
        st = sf / fps
        et = (ef + 1) / fps
        btn_w = 600
        btn_h = int(btn_w * 1920 / 1080)
        x_pos = (1920 - btn_w) // 2
        y_pos = ((1080 - btn_h) // 2) + 200  # moved down 200px
        filter_complex = (
            f"[0:v]drawbox=x={x_pos}:y={y_pos}:w={btn_w}:h={btn_h}:"
            f"color=black@0:t=fill:enable='between(t,{st},{et})'[bg];"
            f"[1:v]format=yuva420p,chromakey=0x00FF00:0.4:0.2,scale={btn_w}:-1[btn];"
            f"[bg][btn]overlay=enable='between(t,{st},{et})':"
            f"x={x_pos}:y={y_pos}:shortest=1[finalv]"
        )
        cmd = [
            self.ffmpeg_cmd,
            '-i', in_path,
            '-stream_loop', '-1',
            '-i', self.subscribe_animation_path,
            '-filter_complex', filter_complex,
            '-map', '[finalv]',
            '-map', '0:a',
            '-c:v', self.encoder,
            '-crf', self.quality_setting,
            '-preset', self.preset,
            '-c:a', 'copy',
            out_path
        ]
        print(f"DEBUG: Running command: {' '.join(cmd)}")
        return cmd
    def process_video_task(self, video_path):
        name = os.path.splitext(os.path.basename(video_path))[0]
        out_file = os.path.join(self.output_folder, f"{name}_clip.mp4")
        dur, fps = self.get_video_info(video_path)
        print(f"INFO: {name} duration={dur:.2f}s fps={fps:.2f}")
        if dur > 40:
            start_time = 30
        elif dur >= 15:
            start_time = 15
        else:
            return f"Skipped '{name}': Too short"
        sf = int(start_time * fps)
        ef = sf + int(5 * fps) - 1
        cmd = self._build_ffmpeg_command(video_path, out_file, sf, ef, fps)
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return "Success"
        except subprocess.CalledProcessError as e:
            if LOG_ERRORS_CTA:
                err_log = os.path.join(self.output_folder, f"{name}_error.log")
                with open(err_log, 'w') as f:
                    f.write(f"CMD: {' '.join(cmd)}\nSTDERR:\n{e.stderr}")
                return f"Failed '{name}': See {err_log}"
            return f"Failed '{name}'"
    def process_all_videos(self):
        print("Starting video processing...")
        if not self.ffmpeg_cmd or not self.ffprobe_cmd:
            print("ERROR: ffmpeg or ffprobe not found!")
            return
        files = glob.glob(os.path.join(self.folder_name, "*.mp4"))
        print(f"Found {len(files)} files to process.")
        if not files:
            return
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_CTA) as executor: #Tyler update this line
            for res in tqdm(executor.map(self.process_video_task, files), total=len(files)):
                results.append(res)
        succ = results.count("Success")
        skip = sum(1 for r in results if r.startswith("Skipped"))
        fail = len(results) - succ - skip
        print(f"Done: {succ} success, {skip} skipped, {fail} failed.")

def extract_first_frame(video_path, output_image_path):
        """
        Extracts the first frame of a video using FFmpeg and saves it as a JPG.
        Returns the path to the image if successful, otherwise None.
        """
        try:
            # Command to extract the first frame (-vframes 1) and overwrite if it exists (-y)
            command = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-vframes', '1',
                '-q:v', '2', # High quality JPEG (1 is highest, 31 is lowest)
                output_image_path
            ]

            # This flag prevents a console window from popping up on Windows
            creationflags = 0
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NO_WINDOW

            subprocess.run(command, check=True, capture_output=True, text=True, creationflags=creationflags)

            if os.path.exists(output_image_path):
                print(f"📸 Successfully extracted thumbnail to: {output_image_path}")
                return output_image_path
            return None
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error while extracting frame: {e.stderr}")
            return None
        except FileNotFoundError:
            print("❌ FFmpeg is not installed or not in the system's PATH. Cannot extract thumbnail.")
            return None
        except Exception as e:
            print(f"❌ An unexpected error occurred during frame extraction: {e}")
            return None

class YouTubeUploader:
    """Handles YouTube authentication and video uploading."""
    CLIENT_SECRETS_FILE = "client_secrets.json"
    API_NAME = 'youtube'
    API_VERSION = 'v3'
    SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

    def __init__(self, nickname, oauth_client_id, client_secret):
        if not oauth_client_id or not client_secret:
            raise ValueError("OAuth Client ID and Client Secret are required for uploading.")

        self.nickname = nickname
        self.oauth_client_id = oauth_client_id
        self.client_secret = client_secret
        self.token_file = f"token_{nickname}.json" # Unique token for each account
        self.service = self._get_authenticated_service()

    def _get_authenticated_service(self):
        """Authenticates with the YouTube API and returns the service object."""
        creds = None
        
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    print(f"Refreshing token for {self.nickname}...")
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Token refresh failed: {e}. Re-authentication is required.")
                    creds = None
            else:
                client_config = {
                    "installed": {
                        "client_id": self.oauth_client_id,
                        "client_secret": self.client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": ["http://localhost:8080/"]
                    }
                }
                flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_config(
                    client_config, self.SCOPES)
                creds = flow.run_local_server(port=8090)
            
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
                print(f"Token for {self.nickname} saved to {self.token_file}")

        return googleapiclient.discovery.build(
            self.API_NAME, self.API_VERSION, credentials=creds)

    def upload_video(self, video_path, title, description, thumbnail_path, status):
        """Uploads a video to YouTube with its metadata and thumbnail."""
        if not self.service:
            print("YouTube service is not authenticated. Cannot upload.")
            return

        try:
            upload_type = "scheduled" if 'publishAt' in status else "immediate"
            print(f"🚀 Starting {upload_type} upload for: {title}")

            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'tags': ['creepypasta', 'horrorstories', 'scarystories', 'nosleep', 'reddit', 'redditstories'],
                    'categoryId': '24' # Entertainment category
                },
                'status': status
            }

            media = googleapiclient.http.MediaFileUpload(video_path, chunksize=-1, resumable=True)

            request = self.service.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = None
            while response is None:
                upload_status, response = request.next_chunk()
                if upload_status:
                    print(f"Uploaded {int(upload_status.progress() * 100)}%")
            
            video_id = response.get('id')
            print(f"✅ Video uploaded successfully! Video ID: {video_id}")

            if thumbnail_path and os.path.exists(thumbnail_path):
                print(f"🖼️ Setting thumbnail from: {thumbnail_path}")
                self.service.thumbnails().set(
                    videoId=video_id,
                    media_body=googleapiclient.http.MediaFileUpload(thumbnail_path)
                ).execute()
                print("✅ Thumbnail set successfully!")
            else:
                print("⚠️ Thumbnail not found or not provided. Skipping thumbnail upload.")

        except googleapiclient.errors.HttpError as e:
            print(f"An HTTP error occurred: {e}\nDetails: {e.content}")
        except Exception as e:
            print(f"An unexpected error occurred during upload: {e}")

def run_reddit(VID_ID, credentials=None):
    """
    Orchestrates the entire Reddit video creation pipeline from start to finish.
    """
    # --- Configuration ---
    AUDIO_PROMPT = "util/0602-vocals.mp3"
    FONT_PATH = "C:/Windows/Fonts/Cooper Black.ttf"
    SUBSCRIBE_ANIMATION_PATH = "Subscribe_transparent.mov"

    print(f"--- STARTING REDDIT VIDEO PIPELINE FOR VID_ID: {VID_ID} ---")

    # === Step 1-8 (Phases 1-8 are unchanged) ===
    print("\n[PHASE 1/9] Scraping Reddit for stories...")
    scraper_config = RedditConfig(vid_id_from_gui=VID_ID)
    scraper = RedditScraper(scraper_config)
    scraper.run()

    print("\n[PHASE 2/9] Cleaning the story database...")
    db_cleaner_ui = UserInterfaceStoryFilter(VID_ID)
    db_cleaner_ui.run()

    print("\n[PHASE 3/9] Generating Text-to-Speech audio...")
    tts_ui = UserInterfaceTTS(VID_ID, AUDIO_PROMPT)
    tts_ui.run()

    print("\n[PHASE 4/9] Cleaning and combining audio files...")
    mp.freeze_support()
    audio_base_folder = f"./{VID_ID}_Reddit_Audio"
    audio_cleaner = AudioCleaner(audio_base_folder)
    audio_cleaner.process_all_audio_parallel()
    audio_combiner = AudioCombiner(VID_ID)
    audio_combiner.combine_audio_files()

    print("\n[PHASE 5/9] Creating Reddit title card images...")
    post_generator = FakeRedditPostGenerator(vid_id=VID_ID)
    post_generator.run()

    print("\n[PHASE 6/9] Assembling background video and adding captions...")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Multiprocessing context already set.")
    initial_cleaner = VideoCreator(vid_id=VID_ID)
    #
    initial_cleaner.cleanup_instance()
    video_creator = VideoCreator(vid_id=VID_ID)
    video_creator.create_videos(use_parallel=True, max_workers=None)
    captioner = VideoCaptioner(vid_id=VID_ID, model_size="base")
    captioner.process_all_videos()

    print("\n[PHASE 7/9] Overlaying title cards onto videos...")
    initial_cleaner.cleanup_stale_temp_files()
    overlay_processor = RedditVideoProcessor(
        fake_reddit_folder=f"{VID_ID}_fake_reddit_posts",
        captioned_folder=f"{VID_ID}_Captioned",
        output_folder_p2v=f"{VID_ID}_Ready_To_Post"
    )
    overlay_processor.process_all_videos()

    print("\n[PHASE 8/9] Generating final short-form clips...")
    clips_processor = VideoClipProcessorCC(
        vid_id=VID_ID, 
        font_path=FONT_PATH, 
        subscribe_path=SUBSCRIBE_ANIMATION_PATH
    )
    clips_processor.process_all_videos()
    cta_processor = VideoClipProcessorCTA(VID_ID, SUBSCRIBE_ANIMATION_PATH)
    cta_processor.process_all_videos()
    
    # === Step 8.5: Upload Videos to YouTube with Dynamic Metadata ===
    if credentials:
        print("\n[PHASE 8.5/9] Uploading videos to YouTube...")
        try:
            uploader = YouTubeUploader(
                nickname=credentials['nickname'],
                oauth_client_id=credentials['oauth_client_id'],
                client_secret=credentials['client_secret']
            )

            # Define folder paths
            ready_to_post_folder = f"{VID_ID}_Ready_To_Post"
            long_videos_folder = os.path.join(ready_to_post_folder, "Long_Vid_With_CTA")
            shorts_folder = os.path.join(ready_to_post_folder, "Long_Vid_Shorts")
            
            temp_thumbnail_folder = os.path.join(ready_to_post_folder, "temp_thumbnails")
            os.makedirs(temp_thumbnail_folder, exist_ok=True)
            
            db_path = f'{VID_ID}_reddit_nosleep.db'
            
            next_publish_time = None
            video_upload_count = 0

            if os.path.exists(long_videos_folder):
                video_files_to_upload = sorted([f for f in os.listdir(long_videos_folder) if f.endswith('.mp4')])

                for long_video_filename in video_files_to_upload:
                    video_upload_count += 1
                    long_video_path = os.path.join(long_videos_folder, long_video_filename)
                    
                    # --- DYNAMIC METADATA & THUMBNAIL LOGIC ---
                    base_name = os.path.splitext(long_video_filename)[0].replace('_clip', '')
                    
                    # Parse filename to get story identifiers
                    parts = base_name.replace(f"final_final_captioned_{VID_ID}_", "").split('_')
                    story_id = parts[-1]
                    is_long = "long" in parts
                    subreddit = "_".join(parts[:-2] if is_long else parts[:-1])
                    table_name = f"{subreddit}_long" if is_long else subreddit
                    
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT title, body FROM {table_name} WHERE id=?", (story_id,))
                    row = cursor.fetchone()
                    conn.close()

                    if row:
                        # 1. TITLE from Database
                        title, body = row
                        
                        # 2. DESCRIPTION from first 3 sentences of the body + hashtags
                        sentences = re.split(r'(?<=[.!?])\s+', body)
                        description_snippet = ' '.join(sentences[:3])
                        hashtags = "#creepypasta #horrorstories #scarystories #nosleep"
                        description = f"{description_snippet}\n\n{hashtags}"

                        # 3. THUMBNAIL from first frame of the video
                        temp_thumbnail_path = os.path.join(temp_thumbnail_folder, f"{base_name}_thumb.jpg")
                        thumbnail_path = extract_first_frame(long_video_path, temp_thumbnail_path)
                        
                        # --- DYNAMIC STATUS DETERMINATION (Scheduling) ---
                        status_for_long_video = {}
                        if video_upload_count == 1:
                            print(f"\n--- Uploading Long Video (Immediate): {title} ---")
                            status_for_long_video = {'privacyStatus': 'public', 'selfDeclaredMadeForKids': False}
                            next_publish_time = datetime.now(timezone.utc) + timedelta(hours=2)
                        else:
                            print(f"\n--- Uploading Long Video (Scheduled for {next_publish_time.strftime('%Y-%m-%d %H:%M:%S UTC')}): {title} ---")
                            status_for_long_video = {
                                'privacyStatus': 'private',
                                'publishAt': next_publish_time.isoformat(),
                                'selfDeclaredMadeForKids': False
                            }
                            next_publish_time += timedelta(hours=2)
                        
                        uploader.upload_video(long_video_path, title, description, thumbnail_path, status_for_long_video)

                        # --- Corresponding Short Video ---
                        short_base_name = os.path.splitext(long_video_filename)[0].replace('_clip', '')
                        short_video_filename = f"{short_base_name}_55sec_clip.mp4"
                        short_video_path = os.path.join(shorts_folder, short_video_filename)

                        if os.path.exists(short_video_path):
                            shorts_title = f"{title} #shorts"
                            shorts_description = f"Full story is on our channel!\n\n{title}\n\n#shorts #creepypasta #horrorstories #scarystories #nosleep"
                            status_for_short = {'privacyStatus': 'public', 'selfDeclaredMadeForKids': False}
                            print(f"--- Uploading Short Video (Immediate): {shorts_title} ---")
                            uploader.upload_video(short_video_path, shorts_title, shorts_description, thumbnail_path, status_for_short)
                        else:
                            print(f"⚠️ Could not find corresponding short video: {short_video_path}")
                    else:
                        print(f"❌ Could not find metadata in DB for {long_video_filename}")
            else:
                print(f"Final videos folder not found: {long_videos_folder}")

        except Exception as e:
            print(f"An error occurred during the upload phase: {e}")
            traceback.print_exc()
    else:
        print("\n[PHASE 8.5/9] SKIPPED: No credentials provided for YouTube upload.")

    # === Step 9 (Cleanup is unchanged) ===
    print("\n[PHASE 9/9] Cleaning up intermediate files and folders...")
    base_dir = "."
    prefix_to_delete = f"{VID_ID}_"
    folder_to_keep = f"{VID_ID}_Ready_To_Post"
    
    # Clean up intermediate project folders
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith(prefix_to_delete) and item != folder_to_keep:
            try:
                print(f"Deleting intermediate folder: {item_path}")
                shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error deleting folder {item_path}: {e}")
                
    # Clean up intermediate videos inside the final folder
    ready_to_post_dir = folder_to_keep
    if os.path.isdir(ready_to_post_dir):
        # Clean up temp thumbnails
        temp_thumb_path = os.path.join(ready_to_post_dir, "temp_thumbnails")
        if os.path.exists(temp_thumb_path):
            shutil.rmtree(temp_thumb_path)

        for item in os.listdir(ready_to_post_dir):
            item_path = os.path.join(ready_to_post_dir, item)
            if os.path.isfile(item_path) and "final_final_captioned" in item:
                print(f"Deleting intermediate video file: {item_path}")
                os.remove(item_path)
                
    # Clean up database file
    db_file = f"{VID_ID}_reddit_nosleep.db"
    if os.path.isfile(db_file):
        print(f"Deleting database file: {db_file}")
        os.remove(db_file)

    print("\n--- REDDIT VIDEO PIPELINE FINISHED ---")



if __name__ == "__main__":
    video_id_for_testing = "V20"
    print(f"--- EXECUTING STANDALONE SCRIPT FOR VID_ID: {video_id_for_testing} ---")

    # Mock credentials for local testing (won't actually upload)
    mock_credentials = {
        "nickname": "test_account",
        "oauth_client_id": "YOUR_TEST_CLIENT_ID",
        "client_secret": "YOUR_TEST_CLIENT_SECRET"
    }

    try:
        # Pass the mock credentials to the function
        run_reddit(VID_ID=video_id_for_testing, credentials=mock_credentials)
        print(f"--- SCRIPT FINISHED SUCCESSFULLY FOR VID_ID: {video_id_for_testing} ---")
    except Exception as e:
        print(f"--- SCRIPT FAILED FOR VID_ID: {video_id_for_testing} ---")
        print(f"An error occurred: {e}")