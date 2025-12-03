"""
Optimized Long-Form Script Generator
Generates chapter-based scripts for long-form content using OpenAI API and stores them in SQLite database
"""


import openai
import sqlite3
import os
import json
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Set, Any
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import subprocess
import time
import re
import torch
import torchaudio
import threading
from queue import Queue
import gc
import numpy as np
from functools import lru_cache
from chatterbox.tts import ChatterboxTTS
import soundfile as sf
import librosa
from pathlib import Path
import multiprocessing as mp
import logging
from dataclasses import dataclass
from collections import defaultdict
import tempfile
import shutil
from mutagen.wave import WAVE
import glob
from contextlib import contextmanager, asynccontextmanager
import aiofiles
import random
import cv2
from PIL import Image
import whisper
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from tqdm import tqdm
import threading
import natsort
from enum import Enum

def run_storyform(VID_ID, TOPIC, CHAPTER_NUM):
    # =============================================================================
    # CONFIGURATION - Change these values as needed
    # =============================================================================
    # VID_ID = "V6"  # Change this to your desired video ID
    # TOPIC = "Modern Day Biblical Prophesies and What They Mean for Our Future"  # Change this to your topic
    # Put your OpenAI API key here, or leave empty to use environment variable
    # CHAPTER_NUM = 6  # Number of chapters to generate
    # =============================================================================

    # =============================================================================
    # CONFIGURATION - Change these values as needed
    # =============================================================================
    API_KEY = ""  # Put your OpenAI API key here, or leave empty to use environment variable
    # Script chunking configuration
    WORDS_PER_20_SECONDS = 55  # Average speaking rate for 20 seconds
    CHARS_PER_20_SECONDS = 350  # Approximate characters for 20 seconds
    # =============================================================================

    AUDIO_PROMPT = "0602-vocals.mp3"

    class OptimizedScriptGenerator:
        def __init__(self, api_key: str, vid_id: str = "DEFAULT"):
            """
            Initialize the Optimized Script Generator
            
            Args:
                api_key (str): OpenAI API key
                vid_id (str): Video ID for database naming
            """
            self.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key)
            self.vid_id = vid_id
            self.db_name = f"{vid_id}_longform.db"
            self.setup_database()
        
        def setup_database(self):
            """Create the database and table if they don't exist with optimized settings"""
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Enable WAL mode for better concurrent access
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS video_ideas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_topic TEXT NOT NULL,
                    chapter_number INTEGER NOT NULL,
                    chapter_title TEXT NOT NULL,
                    chapter_content TEXT NOT NULL,
                    key_points TEXT,
                    estimated_duration TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_chapter_number ON video_ideas(chapter_number)
            ''')
            
            conn.commit()
            conn.close()
            print(f"Database '{self.db_name}' initialized successfully!")
        
        def generate_video_ideas_fast(self, topic: str) -> List[Dict]:
            """
            Generate script chapters using faster model and optimized prompt
            
            Args:
                topic (str): The main topic for the long-form content
                
            Returns:
                List[Dict]: List of script chapters with titles and content
            """
            # Streamlined prompt for faster processing
            prompt = f"""Generate a {CHAPTER_NUM}-chapter script for: "{topic}"

    Return ONLY valid JSON array with this exact structure:
    [
    {{
        "chapter_number": 1,
        "chapter_title": "Title Here",
        "content_outline": "Brief outline here",
        "key_points": ["Point 1", "Point 2", "Point 3"],
        "estimated_duration": "7 minutes"
    }}
    ]

    Make chapters flow logically, 45-60 min total duration."""
            
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Faster than gpt-4
                    messages=[
                        {"role": "system", "content": "You are an expert scriptwriter. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000,  # Reduced for faster response
                    timeout=30  # Add timeout
                )
                
                content = response.choices[0].message.content.strip()
                
                # Fast JSON extraction
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1]
                
                video_ideas = json.loads(content.strip())
                
                print(f"API call completed in {time.time() - start_time:.2f} seconds")
                return video_ideas
                
            except Exception as e:
                print(f"Error generating script chapters: {e}")
                return []
        
        async def generate_chapters_async(self, topic: str, num_chapters: int) -> List[Dict]:
            """
            Generate chapters asynchronously for maximum speed
            
            Args:
                topic (str): The main topic
                num_chapters (int): Number of chapters to generate
                
            Returns:
                List[Dict]: Generated chapters
            """
            chapters_per_batch = 4  # Generate 4 chapters per API call
            batches = []
            
            for i in range(0, num_chapters, chapters_per_batch):
                batch_size = min(chapters_per_batch, num_chapters - i)
                batch_start = i + 1
                batch_end = i + batch_size
                
                batch_prompt = f"""Generate chapters {batch_start}-{batch_end} for: "{topic}"

    Return ONLY valid JSON array:
    [
    {{
        "chapter_number": {batch_start},
        "chapter_title": "Title",
        "content_outline": "Brief outline",
        "key_points": ["Point 1", "Point 2"],
        "estimated_duration": "7 minutes"
    }}
    ]"""
                batches.append(batch_prompt)
            
            # Run batches concurrently
            tasks = []
            for batch_prompt in batches:
                task = self._generate_batch_async(batch_prompt)
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            all_chapters = []
            for result in batch_results:
                if isinstance(result, list):
                    all_chapters.extend(result)
                elif isinstance(result, Exception):
                    print(f"Batch error: {result}")
            
            return sorted(all_chapters, key=lambda x: x.get('chapter_number', 0))
        
        async def _generate_batch_async(self, prompt: str) -> List[Dict]:
            """Generate a batch of chapters asynchronously"""
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500,
                    timeout=25
                )
                
                content = response.choices[0].message.content.strip()
                
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1]
                
                return json.loads(content.strip())
                
            except Exception as e:
                print(f"Batch generation error: {e}")
                return []
        
        def save_chapters_to_database_fast(self, topic: str, chapters: List[Dict]):
            """
            Save chapters to database with batch insert for speed
            
            Args:
                topic (str): Original topic
                chapters (List[Dict]): Chapters to save
            """
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Prepare data for batch insert
            data_to_insert = []
            for chapter in chapters:
                key_points = chapter.get('key_points', '')
                if isinstance(key_points, list):
                    key_points = '\n'.join([f"• {point}" for point in key_points])
                
                data_to_insert.append((
                    topic,
                    chapter.get('chapter_number', 0),
                    chapter.get('chapter_title', ''),
                    chapter.get('content_outline', ''),
                    key_points,
                    chapter.get('estimated_duration', '')
                ))
            
            # Batch insert - much faster than individual inserts
            cursor.executemany('''
                INSERT INTO video_ideas (original_topic, chapter_number, chapter_title, chapter_content, key_points, estimated_duration)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', data_to_insert)
            
            conn.commit()
            conn.close()
            print(f"Batch saved {len(chapters)} chapters to database!")
        
        def display_chapters_fast(self, chapters: List[Dict]):
            """Display chapters with minimal formatting for speed"""
            print("\n" + "="*60)
            print("GENERATED SCRIPT CHAPTERS")
            print("="*60)
            
            for chapter in chapters:
                print(f"\nCHAPTER {chapter.get('chapter_number', 'N/A')}: {chapter.get('chapter_title', 'No title')}")
                print(f"Duration: {chapter.get('estimated_duration', 'N/A')}")
                print(f"Content: {chapter.get('content_outline', 'No content')}")
                
                key_points = chapter.get('key_points', [])
                if key_points:
                    if isinstance(key_points, list):
                        print(f"Key Points: {', '.join(key_points)}")
                    else:
                        print(f"Key Points: {key_points}")
                print("-" * 40)
        
        def generate_with_parallel_processing(self, topic: str) -> List[Dict]:
            """
            Use parallel processing for CPU-bound tasks
            """
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit the API call to thread pool
                future = executor.submit(self.generate_video_ideas_fast, topic)
                
                # While waiting, we could do other prep work
                # (This is just an example - in practice you'd do actual prep work)
                
                # Get the result
                chapters = future.result()
                
                return chapters

    # Convenience function for async usage
    async def generate_script_async(api_key: str, vid_id: str, topic: str, chapter_num: int):
        """
        Async wrapper for the entire script generation process
        """
        generator = OptimizedScriptGenerator(api_key, vid_id)
        
        start_time = time.time()
        print(f"Starting async generation for {chapter_num} chapters...")
        
        chapters = await generator.generate_chapters_async(topic, chapter_num)
        
        if chapters:
            generator.display_chapters_fast(chapters)
            generator.save_chapters_to_database_fast(topic, chapters)
            
            total_time = time.time() - start_time
            print(f"\nCompleted in {total_time:.2f} seconds!")
            print(f"Generated {len(chapters)} chapters and saved to '{vid_id}_longform.db'")
        
        return chapters

    class ChapterScriptGenerator:
        def __init__(self, api_key: str, vid_id: str):
            """
            Initialize the Chapter Script Generator
            
            Args:
                api_key (str): OpenAI API key
                vid_id (str): Video ID for database naming
            """
            self.client = openai.OpenAI(api_key=api_key)
            self.vid_id = vid_id
            self.db_name = f"{vid_id}_longform.db"
            self.setup_database()
        
        def setup_database(self):
            """Add script and chunk columns to existing database if they don't exist"""
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Check existing columns
            cursor.execute("PRAGMA table_info(video_ideas)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Add script column if it doesn't exist
            if 'script' not in columns:
                cursor.execute('ALTER TABLE video_ideas ADD COLUMN script TEXT')
                print("Added 'script' column to database")
            
            # Add chunk columns (C1 through C30 for longer chapters)
            chunk_columns_to_add = []
            for i in range(1, 31):  # C1 to C30
                col_name = f"C{i}"
                if col_name not in columns:
                    chunk_columns_to_add.append(col_name)
            
            for col_name in chunk_columns_to_add:
                cursor.execute(f'ALTER TABLE video_ideas ADD COLUMN {col_name} TEXT')
                print(f"Added '{col_name}' column to database")
            
            conn.commit()
            conn.close()
        
        def split_script_into_chunks(self, script: str) -> List[str]:
            """
            Split script into approximately 20-second chunks
            
            Args:
                script (str): The full script text
                
            Returns:
                List[str]: List of script chunks
            """
            # Split into sentences using proper punctuation
            sentences = re.split(r'[.!?]+', script)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            chunks = []
            current_chunk = ""
            current_word_count = 0
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                
                # If adding this sentence would make the chunk too long, start a new chunk
                if current_word_count + sentence_words > WORDS_PER_20_SECONDS and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_word_count = sentence_words
                else:
                    if current_chunk:
                        current_chunk += ". " + sentence
                    else:
                        current_chunk = sentence
                    current_word_count += sentence_words
            
            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        
        def get_chapters_without_scripts(self) -> List[Dict]:
            """
            Get all chapters that don't have scripts yet, ordered by chapter number
            Each chapter is a separate row in the database
            
            Returns:
                List[Dict]: List of chapters without scripts
            """
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # First check if script column exists, if not we need to add it
            cursor.execute("PRAGMA table_info(video_ideas)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'script' not in columns:
                # If script column doesn't exist, return all chapters
                cursor.execute('''
                    SELECT id, chapter_number, chapter_title, chapter_content, key_points, original_topic, estimated_duration
                    FROM video_ideas 
                    ORDER BY chapter_number ASC
                ''')
            else:
                # If script column exists, filter by empty scripts
                cursor.execute('''
                    SELECT id, chapter_number, chapter_title, chapter_content, key_points, original_topic, estimated_duration
                    FROM video_ideas 
                    WHERE script IS NULL OR script = ""
                    ORDER BY chapter_number ASC
                ''')
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'id': row[0],
                    'chapter_number': row[1],
                    'chapter_title': row[2],
                    'chapter_content': row[3],
                    'key_points': row[4] if row[4] is not None else "",
                    'original_topic': row[5],
                    'estimated_duration': row[6] if row[6] is not None else "5-7 minutes"
                }
                for row in results
            ]
        
        def get_previous_chapters_summary(self, current_chapter_num: int, topic: str) -> str:
            """
            Get a summary of previous chapters to maintain continuity
            
            Args:
                current_chapter_num (int): Current chapter number
                topic (str): Main topic
                
            Returns:
                str: Summary of previous chapters
            """
            if current_chapter_num <= 1:
                return ""
            
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT chapter_number, chapter_title, chapter_content
                FROM video_ideas 
                WHERE chapter_number < ? AND original_topic = ?
                ORDER BY chapter_number ASC
            ''', (current_chapter_num, topic))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return ""
            
            summary = f"PREVIOUS CHAPTERS COVERED:\n"
            for chapter_num, title, content in results:
                summary += f"Chapter {chapter_num}: {title}\n- {content[:200]}...\n\n"
            
            return summary
        
        async def generate_chapter_script(self, chapter: Dict, previous_chapters_summary: str) -> Dict:
            """
            Generate a long-form script for a chapter (async)
            Each chapter script will be stored in its own database row
            
            Args:
                chapter (Dict): Chapter information
                previous_chapters_summary (str): Summary of previous chapters
                
            Returns:
                Dict: Contains chapter info, script, and metadata
            """
            chapter_num = chapter['chapter_number']
            title = chapter['chapter_title']
            content = chapter['chapter_content']
            key_points = chapter['key_points']
            topic = chapter['original_topic']
            duration = chapter['estimated_duration']
            
            # Build context for continuity
            context_section = ""
            if previous_chapters_summary:
                context_section = f"""
            CONTEXT FROM PREVIOUS CHAPTERS:
            {previous_chapters_summary}
            
            IMPORTANT: Reference and build upon the previous chapters naturally. Avoid repeating information already covered. Create smooth transitions that acknowledge what the audience already knows.
            """
            
            prompt = f"""
            Create a compelling, long-form YouTube script for a documentary chapter:
            
            CHAPTER INFORMATION:
            Chapter Title: "{title}"
            Content Outline: {content}
            Key Points to Cover: {key_points}
            Main Topic: {topic}
            Target Duration: {duration}
            
            {context_section}
            
            SCRIPT REQUIREMENTS:
            - Length: 1500-2500 words (adjust based on estimated duration)
            - Use storytelling vocabulary and narrative techniques
            - Build suspense and maintain engagement throughout
            - Use phrases like "But here's where it gets fascinating", "The truth becomes even more intriguing"
            - Create smooth transitions that reference previous chapters when appropriate
            - Include natural pacing for YouTube narration
            - Make it engaging and educational without being misleading
            - Use present tense for immediacy and engagement
            - Include rhetorical questions to engage the audience
            - Build layers of intrigue appropriate for this chapter's focus
            
            CONTINUITY REQUIREMENTS:
            {"- Start with a brief transition from the previous chapter if this isn't the first chapter" if chapter_num > 1 else "- Start with an engaging hook since this is the opening chapter"}
            - Reference previous chapters naturally when relevant
            - Avoid repeating information already covered
            - Build upon established knowledge and narrative
            - Create anticipation for future chapters when appropriate
            
            CRITICAL FORMATTING REQUIREMENTS:
            - DO NOT mention chapter numbers anywhere in the script
            - DO NOT include any stage directions, labels, or meta-commentary
            - DO NOT use words like "Hook", "Introduction", "Conclusion", "Transition", "Long pause", etc.
            - DO NOT include section headers or labels like "(Opening)", "(Main Content)", "(Closing)"
            - Write ONLY the actual words to be spoken by the narrator
            - Use complete sentences with proper punctuation
            - Structure the content in flowing paragraphs
            - Make sure sentences flow naturally for voice narration
            - Use periods, exclamation marks, and question marks naturally
            - The script should be pure speech content that can be read directly by a narrator
            
            Create a script that maintains the documentary's narrative flow while diving deep into this chapter's specific content. Remember: this script will be stored as a separate row in the database for this specific chapter.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",  # Using GPT-4 for better quality scripts
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a master documentary scriptwriter specializing in educational long-form content. You create clean, narrator-ready scripts without any stage directions, chapter numbers, or meta-commentary. Your scripts contain only the words to be spoken, formatted as flowing paragraphs of natural speech. You excel at creating compelling narratives that maintain continuity across chapters while building suspense and keeping viewers engaged."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                script = response.choices[0].message.content.strip()
                
                # Clean up any remaining unwanted elements
                script = self.clean_script(script)
                
                return {
                    'chapter_id': chapter['id'],
                    'chapter_number': chapter_num,
                    'chapter_title': title,
                    'script': script,
                    'success': True,
                    'error': None
                }
                
            except Exception as e:
                print(f"Error generating script for Chapter {chapter_num} '{title}': {e}")
                return {
                    'chapter_id': chapter['id'],
                    'chapter_number': chapter_num,
                    'chapter_title': title,
                    'script': "",
                    'success': False,
                    'error': str(e)
                }
        
        def clean_script(self, script: str) -> str:
            """
            Clean the script to remove any unwanted elements that might have slipped through
            
            Args:
                script (str): Raw script from AI
                
            Returns:
                str: Cleaned script
            """
            # Remove common unwanted patterns
            unwanted_patterns = [
                r'\(.*?\)',  # Remove anything in parentheses
                r'\[.*?\]',  # Remove anything in square brackets
                r'Chapter \d+:?',  # Remove chapter references
                r'CHAPTER \d+:?',  # Remove uppercase chapter references
                r'^(Hook|Introduction|Transition|Conclusion|Opening|Closing):?.*$',  # Remove section labels
                r'^[A-Z\s]+:',  # Remove all-caps labels followed by colons
                r'\*.*?\*',  # Remove text between asterisks
                r'Long pause\.?',  # Remove pause instructions
                r'Dramatic pause\.?',  # Remove pause instructions
            ]
            
            cleaned_script = script
            for pattern in unwanted_patterns:
                cleaned_script = re.sub(pattern, '', cleaned_script, flags=re.MULTILINE | re.IGNORECASE)
            
            # Clean up extra whitespace and empty lines
            lines = [line.strip() for line in cleaned_script.split('\n') if line.strip()]
            cleaned_script = '\n\n'.join(lines)
            
            return cleaned_script
        
        def save_script_and_chunks_to_database(self, chapter_id: int, script: str, chunks: List[str]):
            """
            Save the generated script and its chunks to the database
            Each chapter is stored in its own row
            
            Args:
                chapter_id (int): ID of the chapter row
                script (str): Generated script
                chunks (List[str]): Script chunks
            """
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Prepare the update query for this specific chapter row
            update_parts = ['script = ?']
            values = [script]
            
            # Add chunks to the update
            for i, chunk in enumerate(chunks, 1):
                if i <= 30:  # Only save up to C30
                    update_parts.append(f'C{i} = ?')
                    values.append(chunk)
            
            # Clear any remaining chunk columns if there are fewer chunks
            for i in range(len(chunks) + 1, 31):
                update_parts.append(f'C{i} = ?')
                values.append(None)
            
            values.append(chapter_id)  # For WHERE clause
            
            query = f'''
                UPDATE video_ideas 
                SET {', '.join(update_parts)}
                WHERE id = ?
            '''
            
            cursor.execute(query, values)
            conn.commit()
            conn.close()
            
            print(f"✓ Script and chunks saved to database row for chapter ID {chapter_id}")
        
        async def process_all_chapters(self):
            """
            Process all chapters without scripts and generate scripts sequentially for continuity
            Each chapter will be stored in its own database row
            """
            chapters = self.get_chapters_without_scripts()
            
            if not chapters:
                print("No chapters found without scripts!")
                return
            
            print(f"Found {len(chapters)} chapters that need scripts.")
            print("Each chapter will be stored as a separate row in the database.")
            print("Generating scripts sequentially to maintain continuity...\n")
            
            successful_results = []
            failed_results = []
            
            # Process chapters sequentially to maintain continuity
            for i, chapter in enumerate(chapters, 1):
                print(f"Processing Chapter {chapter['chapter_number']}/{len(chapters)}: {chapter['chapter_title']}")
                print(f"Database Row ID: {chapter['id']}")
                print("-" * 80)
                
                # Get previous chapters summary for continuity
                previous_summary = self.get_previous_chapters_summary(
                    chapter['chapter_number'], 
                    chapter['original_topic']
                )
                
                # Generate script for this chapter
                result = await self.generate_chapter_script(chapter, previous_summary)
                
                if result['success']:
                    successful_results.append(result)
                    
                    if result['script']:
                        # Split script into chunks
                        chunks = self.split_script_into_chunks(result['script'])
                        
                        # Save to database (this specific chapter's row)
                        self.save_script_and_chunks_to_database(result['chapter_id'], result['script'], chunks)
                        
                        print(f"✓ Clean script generated ({len(result['script'])} characters)")
                        print(f"✓ Split into {len(chunks)} chunks (~20 seconds each)")
                        print(f"✓ Saved to database row ID {result['chapter_id']}")
                        
                        # Display chunk preview
                        print(f"Chunk preview:")
                        for j, chunk in enumerate(chunks[:2], 1):  # Show first 2 chunks
                            preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                            print(f"  C{j}: {preview}")
                        if len(chunks) > 2:
                            print(f"  ... and {len(chunks) - 2} more chunks")
                    else:
                        print("✗ Empty script received")
                        failed_results.append(result)
                else:
                    failed_results.append(result)
                    print(f"✗ Failed: {result['error']}")
                
                print()
                
                # Small delay between chapters to avoid rate limiting
                if i < len(chapters):
                    await asyncio.sleep(1)
            
            # Report results
            print("="*80)
            print(f"Chapter script generation complete!")
            print(f"✓ Successfully processed: {len(successful_results)} chapters")
            print(f"✗ Failed: {len(failed_results)} chapters")
            print(f"Each chapter script is stored in its own database row")
            
            if failed_results:
                print("\nFailed chapter script generations:")
                for result in failed_results:
                    print(f"✗ Chapter {result['chapter_number']}: {result['chapter_title']} - {result['error']}")
            
            print(f"\nAll scripts and chunks saved to database: {self.db_name}")
        
        def get_chapters_with_scripts(self, limit: int = 10) -> List[Dict]:
            """
            Get chapters that have scripts and chunks
            Each result represents a separate database row
            
            Args:
                limit (int): Number of chapters to retrieve
                
            Returns:
                List[Dict]: List of chapters with scripts and chunks
            """
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Check if script column exists
            cursor.execute("PRAGMA table_info(video_ideas)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'script' not in columns:
                print("No script column found in database. Run the script generation first.")
                conn.close()
                return []
            
            # Get all chunk columns that exist
            chunk_columns = []
            for i in range(1, 31):
                col_name = f'C{i}'
                if col_name in columns:
                    chunk_columns.append(col_name)
            
            chunk_select = ', ' + ', '.join(chunk_columns) if chunk_columns else ''
            
            cursor.execute(f'''
                SELECT id, chapter_number, chapter_title, chapter_content, script, created_at{chunk_select}
                FROM video_ideas 
                WHERE script IS NOT NULL AND script != ""
                ORDER BY chapter_number ASC 
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            chapters = []
            for row in results:
                chapter = {
                    'id': row[0],
                    'chapter_number': row[1],
                    'chapter_title': row[2],
                    'chapter_content': row[3],
                    'script': row[4],
                    'created_at': row[5],
                    'chunks': {}
                }
                
                # Add non-empty chunks if they exist
                if chunk_columns:
                    for i, chunk_col in enumerate(chunk_columns):
                        chunk_index = 6 + i  # Start after the basic columns
                        if len(row) > chunk_index and row[chunk_index]:
                            chunk_num = chunk_col
                            chapter['chunks'][chunk_num] = row[chunk_index]
                
                chapters.append(chapter)
            
            return chapters
        
        def display_script_preview(self, limit: int = 5):
            """Display a preview of generated scripts and their chunks"""
            chapters_with_scripts = self.get_chapters_with_scripts(limit)
            
            if not chapters_with_scripts:
                print("No chapter scripts found in database.")
                return
            
            print("\n" + "="*80)
            print("GENERATED CHAPTER SCRIPTS WITH CHUNKS PREVIEW")
            print("(Each chapter is stored in its own database row)")
            print("="*80)
            
            for chapter in chapters_with_scripts:
                print(f"\nDatabase Row ID: {chapter['id']}")
                print(f"Chapter {chapter['chapter_number']}: {chapter['chapter_title']}")
                print(f"Created: {chapter['created_at']}")
                print(f"Total Script Length: {len(chapter['script'])} characters")
                print(f"Number of Chunks: {len(chapter['chunks'])}")
                
                # Show script preview (first 200 characters)
                script_preview = chapter['script'][:200] + "..." if len(chapter['script']) > 200 else chapter['script']
                print(f"\nScript Preview: {script_preview}")
                
                # Show first few chunks
                chunk_preview_count = min(3, len(chapter['chunks']))
                print(f"\nFirst {chunk_preview_count} chunks:")
                
                for i in range(1, chunk_preview_count + 1):
                    chunk_key = f'C{i}'
                    if chunk_key in chapter['chunks']:
                        chunk = chapter['chunks'][chunk_key]
                        preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
                        print(f"  {chunk_key} ({len(chunk)} chars): {preview}")
                
                if len(chapter['chunks']) > chunk_preview_count:
                    print(f"  ... and {len(chapter['chunks']) - chunk_preview_count} more chunks")
                
                print("-" * 80)
        
        def export_full_script(self, filename: str = None):
            """
            Export all chapter scripts as one complete script
            Combines scripts from all database rows
            
            Args:
                filename (str): Optional filename, will auto-generate if not provided
            """
            chapters = self.get_chapters_with_scripts(100)  # Get all chapters
            
            if not chapters:
                print("No chapter scripts found to export.")
                return
            
            if not filename:
                filename = f"{self.vid_id}_complete_script.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Complete Script for: {chapters[0]['chapter_content']}\n")
                f.write(f"Generated from {len(chapters)} database rows\n")
                f.write("=" * 80 + "\n\n")
                
                for chapter in chapters:
                    f.write(f"CHAPTER {chapter['chapter_number']}: {chapter['chapter_title']}\n")
                    f.write(f"(Database Row ID: {chapter['id']})\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"{chapter['script']}\n\n")
                    f.write("=" * 80 + "\n\n")
            
            print(f"Complete script exported to: {filename}")
            print(f"Total chapters: {len(chapters)}")
            print(f"Each chapter was stored in its own database row")

        async def run_generator(self, api_key: str = None, vid_id: str = None):
            """
            Convenience method to run the complete script generation process
            
            Args:
                api_key (str): Optional API key override
                vid_id (str): Optional VID_ID override
            """
            # Use provided parameters or fall back to class/config values
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
            if vid_id:
                self.vid_id = vid_id
                self.db_name = f"{vid_id}_longform.db"
                self.setup_database()
            
            # Get API key from config or environment variable if not provided
            if not hasattr(self.client, 'api_key') or not self.client.api_key:
                api_key = API_KEY if API_KEY else os.getenv('OPENAI_API_KEY')
                if not api_key:
                    print("Error: No OpenAI API key found. Please set API_KEY in the configuration section or set OPENAI_API_KEY environment variable.")
                    return
                self.client = openai.OpenAI(api_key=api_key)
            
            # Check if database exists
            if not os.path.exists(self.db_name):
                print(f"Error: Database '{self.db_name}' not found!")
                print("Please run the video ideas generator first to create video ideas.")
                return
            
            print(f"Using VID_ID: {self.vid_id}")
            print(f"Database: {self.db_name}")
            print(f"Chunk size: ~{WORDS_PER_20_SECONDS} words ({CHARS_PER_20_SECONDS} chars) per 20-second segment")
            
            # Process all ideas and generate scripts concurrently
            await self.process_all_ideas()
            
            # Show preview of generated scripts and chunks
            self.display_script_preview()

    async def ChapterScript():
        """Main function to run the chapter script generator"""
        
        # Get API key from config or environment variable
        api_key = API_KEY if API_KEY else os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: No OpenAI API key found. Please set API_KEY in the configuration section or set OPENAI_API_KEY environment variable.")
            return
        
        # Check if database exists
        db_name = f"{VID_ID}_longform.db"
        if not os.path.exists(db_name):
            print(f"Error: Database '{db_name}' not found!")
            print("Please run the script generator first to create chapter outlines.")
            return
        
        print(f"Using VID_ID: {VID_ID}")
        print(f"Database: {db_name}")
        print(f"Chunk size: ~{WORDS_PER_20_SECONDS} words ({CHARS_PER_20_SECONDS} chars) per 20-second segment")
        print("Each chapter will be stored as a separate row in the database")
        
        # Initialize the generator
        generator = ChapterScriptGenerator(api_key, VID_ID)
        
        # Process all chapters and generate scripts sequentially for continuity
        await generator.process_all_chapters()
        
        # Show preview of generated scripts and chunks
        generator.display_script_preview()
        
        # Ask if user wants to export complete script
        try:
            export_choice = 'n'
            if export_choice == 'y':
                generator.export_full_script()
        except:
            pass  # Skip if running in non-interactive environment

    # ==============================================================================
    # ULTRA-FAST OPTIMIZATIONS
    # ==============================================================================

    class DatabasePool:
        """Connection pool for database access"""
        def __init__(self, db_path, pool_size=4):
            self.db_path = db_path
            self.pool = Queue(maxsize=pool_size)
            self._initialize_pool(pool_size)
        
        def _initialize_pool(self, size):
            for _ in range(size):
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.execute('PRAGMA journal_mode=WAL;')
                conn.execute('PRAGMA synchronous=OFF;')  # Faster than NORMAL
                conn.execute('PRAGMA cache_size=50000;')  # Larger cache
                conn.execute('PRAGMA temp_store=MEMORY;')
                conn.execute('PRAGMA mmap_size=268435456;')  # 256MB mmap
                self.pool.put(conn)
        
        def get_connection(self):
            return self.pool.get()
        
        def return_connection(self, conn):
            self.pool.put(conn)

    @lru_cache(maxsize=1000)
    def clean_script_text_cached(text):
        """Cached version of text cleaning for repeated texts"""
        if not text:
            return ""
        
        # Single-pass cleaning
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\.,!?;:\-\'"()]', '', text)
        
        # Faster truncation
        if len(text) > 500:
            text = text[:497] + "..."
        
        return text

    def load_scripts_ultra_fast(vid_id, db_pool):
        """Ultra-fast database loading with aggressive optimizations"""
        conn = db_pool.get_connection()
        
        try:
            cursor = conn.cursor()
            
            # Get column info with single query
            cursor.execute("SELECT name FROM PRAGMA_TABLE_INFO('video_ideas') WHERE name LIKE 'C%'")
            c_columns = sorted([row[0] for row in cursor.fetchall() 
                            if row[0][1:].isdigit()], key=lambda x: int(x[1:]))
            
            if not c_columns:
                return []
            
            # Optimized query with COALESCE for non-null values
            columns_str = ', '.join(c_columns)
            query = f"""
            SELECT rowid, {columns_str} 
            FROM video_ideas 
            WHERE ({' IS NOT NULL OR '.join(c_columns)} IS NOT NULL)
            ORDER BY rowid
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Vectorized processing
            scripts_data = []
            for row in rows:
                table_index = row[0]
                
                for col_idx, c_column in enumerate(c_columns):
                    script_content = row[col_idx + 1]
                    
                    if script_content and script_content.strip():
                        cleaned_script = clean_script_text_cached(script_content)
                        if cleaned_script:
                            scripts_data.append({
                                'table_index': table_index,
                                'column_name': c_column,
                                'column_number': int(c_column[1:]),
                                'script': cleaned_script,
                                'filename': f"{vid_id}_{table_index}_{c_column[1:]}.wav"
                            })
            
            return scripts_data
            
        finally:
            db_pool.return_connection(conn)

    class ModelManager:
        """Singleton model manager with advanced optimizations"""
        _instance = None
        _model = None
        _device = None
        
        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
        
        def get_model(self, device=None):
            if self._model is None:
                self._initialize_model(device)
            return self._model
        
        def _initialize_model(self, device=None):
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._device = device
            
            # Extreme GPU optimizations
            if device.startswith("cuda"):
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.deterministic = False
                torch.cuda.empty_cache()
                
                # Set memory allocation strategy
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,roundup_power2_divisions:16'
            
            print(f"Loading model on {device}...")
            self._model = ChatterboxTTS.from_pretrained(device=device)
            
            # Aggressive optimizations
            self._apply_optimizations()
        
        def _apply_optimizations(self):
            """Apply all possible optimizations for ChatterboxTTS"""
            # FP16 precision - be more careful with ChatterboxTTS
            try:
                if hasattr(self._model, 'half'):
                    self._model = self._model.half()
                    print("✓ FP16 enabled")
                else:
                    print("✗ FP16 not supported by this model")
            except Exception as e:
                print(f"✗ FP16 failed: {str(e)[:50]}")
            
            # Model compilation (PyTorch 2.0+) - more conservative
            try:
                if hasattr(torch, 'compile') and hasattr(self._model, 'forward'):
                    self._model = torch.compile(self._model, mode='default')  # Less aggressive
                    print("✓ Model compiled")
            except Exception as e:
                print(f"✗ torch.compile failed: {str(e)[:50]}")
            
            # Set to eval mode only if the method exists
            try:
                if hasattr(self._model, 'eval'):
                    self._model.eval()
                    print("✓ Model set to eval mode")
            except:
                print("✗ Model doesn't have eval() method")
            
            # Disable gradients for all parameters if possible
            try:
                if hasattr(self._model, 'parameters'):
                    for param in self._model.parameters():
                        param.requires_grad = False
                    print("✓ Gradients disabled")
            except:
                print("✗ Could not disable gradients")

    class BatchProcessor:
        """Optimized batch processing with memory management"""
        
        def __init__(self, model_manager, audio_prompt_path, output_dir):
            self.model_manager = model_manager
            self.audio_prompt_path = audio_prompt_path
            self.output_dir = output_dir
            self.model = model_manager.get_model()
            
            # Pre-warm model
            self._warmup_model()
        
        def _warmup_model(self):
            """Warm up model with dummy input"""
            try:
                with torch.no_grad():
                    dummy_text = "Hello world"
                    _ = self.model.generate(dummy_text, audio_prompt_path=self.audio_prompt_path)
                print("✓ Model warmed up")
            except:
                print("✗ Model warmup failed")
        
        def process_batch(self, batch_scripts):
            """Process a batch of scripts with aggressive per-file cache cleanup"""
            results = []
            
            # Process in micro-batches to optimize memory
            micro_batch_size = 1  # TTS models typically work best with batch size 1
            
            for i in range(0, len(batch_scripts), micro_batch_size):
                micro_batch = batch_scripts[i:i + micro_batch_size]
                
                try:
                    with torch.no_grad():
                        for script_info in micro_batch:
                            output_path = os.path.join(self.output_dir, script_info['filename'])
                            
                            # Skip if exists
                            if os.path.exists(output_path):
                                results.append(True)
                                continue
                            
                            # Generate audio with memory optimization
                            if torch.cuda.is_available():
                                with torch.cuda.amp.autocast():
                                    wav = self.model.generate(script_info['script'], 
                                                            audio_prompt_path=self.audio_prompt_path)
                            else:
                                wav = self.model.generate(script_info['script'], 
                                                        audio_prompt_path=self.audio_prompt_path)
                            
                            if wav is not None and wav.numel() > 0:
                                # Optimize saving
                                if wav.dtype == torch.float16:
                                    wav = wav.float()
                                
                                # Save directly without CPU transfer if possible
                                torchaudio.save(output_path, wav.cpu(), self.model.sr)
                                results.append(True)
                            else:
                                results.append(False)
                            
                            # AGGRESSIVE CLEANUP AFTER EACH FILE
                            self._cleanup_after_file()
                    
                    # Additional cleanup every 10 iterations for safety
                    if i % 10 == 0:
                        self.deep_cleanup()
                        
                except Exception as e:
                    print(f"Batch error: {str(e)[:100]}")
                    results.extend([False] * len(micro_batch))
            
            return results
        
        def _cleanup_after_file(self):
            """Aggressive cleanup after each audio file generation"""
            # Clear CUDA cache immediately
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete
            
            # Force garbage collection
            gc.collect()
            
            # Clear any temporary variables that might accumulate
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
        
        def deep_cleanup(self):
            """Deep cleanup every 10 files to prevent memory leaks - cross-platform"""
            print("🧹 Deep cleanup...")

            # Aggressive CUDA cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                try:
                    torch.cuda.ipc_collect()  # Clean up inter-process CUDA memory
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                except:
                    pass
                
            # Multiple garbage collection passes
            for _ in range(3):
                gc.collect()

            # Clear Python's internal caches
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()

            # Skip OS-specific cleanup that might cause errors
            print("✓ Deep cleanup complete (cross-platform)")

    def generate_audio_hyper_optimized(vid_id, audio_prompt_path, max_workers=None):
        """Hyper-optimized version with single model and sequential processing"""
        
        print("\n=== HYPER-OPTIMIZED TTS GENERATION ===")
        
        # Verify CUDA
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available. Performance will be severely limited.")
            return
        
        device_name = torch.cuda.get_device_name()
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: {device_name} ({memory_gb:.1f}GB)")
        
        # Verify audio prompt
        if not os.path.exists(audio_prompt_path):
            raise FileNotFoundError(f"Audio prompt file {audio_prompt_path} not found")
        
        # Create output directory
        output_dir = f"{vid_id}_Audio"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize database pool
        db_path = f"{vid_id}_longform.db"
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file {db_path} not found")
        
        db_pool = DatabasePool(db_path)
        
        # Load scripts with pooled connections
        print("Loading scripts...")
        scripts_data = load_scripts_ultra_fast(vid_id, db_pool)
        
        if not scripts_data:
            print("No scripts found!")
            return
        
        # Sort for consistent processing
        scripts_data.sort(key=lambda x: (x['table_index'], x['column_number']))
        
        # Filter existing files
        remaining_scripts = [s for s in scripts_data 
                            if not os.path.exists(os.path.join(output_dir, s['filename']))]
        
        total_scripts = len(scripts_data)
        remaining_count = len(remaining_scripts)
        
        print(f"Total scripts: {total_scripts}")
        print(f"Remaining: {remaining_count}")
        
        if remaining_count == 0:
            print("All files already exist!")
            return
        
        # Initialize SINGLE model manager (no threading for now)
        print("Initializing single model...")
        model_manager = ModelManager()
        
        # Create single processor
        processor = BatchProcessor(model_manager, audio_prompt_path, output_dir)
        
        print("Processing files sequentially with aggressive cleanup...")
        
        # Process sequentially with per-file cleanup
        start_time = time.time()
        successful = 0
        
        for i, script_info in enumerate(remaining_scripts):
            try:
                # Process single file
                results = processor.process_batch([script_info])
                if results and results[0]:
                    successful += 1
                
                # Progress update every 5 files
                if (i + 1) % 5 == 0 or i == remaining_count - 1:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (remaining_count - i - 1) / rate if rate > 0 else 0
                    
                    print(f"Progress: {i+1}/{remaining_count} ({(i+1)/remaining_count*100:.1f}%) - "
                        f"Rate: {rate:.1f}/sec - ETA: {eta:.0f}s - Success: {successful}")
                    
                    # Extra cleanup every 5 files
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                    
            except Exception as e:
                print(f"Error processing {script_info['filename']}: {str(e)[:100]}")
        
        # Final cleanup
        cleanup_resources()
        
        duration = time.time() - start_time
        
        print(f"\n=== HYPER-OPTIMIZED COMPLETE ===")
        print(f"Processed: {successful}/{remaining_count} files")
        print(f"Success rate: {successful/remaining_count*100:.1f}%")
        print(f"Total time: {duration:.1f}s")
        print(f"Average: {duration/remaining_count:.2f}s per file")
        print(f"Speed: {remaining_count/duration:.1f} files/sec")
        
        print(f"\n=== HYPER-OPTIMIZED COMPLETE ===")
        print(f"Processed: {successful}/{remaining_count} files")
        print(f"Success rate: {successful/remaining_count*100:.1f}%")
        print(f"Total time: {duration:.1f}s")
        print(f"Average: {duration/remaining_count:.2f}s per file")
        print(f"Speed: {remaining_count/duration:.1f} files/sec")

    def generate_audio_single_threaded_optimized(vid_id, audio_prompt_path):
        """Single-threaded version optimized for memory efficiency and speed"""
        
        print("\n=== SINGLE-THREADED OPTIMIZED TTS GENERATION ===")
        
        # Verify CUDA
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available. Performance will be severely limited.")
            device = "cpu"
        else:
            device = "cuda"
            device_name = torch.cuda.get_device_name()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Device: {device_name} ({memory_gb:.1f}GB)")
        
        # Verify audio prompt
        if not os.path.exists(audio_prompt_path):
            raise FileNotFoundError(f"Audio prompt file {audio_prompt_path} not found")
        
        # Create output directory
        output_dir = f"{vid_id}_Audio"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load scripts
        db_path = f"{vid_id}_longform.db"
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file {db_path} not found")
        
        db_pool = DatabasePool(db_path)
        
        print("Loading scripts...")
        scripts_data = load_scripts_ultra_fast(vid_id, db_pool)
        
        if not scripts_data:
            print("No scripts found!")
            return
        
        scripts_data.sort(key=lambda x: (x['table_index'], x['column_number']))
        
        # Filter existing files
        remaining_scripts = [s for s in scripts_data 
                            if not os.path.exists(os.path.join(output_dir, s['filename']))]
        
        total_scripts = len(scripts_data)
        remaining_count = len(remaining_scripts)
        
        print(f"Total scripts: {total_scripts}")
        print(f"Remaining: {remaining_count}")
        
        if remaining_count == 0:
            print("All files already exist!")
            return
        
        # Initialize model
        print("Loading model...")
        model = ChatterboxTTS.from_pretrained(device=device)
        
        # Basic optimizations that work with ChatterboxTTS
        try:
            if device == "cuda":
                # Try FP16 if supported
                model = model.half()
                print("✓ FP16 enabled")
        except:
            print("✗ FP16 not supported, using FP32")
        
        print("Processing files with aggressive cleanup...")
        
        start_time = time.time()
        successful = 0
        
        for i, script_info in enumerate(remaining_scripts):
            try:
                output_path = os.path.join(output_dir, script_info['filename'])
                
                # Generate audio
                with torch.no_grad():
                    if device == "cuda":
                        try:
                            with torch.cuda.amp.autocast():
                                wav = model.generate(script_info['script'], 
                                                audio_prompt_path=audio_prompt_path)
                        except:
                            # Fallback without autocast
                            wav = model.generate(script_info['script'], 
                                            audio_prompt_path=audio_prompt_path)
                    else:
                        wav = model.generate(script_info['script'], 
                                        audio_prompt_path=audio_prompt_path)
                
                # Save audio
                if wav is not None and wav.numel() > 0:
                    if wav.dtype == torch.float16:
                        wav = wav.float()
                    
                    torchaudio.save(output_path, wav.cpu(), model.sr)
                    successful += 1
                    
                    # Aggressive cleanup after each file
                    del wav
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                
                # Progress update every 5 files
                if (i + 1) % 5 == 0 or i == remaining_count - 1:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (remaining_count - i - 1) / rate if rate > 0 else 0
                    
                    # Memory stats
                    if device == "cuda":
                        allocated = torch.cuda.memory_allocated() / 1e9
                        cached = torch.cuda.memory_reserved() / 1e9
                        print(f"Progress: {i+1}/{remaining_count} ({(i+1)/remaining_count*100:.1f}%) - "
                            f"Rate: {rate:.1f}/sec - ETA: {eta:.0f}s - GPU: {allocated:.1f}GB/{cached:.1f}GB")
                    else:
                        print(f"Progress: {i+1}/{remaining_count} ({(i+1)/remaining_count*100:.1f}%) - "
                            f"Rate: {rate:.1f}/sec - ETA: {eta:.0f}s")
                    
                    # Deep cleanup every 10 files
                    if (i + 1) % 10 == 0:
                        print("🧹 Deep cleanup...")
                        cleanup_resources()
                        
            except Exception as e:
                print(f"Error processing {script_info['filename']}: {str(e)[:100]}")
        
        duration = time.time() - start_time
        
        print(f"\n=== SINGLE-THREADED COMPLETE ===")
        print(f"Processed: {successful}/{remaining_count} files")
        print(f"Success rate: {successful/remaining_count*100:.1f}%")
        print(f"Total time: {duration:.1f}s")
        print(f"Average: {duration/remaining_count:.2f}s per file")
        print(f"Speed: {remaining_count/duration:.1f} files/sec")
        """Benchmark different optimization strategies"""
        print("\n=== OPTIMIZATION BENCHMARK ===")
        
        # Test script loading speed
        start = time.time()
        db_pool = DatabasePool(f"{vid_id}_longform.db")
        scripts = load_scripts_ultra_fast(vid_id, db_pool)
        load_time = time.time() - start
        print(f"Script loading: {len(scripts)} scripts in {load_time:.2f}s ({len(scripts)/load_time:.1f}/sec)")
        
        # Test model initialization
        start = time.time()
        model_manager = ModelManager()
        model = model_manager.get_model()
        init_time = time.time() - start
        print(f"Model initialization: {init_time:.2f}s")
        
        # Test single generation
        if scripts:
            start = time.time()
            with torch.no_grad():
                wav = model.generate(scripts[0]['script'][:100], audio_prompt_path=audio_prompt_path)
            gen_time = time.time() - start
            print(f"Single generation: {gen_time:.2f}s")
            print(f"Estimated throughput: {1/gen_time:.1f} files/sec")

    # Additional utility functions
    def profile_memory_usage():
        """Profile memory usage during generation"""
        import psutil
        import GPUtil
        
        process = psutil.Process()
        
        print("=== MEMORY PROFILING ===")
        print(f"RAM: {process.memory_info().rss / 1e9:.2f}GB")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu = GPUtil.getGPUs()[i]
                print(f"GPU {i}: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")

    def cleanup_resources():
        """Comprehensive cleanup to maintain consistent performance - cross-platform"""
        print("🧹 Cleaning up resources...")
        
        # CUDA cleanup
        if torch.cuda.is_available():
            # Empty all caches
            torch.cuda.empty_cache()
            
            # Synchronize all devices
            torch.cuda.synchronize()
            
            # Clean up inter-process memory (if using multiprocessing)
            try:
                torch.cuda.ipc_collect()
            except:
                pass
            
            # Reset memory statistics to prevent tracking overhead
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            except:
                pass
            
            # Print current GPU memory usage
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                cached = torch.cuda.memory_reserved(i) / 1e9
                print(f"GPU {i}: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
        
        # Python garbage collection (multiple passes for thorough cleanup)
        collected = 0
        for _ in range(3):
            collected += gc.collect()
        
        if collected > 0:
            print(f"🗑️  Collected {collected} objects")
        
        # Clear Python internal caches
        import sys
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
        
        # OS-level memory cleanup (cross-platform)
        try:
            import platform
            system = platform.system().lower()
            
            if system == 'linux':
                # Linux-specific malloc_trim
                import ctypes
                try:
                    # Try different common paths for libc
                    libc_paths = [
                        "libc.so.6",
                        "/lib/x86_64-linux-gnu/libc.so.6",
                        "/lib64/libc.so.6",
                        "/usr/lib64/libc.so.6"
                    ]
                    
                    libc = None
                    for path in libc_paths:
                        try:
                            libc = ctypes.CDLL(path)
                            break
                        except OSError:
                            continue
                    
                    if libc and hasattr(libc, 'malloc_trim'):
                        libc.malloc_trim(0)
                        print("✓ Linux malloc_trim applied")
                    else:
                        print("ℹ️  malloc_trim not available")
                        
                except Exception as e:
                    print(f"ℹ️  malloc_trim failed: {str(e)[:50]}")
            
            elif system == 'windows':
                # Windows-specific memory cleanup
                try:
                    import ctypes
                    from ctypes import wintypes
                    
                    # Try to trim working set
                    kernel32 = ctypes.windll.kernel32
                    handle = kernel32.GetCurrentProcess()
                    
                    # SetProcessWorkingSetSize with -1, -1 to trim working set
                    if kernel32.SetProcessWorkingSetSize(handle, -1, -1):
                        print("✓ Windows working set trimmed")
                    else:
                        print("ℹ️  Windows working set trim failed")
                        
                except Exception as e:
                    print(f"ℹ️  Windows cleanup failed: {str(e)[:50]}")
            
            elif system == 'darwin':  # macOS
                # macOS doesn't have direct equivalent, but we can try to force memory pressure
                try:
                    import subprocess
                    # Try to run memory pressure command (if available)
                    result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        print("✓ macOS memory stats checked")
                    else:
                        print("ℹ️  macOS memory cleanup limited")
                except Exception as e:
                    print(f"ℹ️  macOS cleanup limited: {str(e)[:30]}")
            
            else:
                print(f"ℹ️  OS-level cleanup not available for {system}")
                
        except Exception as e:
            print(f"ℹ️  OS-level cleanup skipped: {str(e)[:50]}")
        
        print("✓ Cleanup complete")

    def monitor_memory_growth():
        """Monitor memory growth patterns to detect leaks"""
        if not torch.cuda.is_available():
            return
        
        memory_snapshots = []
        
        def take_snapshot(label=""):
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            memory_snapshots.append((label, allocated, cached))
            return allocated, cached
        
        return take_snapshot

    # Enhanced memory monitoring function
    def print_memory_stats(label="Memory Stats"):
        """Print detailed memory statistics"""
        print(f"\n=== {label} ===")
        
        # System memory
        import psutil
        process = psutil.Process()
        ram_gb = process.memory_info().rss / 1e9
        print(f"System RAM: {ram_gb:.2f}GB")
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                cached = torch.cuda.memory_reserved(i) / 1e9
                max_allocated = torch.cuda.max_memory_allocated(i) / 1e9
                print(f"GPU {i}: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {max_allocated:.2f}GB peak")
        
        # Garbage collection stats
        import gc
        print(f"GC: {gc.get_count()} objects in generations")
        print("=" * (len(label) + 8))

    def benchmark_optimizations(vid_id, audio_prompt_path):
        """Benchmark different optimization strategies"""
        print("\n=== OPTIMIZATION BENCHMARK ===")
        
        # Verify files exist
        db_path = f"{vid_id}_longform.db"
        if not os.path.exists(db_path):
            print(f"❌ Database file {db_path} not found")
            return
        
        if not os.path.exists(audio_prompt_path):
            print(f"❌ Audio prompt file {audio_prompt_path} not found")
            return
        
        # Test 1: Script loading speed
        print("\n1. Testing script loading speed...")
        start = time.time()
        db_pool = DatabasePool(db_path)
        scripts = load_scripts_ultra_fast(vid_id, db_pool)
        load_time = time.time() - start
        print(f"   ✓ Script loading: {len(scripts)} scripts in {load_time:.2f}s ({len(scripts)/load_time:.1f}/sec)")
        
        if not scripts:
            print("   ❌ No scripts found for benchmarking")
            return
        
        # Test 2: Model initialization speed
        print("\n2. Testing model initialization...")
        start = time.time()
        model_manager = ModelManager()
        model = model_manager.get_model()
        init_time = time.time() - start
        print(f"   ✓ Model initialization: {init_time:.2f}s")
        
        # Test 3: Single generation speed (with different text lengths)
        print("\n3. Testing generation speeds...")
        test_texts = [
            ("Short", scripts[0]['script'][:50]),
            ("Medium", scripts[0]['script'][:200]),
            ("Long", scripts[0]['script'][:500])
        ]
        
        generation_times = []
        
        for label, text in test_texts:
            try:
                # Warmup
                with torch.no_grad():
                    _ = model.generate("Hello", audio_prompt_path=audio_prompt_path)
                
                # Actual test
                start = time.time()
                with torch.no_grad():
                    wav = model.generate(text, audio_prompt_path=audio_prompt_path)
                gen_time = time.time() - start
                generation_times.append(gen_time)
                
                if wav is not None:
                    audio_length = wav.shape[-1] / model.sr
                    rtf = gen_time / audio_length  # Real-time factor
                    print(f"   ✓ {label} text ({len(text)} chars): {gen_time:.2f}s -> {audio_length:.2f}s audio (RTF: {rtf:.2f}x)")
                else:
                    print(f"   ❌ {label} text failed to generate")
                
                # Cleanup after each test
                del wav
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                    
            except Exception as e:
                print(f"   ❌ {label} text failed: {str(e)[:100]}")
        
        # Test 4: Memory efficiency test
        print("\n4. Testing memory efficiency...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
            initial_memory = torch.cuda.memory_allocated() / 1e9
            print(f"   Initial GPU memory: {initial_memory:.2f}GB")
            
            # Generate multiple files to test memory growth
            test_scripts = scripts[:min(5, len(scripts))]
            start_memory = torch.cuda.memory_allocated() / 1e9
            
            for i, script_info in enumerate(test_scripts):
                with torch.no_grad():
                    wav = model.generate(script_info['script'][:200], audio_prompt_path=audio_prompt_path)
                
                current_memory = torch.cuda.memory_allocated() / 1e9
                memory_growth = current_memory - start_memory
                
                if i == 0:
                    base_growth = memory_growth
                
                print(f"   File {i+1}: {current_memory:.2f}GB ({memory_growth:.2f}GB growth)")
                
                # Cleanup
                del wav
                torch.cuda.empty_cache()
                gc.collect()
            
            final_memory = torch.cuda.memory_allocated() / 1e9
            total_growth = final_memory - initial_memory
            print(f"   Final GPU memory: {final_memory:.2f}GB (net growth: {total_growth:.2f}GB)")
            
            if total_growth < 0.1:
                print("   ✓ Excellent memory management")
            elif total_growth < 0.5:
                print("   ⚠️  Moderate memory growth")
            else:
                print("   ❌ High memory growth - potential leak")
        
        # Test 5: Database connection pool efficiency
        print("\n5. Testing database connection pool...")
        start = time.time()
        
        # Test multiple concurrent database accesses
        for _ in range(10):
            conn = db_pool.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM video_ideas")
            result = cursor.fetchone()
            db_pool.return_connection(conn)
        
        db_time = time.time() - start
        print(f"   ✓ 10 database operations: {db_time:.3f}s ({10/db_time:.1f} ops/sec)")
        
        # Test 6: Cleanup efficiency
        print("\n6. Testing cleanup efficiency...")
        if torch.cuda.is_available():
            # Create some GPU memory usage
            dummy_tensors = []
            for i in range(5):
                dummy_tensors.append(torch.randn(1000, 1000, device='cuda'))
            
            before_cleanup = torch.cuda.memory_allocated() / 1e9
            
            # Test cleanup
            start = time.time()
            del dummy_tensors
            cleanup_resources()
            cleanup_time = time.time() - start
            
            after_cleanup = torch.cuda.memory_allocated() / 1e9
            memory_freed = before_cleanup - after_cleanup
            
            print(f"   ✓ Cleanup time: {cleanup_time:.3f}s")
            print(f"   ✓ Memory freed: {memory_freed:.2f}GB")
        
        # Summary and recommendations
        print("\n=== BENCHMARK SUMMARY ===")
        
        if generation_times:
            avg_gen_time = sum(generation_times) / len(generation_times)
            estimated_throughput = 1 / avg_gen_time if avg_gen_time > 0 else 0
            
            print(f"Average generation time: {avg_gen_time:.2f}s")
            print(f"Estimated throughput: {estimated_throughput:.1f} files/sec")
            
            # Estimate total processing time
            total_files = len(scripts)
            estimated_total_time = total_files * avg_gen_time
            
            print(f"Estimated time for {total_files} files: {estimated_total_time/60:.1f} minutes")
            
            # Performance recommendations
            print(f"\n=== RECOMMENDATIONS ===")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"Hardware: {gpu_name} ({gpu_memory:.1f}GB)")
                
                if estimated_throughput > 2.0:
                    print("✓ Excellent performance - hardware is well-optimized")
                elif estimated_throughput > 1.0:
                    print("⚠️  Good performance - consider enabling more optimizations")
                else:
                    print("❌ Poor performance - check for memory bottlenecks")
            else:
                print("⚠️  CPU-only mode detected - GPU acceleration strongly recommended")
            
            # Memory recommendations
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1e9
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                memory_usage = current_memory / total_memory * 100
                
                if memory_usage < 50:
                    print("✓ Memory usage is efficient")
                elif memory_usage < 80:
                    print("⚠️  Memory usage is moderate - monitor for growth")
                else:
                    print("❌ High memory usage - consider batch size reduction")
            
            print(f"✓ Benchmark complete")
        else:
            print("❌ Benchmark failed - no successful generations")
        
        # Final cleanup
        cleanup_resources()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    @dataclass
    class TrimResult:
        """Data class to store trimming results"""
        success: bool
        original_duration: float
        new_duration: float
        time_saved: float
        file_path: str
        error_message: Optional[str] = None

    class OptimizedAudioTrimmer:
        """
        High-performance audio trimmer with parallel processing and optimizations.
        """
        
        def __init__(self, 
                    vid_id: str = "SolarFlare",
                    silence_threshold: float = 0.1,
                    silence_duration: float = 0.2,
                    backup_originals: bool = False,
                    max_workers: Optional[int] = None,
                    chunk_size: int = 1024 * 1024,  # 1MB chunks for large files
                    enable_cache: bool = True):
            """
            Initialize the audio trimmer with optimized settings.
            
            Args:
                vid_id: Video ID for folder naming
                silence_threshold: Amplitude threshold for detecting silence
                silence_duration: Seconds to wait after speech stops before trimming
                backup_originals: Whether to create backup files
                max_workers: Number of parallel workers (None = auto-detect)
                chunk_size: Size of audio chunks for processing large files
                enable_cache: Enable caching for repeated operations
            """
            self.vid_id = vid_id
            self.silence_threshold = silence_threshold
            self.silence_duration = silence_duration
            self.backup_originals = backup_originals
            self.max_workers = max_workers or min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
            self.chunk_size = chunk_size
            self.enable_cache = enable_cache
            
            # Audio extensions to process
            self.audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
            
            # Cache for repeated operations
            self._cache = {} if enable_cache else None
            
            # Audio folder path
            self.audio_folder = Path(f"{vid_id}_Audio")
            
            logger.info(f"Initialized AudioTrimmer with {self.max_workers} workers")
        
        def _get_cache_key(self, file_path: Path) -> str:
            """Generate cache key based on file path and modification time"""
            return f"{file_path}_{file_path.stat().st_mtime}"
        
        def detect_speech_end_optimized(self, audio_data: np.ndarray, sample_rate: int) -> int:
            """
            Optimized speech end detection using vectorized operations.
            
            Args:
                audio_data: Audio data array
                sample_rate: Sample rate of the audio
                
            Returns:
                Sample index where to cut the audio
            """
            # Convert to mono if stereo (optimized)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Use vectorized operations for better performance
            abs_audio = np.abs(audio_data)
            
            # Find all points above threshold using vectorized operations
            above_threshold = abs_audio > self.silence_threshold
            
            if not np.any(above_threshold):
                # No speech detected, return original length
                return len(audio_data)
            
            # Find the last point where audio is above threshold (optimized)
            last_speech_indices = np.where(above_threshold)[0]
            last_speech_sample = last_speech_indices[-1] if len(last_speech_indices) > 0 else 0
            
            # Add silence duration after the last speech
            silence_samples = int(self.silence_duration * sample_rate)
            cut_point = min(last_speech_sample + silence_samples, len(audio_data))
            
            return cut_point
        
        def detect_speech_end_chunked(self, audio_data: np.ndarray, sample_rate: int) -> int:
            """
            Memory-efficient speech detection for very large files using chunked processing.
            """
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Process in chunks to save memory
            chunk_samples = self.chunk_size
            last_speech_sample = 0
            
            # Process chunks from end to beginning for efficiency
            for start_idx in range(len(audio_data) - chunk_samples, -1, -chunk_samples):
                end_idx = min(start_idx + chunk_samples, len(audio_data))
                chunk = audio_data[start_idx:end_idx]
                
                abs_chunk = np.abs(chunk)
                above_threshold = abs_chunk > self.silence_threshold
                
                if np.any(above_threshold):
                    chunk_indices = np.where(above_threshold)[0]
                    last_speech_sample = start_idx + chunk_indices[-1]
                    break
            
            # Add silence duration
            silence_samples = int(self.silence_duration * sample_rate)
            cut_point = min(last_speech_sample + silence_samples, len(audio_data))
            
            return cut_point
        
        def trim_audio_file_optimized(self, file_path: Path) -> TrimResult:
            """
            Optimized audio file trimming with better error handling and performance.
            """
            start_time = time.time()
            
            try:
                # Check cache first
                cache_key = self._get_cache_key(file_path) if self.enable_cache else None
                if cache_key and self._cache and cache_key in self._cache:
                    logger.info(f"Using cached result for {file_path.name}")
                    return self._cache[cache_key]
                
                logger.info(f"Processing: {file_path.name}")
                
                # Load audio with optimized settings
                audio_data, sample_rate = librosa.load(
                    str(file_path), 
                    sr=None, 
                    mono=False,
                    dtype=np.float32  # Use float32 for better memory efficiency
                )
                
                # Handle stereo audio shape
                if len(audio_data.shape) > 1 and audio_data.shape[0] < audio_data.shape[1]:
                    audio_data = audio_data.T
                
                original_length = len(audio_data) if len(audio_data.shape) == 1 else len(audio_data)
                original_duration = original_length / sample_rate
                
                # Choose detection method based on file size
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > 100:  # Use chunked processing for files > 100MB
                    logger.info(f"Using chunked processing for large file ({file_size_mb:.1f}MB)")
                    cut_point = self.detect_speech_end_chunked(audio_data, sample_rate)
                else:
                    cut_point = self.detect_speech_end_optimized(audio_data, sample_rate)
                
                # Trim the audio
                if len(audio_data.shape) == 1:
                    trimmed_audio = audio_data[:cut_point]
                else:
                    trimmed_audio = audio_data[:cut_point]
                
                new_duration = len(trimmed_audio) / sample_rate
                time_saved = original_duration - new_duration
                
                # Create backup if requested
                if self.backup_originals:
                    backup_path = file_path.parent / f"{file_path.stem}_original{file_path.suffix}"
                    if not backup_path.exists():
                        sf.write(str(backup_path), audio_data, sample_rate)
                        logger.info(f"Created backup: {backup_path.name}")
                
                # Save trimmed audio (replace original)
                sf.write(str(file_path), trimmed_audio, sample_rate)
                
                # Clean up memory
                del audio_data, trimmed_audio
                gc.collect()
                
                processing_time = time.time() - start_time
                logger.info(f"✓ {file_path.name}: {original_duration:.2f}s → {new_duration:.2f}s "
                        f"(saved {time_saved:.2f}s) in {processing_time:.2f}s")
                
                result = TrimResult(
                    success=True,
                    original_duration=original_duration,
                    new_duration=new_duration,
                    time_saved=time_saved,
                    file_path=str(file_path)
                )
                
                # Cache the result
                if cache_key and self._cache is not None:
                    self._cache[cache_key] = result
                
                return result
                
            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {str(e)}"
                logger.error(error_msg)
                return TrimResult(
                    success=False,
                    original_duration=0,
                    new_duration=0,
                    time_saved=0,
                    file_path=str(file_path),
                    error_message=error_msg
                )
        
        def get_audio_files(self) -> List[Path]:
            """Get all audio files in the target folder, excluding backups."""
            if not self.audio_folder.exists():
                raise FileNotFoundError(f"Audio folder '{self.audio_folder}' not found!")
            
            audio_files = []
            for file_path in self.audio_folder.iterdir():
                if (file_path.is_file() and 
                    file_path.suffix.lower() in self.audio_extensions and
                    not file_path.stem.endswith('_original')):
                    audio_files.append(file_path)
            
            return sorted(audio_files)
        
        def trim_all_files_parallel(self) -> Dict[str, any]:
            """
            Trim all audio files using parallel processing.
            
            Returns:
                Dictionary with processing statistics
            """
            start_time = time.time()
            audio_files = self.get_audio_files()
            
            if not audio_files:
                logger.warning("No audio files found to process!")
                return {"success": False, "message": "No audio files found"}
            
            logger.info(f"Found {len(audio_files)} files to process with {self.max_workers} workers")
            logger.warning("⚠️  Original files will be REPLACED!")
            
            results = []
            successful_trims = 0
            total_time_saved = 0
            
            # Process files in parallel
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs
                future_to_file = {
                    executor.submit(self.trim_audio_file_optimized, file_path): file_path 
                    for file_path in audio_files
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_file):
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        successful_trims += 1
                        total_time_saved += result.time_saved
            
            # Calculate statistics
            total_processing_time = time.time() - start_time
            avg_time_saved = total_time_saved / max(successful_trims, 1)
            
            # Summary
            logger.info("=" * 60)
            logger.info("PROCESSING COMPLETE!")
            logger.info(f"Successfully processed: {successful_trims}/{len(audio_files)} files")
            logger.info(f"Total audio time trimmed: {total_time_saved:.2f} seconds")
            logger.info(f"Average time saved per file: {avg_time_saved:.2f} seconds")
            logger.info(f"Total processing time: {total_processing_time:.2f} seconds")
            logger.info(f"Processing speed: {len(audio_files)/total_processing_time:.2f} files/second")
            logger.info("✓ All original files replaced with trimmed versions")
            
            return {
                "success": True,
                "files_processed": len(audio_files),
                "successful_trims": successful_trims,
                "total_time_saved": total_time_saved,
                "total_processing_time": total_processing_time,
                "avg_time_saved": avg_time_saved,
                "processing_speed": len(audio_files) / total_processing_time,
                "results": results
            }
        
        def trim_single_file(self, filename: str) -> TrimResult:
            """Trim a single audio file by name."""
            file_path = self.audio_folder / filename
            
            if not file_path.exists():
                error_msg = f"File '{file_path}' not found!"
                logger.error(error_msg)
                return TrimResult(
                    success=False,
                    original_duration=0,
                    new_duration=0,
                    time_saved=0,
                    file_path=str(file_path),
                    error_message=error_msg
                )
            
            logger.info(f"Trimming single file: {filename}")
            logger.warning("⚠️  Original file will be REPLACED")
            
            return self.trim_audio_file_optimized(file_path)
        
        def preview_analysis(self) -> Dict[str, any]:
            """
            Analyze all files without modifying them - shows what would be trimmed.
            """
            audio_files = self.get_audio_files()
            
            if not audio_files:
                logger.warning("No audio files found!")
                return {"success": False, "message": "No audio files found"}
            
            logger.info(f"PREVIEW MODE - Analyzing {len(audio_files)} files")
            logger.info(f"Silence threshold: {self.silence_threshold}")
            logger.info(f"Silence duration after speech: {self.silence_duration}s")
            logger.info("-" * 60)
            
            total_original_duration = 0
            total_trimmed_duration = 0
            analysis_results = []
            
            for file_path in audio_files:
                try:
                    # Load audio for analysis
                    audio_data, sample_rate = librosa.load(str(file_path), sr=None, mono=False, dtype=np.float32)
                    
                    original_length = len(audio_data) if len(audio_data.shape) == 1 else len(audio_data)
                    original_duration = original_length / sample_rate
                    
                    # Detect cut point
                    cut_point = self.detect_speech_end_optimized(audio_data, sample_rate)
                    new_duration = cut_point / sample_rate
                    time_to_trim = original_duration - new_duration
                    
                    total_original_duration += original_duration
                    total_trimmed_duration += new_duration
                    
                    result = {
                        "filename": file_path.name,
                        "original_duration": original_duration,
                        "new_duration": new_duration,
                        "time_to_trim": time_to_trim,
                        "percentage_trimmed": (time_to_trim / original_duration * 100) if original_duration > 0 else 0
                    }
                    analysis_results.append(result)
                    
                    logger.info(f"{file_path.name}:")
                    logger.info(f"  Original: {original_duration:.2f}s → Trimmed: {new_duration:.2f}s")
                    logger.info(f"  Will trim: {time_to_trim:.2f}s ({result['percentage_trimmed']:.1f}%)")
                    
                    # Clean up memory
                    del audio_data
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"{file_path.name}: Error - {e}")
                    analysis_results.append({
                        "filename": file_path.name,
                        "error": str(e)
                    })
            
            total_time_to_save = total_original_duration - total_trimmed_duration
            overall_reduction = (total_time_to_save / total_original_duration * 100) if total_original_duration > 0 else 0
            
            logger.info("-" * 60)
            logger.info("TOTAL ANALYSIS:")
            logger.info(f"Original total duration: {total_original_duration:.2f}s")
            logger.info(f"Trimmed total duration: {total_trimmed_duration:.2f}s")
            logger.info(f"Total time that will be saved: {total_time_to_save:.2f}s")
            logger.info(f"Overall reduction: {overall_reduction:.1f}%")
            
            return {
                "success": True,
                "total_files": len(audio_files),
                "total_original_duration": total_original_duration,
                "total_trimmed_duration": total_trimmed_duration,
                "total_time_to_save": total_time_to_save,
                "overall_reduction_percentage": overall_reduction,
                "file_results": analysis_results
            }
        
        def clear_cache(self):
            """Clear the internal cache."""
            if self._cache:
                self._cache.clear()
                logger.info("Cache cleared")

    @dataclass
    class AudioFileInfo:
        """Data class to hold audio file information"""
        path: Path
        session_num: int
        part_num: int
        duration: Optional[float] = None
        sample_rate: Optional[int] = None
        channels: Optional[int] = None
        file_size: Optional[int] = None

    class FastAudioCombiner:
        """
        High-performance audio file combiner with parallel processing and memory optimization
        """
        
        def __init__(self, 
                    vid_id: str = "V3",
                    output_format: str = "wav",
                    delete_original_parts: bool = True,
                    silence_between_parts: float = 0.1,
                    max_workers: int = None,
                    chunk_size: int = 8192,
                    use_memory_mapping: bool = True,
                    log_level: str = "INFO"):
            """
            Initialize the FastAudioCombiner
            
            Args:
                vid_id: Video ID prefix for files
                output_format: Output format (wav, mp3, etc.)
                delete_original_parts: Whether to delete original files after combining
                silence_between_parts: Seconds of silence between parts
                max_workers: Number of parallel workers (None = auto)
                chunk_size: Audio processing chunk size
                use_memory_mapping: Use memory mapping for large files
                log_level: Logging level
            """
            self.vid_id = vid_id
            self.output_format = output_format.lower()
            self.delete_original_parts = delete_original_parts
            self.silence_between_parts = silence_between_parts
            self.max_workers = max_workers or min(8, os.cpu_count() or 1)
            self.chunk_size = chunk_size
            self.use_memory_mapping = use_memory_mapping
            
            # Setup logging
            logging.basicConfig(level=getattr(logging, log_level.upper()),
                            format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
            
            # Audio folder path
            self.audio_folder = Path(f"{vid_id}_Audio")
            
            # Cache for file info
            self._file_info_cache = {}
            
            # Performance tracking
            self.stats = {
                'total_files_processed': 0,
                'total_duration_combined': 0.0,
                'processing_time': 0.0,
                'files_deleted': 0
            }
        
        @lru_cache(maxsize=128)
        def _get_file_pattern(self) -> re.Pattern:
            """Get compiled regex pattern for filename matching (cached)"""
            return re.compile(rf"^{re.escape(self.vid_id)}_(\d+)_(\d+)$")
        
        def _parse_filename(self, filename: str) -> Optional[Tuple[str, int, int]]:
            """
            Parse filename to extract VID_ID, session number, and part number.
            Expected format: {VID_ID}_{session}_{part}.wav
            Returns: (vid_id, session_num, part_num) or None
            """
            name_without_ext = Path(filename).stem
            pattern = self._get_file_pattern()
            match = pattern.match(name_without_ext)
            
            if match:
                session_num = int(match.group(1))
                part_num = int(match.group(2))
                return self.vid_id, session_num, part_num
            
            return None
        
        def _get_audio_info_fast(self, file_path: Path) -> AudioFileInfo:
            """Get audio file info with caching and optimized loading"""
            if str(file_path) in self._file_info_cache:
                return self._file_info_cache[str(file_path)]
            
            try:
                # Use soundfile for faster metadata reading
                with sf.SoundFile(file_path) as f:
                    duration = len(f) / f.samplerate
                    sample_rate = f.samplerate
                    channels = f.channels
                
                parsed = self._parse_filename(file_path.name)
                if not parsed:
                    raise ValueError(f"Invalid filename format: {file_path.name}")
                
                _, session_num, part_num = parsed
                
                info = AudioFileInfo(
                    path=file_path,
                    session_num=session_num,
                    part_num=part_num,
                    duration=duration,
                    sample_rate=sample_rate,
                    channels=channels,
                    file_size=file_path.stat().st_size
                )
                
                self._file_info_cache[str(file_path)] = info
                return info
                
            except Exception as e:
                self.logger.warning(f"Could not get info for {file_path}: {e}")
                return None
        
        def scan_audio_files(self) -> Dict[int, List[AudioFileInfo]]:
            """
            Scan the audio folder and group files by session number (parallelized).
            Returns a dictionary: {session_num: [AudioFileInfo, ...]}
            """
            if not self.audio_folder.exists():
                self.logger.error(f"Audio folder '{self.audio_folder}' not found!")
                return {}
            
            # Find all audio files
            audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
            audio_files = []
            
            for file_path in self.audio_folder.iterdir():
                if (file_path.is_file() and 
                    file_path.suffix.lower() in audio_extensions and
                    'combined' not in file_path.stem.lower() and
                    not file_path.stem.endswith('_original')):
                    audio_files.append(file_path)
            
            self.logger.info(f"Found {len(audio_files)} audio files to process")
            
            # Process files in parallel
            sessions = defaultdict(list)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                file_infos = list(executor.map(self._get_audio_info_fast, audio_files))
            
            # Group by session
            for info in file_infos:
                if info:
                    sessions[info.session_num].append(info)
                    self.logger.debug(f"Found: Session {info.session_num}, Part {info.part_num} - {info.path.name}")
            
            # Sort parts within each session
            for session_num in sessions:
                sessions[session_num].sort(key=lambda x: x.part_num)
            
            return dict(sessions)
        
        def _combine_audio_files_optimized(self, 
                                        file_infos: List[AudioFileInfo], 
                                        output_path: Path) -> Tuple[bool, Union[float, str]]:
            """
            Optimized audio combining with streaming and memory efficiency
            """
            if not file_infos:
                return False, "No files to combine"
            
            self.logger.info(f"Combining {len(file_infos)} files to {output_path.name}")
            
            try:
                # Get target sample rate from first file
                target_sr = file_infos[0].sample_rate
                target_channels = file_infos[0].channels
                
                # Use temporary file for writing
                with tempfile.NamedTemporaryFile(suffix=f'.{self.output_format}', delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                
                total_duration = 0.0
                silence_samples = int(self.silence_between_parts * target_sr) if self.silence_between_parts > 0 else 0
                
                # Create silence array once if needed
                silence_array = None
                if silence_samples > 0:
                    if target_channels == 1:
                        silence_array = np.zeros(silence_samples, dtype=np.float32)
                    else:
                        silence_array = np.zeros((silence_samples, target_channels), dtype=np.float32)
                
                # Open output file for writing
                with sf.SoundFile(temp_path, 'w', samplerate=target_sr, 
                                channels=target_channels, format=self.output_format.upper()) as output_file:
                    
                    for i, file_info in enumerate(file_infos):
                        self.logger.debug(f"Processing part {i+1}/{len(file_infos)}: {file_info.path.name}")
                        
                        # Stream audio in chunks to save memory
                        with sf.SoundFile(file_info.path) as input_file:
                            # Resample if necessary
                            if input_file.samplerate != target_sr:
                                self.logger.warning(f"Resampling {file_info.path.name} from {input_file.samplerate} to {target_sr}")
                            
                            # Process in chunks
                            while True:
                                chunk = input_file.read(self.chunk_size, dtype=np.float32)
                                if len(chunk) == 0:
                                    break
                                
                                # Resample chunk if needed
                                if input_file.samplerate != target_sr:
                                    chunk = librosa.resample(chunk.T, orig_sr=input_file.samplerate, 
                                                        target_sr=target_sr).T
                                
                                # Handle channel mismatch
                                if len(chunk.shape) == 1 and target_channels > 1:
                                    chunk = np.column_stack([chunk] * target_channels)
                                elif len(chunk.shape) == 2 and chunk.shape[1] != target_channels:
                                    if chunk.shape[1] > target_channels:
                                        chunk = chunk[:, :target_channels]  # Downmix
                                    else:
                                        # Upmix by repeating channels
                                        chunk = np.column_stack([chunk] * (target_channels // chunk.shape[1] + 1))
                                        chunk = chunk[:, :target_channels]
                                
                                output_file.write(chunk)
                        
                        total_duration += file_info.duration
                        
                        # Add silence between parts (except after last part)
                        if i < len(file_infos) - 1 and silence_array is not None:
                            output_file.write(silence_array)
                
                # Move temp file to final location
                shutil.move(temp_path, output_path)
                
                self.logger.info(f"✓ Combined file created: {output_path.name} ({total_duration:.2f}s)")
                return True, total_duration
                
            except Exception as e:
                self.logger.error(f"Error combining files: {e}")
                # Clean up temp file if it exists
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink()
                return False, str(e)
        
        def _combine_session(self, session_data: Tuple[int, List[AudioFileInfo]]) -> Tuple[int, bool, str]:
            """Combine a single session (for parallel processing)"""
            session_num, file_infos = session_data
            
            output_filename = f"{self.vid_id}_{session_num}_combined.{self.output_format}"
            output_path = self.audio_folder / output_filename
            
            self.logger.info(f"[Session {session_num}] Starting combination of {len(file_infos)} parts")
            
            success, result = self._combine_audio_files_optimized(file_infos, output_path)
            
            if success:
                self.stats['total_files_processed'] += len(file_infos)
                self.stats['total_duration_combined'] += result
                return session_num, True, f"Combined successfully ({result:.2f}s)"
            else:
                return session_num, False, f"Failed: {result}"
        
        def combine_all_sessions(self) -> Dict[str, any]:
            """
            Main function to combine all audio sessions with parallel processing
            """
            start_time = time.time()
            
            # Scan for audio files
            sessions = self.scan_audio_files()
            
            if not sessions:
                self.logger.warning("No audio files found to combine!")
                return {'success': False, 'message': 'No files found'}
            
            self.logger.info(f"Found {len(sessions)} sessions to combine")
            self.logger.info(f"Using {self.max_workers} parallel workers")
            self.logger.info(f"Output format: {self.output_format}")
            self.logger.info(f"Silence between parts: {self.silence_between_parts}s")
            
            # Prepare session data for parallel processing
            session_items = list(sessions.items())
            
            # Process sessions in parallel
            results = []
            files_to_delete = []
            
            # Use ThreadPoolExecutor for I/O bound operations
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(sessions))) as executor:
                session_results = executor.map(self._combine_session, session_items)
            
            # Process results
            successful_combines = 0
            for session_num, success, message in session_results:
                if success:
                    successful_combines += 1
                    self.logger.info(f"✓ Session {session_num}: {message}")
                    
                    # Mark files for deletion
                    if self.delete_original_parts:
                        files_to_delete.extend([info.path for info in sessions[session_num]])
                else:
                    self.logger.error(f"✗ Session {session_num}: {message}")
            
            # Delete original files if requested
            if self.delete_original_parts and files_to_delete:
                self._delete_files_parallel(files_to_delete)
            
            # Calculate final statistics
            processing_time = time.time() - start_time
            self.stats['processing_time'] = processing_time
            
            # Log final summary
            self.logger.info("=" * 60)
            self.logger.info("COMBINATION COMPLETE!")
            self.logger.info(f"Successfully combined: {successful_combines}/{len(sessions)} sessions")
            self.logger.info(f"Total files processed: {self.stats['total_files_processed']}")
            self.logger.info(f"Total duration combined: {self.stats['total_duration_combined']:.2f}s")
            self.logger.info(f"Processing time: {processing_time:.2f}s")
            self.logger.info(f"Average speed: {self.stats['total_duration_combined']/processing_time:.1f}x realtime")
            
            if self.delete_original_parts:
                self.logger.info(f"Files deleted: {self.stats['files_deleted']}")
            
            # List final combined files
            self._list_combined_files(sessions.keys())
            
            return {
                'success': True,
                'sessions_combined': successful_combines,
                'total_sessions': len(sessions),
                'processing_time': processing_time,
                'stats': self.stats
            }
        
        def _delete_files_parallel(self, file_paths: List[Path]):
            """Delete files in parallel"""
            self.logger.info(f"Deleting {len(file_paths)} original files...")
            
            def delete_file(file_path):
                try:
                    file_path.unlink()
                    return True, file_path.name
                except Exception as e:
                    return False, f"{file_path.name}: {e}"
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = executor.map(delete_file, file_paths)
            
            deleted_count = 0
            for success, message in results:
                if success:
                    deleted_count += 1
                    self.logger.debug(f"Deleted: {message}")
                else:
                    self.logger.warning(f"Failed to delete {message}")
            
            self.stats['files_deleted'] = deleted_count
            self.logger.info(f"✓ Deleted {deleted_count}/{len(file_paths)} files")
        
        def _list_combined_files(self, session_nums):
            """List information about combined files"""
            self.logger.info("\nFinal combined files:")
            
            def get_file_info(session_num):
                output_filename = f"{self.vid_id}_{session_num}_combined.{self.output_format}"
                output_path = self.audio_folder / output_filename
                
                if output_path.exists():
                    try:
                        with sf.SoundFile(output_path) as f:
                            duration = len(f) / f.samplerate
                        file_size = output_path.stat().st_size / (1024 * 1024)
                        return f"  {output_filename} - {duration:.2f}s, {file_size:.1f}MB"
                    except:
                        return f"  {output_filename} - Created successfully"
                return None
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                file_infos = executor.map(get_file_info, sorted(session_nums))
            
            for info in file_infos:
                if info:
                    self.logger.info(info)
        
        def preview_combination(self) -> Dict[str, any]:
            """
            Preview what files would be combined without actually combining them
            """
            self.logger.info("PREVIEW MODE - Files that would be combined:")
            self.logger.info("=" * 60)
            
            sessions = self.scan_audio_files()
            
            if not sessions:
                self.logger.warning("No audio files found!")
                return {'sessions': {}, 'total_sessions': 0, 'total_parts': 0}
            
            total_sessions = len(sessions)
            total_parts = sum(len(parts) for parts in sessions.values())
            
            self.logger.info(f"Found {total_sessions} sessions with {total_parts} total parts")
            self.logger.info(f"Output format will be: {self.output_format}")
            self.logger.info(f"Original parts will be {'DELETED' if self.delete_original_parts else 'KEPT'}")
            self.logger.info("-" * 60)
            
            preview_data = {}
            
            for session_num in sorted(sessions.keys()):
                parts = sessions[session_num]
                output_filename = f"{self.vid_id}_{session_num}_combined.{self.output_format}"
                
                self.logger.info(f"\nSession {session_num} -> {output_filename}")
                self.logger.info(f"  Will combine {len(parts)} parts:")
                
                total_duration = sum(info.duration for info in parts)
                
                for info in parts:
                    self.logger.info(f"    Part {info.part_num}: {info.path.name} ({info.duration:.2f}s)")
                
                if self.silence_between_parts > 0:
                    silence_total = self.silence_between_parts * (len(parts) - 1)
                    total_duration += silence_total
                    self.logger.info(f"  + {silence_total:.2f}s silence between parts")
                
                self.logger.info(f"  Total estimated duration: {total_duration:.2f}s")
                
                preview_data[session_num] = {
                    'parts': len(parts),
                    'duration': total_duration,
                    'output_filename': output_filename
                }
            
            return {
                'sessions': preview_data,
                'total_sessions': total_sessions,
                'total_parts': total_parts
            }
        
    class AudioDurationProcessor:
        """
        Optimized class for processing WAV files and updating database with audio duration data.
        
        Key optimizations:
        - Batch database operations for better performance
        - Connection pooling with context managers
        - Pre-compiled SQL statements
        - Efficient file filtering and processing
        - Better error handling and logging
        """
        
        def __init__(self, vid_id: str, log_level: int = logging.INFO):
            """
            Initialize the AudioDurationProcessor.
            
            Args:
                vid_id (str): The video ID used in file and database naming
                log_level (int): Logging level (default: INFO)
            """
            self.vid_id = vid_id
            self.db_path = Path(f"{vid_id}_longform.db")
            self.audio_folder = Path(f"{vid_id}_Audio")
            
            # Set up logging
            logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
            self.logger = logging.getLogger(__name__)
            
            # Pre-compile SQL statements for better performance
            self.sql_statements = {
                'add_column': "ALTER TABLE video_ideas ADD COLUMN total_images INTEGER",
                'update_images': "UPDATE video_ideas SET total_images = ? WHERE rowid = ?",
                'select_updated': "SELECT rowid, total_images FROM video_ideas WHERE total_images IS NOT NULL ORDER BY rowid",
                'check_column_exists': "PRAGMA table_info(video_ideas)"
            }
        
        @contextmanager
        def get_db_connection(self):
            """
            Context manager for database connections with optimized settings.
            
            Yields:
                sqlite3.Connection: Database connection with optimized pragmas
            """
            conn = None
            try:
                conn = sqlite3.connect(str(self.db_path))
                # Optimize SQLite for better performance
                conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
                conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
                conn.execute("PRAGMA cache_size=10000")    # Larger cache
                conn.execute("PRAGMA temp_store=MEMORY")   # Use memory for temp tables
                yield conn
            except sqlite3.Error as e:
                self.logger.error(f"Database connection error: {e}")
                if conn:
                    conn.rollback()
                raise
            finally:
                if conn:
                    conn.close()
        
        def _validate_setup(self) -> bool:
            """
            Validate that required files and folders exist.
            
            Returns:
                bool: True if setup is valid, False otherwise
            """
            if not self.vid_id:
                self.logger.error("VID_ID cannot be empty!")
                return False
            
            if not self.audio_folder.exists():
                self.logger.error(f"Audio folder '{self.audio_folder}' not found!")
                return False
            
            if not self.db_path.exists():
                self.logger.error(f"Database '{self.db_path}' not found!")
                return False
            
            return True
        
        def _ensure_column_exists(self, cursor: sqlite3.Cursor) -> None:
            """
            Ensure the total_images column exists in the database.
            
            Args:
                cursor (sqlite3.Cursor): Database cursor
            """
            # Check if column already exists
            cursor.execute(self.sql_statements['check_column_exists'])
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'total_images' not in columns:
                try:
                    cursor.execute(self.sql_statements['add_column'])
                    self.logger.info("Added total_images column to database")
                except sqlite3.OperationalError as e:
                    self.logger.warning(f"Could not add column: {e}")
            else:
                self.logger.debug("total_images column already exists")
        
        def _get_wav_files(self) -> List[Path]:
            """
            Get all WAV files matching the expected pattern.
            
            Returns:
                List[Path]: List of WAV file paths
            """
            pattern = f"{self.vid_id}_*_combined.wav"
            wav_files = list(self.audio_folder.glob(pattern))
            
            if not wav_files:
                self.logger.warning(f"No WAV files found matching pattern: {pattern}")
                return []
            
            # Sort files by index for consistent processing order
            def extract_index(file_path: Path) -> int:
                try:
                    parts = file_path.stem.split('_')
                    return int(parts[-2]) if len(parts) >= 3 else float('inf')
                except (ValueError, IndexError):
                    return float('inf')
            
            wav_files.sort(key=extract_index)
            self.logger.info(f"Found {len(wav_files)} WAV files to process")
            return wav_files
        
        def _extract_index_from_filename(self, filename: str) -> Optional[int]:
            """
            Extract index from WAV filename.
            
            Args:
                filename (str): The filename to parse
                
            Returns:
                Optional[int]: The extracted index, or None if parsing failed
            """
            try:
                parts = filename.replace('.wav', '').split('_')
                if len(parts) >= 3:
                    return int(parts[-2])
            except (ValueError, IndexError):
                pass
            
            self.logger.warning(f"Could not extract index from filename: {filename}")
            return None
        
        def _calculate_total_images(self, duration_minutes: float) -> int:
            """
            Calculate total_images value based on audio duration.
            
            Args:
                duration_minutes (float): Duration in minutes
                
            Returns:
                int: Calculated total_images value
            """
            # If duration is about 4 minutes, store minutes + 1
            if 3.5 <= duration_minutes <= 4.5:
                total_images = int(duration_minutes) + 1
                self.logger.debug(f"~4 minutes detected, total_images = {total_images}")
            else:
                # For other durations, use minutes + 1, minimum of 1
                total_images = max(1, int(duration_minutes) + 1)
                self.logger.debug(f"Not ~4 minutes, total_images = {total_images}")
            
            return total_images
        
        def _process_single_wav_file(self, wav_file: Path) -> Optional[Tuple[int, int]]:
            """
            Process a single WAV file and return the data for database update.
            
            Args:
                wav_file (Path): Path to the WAV file
                
            Returns:
                Optional[Tuple[int, int]]: (index, total_images) or None if processing failed
            """
            try:
                # Extract index from filename
                index = self._extract_index_from_filename(wav_file.name)
                if index is None:
                    return None
                
                # Get WAV duration using mutagen
                audio = WAVE(str(wav_file))
                duration_seconds = audio.info.length
                duration_minutes = duration_seconds / 60
                
                self.logger.debug(f"File: {wav_file.name}, Duration: {duration_minutes:.2f} minutes")
                
                # Calculate total_images value
                total_images = self._calculate_total_images(duration_minutes)
                
                return (index, total_images)
                
            except Exception as e:
                self.logger.error(f"Error processing {wav_file}: {e}")
                return None
        
        def _batch_update_database(self, update_data: List[Tuple[int, int]]) -> int:
            """
            Perform batch update of database records.
            
            Args:
                update_data (List[Tuple[int, int]]): List of (total_images, index) tuples
                
            Returns:
                int: Number of successfully updated records
            """
            if not update_data:
                return 0
            
            updated_count = 0
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Ensure column exists
                self._ensure_column_exists(cursor)
                
                # Prepare data for batch update (swap order for SQL parameters)
                batch_data = [(total_images, index) for index, total_images in update_data]
                
                # Execute batch update
                cursor.executemany(self.sql_statements['update_images'], batch_data)
                updated_count = cursor.rowcount
                
                # Commit all changes at once
                conn.commit()
                
                self.logger.info(f"Batch updated {updated_count} records in database")
                
            return updated_count
        
        def _display_results(self) -> None:
            """Display the updated records from the database."""
            try:
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(self.sql_statements['select_updated'])
                    results = cursor.fetchall()
                    
                    if results:
                        self.logger.info("Updated records:")
                        for row_id, total_images in results:
                            self.logger.info(f"  Row {row_id}: total_images = {total_images}")
                    else:
                        self.logger.info("No records were updated")
                        
            except sqlite3.Error as e:
                self.logger.error(f"Error displaying results: {e}")
        
        def process_audio_files(self) -> bool:
            """
            Main method to process all WAV files and update the database.
            
            Returns:
                bool: True if processing was successful, False otherwise
            """
            self.logger.info(f"Processing audio files for VID_ID: {self.vid_id}")
            
            # Validate setup
            if not self._validate_setup():
                return False
            
            # Get WAV files to process
            wav_files = self._get_wav_files()
            if not wav_files:
                return False
            
            # Process all WAV files and collect update data
            update_data = []
            processed_count = 0
            
            for wav_file in wav_files:
                result = self._process_single_wav_file(wav_file)
                if result:
                    update_data.append(result)
                    processed_count += 1
            
            if not update_data:
                self.logger.warning("No files were successfully processed")
                return False
            
            self.logger.info(f"Successfully processed {processed_count}/{len(wav_files)} files")
            
            # Batch update database
            try:
                updated_count = self._batch_update_database(update_data)
                
                if updated_count > 0:
                    self.logger.info("Database updated successfully!")
                    self._display_results()
                    return True
                else:
                    self.logger.warning("No database records were updated")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Failed to update database: {e}")
                return False

    class OptimizedImageGenerator:
        def __init__(self, vid_id: str):
            self.vid_id = vid_id
            self.db_path = f"{vid_id}_longform.db"
            self.api_key = "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"
            self.base_style_prompt = """A radiant, glowing human silhouette standing in the center, facing forward, body filled with glowing white neural pathways or light lines, radiating divine energy. The background is an abstract, golden parchment texture with spiritual and scientific drawings, including anatomical sketches, cryptic handwritten notes, Asian calligraphy, and esoteric symbols. Surrounding the figure are luminous beams of light that burst outward, creating a halo effect. The overall color palette is warm gold, amber, and sepia tones. The mood is mystical, sacred, and metaphysical, blending enlightenment themes with ancient knowledge and modern illustration."""
            
            # Create images directory
            self.images_dir = f"{vid_id}_Images"
            os.makedirs(self.images_dir, exist_ok=True)
            
            # Connection pool and rate limiting
            self.max_concurrent_requests = 50  # Increased from 20
            self.semaphore = None
            self.rate_limit_delay = 0.1  # Reduced from 60 seconds between batches
            
        def get_database_data(self) -> List[tuple]:
            """
            Retrieve rows from video_ideas table with total_images column
            Returns list of tuples: (rowid, total_images, [C1, C2, C3, ...])
            """
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get column names to identify C columns
                cursor.execute("PRAGMA table_info(video_ideas)")
                columns_info = cursor.fetchall()
                
                # Find C columns (C1, C2, C3, etc.)
                c_columns = []
                for col_info in columns_info:
                    col_name = col_info[1]  # Column name is at index 1
                    if col_name.startswith('C') and col_name[1:].isdigit():
                        c_columns.append(col_name)
                
                # Sort C columns numerically (C1, C2, C3, etc.)
                c_columns.sort(key=lambda x: int(x[1:]))
                
                if not c_columns:
                    print("No C columns found in video_ideas table!")
                    return []
                
                # Build query to get rowid, total_images, and all C columns
                c_columns_str = ", ".join(c_columns)
                query = f"SELECT rowid, total_images, {c_columns_str} FROM video_ideas WHERE total_images IS NOT NULL AND total_images > 0"
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                conn.close()
                return results
                
            except sqlite3.Error as e:
                print(f"Database error: {str(e)}")
                return []
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return []
        
        def get_random_theme_from_row(self, c_values: tuple) -> Optional[str]:
            """
            Randomly select a non-empty theme from the available C columns
            Returns the selected theme or None if no valid themes found
            """
            # Filter out None and empty string values
            valid_themes = [theme for theme in c_values if theme and str(theme).strip()]
            
            if not valid_themes:
                return None
            
            # Randomly select one theme
            selected_theme = random.choice(valid_themes)
            return str(selected_theme).strip()
        
        def create_enhanced_prompt(self, theme: str) -> str:
            """
            Create an enhanced prompt where the theme dominates the style
            The theme becomes the primary focus with style elements as supporting details
            """
            if not theme or theme.strip() == "":
                return self.base_style_prompt
            
            # Shorter, more efficient prompt
            enhanced_prompt = f"""{theme.strip()}. Radiant glowing human silhouette with neural pathways, golden parchment background with mystical symbols, warm gold/amber tones, sacred metaphysical mood."""
            
            return enhanced_prompt
        
        def clean_filename_text(self, text: str, max_length: int = 30) -> str:
            """
            Clean text to be safe for use in filenames (optimized version)
            """
            if not text:
                return "empty"
            
            # More efficient character replacement using translate
            invalid_chars = '<>:"/\\|?*\n\r\t\'"'
            translator = str.maketrans(invalid_chars, '_' * len(invalid_chars))
            clean_text = text.translate(translator)
            
            # Single regex operation for multiple replacements
            import re
            clean_text = re.sub(r'[_\s]+', '_', clean_text).strip('_.')
            
            return clean_text[:max_length] if clean_text else "theme"
        
        @asynccontextmanager
        async def get_session(self):
            """
            Context manager for aiohttp session with optimized settings
            """
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=50,  # Connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                keepalive_timeout=60,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(
                total=120,  # Reduced from 300
                connect=30,
                sock_read=30
            )
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Authorization": f"Bearer {self.api_key}"}
            ) as session:
                yield session

        async def generate_image_optimized(self, session: aiohttp.ClientSession, prompt: str, index: int, image_number: int, theme_used: str) -> Optional[str]:
            """
            Optimized image generation with semaphore control and async file I/O
            """
            async with self.semaphore:  # Control concurrency
                try:
                    url = "https://api.openai.com/v1/images/generations"
                    
                    data = {
                        "model": "gpt-image-1",
                        "prompt": prompt,
                        "n": 1,
                        "size": "1024x1024",
                        "quality": "low"  # Use 'low' for faster generation
                    }
                    
                    async with session.post(url, json=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            if 'data' not in result or not result['data']:
                                return None
                            
                            image_data = result['data'][0]
                            
                            if 'url' in image_data:
                                image_url = image_data['url']
                                
                                # Download image
                                async with session.get(image_url) as img_response:
                                    if img_response.status == 200:
                                        clean_theme = self.clean_filename_text(theme_used)
                                        filename = f"{self.vid_id}_{index}_{image_number}_theme-{clean_theme}.png"
                                        filepath = os.path.join(self.images_dir, filename)
                                        
                                        # Async file writing
                                        content = await img_response.read()
                                        async with aiofiles.open(filepath, 'wb') as f:
                                            await f.write(content)
                                        
                                        return filename
                            
                            elif 'b64_json' in image_data:
                                import base64
                                b64_data = image_data['b64_json']
                                clean_theme = self.clean_filename_text(theme_used)
                                filename = f"{self.vid_id}_{index}_{image_number}_theme-{clean_theme}.png"
                                filepath = os.path.join(self.images_dir, filename)
                                
                                # Async file writing for base64
                                decoded_data = base64.b64decode(b64_data)
                                async with aiofiles.open(filepath, "wb") as f:
                                    await f.write(decoded_data)
                                
                                return filename
                        
                        else:
                            print(f"API Error {response.status} for image {image_number}")
                            return None
                            
                except Exception as e:
                    print(f"Error generating image {image_number}: {str(e)}")
                    return None
        
        async def process_batch_optimized(self, session: aiohttp.ClientSession, tasks: List[tuple]) -> tuple:
            """
            Process batch with better error handling and progress tracking
            """
            print(f"Processing batch of {len(tasks)} images...")
            
            # Create tasks with progress tracking
            async_tasks = []
            for i, (prompt, index, image_number, theme_used) in enumerate(tasks):
                task = self.generate_image_optimized(session, prompt, index, image_number, theme_used)
                async_tasks.append(task)
            
            # Execute with progress updates
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
            failed = len(results) - successful
            
            return successful, failed
        
        async def process_all_rows_optimized(self):
            """
            Optimized processing with better concurrency control
            """
            print(f"Starting optimized image generation for VID_ID: {self.vid_id}")
            
            data = self.get_database_data()
            if not data:
                print("No data found to process!")
                return
            
            # Prepare all tasks
            all_tasks = []
            for row in data:
                row_id = row[0]
                total_images = row[1]
                c_values = row[2:]
                
                for i in range(total_images):
                    selected_theme = self.get_random_theme_from_row(c_values)
                    if selected_theme:
                        prompt = self.create_enhanced_prompt(selected_theme)
                        image_number = i + 1
                        all_tasks.append((prompt, row_id, image_number, selected_theme))
            
            print(f"Total tasks: {len(all_tasks)} images")
            
            if not all_tasks:
                return
            
            # Initialize semaphore for concurrency control
            self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            
            total_successful = 0
            total_failed = 0
            
            async with self.get_session() as session:
                # Process all tasks with controlled concurrency
                batch_size = 100  # Larger batches since we control concurrency with semaphore
                
                for i in range(0, len(all_tasks), batch_size):
                    batch = all_tasks[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(all_tasks) + batch_size - 1) // batch_size
                    
                    print(f"\n=== Processing Batch {batch_num}/{total_batches} ===")
                    
                    successful, failed = await self.process_batch_optimized(session, batch)
                    total_successful += successful
                    total_failed += failed
                    
                    print(f"Batch {batch_num} complete: {successful} successful, {failed} failed")
                    
                    # Much shorter delay between batches
                    if i + batch_size < len(all_tasks):
                        await asyncio.sleep(self.rate_limit_delay)
            
            print(f"\n=== Generation Complete ===")
            print(f"Successfully generated: {total_successful} images")
            print(f"Failed generations: {total_failed} images")

    class AudioLengthProcessor:
        """
        Optimized class for processing audio files and updating database with precise audio length data.
        
        Key optimizations:
        - Batch database operations for better performance
        - Connection pooling with context managers
        - Pre-compiled SQL statements
        - Efficient audio processing with librosa
        - Comprehensive error handling and logging
        - Memory-efficient audio loading
        """
        
        def __init__(self, vid_id: str, log_level: int = logging.INFO):
            """
            Initialize the AudioLengthProcessor.
            
            Args:
                vid_id (str): The video ID used in file and database naming
                log_level (int): Logging level (default: INFO)
            """
            self.vid_id = vid_id
            self.db_path = Path(f"{vid_id}_longform.db")
            self.audio_folder = Path(f"{vid_id}_Audio")
            
            # Set up logging
            logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
            self.logger = logging.getLogger(__name__)
            
            # Pre-compile SQL statements for better performance
            self.sql_statements = {
                'check_column_exists': "PRAGMA table_info(video_ideas)",
                'add_column': "ALTER TABLE video_ideas ADD COLUMN audio_length REAL",
                'get_all_rows': "SELECT rowid FROM video_ideas ORDER BY rowid",
                'update_length': "UPDATE video_ideas SET audio_length = ? WHERE rowid = ?",
                'verify_updates': "SELECT rowid, audio_length FROM video_ideas WHERE audio_length IS NOT NULL ORDER BY rowid",
                'get_stats': "SELECT COUNT(*), MIN(audio_length), MAX(audio_length), AVG(audio_length) FROM video_ideas WHERE audio_length IS NOT NULL"
            }
            
            # Cache for audio processing settings
            self.audio_cache = {}
        
        @contextmanager
        def get_db_connection(self):
            """
            Context manager for database connections with optimized settings.
            
            Yields:
                sqlite3.Connection: Database connection with optimized pragmas
            """
            conn = None
            try:
                conn = sqlite3.connect(str(self.db_path))
                # Optimize SQLite for better performance
                conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
                conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
                conn.execute("PRAGMA cache_size=10000")    # Larger cache
                conn.execute("PRAGMA temp_store=MEMORY")   # Use memory for temp tables
                yield conn
            except sqlite3.Error as e:
                self.logger.error(f"Database connection error: {e}")
                if conn:
                    conn.rollback()
                raise
            finally:
                if conn:
                    conn.close()
        
        def _validate_setup(self) -> bool:
            """
            Validate that required files and folders exist.
            
            Returns:
                bool: True if setup is valid, False otherwise
            """
            if not self.vid_id:
                self.logger.error("VID_ID cannot be empty!")
                return False
            
            if not self.audio_folder.exists():
                self.logger.error(f"Audio folder not found: {self.audio_folder}")
                return False
            
            if not self.db_path.exists():
                self.logger.error(f"Database not found: {self.db_path}")
                return False
            
            return True
        
        def _ensure_column_exists(self, cursor: sqlite3.Cursor) -> None:
            """
            Ensure the audio_length column exists in the database.
            
            Args:
                cursor (sqlite3.Cursor): Database cursor
            """
            # Check if column already exists
            cursor.execute(self.sql_statements['check_column_exists'])
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'audio_length' not in columns:
                try:
                    cursor.execute(self.sql_statements['add_column'])
                    self.logger.info("Added audio_length column to video_ideas table")
                except sqlite3.OperationalError as e:
                    self.logger.warning(f"Could not add column: {e}")
            else:
                self.logger.debug("audio_length column already exists")
        
        def _get_database_rows(self) -> List[int]:
            """
            Get all row IDs from the video_ideas table.
            
            Returns:
                List[int]: List of row IDs from the database
            """
            try:
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(self.sql_statements['get_all_rows'])
                    rows = cursor.fetchall()
                    return [row[0] for row in rows]
            except sqlite3.Error as e:
                self.logger.error(f"Error getting database rows: {e}")
                return []
        
        def _get_audio_length_optimized(self, audio_file_path: Path) -> Optional[float]:
            """
            Get the length of an audio file in seconds using optimized librosa loading.
            
            Args:
                audio_file_path (Path): Path to the audio file
                
            Returns:
                Optional[float]: Length of audio in seconds, or None if error occurs
            """
            try:
                if not audio_file_path.exists():
                    self.logger.debug(f"Audio file not found: {audio_file_path}")
                    return None
                
                # Check cache first (if we've processed this file before)
                file_key = str(audio_file_path)
                if file_key in self.audio_cache:
                    return self.audio_cache[file_key]
                
                # Load audio file with optimized settings
                # duration_only=True for faster loading when we only need duration
                try:
                    # First try to get duration without loading the entire file
                    duration = librosa.get_duration(path=str(audio_file_path))
                    
                    # Cache the result
                    self.audio_cache[file_key] = duration
                    return duration
                    
                except Exception:
                    # Fallback to loading the file if direct duration fails
                    y, sr = librosa.load(str(audio_file_path), sr=None, mono=True)
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    # Cache the result
                    self.audio_cache[file_key] = duration
                    return duration
            
            except Exception as e:
                self.logger.warning(f"Error reading audio file {audio_file_path}: {e}")
                return None
        
        def _process_audio_files_batch(self, row_ids: List[int]) -> List[Tuple[float, int]]:
            """
            Process multiple audio files and return batch data for database update.
            
            Args:
                row_ids (List[int]): List of database row IDs to process
                
            Returns:
                List[Tuple[float, int]]: List of (audio_length, row_id) tuples for batch update
            """
            batch_data = []
            processed_count = 0
            
            self.logger.info(f"Processing {len(row_ids)} audio files...")
            
            for i, table_index in enumerate(row_ids, 1):
                # Construct audio file path
                audio_filename = f"{self.vid_id}_{table_index}_combined.wav"
                audio_file_path = self.audio_folder / audio_filename
                
                # Get audio length
                audio_length = self._get_audio_length_optimized(audio_file_path)
                
                if audio_length is not None:
                    batch_data.append((audio_length, table_index))
                    processed_count += 1
                    self.logger.debug(f"Processed index {table_index}: {audio_length:.2f} seconds")
                else:
                    self.logger.debug(f"Skipped index {table_index}: audio file not found or error occurred")
                
                # Progress logging for large batches
                if i % 10 == 0 or i == len(row_ids):
                    self.logger.info(f"Progress: {i}/{len(row_ids)} files checked, {processed_count} processed")
            
            return batch_data
        
        def _batch_update_database(self, batch_data: List[Tuple[float, int]]) -> int:
            """
            Perform batch update of database records with audio lengths.
            
            Args:
                batch_data (List[Tuple[float, int]]): List of (audio_length, row_id) tuples
                
            Returns:
                int: Number of successfully updated records
            """
            if not batch_data:
                return 0
            
            try:
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Ensure column exists
                    self._ensure_column_exists(cursor)
                    
                    # Execute batch update
                    cursor.executemany(self.sql_statements['update_length'], batch_data)
                    updated_count = cursor.rowcount
                    
                    # Commit all changes at once
                    conn.commit()
                    
                    self.logger.info(f"Successfully updated {updated_count} records with audio lengths")
                    return updated_count
                    
            except sqlite3.Error as e:
                self.logger.error(f"Database error during batch update: {e}")
                return 0
        
        def _get_processing_statistics(self) -> Dict[str, float]:
            """
            Get statistics about the processed audio lengths.
            
            Returns:
                Dict[str, float]: Dictionary containing processing statistics
            """
            try:
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(self.sql_statements['get_stats'])
                    result = cursor.fetchone()
                    
                    if result and result[0] > 0:
                        return {
                            'count': result[0],
                            'min_length': result[1],
                            'max_length': result[2],
                            'avg_length': result[3]
                        }
                    return {'count': 0}
                    
            except sqlite3.Error as e:
                self.logger.error(f"Error getting statistics: {e}")
                return {'count': 0}
        
        def verify_updates(self) -> bool:
            """
            Verify that the audio lengths were properly stored in the database.
            
            Returns:
                bool: True if verification successful, False otherwise
            """
            try:
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(self.sql_statements['verify_updates'])
                    results = cursor.fetchall()
                    
                    if results:
                        self.logger.info(f"Verification - Found {len(results)} records with audio lengths:")
                        
                        # Show first few and last few records, or all if small dataset
                        if len(results) <= 10:
                            for rowid, length in results:
                                self.logger.info(f"  Index {rowid}: {length:.2f} seconds")
                        else:
                            # Show first 3 and last 3
                            for rowid, length in results[:3]:
                                self.logger.info(f"  Index {rowid}: {length:.2f} seconds")
                            self.logger.info(f"  ... ({len(results) - 6} more records) ...")
                            for rowid, length in results[-3:]:
                                self.logger.info(f"  Index {rowid}: {length:.2f} seconds")
                        
                        # Show statistics
                        stats = self._get_processing_statistics()
                        if stats['count'] > 0:
                            self.logger.info(f"Statistics:")
                            self.logger.info(f"  Total records: {int(stats['count'])}")
                            self.logger.info(f"  Min length: {stats['min_length']:.2f} seconds")
                            self.logger.info(f"  Max length: {stats['max_length']:.2f} seconds")
                            self.logger.info(f"  Average length: {stats['avg_length']:.2f} seconds")
                        
                        return True
                    else:
                        self.logger.warning("No records found with audio lengths")
                        return False
                        
            except sqlite3.Error as e:
                self.logger.error(f"Database error during verification: {e}")
                return False
        
        def process_audio_lengths(self) -> bool:
            """
            Main method to process all audio files and update the database with lengths.
            
            Returns:
                bool: True if processing was successful, False otherwise
            """
            self.logger.info(f"Starting audio length update process for VID_ID: {self.vid_id}")
            self.logger.info(f"Looking for audio folder: {self.audio_folder}")
            self.logger.info(f"Database file: {self.db_path}")
            self.logger.info("-" * 50)
            
            # Validate setup
            if not self._validate_setup():
                return False
            
            # Get all database rows
            row_ids = self._get_database_rows()
            if not row_ids:
                self.logger.error("No rows found in database")
                return False
            
            self.logger.info(f"Found {len(row_ids)} rows in database to process")
            
            # Process audio files in batch
            batch_data = self._process_audio_files_batch(row_ids)
            
            if not batch_data:
                self.logger.warning("No audio files were successfully processed")
                return False
            
            # Batch update database
            updated_count = self._batch_update_database(batch_data)
            
            if updated_count > 0:
                self.logger.info("-" * 50)
                success = self.verify_updates()
                self.logger.info("-" * 50)
                self.logger.info("Process completed successfully!")
                return success
            else:
                self.logger.error("Failed to update database")
                return False
        
        def clear_cache(self) -> None:
            """Clear the audio processing cache to free memory."""
            self.audio_cache.clear()
            self.logger.debug("Audio cache cleared")
        
        def get_cache_info(self) -> Dict[str, int]:
            """
            Get information about the current cache state.
            
            Returns:
                Dict[str, int]: Cache statistics
            """
            return {
                'cached_files': len(self.audio_cache),
                'memory_usage_estimate': len(self.audio_cache) * 64  # Rough estimate in bytes
            }

    class OptimizedVideoCreator:
        """
        High-performance video creator with parallel processing and memory optimization.
        """
        
        def __init__(self, vid_id: str, output_width: int = 1920, output_height: int = 1080, 
                    fps: int = 30, fade_duration: float = 0.5, zoom_factor: float = 1.2,
                    max_workers: Optional[int] = None):
            """
            Initialize the video creator with optimized settings.
            
            Args:
                vid_id: Video ID identifier
                output_width: Output video width
                output_height: Output video height
                fps: Frames per second
                fade_duration: Fade in/out duration in seconds
                zoom_factor: Zoom effect factor
                max_workers: Maximum number of parallel workers (None for auto-detect)
            """
            self.vid_id = vid_id
            self.output_width = output_width
            self.output_height = output_height
            self.fps = fps
            self.fade_duration = fade_duration
            self.zoom_factor = zoom_factor
            self.fade_frames = int(fade_duration * fps)
            
            # Optimize worker count based on CPU cores
            if max_workers is None:
                self.max_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
            else:
                self.max_workers = max_workers
                
            # Pre-calculate common values
            self.target_size = (output_width, output_height)
            
            # Paths
            self.images_folder = f"{vid_id}_Images"
            self.output_folder = f"{vid_id}_Videos"
            self.database_path = f"{vid_id}_longform.db"
            
            print(f"Initialized OptimizedVideoCreator with {self.max_workers} workers")
        
        def get_audio_lengths_from_db(self) -> Dict[int, float]:
            """
            Get audio lengths for all indices from the database.
            
            Returns:
                Dictionary mapping index to audio length
            """
            audio_lengths = {}
            
            try:
                with sqlite3.connect(self.database_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT rowid, audio_length FROM video_ideas WHERE audio_length IS NOT NULL")
                    results = cursor.fetchall()
                    
                    for rowid, length in results:
                        audio_lengths[rowid] = length
                        
                print(f"Retrieved audio lengths for {len(audio_lengths)} indices")
                
            except sqlite3.Error as e:
                print(f"Database error: {e}")
            
            return audio_lengths
        
        def get_images_for_index(self, index: int) -> List[str]:
            """
            Get all images for a specific index, sorted by photo number.
            
            Args:
                index: Table index
                
            Returns:
                Sorted list of image file paths
            """
            pattern = f"{self.vid_id}_{index}_*.png"
            search_path = os.path.join(self.images_folder, pattern)
            image_files = glob.glob(search_path)
            
            # Sort by photo number using regex
            def extract_photo_number(filename):
                match = re.search(rf'{self.vid_id}_{index}_(\d+)-', filename)
                return int(match.group(1)) if match else 0
            
            image_files.sort(key=extract_photo_number)
            return image_files
        
        @staticmethod
        def resize_and_center_image_optimized(image_path: str, target_width: int, target_height: int) -> np.ndarray:
            """
            Optimized image resizing with OpenCV for better performance.
            
            Args:
                image_path: Path to image file
                target_width: Target width
                target_height: Target height
                
            Returns:
                Resized and centered image as OpenCV array
            """
            # Load image directly with OpenCV for better performance
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            img_height, img_width = image.shape[:2]
            
            # Calculate scaling to fit within target dimensions
            img_ratio = img_width / img_height
            target_ratio = target_width / target_height
            
            if img_ratio > target_ratio:
                # Image is wider relative to target
                new_width = target_width
                new_height = int(target_width / img_ratio)
            else:
                # Image is taller relative to target
                new_height = target_height
                new_width = int(target_height * img_ratio)
            
            # Resize image using OpenCV (faster than PIL)
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Create black background and center the image
            background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
            
            return background
        
        def create_zoom_and_fade_frame(self, base_image: np.ndarray, zoom_progress: float, fade_alpha: float) -> np.ndarray:
            """
            Create a single frame with zoom and fade effects applied.
            Optimized to work directly with OpenCV arrays.
            
            Args:
                base_image: OpenCV image array
                zoom_progress: Progress from 0.0 to 1.0
                fade_alpha: Fade level from 0.0 to 1.0
                
            Returns:
                Processed frame
            """
            # Calculate current zoom level (starts at zoom_factor, ends at 1.0)
            current_zoom = self.zoom_factor - (self.zoom_factor - 1.0) * zoom_progress
            
            # Get image dimensions
            height, width = base_image.shape[:2]
            
            # Calculate new dimensions for zoom
            new_width = int(width * current_zoom)
            new_height = int(height * current_zoom)
            
            # Resize for zoom effect
            zoomed = cv2.resize(base_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Crop to original size from center
            start_x = (new_width - width) // 2
            start_y = (new_height - height) // 2
            cropped = zoomed[start_y:start_y + height, start_x:start_x + width]
            
            # Apply fade effect efficiently
            if fade_alpha < 1.0:
                # Use in-place multiplication for better performance
                return (cropped.astype(np.float32) * fade_alpha).astype(np.uint8)
            else:
                return cropped
        
        def process_image_batch(self, image_paths: List[str]) -> List[np.ndarray]:
            """
            Process a batch of images in parallel for better performance.
            
            Args:
                image_paths: List of image file paths
                
            Returns:
                List of processed images
            """
            with ThreadPoolExecutor(max_workers=min(4, len(image_paths))) as executor:
                futures = [
                    executor.submit(self.resize_and_center_image_optimized, img_path, 
                                self.output_width, self.output_height)
                    for img_path in image_paths
                ]
                
                processed_images = []
                for i, future in enumerate(futures):
                    try:
                        processed_images.append(future.result())
                        print(f"    Loaded image {i + 1}/{len(image_paths)}")
                    except Exception as e:
                        print(f"    Error loading image {i + 1}: {e}")
                        # Create a black frame as fallback
                        processed_images.append(np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8))
            
            return processed_images
        
        def create_video_for_index(self, index: int, audio_length: float) -> Optional[str]:
            """
            Create a video for a specific index using its images and audio length.
            Highly optimized version with parallel processing.
            
            Args:
                index: Table index
                audio_length: Length of audio in seconds
                
            Returns:
                Path to created video file, or None if failed
            """
            start_time = time.time()
            
            # Get images for this index
            image_files = self.get_images_for_index(index)
            
            if not image_files:
                print(f"No images found for index {index}")
                return None
            
            print(f"Processing index {index}: {len(image_files)} images, {audio_length:.2f}s audio")
            
            # Calculate timing
            total_images = len(image_files)
            image_duration = audio_length / total_images
            frames_per_image = int(image_duration * self.fps)
            total_frames = frames_per_image * total_images
            
            print(f"  Will generate {total_frames:,} frames ({frames_per_image} per image)")
            
            # Output video path
            output_path = os.path.join(self.output_folder, f"{self.vid_id}_{index}_video.mp4")
            
            # Use x264 codec for better compression and compatibility
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, self.target_size)
            
            if not out.isOpened():
                print(f"Failed to open video writer for index {index}")
                return None
            
            try:
                # Process all images in parallel batches
                print(f"  Pre-processing {total_images} images...")
                processed_images = self.process_image_batch(image_files)
                
                print(f"  Generating video frames...")
                frames_written = 0
                
                # Pre-allocate arrays for better performance
                progress_points = [total_frames // 10 * i for i in range(1, 11)]
                
                for img_idx, base_image in enumerate(processed_images):
                    print(f"  Processing image {img_idx + 1}/{total_images}")
                    
                    # Generate frames for this image
                    for frame_idx in range(frames_per_image):
                        # Calculate progress through this image (0.0 to 1.0)
                        progress = frame_idx / frames_per_image if frames_per_image > 1 else 0.0
                        
                        # Calculate fade alpha
                        fade_alpha = 1.0
                        
                        # Fade in at the beginning
                        if frame_idx < self.fade_frames:
                            fade_alpha = frame_idx / self.fade_frames if self.fade_frames > 0 else 1.0
                        
                        # Fade out at the end
                        elif frame_idx >= frames_per_image - self.fade_frames:
                            fade_alpha = (frames_per_image - frame_idx) / self.fade_frames if self.fade_frames > 0 else 1.0
                        
                        # Create frame with effects
                        frame = self.create_zoom_and_fade_frame(base_image, progress, fade_alpha)
                        
                        # Write frame
                        out.write(frame)
                        frames_written += 1
                        
                        # Progress update at specific points
                        if frames_written in progress_points:
                            percent = (frames_written / total_frames) * 100
                            elapsed = time.time() - start_time
                            fps_processing = frames_written / elapsed if elapsed > 0 else 0
                            print(f"    Progress: {percent:.1f}% ({frames_written:,}/{total_frames:,} frames) - {elapsed:.1f}s elapsed - {fps_processing:.1f} fps")
                
                processing_time = time.time() - start_time
                print(f"  Video created in {processing_time:.1f}s: {output_path}")
                print(f"  Performance: {frames_written/processing_time:.1f} fps processing speed")
                return output_path
                
            except Exception as e:
                print(f"Error creating video for index {index}: {e}")
                return None
            finally:
                out.release()
        
        def process_index_wrapper(self, args: Tuple[int, float]) -> Tuple[int, Optional[str]]:
            """
            Wrapper function for parallel processing of indices.
            
            Args:
                args: Tuple of (index, audio_length)
                
            Returns:
                Tuple of (index, video_path or None)
            """
            index, audio_length = args
            video_path = self.create_video_for_index(index, audio_length)
            return index, video_path
        
        def process_all_videos(self, parallel: bool = True) -> Tuple[List[Tuple[int, str]], List[int]]:
            """
            Process all indices to create videos with optional parallel processing.
            
            Args:
                parallel: Whether to use parallel processing for multiple videos
                
            Returns:
                Tuple of (successful_videos, failed_indices)
            """
            # Check if images folder exists
            if not os.path.exists(self.images_folder):
                print(f"Images folder not found: {self.images_folder}")
                return [], []
            
            # Create output folder
            os.makedirs(self.output_folder, exist_ok=True)
            
            # Get audio lengths from database
            audio_lengths = self.get_audio_lengths_from_db()
            
            if not audio_lengths:
                print("No audio lengths found in database")
                return [], []
            
            # Process indices
            successful_videos = []
            failed_videos = []
            total_start_time = time.time()
            
            if parallel and len(audio_lengths) > 1:
                # Parallel processing for multiple videos
                print(f"Processing {len(audio_lengths)} videos in parallel with {min(self.max_workers, len(audio_lengths))} workers")
                
                with ProcessPoolExecutor(max_workers=min(self.max_workers, len(audio_lengths))) as executor:
                    # Submit all tasks
                    future_to_index = {
                        executor.submit(self.create_video_for_index, index, audio_length): index
                        for index, audio_length in audio_lengths.items()
                    }
                    
                    # Collect results
                    for future in future_to_index:
                        index = future_to_index[future]
                        try:
                            video_path = future.result()
                            if video_path:
                                successful_videos.append((index, video_path))
                            else:
                                failed_videos.append(index)
                        except Exception as e:
                            print(f"Error processing index {index}: {e}")
                            failed_videos.append(index)
            else:
                # Sequential processing
                for index, audio_length in audio_lengths.items():
                    print(f"\nProcessing index {index} (audio length: {audio_length:.2f}s)")
                    
                    video_path = self.create_video_for_index(index, audio_length)
                    
                    if video_path:
                        successful_videos.append((index, video_path))
                    else:
                        failed_videos.append(index)
            
            total_time = time.time() - total_start_time
            
            # Summary
            print(f"\n{'='*50}")
            print(f"PROCESSING COMPLETE - Total time: {total_time:.1f}s")
            print(f"{'='*50}")
            print(f"Successful videos: {len(successful_videos)}")
            print(f"Failed videos: {len(failed_videos)}")
            
            if successful_videos:
                print(f"\nSuccessful videos:")
                for index, path in successful_videos:
                    print(f"  Index {index}: {path}")
            
            if failed_videos:
                print(f"\nFailed indices: {failed_videos}")
            
            return successful_videos, failed_videos
        
        def run(self, parallel: bool = True):
            """
            Main execution method.
            
            Args:
                parallel: Whether to use parallel processing
            """
            print(f"Starting OPTIMIZED video creation process for VID_ID: {self.vid_id}")
            print(f"Looking for images in: {self.images_folder}")
            print(f"Output folder: {self.output_folder}")
            print(f"Video settings: {self.output_width}x{self.output_height} at {self.fps} FPS")
            print(f"Fade duration: {self.fade_duration}s, Zoom factor: {self.zoom_factor}x")
            print(f"Parallel processing: {'Enabled' if parallel else 'Disabled'}")
            print("-" * 50)
            
            successful_videos, failed_videos = self.process_all_videos(parallel)
            
            print("-" * 50)
            print("Video creation process completed!")
            
            return successful_videos, failed_videos
        
    @dataclass
    class CombinationResult:
        """Data class to store combination results."""
        index: int
        success: bool
        output_path: Optional[str] = None
        error_message: Optional[str] = None
        file_size_mb: Optional[float] = None
        processing_time: Optional[float] = None

    class AudioVideoCombiner:
        """
        Optimized class for combining audio and video files using FFmpeg.
        Features parallel processing, better error handling, and performance optimizations.
        """
        
        def __init__(self, vid_id: str, max_workers: int = None, timeout: int = 300):
            """
            Initialize the AudioVideoCombiner.
            
            Args:
                vid_id (str): Video ID
                max_workers (int): Maximum number of parallel workers (default: CPU count)
                timeout (int): Timeout for FFmpeg operations in seconds (default: 300)
            """
            self.vid_id = vid_id
            self.max_workers = max_workers or os.cpu_count()
            self.timeout = timeout
            
            # Define folder paths
            self.audio_folder = Path(f"{vid_id}_Audio")
            self.video_folder = Path(f"{vid_id}_Videos")
            self.output_folder = Path(f"{vid_id}_Final")
            
            # Setup logging
            self._setup_logging()
            
            # Cache for file existence checks
            self._audio_files_cache = None
            self._video_files_cache = None
        
        def _setup_logging(self):
            """Setup logging configuration."""
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(f'{self.vid_id}_combination.log')
                ]
            )
            self.logger = logging.getLogger(__name__)
        
        def _get_audio_files(self) -> Set[int]:
            """
            Get available audio file indices with caching.
            
            Returns:
                Set[int]: Set of audio file indices
            """
            if self._audio_files_cache is None:
                audio_pattern = self.audio_folder / f"{self.vid_id}_*_combined.wav"
                audio_files = glob.glob(str(audio_pattern))
                
                self._audio_files_cache = set()
                for audio_file in audio_files:
                    try:
                        filename = Path(audio_file).stem
                        # Extract index from filename like "VID_ID_123_combined"
                        index = int(filename.split('_')[1])
                        self._audio_files_cache.add(index)
                    except (IndexError, ValueError) as e:
                        self.logger.warning(f"Could not parse audio file {audio_file}: {e}")
            
            return self._audio_files_cache
        
        def _get_video_files(self) -> Set[int]:
            """
            Get available video file indices with caching.
            
            Returns:
                Set[int]: Set of video file indices
            """
            if self._video_files_cache is None:
                video_pattern = self.video_folder / f"{self.vid_id}_*_video.mp4"
                video_files = glob.glob(str(video_pattern))
                
                self._video_files_cache = set()
                for video_file in video_files:
                    try:
                        filename = Path(video_file).stem
                        # Extract index from filename like "VID_ID_123_video"
                        index = int(filename.split('_')[1])
                        self._video_files_cache.add(index)
                    except (IndexError, ValueError) as e:
                        self.logger.warning(f"Could not parse video file {video_file}: {e}")
            
            return self._video_files_cache
        
        def get_available_indices(self) -> List[int]:
            """
            Get indices that have both audio and video files available.
            
            Returns:
                List[int]: Sorted list of indices that have both audio and video files
            """
            audio_indices = self._get_audio_files()
            video_indices = self._get_video_files()
            
            # Find intersection - indices that have both audio and video
            available_indices = list(audio_indices.intersection(video_indices))
            available_indices.sort()
            
            self.logger.info(f"Found audio files for indices: {sorted(audio_indices)}")
            self.logger.info(f"Found video files for indices: {sorted(video_indices)}")
            self.logger.info(f"Available for combining: {available_indices}")
            
            return available_indices
        
        def _combine_single_file(self, index: int) -> CombinationResult:
            """
            Combine audio and video files for a specific index.
            
            Args:
                index (int): Index number
                
            Returns:
                CombinationResult: Result of the combination operation
            """
            start_time = time.time()
            
            # Define file paths
            audio_file = self.audio_folder / f"{self.vid_id}_{index}_combined.wav"
            video_file = self.video_folder / f"{self.vid_id}_{index}_video.mp4"
            output_file = self.output_folder / f"{self.vid_id}_{index}_final.mp4"
            
            # Check if input files exist
            if not audio_file.exists():
                return CombinationResult(
                    index=index,
                    success=False,
                    error_message=f"Audio file not found: {audio_file}"
                )
            
            if not video_file.exists():
                return CombinationResult(
                    index=index,
                    success=False,
                    error_message=f"Video file not found: {video_file}"
                )
            
            self.logger.info(f"Processing index {index}: {audio_file.name} + {video_file.name}")
            
            try:
                # Optimized FFmpeg command
                cmd = [
                    'ffmpeg',
                    '-i', str(video_file),
                    '-i', str(audio_file),
                    '-c:v', 'copy',  # Copy video stream (no re-encoding)
                    '-c:a', 'aac',   # AAC audio codec
                    '-b:a', '128k',  # Audio bitrate
                    '-movflags', '+faststart',  # Optimize for streaming
                    '-shortest',     # Finish when shortest input ends
                    '-y',           # Overwrite output file
                    str(output_file)
                ]
                
                # Run FFmpeg with optimized settings
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False  # Don't raise exception on non-zero return code
                )
                
                processing_time = time.time() - start_time
                
                if result.returncode == 0 and output_file.exists():
                    file_size_mb = output_file.stat().st_size / (1024 * 1024)
                    return CombinationResult(
                        index=index,
                        success=True,
                        output_path=str(output_file),
                        file_size_mb=file_size_mb,
                        processing_time=processing_time
                    )
                else:
                    return CombinationResult(
                        index=index,
                        success=False,
                        error_message=f"FFmpeg error (code {result.returncode}): {result.stderr}",
                        processing_time=processing_time
                    )
                    
            except subprocess.TimeoutExpired:
                return CombinationResult(
                    index=index,
                    success=False,
                    error_message=f"Timeout: FFmpeg took longer than {self.timeout} seconds",
                    processing_time=time.time() - start_time
                )
            except Exception as e:
                return CombinationResult(
                    index=index,
                    success=False,
                    error_message=f"Unexpected error: {str(e)}",
                    processing_time=time.time() - start_time
                )
        
        def check_ffmpeg(self) -> bool:
            """
            Check if FFmpeg is available and working.
            
            Returns:
                bool: True if FFmpeg is available, False otherwise
            """
            try:
                result = subprocess.run(
                    ['ffmpeg', '-version'], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                return result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return False
        
        def validate_environment(self) -> bool:
            """
            Validate that the environment is ready for processing.
            
            Returns:
                bool: True if environment is valid, False otherwise
            """
            # Check FFmpeg availability
            if not self.check_ffmpeg():
                self.logger.error("FFmpeg is not installed or not in PATH")
                self.logger.error("Please install FFmpeg from https://ffmpeg.org/download.html")
                return False
            
            # Check if required folders exist
            if not self.audio_folder.exists():
                self.logger.error(f"Audio folder not found: {self.audio_folder}")
                return False
            
            if not self.video_folder.exists():
                self.logger.error(f"Video folder not found: {self.video_folder}")
                return False
            
            # Create output folder
            self.output_folder.mkdir(exist_ok=True)
            
            return True
        
        def process_all_combinations(self) -> Tuple[List[CombinationResult], List[CombinationResult]]:
            """
            Process all available audio-video combinations using parallel processing.
            
            Returns:
                Tuple[List[CombinationResult], List[CombinationResult]]: 
                (successful_results, failed_results)
            """
            start_time = time.time()
            
            # Validate environment
            if not self.validate_environment():
                return [], []
            
            # Get available indices
            available_indices = self.get_available_indices()
            
            if not available_indices:
                self.logger.warning("No matching audio-video pairs found!")
                return [], []
            
            self.logger.info(f"Starting parallel processing with {self.max_workers} workers")
            self.logger.info(f"Processing {len(available_indices)} combinations")
            
            # Process combinations in parallel
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs
                future_to_index = {
                    executor.submit(self._combine_single_file, index): index 
                    for index in available_indices
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_index):
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result.success:
                            self.logger.info(
                                f"✓ Index {result.index}: {result.file_size_mb:.1f} MB "
                                f"({result.processing_time:.1f}s)"
                            )
                        else:
                            self.logger.error(f"✗ Index {result.index}: {result.error_message}")
                            
                    except Exception as e:
                        index = future_to_index[future]
                        self.logger.error(f"✗ Index {index}: Unexpected error: {e}")
                        results.append(CombinationResult(
                            index=index,
                            success=False,
                            error_message=f"Unexpected error: {str(e)}"
                        ))
            
            # Separate successful and failed results
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            # Log summary
            total_time = time.time() - start_time
            self._log_summary(successful_results, failed_results, total_time)
            
            return successful_results, failed_results
        
        def _log_summary(self, successful_results: List[CombinationResult], 
                        failed_results: List[CombinationResult], total_time: float):
            """
            Log processing summary.
            
            Args:
                successful_results: List of successful results
                failed_results: List of failed results
                total_time: Total processing time
            """
            self.logger.info("=" * 60)
            self.logger.info("AUDIO-VIDEO COMBINATION COMPLETE")
            self.logger.info("=" * 60)
            self.logger.info(f"Total processing time: {total_time:.1f} seconds")
            self.logger.info(f"Successful combinations: {len(successful_results)}")
            self.logger.info(f"Failed combinations: {len(failed_results)}")
            
            if successful_results:
                total_size = sum(r.file_size_mb for r in successful_results)
                avg_time = sum(r.processing_time for r in successful_results) / len(successful_results)
                
                self.logger.info(f"\nSuccessful final videos:")
                self.logger.info(f"Total size: {total_size:.1f} MB")
                self.logger.info(f"Average processing time: {avg_time:.1f} seconds")
                
                for result in sorted(successful_results, key=lambda x: x.index):
                    self.logger.info(
                        f"  Index {result.index}: {Path(result.output_path).name} "
                        f"({result.file_size_mb:.1f} MB, {result.processing_time:.1f}s)"
                    )
            
            if failed_results:
                failed_indices = [r.index for r in failed_results]
                self.logger.error(f"\nFailed indices: {failed_indices}")
                
                for result in sorted(failed_results, key=lambda x: x.index):
                    self.logger.error(f"  Index {result.index}: {result.error_message}")
            
            self.logger.info(f"\nFinal videos saved in: {self.output_folder}")
        
        def get_processing_stats(self) -> dict:
            """
            Get processing statistics.
            
            Returns:
                dict: Processing statistics
            """
            audio_count = len(self._get_audio_files())
            video_count = len(self._get_video_files())
            available_count = len(self.get_available_indices())
            
            return {
                'vid_id': self.vid_id,
                'audio_files_count': audio_count,
                'video_files_count': video_count,
                'available_combinations': available_count,
                'max_workers': self.max_workers,
                'timeout': self.timeout,
                'audio_folder': str(self.audio_folder),
                'video_folder': str(self.video_folder),
                'output_folder': str(self.output_folder)
            }
        

    class VideoCaptionGenerator:
        """
        Generates dynamic, karaoke-style captions with a clean shadow effect for videos.
        Includes optimizations for parallel processing to improve speed.
        """

        def __init__(self, vid_id: str, model_size: str = "base", max_workers: Optional[int] = None):
            """
            Initialize the caption generator.

            Args:
                vid_id: The base ID for your video folders.
                model_size: The size of the Whisper model to use for transcription.
                max_workers: Max threads for parallel processing. Defaults to CPU count.
            """
            self.vid_id = vid_id
            self.model_size = model_size
            self.max_workers = max_workers or mp.cpu_count()

            # Setup directories
            self.input_folder = f"{self.vid_id}_Final"
            self.output_folder = f"{self.vid_id}_Captioned"
            os.makedirs(self.output_folder, exist_ok=True)

            # Load Whisper model once
            print(f"Loading Whisper model: {self.model_size}...")
            self.model = whisper.load_model(model_size)
            print("Whisper model loaded successfully.")

            # --- Concurrency Lock ---
            # Add a lock to ensure the Whisper model is only used by one thread at a time.
            self.transcription_lock = threading.Lock()

            # --- Caption Configuration ---
            self.chunk_size = 5
            self.video_extensions = ('.mp4', '.mov', '.avi', '.mkv')
            self.text_config = {
                'fontsize': 40,
                'color': 'white',
                'font': 'Arial-Bold',
                'align': 'center'
            }
            self.highlight_color = "yellow"

        def _clear_gpu_cache(self):
            """Clears GPU cache if PyTorch is available."""
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            finally:
                gc.collect()

        def _get_video_files(self) -> List[str]:
            """Get all video files from the input folder."""
            print(f"Searching for videos in: {self.input_folder}")
            if not os.path.isdir(self.input_folder):
                raise FileNotFoundError(
                    f"Input folder not found: '{self.input_folder}'. "
                    f"Please ensure it exists and contains your video files."
                )
            
            files = [f for f in os.listdir(self.input_folder)
                    if f.lower().endswith(self.video_extensions)]
            
            if not files:
                raise FileNotFoundError(f"No video files found in '{self.input_folder}'")
            
            return sorted(files)

        def _create_text_clip_with_shadow(self, caption_markup: str, video_width: int, video_height: int,
                                        duration: float, start_time: float) -> List[TextClip]:
            """
            Creates a text clip with a drop shadow using Pango for consistent layout.
            """
            clip_width = int(video_width * 0.9)
            shadow_offset = 3
            # Position the captions in the vertical center of the screen.
            vertical_position = int(video_height * 0.5)

            shadow_markup = caption_markup.replace(f'foreground="{self.highlight_color}"', 'foreground="black"')

            main_config = self.text_config.copy()
            shadow_config = self.text_config.copy()
            shadow_config['color'] = 'black'

            shadow_pos = ('center', vertical_position + shadow_offset)
            main_pos = ('center', vertical_position)

            try:
                shadow_clip = TextClip(
                    shadow_markup, size=(clip_width, None), method='pango', **shadow_config
                ).set_position(shadow_pos).set_duration(duration).set_start(start_time)

                main_clip = TextClip(
                    caption_markup, size=(clip_width, None), method='pango', **main_config
                ).set_position(main_pos).set_duration(duration).set_start(start_time)
                
                return [shadow_clip, main_clip]

            except Exception as e:
                print(f"Warning: Pango rendering failed: {e}. Falling back to simple text.")
                plain_text = re.sub(r'<[^>]+>', '', caption_markup)
                
                fallback_clip = TextClip(
                    plain_text, size=(clip_width, None), method='caption', **main_config
                ).set_position(main_pos).set_duration(duration).set_start(start_time)
                
                return [fallback_clip]

        def _extract_words_from_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Extract and flatten word timestamps from Whisper's segments."""
            words = []
            for segment in segments:
                if "words" in segment:
                    words.extend(segment["words"])
            return words

        def _create_caption_chunks(self, words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
            """Group words into chunks for display."""
            if not words:
                return []
            return [words[i:i + self.chunk_size] for i in range(0, len(words), self.chunk_size)]

        def _generate_caption_text(self, chunk: List[Dict[str, Any]], highlight_index: int) -> str:
            """Generate Pango markup for a caption chunk with one word highlighted."""
            caption_parts = []
            for i, word_info in enumerate(chunk):
                text = word_info['word'].strip()
                if i == highlight_index:
                    caption_parts.append(f'<span foreground="{self.highlight_color}">{text}</span>')
                else:
                    caption_parts.append(text)
            return " ".join(caption_parts)

        def _process_word_chunk(self, chunk: List[Dict[str, Any]], video_width: int, video_height: int) -> List[TextClip]:
            """
            Processes a word chunk to create a continuous caption block.
            This function now creates a persistent base layer of text that stays on screen
            for the entire duration of the chunk, preventing any "flashing".
            Highlighted word clips are then layered on top.
            """
            if not chunk:
                return []

            # Calculate timing and text for the entire chunk
            chunk_start_time = chunk[0].get('start')
            chunk_end_time = chunk[-1].get('end')
            
            # Safety check for missing timestamps
            if chunk_start_time is None or chunk_end_time is None:
                return []
                
            chunk_duration = chunk_end_time - chunk_start_time
            if chunk_duration <= 0:
                return []

            base_text = " ".join([word['word'].strip() for word in chunk])
            all_clips = []

            # 1. Create the persistent base layer (un-highlighted)
            # This clip lasts the entire duration of the chunk, ensuring text is always visible.
            persistent_clips = self._create_text_clip_with_shadow(
                base_text, video_width, video_height, chunk_duration, chunk_start_time
            )
            all_clips.extend(persistent_clips)

            # 2. Create the highlighted word clips to layer on top
            for i, word_info in enumerate(chunk):
                start, end = word_info.get('start'), word_info.get('end')
                if start is None or end is None: continue
                
                duration = end - start
                if duration <= 0.01: continue
                
                # Generate the markup for the whole chunk with the current word highlighted
                highlighted_markup = self._generate_caption_text(chunk, i)
                
                # Create a short clip for this highlighted state
                highlight_clips = self._create_text_clip_with_shadow(
                    highlighted_markup, video_width, video_height, duration, start
                )
                all_clips.extend(highlight_clips)

            return all_clips

        def _process_single_video(self, video_file: str) -> bool:
            """Processes a single video file from transcription to final render."""
            input_path = os.path.join(self.input_folder, video_file)
            output_path = os.path.join(self.output_folder, f"captioned_{video_file}")

            if os.path.exists(output_path):
                print(f"Skipping '{video_file}' - output already exists.")
                return True

            try:
                # --- LOCKING MECHANISM ---
                # Use the lock to ensure that the non-thread-safe transcribe function
                # is only called by one thread at a time.
                print(f"Thread for '{video_file}' is waiting to transcribe...")
                with self.transcription_lock:
                    print(f"Thread for '{video_file}' has the lock. Transcribing...")
                    result = self.model.transcribe(input_path, word_timestamps=True)
                print(f"Transcription for '{video_file}' complete. Lock released.")

                print(f"Loading video file: '{video_file}'")
                video = VideoFileClip(input_path)
                
                words = self._extract_words_from_segments(result.get("segments", []))
                if not words:
                    print(f"Warning: No words found in '{video_file}'. Skipping.")
                    video.close()
                    return True 

                chunks = self._create_caption_chunks(words)
                all_clips = [video]

                print(f"Creating caption clips for '{video_file}'...")
                for chunk in chunks:
                    if chunk:
                        text_clips = self._process_word_chunk(chunk, video.w, video.h)
                        all_clips.extend(text_clips)

                print(f"Compositing final video for '{video_file}'...")
                final_video = CompositeVideoClip(all_clips)
                
                # --- UNIQUE TEMP FILE ---
                # Get a unique identifier for the current thread to prevent file conflicts.
                thread_id = threading.get_ident()
                temp_audio_filename = f'temp-audio-{thread_id}.m4a'

                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=temp_audio_filename,
                    remove_temp=True,
                    verbose=False,
                    logger=None,
                    threads=self.max_workers,
                    preset='ultrafast' 
                )
                
                print(f"✓ Successfully processed and saved to '{output_path}'")
                return True

            except Exception as e:
                print(f"✗ An error occurred while processing '{video_file}': {e}")
                return False
            finally:
                if 'final_video' in locals(): final_video.close()
                if 'video' in locals(): video.close()
                if 'all_clips' in locals():
                    for clip in all_clips:
                        if hasattr(clip, 'close'): clip.close()
                self._clear_gpu_cache()

        def run_sequential(self):
            """Runs the video processing pipeline sequentially for each video."""
            try:
                video_files = self._get_video_files()
                print(f"Found {len(video_files)} video(s) to process sequentially.")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            for video_file in video_files:
                print(f"\n--- Starting processing for: {video_file} ---")
                self._process_single_video(video_file)
            
            print("\n=== All videos processed. ===")

        def run_parallel(self):
            """Runs the video processing pipeline in parallel using a thread pool."""
            try:
                video_files = self._get_video_files()
                print(f"Found {len(video_files)} video(s) to process in parallel with {self.max_workers} workers.")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Wrap executor.map with tqdm for a single, clean progress bar
                results = list(tqdm(executor.map(self._process_single_video, video_files), total=len(video_files), desc="Overall Progress"))

            successful_count = sum(1 for r in results if r)
            print(f"\n=== All videos processed. {successful_count}/{len(video_files)} successful. ===")

    class VideoCaptionGenerator:
        """
        Generates dynamic, karaoke-style captions with a clean shadow effect for videos.
        Includes optimizations for parallel processing to improve speed.
        """

        def __init__(self, vid_id: str, model_size: str = "base", max_workers: Optional[int] = None):
            """
            Initialize the caption generator.

            Args:
                vid_id: The base ID for your video folders.
                model_size: The size of the Whisper model to use for transcription.
                max_workers: Max threads for parallel processing. Defaults to CPU count.
            """
            self.vid_id = vid_id
            self.model_size = model_size
            self.max_workers = max_workers or mp.cpu_count()

            # Setup directories
            self.input_folder = f"{self.vid_id}_Final"
            self.output_folder = f"{self.vid_id}_Captioned"
            os.makedirs(self.output_folder, exist_ok=True)

            # Load Whisper model once
            print(f"Loading Whisper model: {self.model_size}...")
            self.model = whisper.load_model(model_size)
            print("Whisper model loaded successfully.")

            # --- Concurrency Lock ---
            # Add a lock to ensure the Whisper model is only used by one thread at a time.
            self.transcription_lock = threading.Lock()

            # --- Caption Configuration ---
            self.chunk_size = 5
            self.video_extensions = ('.mp4', '.mov', '.avi', '.mkv')
            self.text_config = {
                'fontsize': 40,
                'color': 'white',
                'font': 'Arial-Bold',
                'align': 'center'
            }
            self.highlight_color = "yellow"

        def _clear_gpu_cache(self):
            """Clears GPU cache if PyTorch is available."""
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            finally:
                gc.collect()

        def _get_video_files(self) -> List[str]:
            """Get all video files from the input folder."""
            print(f"Searching for videos in: {self.input_folder}")
            if not os.path.isdir(self.input_folder):
                raise FileNotFoundError(
                    f"Input folder not found: '{self.input_folder}'. "
                    f"Please ensure it exists and contains your video files."
                )
            
            files = [f for f in os.listdir(self.input_folder)
                    if f.lower().endswith(self.video_extensions)]
            
            if not files:
                raise FileNotFoundError(f"No video files found in '{self.input_folder}'")
            
            return sorted(files)

        def _create_text_clip_with_shadow(self, caption_markup: str, video_width: int, video_height: int,
                                        duration: float, start_time: float) -> List[TextClip]:
            """
            Creates a text clip with a drop shadow using Pango for consistent layout.
            """
            clip_width = int(video_width * 0.9)
            shadow_offset = 3
            # Position the captions in the vertical center of the screen.
            vertical_position = int(video_height * 0.5)

            shadow_markup = caption_markup.replace(f'foreground="{self.highlight_color}"', 'foreground="black"')

            main_config = self.text_config.copy()
            shadow_config = self.text_config.copy()
            shadow_config['color'] = 'black'

            shadow_pos = ('center', vertical_position + shadow_offset)
            main_pos = ('center', vertical_position)

            try:
                shadow_clip = TextClip(
                    shadow_markup, size=(clip_width, None), method='pango', **shadow_config
                ).set_position(shadow_pos).set_duration(duration).set_start(start_time)

                main_clip = TextClip(
                    caption_markup, size=(clip_width, None), method='pango', **main_config
                ).set_position(main_pos).set_duration(duration).set_start(start_time)
                
                return [shadow_clip, main_clip]

            except Exception as e:
                print(f"Warning: Pango rendering failed: {e}. Falling back to simple text.")
                plain_text = re.sub(r'<[^>]+>', '', caption_markup)
                
                fallback_clip = TextClip(
                    plain_text, size=(clip_width, None), method='caption', **main_config
                ).set_position(main_pos).set_duration(duration).set_start(start_time)
                
                return [fallback_clip]

        def _extract_words_from_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Extract and flatten word timestamps from Whisper's segments."""
            words = []
            for segment in segments:
                if "words" in segment:
                    words.extend(segment["words"])
            return words

        def _create_caption_chunks(self, words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
            """Group words into chunks for display."""
            if not words:
                return []
            return [words[i:i + self.chunk_size] for i in range(0, len(words), self.chunk_size)]

        def _generate_caption_text(self, chunk: List[Dict[str, Any]], highlight_index: int) -> str:
            """Generate Pango markup for a caption chunk with one word highlighted."""
            caption_parts = []
            for i, word_info in enumerate(chunk):
                text = word_info['word'].strip()
                if i == highlight_index:
                    caption_parts.append(f'<span foreground="{self.highlight_color}">{text}</span>')
                else:
                    caption_parts.append(text)
            return " ".join(caption_parts)

        def _process_word_chunk(self, chunk: List[Dict[str, Any]], video_width: int, video_height: int) -> List[TextClip]:
            """
            Processes a word chunk to create a continuous caption block.
            This function now creates a persistent base layer of text that stays on screen
            for the entire duration of the chunk, preventing any "flashing".
            Highlighted word clips are then layered on top.
            """
            if not chunk:
                return []

            # Calculate timing and text for the entire chunk
            chunk_start_time = chunk[0].get('start')
            chunk_end_time = chunk[-1].get('end')
            
            # Safety check for missing timestamps
            if chunk_start_time is None or chunk_end_time is None:
                return []
                
            chunk_duration = chunk_end_time - chunk_start_time
            if chunk_duration <= 0:
                return []

            base_text = " ".join([word['word'].strip() for word in chunk])
            all_clips = []

            # 1. Create the persistent base layer (un-highlighted)
            # This clip lasts the entire duration of the chunk, ensuring text is always visible.
            persistent_clips = self._create_text_clip_with_shadow(
                base_text, video_width, video_height, chunk_duration, chunk_start_time
            )
            all_clips.extend(persistent_clips)

            # 2. Create the highlighted word clips to layer on top
            for i, word_info in enumerate(chunk):
                start, end = word_info.get('start'), word_info.get('end')
                if start is None or end is None: continue
                
                duration = end - start
                if duration <= 0.01: continue
                
                # Generate the markup for the whole chunk with the current word highlighted
                highlighted_markup = self._generate_caption_text(chunk, i)
                
                # Create a short clip for this highlighted state
                highlight_clips = self._create_text_clip_with_shadow(
                    highlighted_markup, video_width, video_height, duration, start
                )
                all_clips.extend(highlight_clips)

            return all_clips

        def _process_single_video(self, video_file: str) -> bool:
            """Processes a single video file from transcription to final render."""
            input_path = os.path.join(self.input_folder, video_file)
            output_path = os.path.join(self.output_folder, f"captioned_{video_file}")

            if os.path.exists(output_path):
                print(f"Skipping '{video_file}' - output already exists.")
                return True

            try:
                # --- LOCKING MECHANISM ---
                # Use the lock to ensure that the non-thread-safe transcribe function
                # is only called by one thread at a time.
                print(f"Thread for '{video_file}' is waiting to transcribe...")
                with self.transcription_lock:
                    print(f"Thread for '{video_file}' has the lock. Transcribing...")
                    result = self.model.transcribe(input_path, word_timestamps=True)
                print(f"Transcription for '{video_file}' complete. Lock released.")

                print(f"Loading video file: '{video_file}'")
                video = VideoFileClip(input_path)
                
                words = self._extract_words_from_segments(result.get("segments", []))
                if not words:
                    print(f"Warning: No words found in '{video_file}'. Skipping.")
                    video.close()
                    return True 

                chunks = self._create_caption_chunks(words)
                all_clips = [video]

                print(f"Creating caption clips for '{video_file}'...")
                for chunk in chunks:
                    if chunk:
                        text_clips = self._process_word_chunk(chunk, video.w, video.h)
                        all_clips.extend(text_clips)

                print(f"Compositing final video for '{video_file}'...")
                final_video = CompositeVideoClip(all_clips)
                
                # --- UNIQUE TEMP FILE ---
                # Get a unique identifier for the current thread to prevent file conflicts.
                thread_id = threading.get_ident()
                temp_audio_filename = f'temp-audio-{thread_id}.m4a'

                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=temp_audio_filename,
                    remove_temp=True,
                    verbose=False,
                    logger=None,
                    threads=self.max_workers,
                    preset='ultrafast' 
                )
                
                print(f"✓ Successfully processed and saved to '{output_path}'")
                return True

            except Exception as e:
                print(f"✗ An error occurred while processing '{video_file}': {e}")
                return False
            finally:
                if 'final_video' in locals(): final_video.close()
                if 'video' in locals(): video.close()
                if 'all_clips' in locals():
                    for clip in all_clips:
                        if hasattr(clip, 'close'): clip.close()
                self._clear_gpu_cache()

        def run_sequential(self):
            """Runs the video processing pipeline sequentially for each video."""
            try:
                video_files = self._get_video_files()
                print(f"Found {len(video_files)} video(s) to process sequentially.")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            for video_file in video_files:
                print(f"\n--- Starting processing for: {video_file} ---")
                self._process_single_video(video_file)
            
            print("\n=== All videos processed. ===")

        def run_parallel(self):
            """Runs the video processing pipeline in parallel using a thread pool."""
            try:
                video_files = self._get_video_files()
                print(f"Found {len(video_files)} video(s) to process in parallel with {self.max_workers} workers.")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Wrap executor.map with tqdm for a single, clean progress bar
                results = list(tqdm(executor.map(self._process_single_video, video_files), total=len(video_files), desc="Overall Progress"))

            successful_count = sum(1 for r in results if r)
            print(f"\n=== All videos processed. {successful_count}/{len(video_files)} successful. ===")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('video_processing.log')
        ]
    )

    class FastVideoConcatenator:
        def __init__(self, vid_id: str):
            self.vid_id = vid_id
            self.logger = logging.getLogger(__name__)

            self.script_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
            self.caption_videos_dir = self.script_dir / f"{vid_id}_Captioned"
            self.output_dir = self.script_dir / f"{vid_id}_Combined"

            self.output_dir.mkdir(parents=True, exist_ok=True)

        def find_captioned_videos(self):
            caption_pattern = f"captioned_{self.vid_id}_*_final.mp4"
            sorted_videos = natsort.natsorted(self.caption_videos_dir.glob(caption_pattern), key=str)
            return sorted_videos

        def create_concat_file(self, video_paths):
            concat_file_path = self.output_dir / "concat_list.txt"
            with open(concat_file_path, "w") as f:
                for video_path in video_paths:
                    f.write(f"file '{video_path.resolve()}'\n")
            return concat_file_path

        def concatenate_videos(self):
            self.logger.info(f"Starting fast concatenation for project: {self.vid_id}")

            video_paths = self.find_captioned_videos()
            if not video_paths:
                self.logger.warning("No videos found to concatenate.")
                return

            concat_file_path = self.create_concat_file(video_paths)
            output_file = self.output_dir / f"{self.vid_id}_FULL_VIDEO.mp4"

            ffmpeg_cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file_path),
                "-c", "copy",
                "-movflags", "+faststart",
                str(output_file)
            ]

            self.logger.info("Running FFmpeg command...")
            try:
                subprocess.run(ffmpeg_cmd, check=True)
                self.logger.info(f"Successfully created: {output_file}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"FFmpeg failed: {e}")

        def process(self):
            self.concatenate_videos()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    @dataclass
    class VideoQuote:
        start_time: str
        end_time: str
        quote: str
        start_seconds: float = 0.0
        end_seconds: float = 0.0
        
        def to_dict(self) -> Dict[str, str]:
            return {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'quote': self.quote,
                'start_seconds': str(self.start_seconds),
                'end_seconds': str(self.end_seconds)
            }

    @dataclass 
    class WhisperSegment:
        start: float
        end: float
        text: str

    class VideoQuoteExtractor:
        def __init__(self, video_id: str, video_file: str, database_path: str, 
                    openai_api_key: str, total_shorts: int = 7, 
                    whisper_model: str = "base", table_name: str = "short_scripts"):
            self.video_id = video_id
            self.video_file = video_file
            self.database_path = database_path
            self.total_shorts = total_shorts
            self.whisper_model = whisper_model
            self.table_name = table_name
            # Create OpenAI client with the modern syntax
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            self._whisper_model = None
            self.transcript_segments = []
            self.quote_pattern = re.compile(
                r'(\d{2}:\d{2}(?::\d{2})?)\s*-\s*(\d{2}:\d{2}(?::\d{2})?)\s*:\s*["\']([^"\']+)["\']'
            )
        
        @property
        def whisper_model_instance(self):
            if self._whisper_model is None:
                logger.info(f"Loading Whisper model: {self.whisper_model}")
                self._whisper_model = whisper.load_model(self.whisper_model)
            return self._whisper_model
        
        @contextmanager
        def get_db_connection(self):
            conn = None
            try:
                db_dir = os.path.dirname(self.database_path)
                if db_dir:
                    os.makedirs(db_dir, exist_ok=True)
                conn = sqlite3.connect(self.database_path)
                conn.row_factory = sqlite3.Row
                yield conn
            except Exception as e:
                if conn:
                    conn.rollback()
                raise e
            finally:
                if conn:
                    conn.close()
        
        def transcribe_video(self) -> str:
            if not os.path.exists(self.video_file):
                raise FileNotFoundError(f"Video file not found: {self.video_file}")
            
            logger.info(f"Transcribing video: {self.video_file}")
            result = self.whisper_model_instance.transcribe(self.video_file, word_timestamps=True)
            
            self.transcript_segments = []
            for segment in result["segments"]:
                self.transcript_segments.append(WhisperSegment(
                    start=segment["start"],
                    end=segment["end"], 
                    text=segment["text"].strip()
                ))
            
            transcript = result["text"].strip()
            logger.info(f"Transcription complete. Length: {len(transcript)} chars, {len(self.transcript_segments)} segments")
            return transcript
        
        def extract_quotes_with_openai(self, transcript: str) -> List[VideoQuote]:
            prompt = self._build_prompt(transcript)
            
            logger.info("Sending request to OpenAI API...")
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert video editor who extracts compelling quotes with precise timestamps."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1500
                )
                
                content = response.choices[0].message.content.strip()
                quotes = self._parse_quotes(content)
                logger.info(f"Extracted {len(quotes)} quotes from OpenAI response")
                return quotes
            except Exception as e:
                logger.error(f"Error calling OpenAI API: {e}")
                raise
        
        def _build_prompt(self, transcript: str) -> str:
            numbered_segments = []
            for i, segment in enumerate(self.transcript_segments):
                start_time = self._seconds_to_timestamp(segment.start)
                end_time = self._seconds_to_timestamp(segment.end)
                numbered_segments.append(f"[{i+1}] {start_time}-{end_time}: {segment.text}")
            
            numbered_transcript = "\n".join(numbered_segments)
            
            # Truncate if content is too long (aim for ~6000 tokens max)
            max_chars = 24000  # Rough estimate: 4 chars per token
            if len(numbered_transcript) + len(transcript) > max_chars:
                # Prioritize numbered transcript, truncate full transcript
                available_for_full = max_chars - len(numbered_transcript) - 500  # Buffer for prompt text
                if available_for_full > 0:
                    transcript = transcript[:available_for_full] + "...[truncated]"
                else:
                    # If numbered transcript is too long, truncate it too
                    numbered_transcript = numbered_transcript[:max_chars//2] + "...[truncated]"
                    transcript = transcript[:max_chars//2] + "...[truncated]"
            
            return f"""Analyze the timestamped transcript and select the {self.total_shorts} most powerful quotes.
    Each quote should be 20-40 seconds long, self-contained and impactful.

    Use EXACT segment numbers. Reference segments by numbers (e.g., "segments 5-8" or "segment 12").
    Format: ["Segments X-Y: 'Quote text'", "Segment Z: 'Quote text'", ...]

    Timestamped Transcript:
    {numbered_transcript}

    Full transcript: {transcript}"""
        
        def _seconds_to_timestamp(self, seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours > 0 else f"{minutes:02d}:{secs:02d}"
        
        def _timestamp_to_seconds(self, timestamp: str) -> float:
            parts = timestamp.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            return 0.0

        def _parse_quotes(self, content: str) -> List[VideoQuote]:
            quotes = []
            
            try:
                json_match = re.search(r'\[(.*?)\]', content, re.DOTALL)
                if json_match:
                    json_str = '[' + json_match.group(1) + ']'
                    quote_strings = json.loads(json_str)
                else:
                    quote_strings = [line.strip() for line in content.split('\n') if line.strip()]
            except json.JSONDecodeError:
                quote_strings = [line.strip() for line in content.split('\n') if line.strip()]
            
            for quote_str in quote_strings:
                quote = self._parse_single_quote_from_segments(quote_str)
                if quote:
                    quotes.append(quote)
            
            return quotes[:self.total_shorts]
        
        def _parse_single_quote_from_segments(self, quote_str: str) -> Optional[VideoQuote]:
            try:
                quote_str = quote_str.strip().strip('"').strip("'")
                
                segment_match = re.search(r'Segments?\s+(\d+)(?:-(\d+))?\s*:', quote_str, re.IGNORECASE)
                if not segment_match:
                    return self._parse_single_quote_fallback(quote_str)
                
                start_seg = int(segment_match.group(1)) - 1
                end_seg = int(segment_match.group(2)) - 1 if segment_match.group(2) else start_seg
                
                if start_seg < 0 or end_seg >= len(self.transcript_segments) or start_seg > end_seg:
                    logger.warning(f"Invalid segment reference in: {quote_str}")
                    return None
                
                start_time_seconds = self.transcript_segments[start_seg].start
                end_time_seconds = self.transcript_segments[end_seg].end
                start_time = self._seconds_to_timestamp(start_time_seconds)
                end_time = self._seconds_to_timestamp(end_time_seconds)
                quote_text = quote_str.split(':', 1)[1].strip().strip('"').strip("'")
                
                return VideoQuote(
                    start_time=start_time,
                    end_time=end_time,
                    quote=quote_text,
                    start_seconds=start_time_seconds,
                    end_seconds=end_time_seconds
                )
            except Exception as e:
                logger.warning(f"Failed to parse quote from segments: {quote_str}. Error: {e}")
                return None
        
        def _parse_single_quote_fallback(self, quote_str: str) -> Optional[VideoQuote]:
            try:
                match = self.quote_pattern.search(quote_str)
                if match:
                    start_time, end_time, quote_text = match.groups()
                    return VideoQuote(
                        start_time=start_time,
                        end_time=end_time,
                        quote=quote_text.strip(),
                        start_seconds=self._timestamp_to_seconds(start_time),
                        end_seconds=self._timestamp_to_seconds(end_time)
                    )
                
                if ': ' in quote_str:
                    time_part, quote_part = quote_str.split(': ', 1)
                    if ' - ' in time_part:
                        start_time, end_time = time_part.strip().split(' - ')
                        quote_text = quote_part.strip().strip('"').strip("'")
                        return VideoQuote(
                            start_time=start_time.strip(),
                            end_time=end_time.strip(),
                            quote=quote_text,
                            start_seconds=self._timestamp_to_seconds(start_time.strip()),
                            end_seconds=self._timestamp_to_seconds(end_time.strip())
                        )
            except Exception as e:
                logger.warning(f"Failed to parse quote: {quote_str}. Error: {e}")
            return None
        
        def create_database_table(self):
            with self.get_db_connection() as conn:
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_id TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT NOT NULL,
                        quote TEXT NOT NULL,
                        start_seconds REAL,
                        end_seconds REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(video_id, start_time, end_time)
                    )
                """)
                conn.commit()
        
        def save_quotes_to_database(self, quotes: List[VideoQuote]) -> int:
            if not quotes:
                logger.warning("No quotes to save to database")
                return 0
            
            self.create_database_table()
            saved_count = 0
            with self.get_db_connection() as conn:
                for quote in quotes:
                    try:
                        conn.execute(
                            f"""INSERT OR REPLACE INTO {self.table_name} 
                            (video_id, start_time, end_time, quote, start_seconds, end_seconds) 
                            VALUES (?, ?, ?, ?, ?, ?)""",
                            (self.video_id, quote.start_time, quote.end_time, quote.quote, 
                            quote.start_seconds, quote.end_seconds)
                        )
                        saved_count += 1
                    except Exception as e:
                        logger.error(f"Error saving quote to database: {e}")
                conn.commit()
            
            logger.info(f"Successfully saved {saved_count} quotes to database")
            return saved_count
        
        def get_quotes_from_database(self) -> List[VideoQuote]:
            try:
                self.create_database_table()
                with self.get_db_connection() as conn:
                    cursor = conn.execute(
                        f"SELECT start_time, end_time, quote, start_seconds, end_seconds FROM {self.table_name} WHERE video_id = ?",
                        (self.video_id,)
                    )
                    rows = cursor.fetchall()
                    
                    quotes = []
                    for row in rows:
                        quotes.append(VideoQuote(
                            start_time=row['start_time'],
                            end_time=row['end_time'],
                            quote=row['quote'],
                            start_seconds=row['start_seconds'] or 0.0,
                            end_seconds=row['end_seconds'] or 0.0
                        ))
                    return quotes
            except sqlite3.Error as e:
                logger.warning(f"Database error when retrieving quotes: {e}")
                return []
        
        def process_video(self, force_retranscribe: bool = False) -> List[VideoQuote]:
            if not force_retranscribe:
                existing_quotes = self.get_quotes_from_database()
                if existing_quotes:
                    logger.info(f"Found {len(existing_quotes)} existing quotes in database")
                    return existing_quotes
            
            transcript = self.transcribe_video()
            quotes = self.extract_quotes_with_openai(transcript)
            self.save_quotes_to_database(quotes)
            self._whisper_model = None
            return quotes

    @dataclass
    class ClipData:
        """Data class for storing clip information"""
        id: int
        start_time: str
        end_time: str
        quote: str
        parsed_start: str = ""
        parsed_end: str = ""
        safe_filename: str = ""
        duration_seconds: float = 0.0


    class VideoClipExtractor:
        """
        Optimized video clip extractor with parallel processing and improved efficiency.
        """
        
        def __init__(self, vid_id: str, max_workers: int = 4, accurate_seek: bool = True, min_duration: float = 12.0):
            """
            Initialize the video clip extractor.
            
            Args:
                vid_id: Video identifier used for file naming
                max_workers: Maximum number of parallel workers for video processing
                accurate_seek: Whether to use accurate seeking (slower but prevents frozen frames)
                min_duration: Minimum clip duration in seconds (clips shorter than this will be deleted)
            """
            self.vid_id = vid_id
            self.max_workers = max_workers
            self.accurate_seek = accurate_seek
            self.min_duration = min_duration
            
            # File paths
            self.video_file = f"{vid_id}_Combined\{vid_id}_FULL_VIDEO.mp4"
            self.db_path = f"{vid_id}_longform.db"
            self.output_folder = Path(f"{vid_id}_shorts")
            self.table_name = "short_scripts"
            
            # Setup logging
            self._setup_logging()
            
            # Performance tracking
            self.success_count = 0
            self.total_clips = 0
            self.deleted_short_clips = 0
            
        def _setup_logging(self) -> None:
            """Setup logging configuration"""
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(f'{self.vid_id}_extraction.log'),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
            
        def _validate_files(self) -> bool:
            """Validate that required files exist"""
            if not Path(self.video_file).exists():
                self.logger.error(f"Video file '{self.video_file}' not found.")
                return False
                
            if not Path(self.db_path).exists():
                self.logger.error(f"Database '{self.db_path}' not found.")
                return False
                
            return True
            
        def _check_ffmpeg(self) -> bool:
            """Check if ffmpeg is available"""
            try:
                subprocess.run(['ffmpeg', '-version'], 
                            capture_output=True, check=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.logger.error("FFmpeg not found. Please install ffmpeg and add it to PATH.")
                return False
                
        def _time_to_seconds(self, time_str: str) -> float:
            """Convert time string to seconds with better error handling"""
            try:
                parts = time_str.split(':')
                if len(parts) == 2:  # MM:SS
                    minutes, seconds = map(float, parts)
                    return minutes * 60 + seconds
                elif len(parts) == 3:  # HH:MM:SS
                    hours, minutes, seconds = map(float, parts)
                    return hours * 3600 + minutes * 60 + seconds
                else:
                    raise ValueError(f"Invalid time format: {time_str}")
            except ValueError as e:
                self.logger.error(f"Time conversion error: {e}")
                return 0.0
                
        def _parse_time_from_string(self, time_string: str) -> str:
            """Extract time from potentially malformed database strings"""
            # Look for time patterns (HH:MM:SS or MM:SS)
            time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?)'
            match = re.search(time_pattern, str(time_string))
            return match.group(1) if match else "00:00"
            
        def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
            """Create safe filename from text"""
            # Remove or replace problematic characters
            safe_text = re.sub(r'[^\w\s-]', '', text.strip())
            safe_text = re.sub(r'[-\s]+', '_', safe_text)
            return safe_text[:max_length].strip('_')
            
        def _load_clips_from_db(self) -> List[ClipData]:
            """Load and parse clips from database efficiently"""
            clips = []
            
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row  # Enable column access by name
                    cursor = conn.cursor()
                    
                    cursor.execute(f"SELECT id, start_time, end_time, quote FROM {self.table_name}")
                    rows = cursor.fetchall()
                    
                    self.logger.info(f"Found {len(rows)} clips in database")
                    
                    for row in rows:
                        clip = ClipData(
                            id=row['id'],
                            start_time=str(row['start_time']),
                            end_time=str(row['end_time']),
                            quote=str(row['quote'])
                        )
                        
                        # Parse times
                        clip.parsed_start = self._parse_time_from_string(clip.start_time)
                        clip.parsed_end = self._parse_time_from_string(clip.end_time)
                        
                        # Calculate duration
                        start_seconds = self._time_to_seconds(clip.parsed_start)
                        end_seconds = self._time_to_seconds(clip.parsed_end)
                        clip.duration_seconds = end_seconds - start_seconds
                        
                        # Clean quote and create safe filename
                        clean_quote = clip.quote.strip().rstrip('",\']}').strip().strip('"\'')
                        safe_quote = self._sanitize_filename(clean_quote)
                        clip.safe_filename = f"{self.vid_id}_clip_{clip.id:02d}_{safe_quote}.mp4"
                        
                        clips.append(clip)
                        
            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                return []
                
            return clips
            
        def _build_ffmpeg_command(self, clip: ClipData, output_path: Path) -> List[str]:
            """Build optimized FFmpeg command"""
            start_seconds = self._time_to_seconds(clip.parsed_start)
            end_seconds = self._time_to_seconds(clip.parsed_end)
            duration = end_seconds - start_seconds
            
            if duration <= 0:
                self.logger.warning(f"Invalid duration for clip {clip.id}: {duration}s")
                duration = 30  # Default fallback
                
            base_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
            
            if self.accurate_seek:
                # Two-pass seeking for accuracy
                cmd = base_cmd + [
                    '-ss', str(start_seconds),
                    '-i', self.video_file,
                    '-ss', '0',
                    '-t', str(duration),
                    '-c:v', 'libx264',
                    '-preset', 'veryfast',  # Faster than 'fast'
                    '-crf', '23',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-movflags', '+faststart',  # Optimize for streaming
                    '-avoid_negative_ts', 'make_zero',
                    str(output_path),
                    '-y'
                ]
            else:
                # Fast stream copy
                cmd = base_cmd + [
                    '-i', self.video_file,
                    '-ss', str(start_seconds),
                    '-t', str(duration),
                    '-c', 'copy',
                    '-avoid_negative_ts', 'make_zero',
                    str(output_path),
                    '-y'
                ]
                
            return cmd
            
        def _should_delete_clip(self, clip: ClipData) -> bool:
            """Check if clip should be deleted based on duration"""
            return clip.duration_seconds < self.min_duration
            
        def _extract_single_clip(self, clip: ClipData) -> bool:
            """Extract a single video clip and delete if too short"""
            output_path = self.output_folder / clip.safe_filename
            
            # Check if clip should be skipped due to short duration
            if self._should_delete_clip(clip):
                self.logger.info(f"Skipping clip {clip.id}: duration {clip.duration_seconds:.1f}s < {self.min_duration}s minimum")
                self.deleted_short_clips += 1
                return False
            
            try:
                cmd = self._build_ffmpeg_command(clip, output_path)
                
                self.logger.info(f"Processing clip {clip.id}: {clip.parsed_start} - {clip.parsed_end} (duration: {clip.duration_seconds:.1f}s)")
                
                # Run FFmpeg with timeout
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=True,
                    timeout=300  # 5 minute timeout per clip
                )
                
                # Verify output file was created and has reasonable size
                if output_path.exists() and output_path.stat().st_size > 1024:  # At least 1KB
                    # Double-check the actual duration of the created file
                    actual_duration = self._get_video_duration(output_path)
                    if actual_duration is not None and actual_duration < self.min_duration:
                        self.logger.info(f"Deleting clip {clip.id}: actual duration {actual_duration:.1f}s < {self.min_duration}s minimum")
                        output_path.unlink()  # Delete the file
                        self.deleted_short_clips += 1
                        return False
                    else:
                        self.logger.info(f"✓ Successfully created: {clip.safe_filename}")
                        return True
                else:
                    self.logger.error(f"✗ Output file missing or too small: {clip.safe_filename}")
                    return False
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"✗ Timeout processing clip {clip.id}")
                return False
            except subprocess.CalledProcessError as e:
                self.logger.error(f"✗ FFmpeg error for clip {clip.id}: {e.stderr}")
                return False
            except Exception as e:
                self.logger.error(f"✗ Unexpected error processing clip {clip.id}: {e}")
                return False
                
        def _get_video_duration(self, video_path: Path) -> Optional[float]:
            """Get the actual duration of a video file using ffprobe"""
            try:
                cmd = [
                    'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                    '-of', 'csv=p=0', str(video_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return float(result.stdout.strip())
            except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
                self.logger.warning(f"Could not determine duration for {video_path}")
                return None
                
        def extract_all_clips(self) -> Dict[str, int]:
            """
            Extract all clips using parallel processing.
            
            Returns:
                Dictionary with extraction statistics
            """
            # Validate environment
            if not self._validate_files() or not self._check_ffmpeg():
                return {"success": 0, "failed": 0, "total": 0, "deleted_short": 0}
                
            # Create output directory
            self.output_folder.mkdir(exist_ok=True)
            self.logger.info(f"Output folder: {self.output_folder}")
            self.logger.info(f"Accurate seeking: {'Enabled' if self.accurate_seek else 'Disabled'}")
            self.logger.info(f"Max workers: {self.max_workers}")
            self.logger.info(f"Minimum duration: {self.min_duration}s (shorter clips will be deleted)")
            
            # Load clips from database
            clips = self._load_clips_from_db()
            if not clips:
                self.logger.error("No clips found to process")
                return {"success": 0, "failed": 0, "total": 0, "deleted_short": 0}
                
            self.total_clips = len(clips)
            
            # Log duration statistics
            short_clips = sum(1 for clip in clips if clip.duration_seconds < self.min_duration)
            self.logger.info(f"Found {short_clips} clips shorter than {self.min_duration}s that will be skipped")
            
            # Process clips in parallel
            start_time = datetime.now()
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs
                future_to_clip = {
                    executor.submit(self._extract_single_clip, clip): clip 
                    for clip in clips
                }
                
                # Process completed jobs
                for future in as_completed(future_to_clip):
                    clip = future_to_clip[future]
                    try:
                        success = future.result()
                        if success:
                            self.success_count += 1
                    except Exception as e:
                        self.logger.error(f"Worker exception for clip {clip.id}: {e}")
                        
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log final statistics
            failed_count = self.total_clips - self.success_count - self.deleted_short_clips
            self.logger.info(f"\n=== EXTRACTION COMPLETE ===")
            self.logger.info(f"Total clips processed: {self.total_clips}")
            self.logger.info(f"Successful extractions: {self.success_count}")
            self.logger.info(f"Deleted (too short): {self.deleted_short_clips}")
            self.logger.info(f"Failed: {failed_count}")
            self.logger.info(f"Success rate (of valid clips): {(self.success_count/(self.total_clips - self.deleted_short_clips))*100:.1f}%" if (self.total_clips - self.deleted_short_clips) > 0 else "N/A")
            self.logger.info(f"Total time: {duration:.1f} seconds")
            self.logger.info(f"Average time per clip: {duration/self.total_clips:.1f} seconds")
            
            return {
                "success": self.success_count,
                "failed": failed_count,
                "total": self.total_clips,
                "deleted_short": self.deleted_short_clips,
                "duration": duration
            }

    class ProcessingMethod(Enum):
        FFMPEG_ONLY = "ffmpeg_only"
        OPENCV_FFMPEG = "opencv_ffmpeg"
        AUTO = "auto"


    @dataclass
    class VideoConfig:
        """Configuration for video processing."""
        vid_id: str = "SolarFlare"
        blur_strength: int = 51
        processing_method: ProcessingMethod = ProcessingMethod.FFMPEG_ONLY
        output_width: int = 1080
        output_height: int = 1920
        crop_size: int = 1024
        overlay_size: int = 1080
        max_workers: int = 1  # Set to 1 for sequential processing, >1 for parallel
        ffmpeg_preset: str = "fast"  # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        ffmpeg_crf: int = 23  # Quality: 0-51, lower is better quality
        blur_sigma: float = 10.0
        progress_callback: Optional[callable] = None


    @dataclass
    class ProcessingStats:
        """Statistics for video processing."""
        total_files: int = 0
        successful: int = 0
        failed: int = 0
        total_time: float = 0.0
        average_time_per_file: float = 0.0
        errors: List[str] = None
        
        def __post_init__(self):
            if self.errors is None:
                self.errors = []


    class VideoProcessor:
        """
        High-performance video processor for converting landscape videos to portrait format
        with blurred backgrounds.
        """
        
        SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        
        def __init__(self, config: VideoConfig):
            self.config = config
            self.logger = self._setup_logging()
            self._validate_config()
            self._check_dependencies()
        
        def _setup_logging(self) -> logging.Logger:
            """Setup logging configuration."""
            logger = logging.getLogger(f"VideoProcessor_{self.config.vid_id}")
            logger.setLevel(logging.INFO)
            
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            
            return logger
        
        def _validate_config(self) -> None:
            """Validate configuration parameters."""
            if self.config.blur_strength % 2 == 0:
                self.config.blur_strength += 1
                self.logger.warning(f"Adjusted blur strength to {self.config.blur_strength} (must be odd)")
            
            if self.config.blur_strength < 1:
                raise ValueError("Blur strength must be positive")
            
            if self.config.output_width <= 0 or self.config.output_height <= 0:
                raise ValueError("Output dimensions must be positive")
            
            if self.config.crop_size <= 0 or self.config.overlay_size <= 0:
                raise ValueError("Crop and overlay sizes must be positive")
        
        def _check_dependencies(self) -> None:
            """Check if required dependencies are available."""
            try:
                result = subprocess.run(['ffmpeg', '-version'], 
                                    capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    self.logger.warning("FFmpeg check failed, some features may not work")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                if self.config.processing_method == ProcessingMethod.FFMPEG_ONLY:
                    raise RuntimeError("FFmpeg is required but not found")
                self.logger.warning("FFmpeg not found, falling back to OpenCV-only processing")
        
        def get_video_files(self, folder_path: Path) -> List[Path]:
            """
            Get all video files from the specified folder.
            
            Args:
                folder_path: Path to the folder
                
            Returns:
                List of video file paths sorted by name
            """
            if not folder_path.exists():
                raise FileNotFoundError(f"Folder '{folder_path}' does not exist")
            
            video_files = []
            for file_path in folder_path.iterdir():
                if (file_path.is_file() and 
                    file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS):
                    video_files.append(file_path)
            
            return sorted(video_files)
        
        def create_output_filename(self, input_path: Path, output_folder: Path) -> Path:
            """
            Create output filename by adding '_processed' before the file extension.
            
            Args:
                input_path: Input file path
                output_folder: Output folder path
                
            Returns:
                Output file path
            """
            stem = input_path.stem
            suffix = input_path.suffix
            output_name = f"{stem}_processed{suffix}"
            return output_folder / output_name
        
        def process_video_ffmpeg_only(self, input_path: Path, output_path: Path) -> None:
            """
            Process video using FFmpeg only with optimized complex filter.
            This method preserves audio automatically and is generally faster.
            """
            self.logger.info(f"Processing {input_path.name} using FFmpeg-only method")
            
            # Optimized FFmpeg complex filter
            filter_complex = (
                f"[0:v]crop={self.config.crop_size}:{self.config.crop_size}:"
                f"(iw-{self.config.crop_size})/2:(ih-{self.config.crop_size})/2,"
                f"scale={self.config.overlay_size}:{self.config.overlay_size}[overlay];"
                f"[0:v]crop={self.config.crop_size}:{self.config.crop_size}:"
                f"(iw-{self.config.crop_size})/2:(ih-{self.config.crop_size})/2,"
                f"scale={self.config.output_width}:{self.config.output_height},"
                f"gblur=sigma={self.config.blur_sigma}[bg];"
                f"[bg][overlay]overlay=0:{(self.config.output_height - self.config.overlay_size) // 2}"
            )
            
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-filter_complex', filter_complex,
                '-c:v', 'libx264',
                '-preset', self.config.ffmpeg_preset,
                '-crf', str(self.config.ffmpeg_crf),
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-movflags', '+faststart',  # Optimize for streaming
                '-y',  # Overwrite output
                str(output_path)
            ]
            
            try:
                start_time = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                processing_time = time.time() - start_time
                
                self.logger.info(f"FFmpeg processing completed in {processing_time:.2f}s")
                
            except subprocess.CalledProcessError as e:
                error_msg = f"FFmpeg processing failed: {e.stderr}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            except FileNotFoundError:
                raise RuntimeError("FFmpeg not found. Please install FFmpeg")
        
        def process_video_opencv(self, input_path: Path, output_path: Path) -> None:
            """
            Process video using OpenCV for frames and FFmpeg for audio.
            Includes optimizations for better performance.
            """
            # Create temporary file for video without audio
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video_path = Path(temp_video.name)
            
            try:
                # Process video frames
                self._process_video_frames_optimized(input_path, temp_video_path)
                
                # Combine processed video with original audio using ffmpeg
                self._combine_audio_video(input_path, temp_video_path, output_path)
                
            finally:
                # Clean up temporary file
                if temp_video_path.exists():
                    temp_video_path.unlink()
        
        def _process_video_frames_optimized(self, input_path: Path, output_path: Path) -> None:
            """
            Process video frames with optimizations for better performance.
            """
            # Open input video
            cap = cv2.VideoCapture(str(input_path))
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {input_path}")
            
            try:
                # Get video properties
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                self.logger.info(f"Processing {input_path.name}: {input_width}x{input_height}, "
                            f"{fps} FPS, {total_frames} frames")
                
                # Setup video writer with optimized settings
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    str(output_path), fourcc, fps, 
                    (self.config.output_width, self.config.output_height)
                )
                
                if not out.isOpened():
                    raise ValueError("Could not initialize video writer")
                
                # Pre-calculate crop coordinates for better performance
                crop_coords = self._calculate_crop_coordinates(input_width, input_height)
                
                frame_count = 0
                start_time = time.time()
                
                # Process frames with progress reporting
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Create the composite frame
                    composite_frame = self._create_composite_frame_optimized(
                        frame, crop_coords
                    )
                    
                    # Write frame
                    out.write(composite_frame)
                    
                    frame_count += 1
                    
                    # Progress callback and logging
                    if frame_count % 30 == 0:
                        progress = frame_count / total_frames
                        elapsed = time.time() - start_time
                        eta = (elapsed / progress) - elapsed if progress > 0 else 0
                        
                        self.logger.info(f"Progress: {frame_count}/{total_frames} "
                                    f"({progress*100:.1f}%) ETA: {eta:.1f}s")
                        
                        if self.config.progress_callback:
                            self.config.progress_callback(progress, frame_count, total_frames)
                
                out.release()
                
            finally:
                cap.release()
        
        def _calculate_crop_coordinates(self, input_width: int, input_height: int) -> Dict[str, int]:
            """Pre-calculate crop coordinates for performance optimization."""
            center_x = input_width // 2
            center_y = input_height // 2
            half_crop = self.config.crop_size // 2
            
            return {
                'start_x': max(0, center_x - half_crop),
                'end_x': min(input_width, center_x + half_crop),
                'start_y': max(0, center_y - half_crop),
                'end_y': min(input_height, center_y + half_crop),
                'y_offset': (self.config.output_height - self.config.overlay_size) // 2
            }
        
        def _create_composite_frame_optimized(self, original_frame: np.ndarray, 
                                            crop_coords: Dict[str, int]) -> np.ndarray:
            """
            Creates a composite frame with optimized operations.
            """
            # Extract the center crop using pre-calculated coordinates
            center_crop = original_frame[
                crop_coords['start_y']:crop_coords['end_y'],
                crop_coords['start_x']:crop_coords['end_x']
            ]
            
            # Create background by resizing and blurring (optimized order)
            background = cv2.resize(
                center_crop, 
                (self.config.output_width, self.config.output_height),
                interpolation=cv2.INTER_LINEAR  # Faster than default
            )
            background = cv2.GaussianBlur(
                background, 
                (self.config.blur_strength, self.config.blur_strength), 
                0
            )
            
            # Create overlay
            overlay_frame = cv2.resize(
                center_crop, 
                (self.config.overlay_size, self.config.overlay_size),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Composite the frame
            y_start = crop_coords['y_offset']
            y_end = y_start + self.config.overlay_size
            background[y_start:y_end, 0:self.config.overlay_size] = overlay_frame
            
            return background
        
        def _combine_audio_video(self, input_path: Path, video_path: Path, 
                            output_path: Path) -> None:
            """Combine processed video with original audio using ffmpeg."""
            self.logger.info("Combining audio and video...")
            
            cmd = [
                'ffmpeg',
                '-i', str(video_path),      # Input video (no audio)
                '-i', str(input_path),      # Input original (for audio)
                '-c:v', 'copy',             # Copy video stream
                '-c:a', 'aac',              # Encode audio as AAC
                '-map', '0:v:0',            # Map video from first input
                '-map', '1:a:0',            # Map audio from second input
                '-shortest',                # Match shortest stream
                '-y',                       # Overwrite output file
                str(output_path)
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                self.logger.info("Audio combined successfully")
                
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                self.logger.warning(f"Audio combination failed: {e}")
                self.logger.warning("Falling back to video-only output")
                
                # Copy video-only file as fallback
                import shutil
                shutil.copy2(video_path, output_path)
        
        def process_single_video(self, input_path: Path, output_path: Path) -> bool:
            """
            Process a single video file.
            
            Args:
                input_path: Path to input video
                output_path: Path for output video
                
            Returns:
                True if successful, False otherwise
            """
            try:
                start_time = time.time()
                
                if self.config.processing_method == ProcessingMethod.FFMPEG_ONLY:
                    self.process_video_ffmpeg_only(input_path, output_path)
                else:
                    self.process_video_opencv(input_path, output_path)
                
                processing_time = time.time() - start_time
                self.logger.info(f"Completed {input_path.name} in {processing_time:.2f}s")
                return True
                
            except Exception as e:
                self.logger.error(f"Error processing {input_path.name}: {e}")
                return False
        
        def process_batch(self, input_folder: Optional[Path] = None, 
                        output_folder: Optional[Path] = None) -> ProcessingStats:
            """
            Process all video files in a folder.
            
            Args:
                input_folder: Input folder path (defaults to {vid_id}_shorts)
                output_folder: Output folder path (defaults to {vid_id}_shorts_processed)
                
            Returns:
                ProcessingStats object with results
            """
            # Set default paths
            if input_folder is None:
                input_folder = Path(f"{self.config.vid_id}_shorts")
            if output_folder is None:
                output_folder = Path(f"{self.config.vid_id}_shorts_processed")
            
            # Create output folder
            output_folder.mkdir(exist_ok=True)
            
            # Get video files
            video_files = self.get_video_files(input_folder)
            
            if not video_files:
                self.logger.warning(f"No video files found in '{input_folder}'")
                return ProcessingStats()
            
            # Initialize stats
            stats = ProcessingStats(total_files=len(video_files))
            
            self.logger.info(f"Found {len(video_files)} video file(s) in '{input_folder}'")
            self.logger.info(f"Output folder: '{output_folder.name}'")
            self.logger.info(f"Processing method: {self.config.processing_method.value}")
            self.logger.info(f"Max workers: {self.config.max_workers}")
            self.logger.info("-" * 50)
            
            start_time = time.time()
            
            if self.config.max_workers > 1:
                # Parallel processing
                stats = self._process_parallel(video_files, output_folder, stats)
            else:
                # Sequential processing
                stats = self._process_sequential(video_files, output_folder, stats)
            
            # Finalize stats
            stats.total_time = time.time() - start_time
            stats.average_time_per_file = (stats.total_time / stats.total_files 
                                        if stats.total_files > 0 else 0)
            
            self._log_final_stats(stats, output_folder)
            return stats
        
        def _process_sequential(self, video_files: List[Path], output_folder: Path, 
                            stats: ProcessingStats) -> ProcessingStats:
            """Process videos sequentially."""
            for i, video_file in enumerate(video_files, 1):
                self.logger.info(f"[{i}/{len(video_files)}] Processing: {video_file.name}")
                
                output_path = self.create_output_filename(video_file, output_folder)
                
                if self.process_single_video(video_file, output_path):
                    stats.successful += 1
                    self.logger.info(f"✓ Completed: {output_path.name}")
                else:
                    stats.failed += 1
                    stats.errors.append(f"Failed to process {video_file.name}")
            
            return stats
        
        def _process_parallel(self, video_files: List[Path], output_folder: Path, 
                            stats: ProcessingStats) -> ProcessingStats:
            """Process videos in parallel."""
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_video = {
                    executor.submit(
                        self.process_single_video, 
                        video_file, 
                        self.create_output_filename(video_file, output_folder)
                    ): video_file 
                    for video_file in video_files
                }
                
                # Process completed tasks
                for future in as_completed(future_to_video):
                    video_file = future_to_video[future]
                    try:
                        success = future.result()
                        if success:
                            stats.successful += 1
                            self.logger.info(f"✓ Completed: {video_file.name}")
                        else:
                            stats.failed += 1
                            stats.errors.append(f"Failed to process {video_file.name}")
                    except Exception as e:
                        stats.failed += 1
                        error_msg = f"Exception processing {video_file.name}: {e}"
                        stats.errors.append(error_msg)
                        self.logger.error(error_msg)
            
            return stats
        
        def _log_final_stats(self, stats: ProcessingStats, output_folder: Path) -> None:
            """Log final processing statistics."""
            self.logger.info("\n" + "=" * 50)
            self.logger.info("Processing complete!")
            self.logger.info(f"Total files: {stats.total_files}")
            self.logger.info(f"Successful: {stats.successful}")
            self.logger.info(f"Failed: {stats.failed}")
            self.logger.info(f"Total time: {stats.total_time:.2f}s")
            self.logger.info(f"Average time per file: {stats.average_time_per_file:.2f}s")
            self.logger.info(f"Output location: {output_folder.absolute()}")
            
            if stats.errors:
                self.logger.error(f"Errors encountered: {len(stats.errors)}")
                for error in stats.errors[:5]:  # Show first 5 errors
                    self.logger.error(f"  - {error}")
                if len(stats.errors) > 5:
                    self.logger.error(f"  ... and {len(stats.errors) - 5} more errors")
        
        def save_config(self, config_path: Path) -> None:
            """Save current configuration to a JSON file."""
            config_dict = {
                'vid_id': self.config.vid_id,
                'blur_strength': self.config.blur_strength,
                'processing_method': self.config.processing_method.value,
                'output_width': self.config.output_width,
                'output_height': self.config.output_height,
                'crop_size': self.config.crop_size,
                'overlay_size': self.config.overlay_size,
                'max_workers': self.config.max_workers,
                'ffmpeg_preset': self.config.ffmpeg_preset,
                'ffmpeg_crf': self.config.ffmpeg_crf,
                'blur_sigma': self.config.blur_sigma
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"Configuration saved to {config_path}")
        
        @classmethod
        def load_config(cls, config_path: Path) -> 'VideoProcessor':
            """Load configuration from a JSON file and create VideoProcessor instance."""
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            config_dict['processing_method'] = ProcessingMethod(config_dict['processing_method'])
            config = VideoConfig(**config_dict)
            
            return cls(config)

    def main():
        """Main function with multiple speed optimization options"""
        
        # Get API key
        api_key = API_KEY if API_KEY else os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: No OpenAI API key found.")
            return
        
        print(f"Optimized Script Generator")
        print(f"VID_ID: {VID_ID}")
        print(f"Topic: {TOPIC}")
        print(f"Chapters: {CHAPTER_NUM}")
        
        # Method 1: Standard optimized (fastest for most cases)
        print("\n=== METHOD 1: Standard Optimized ===")
        generator = OptimizedScriptGenerator(api_key, VID_ID)
        
        start_time = time.time()
        chapters = generator.generate_video_ideas_fast(TOPIC)
        
        if chapters:
            generator.display_chapters_fast(chapters)
            generator.save_chapters_to_database_fast(TOPIC, chapters)
            
            total_time = time.time() - start_time
            print(f"Method 1 completed in {total_time:.2f} seconds!")
        
        # Method 2: Async (uncomment to use)
        # print("\n=== METHOD 2: Async Generation ===")
        # asyncio.run(generate_script_async(api_key, VID_ID, TOPIC, CHAPTER_NUM))

        asyncio.run(ChapterScript())

        import argparse
        
        parser = argparse.ArgumentParser(description='Ultra-fast TTS generation')
        parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
        parser.add_argument('--workers', type=int, default=None, help='Number of workers')
        parser.add_argument('--single', action='store_true', help='Use single-threaded mode (recommended)')
        args = parser.parse_args()
        
        if args.benchmark:
            benchmark_optimizations(VID_ID, AUDIO_PROMPT)
        elif args.single:
            # Print initial memory stats
            print_memory_stats("Initial State")
            
            try:
                generate_audio_single_threaded_optimized(VID_ID, AUDIO_PROMPT)
            finally:
                # Always cleanup at the end
                print_memory_stats("Final State")
                cleanup_resources()
        else:
            # Use the fixed multi-threaded version (sequential processing with single model)
            print_memory_stats("Initial State")
            
            try:
                generate_audio_hyper_optimized(VID_ID, AUDIO_PROMPT, max_workers=args.workers)
            finally:
                print_memory_stats("Final State")
                cleanup_resources()

        """Main execution function with examples."""
        
        # Configuration
        SILENCE_THRESHOLD = 0.1
        SILENCE_DURATION = 0.2
        BACKUP_ORIGINALS = False
        
        # Initialize the optimized trimmer
        trimmer = OptimizedAudioTrimmer(
            vid_id=VID_ID,
            silence_threshold=SILENCE_THRESHOLD,
            silence_duration=SILENCE_DURATION,
            backup_originals=BACKUP_ORIGINALS,
            max_workers=None,  # Auto-detect optimal number of workers
            enable_cache=True
        )
        
        logger.info("Optimized Audio Trimmer - High Performance Mode")
        logger.info("=" * 60)
        logger.info(f"Video ID: {VID_ID}")
        logger.info(f"Target folder: {VID_ID}_Audio")
        logger.info(f"Silence threshold: {SILENCE_THRESHOLD}")
        logger.info(f"Silence duration after speech: {SILENCE_DURATION}s")
        logger.info(f"Backup originals: {BACKUP_ORIGINALS}")
        logger.info(f"Max workers: {trimmer.max_workers}")
        logger.info("⚠️  WARNING: Original files will be REPLACED!")
        logger.info("=" * 60)
        
        try:
            # Option 1: Preview analysis (recommended first)
            logger.info("\n1. PREVIEW MODE - Analyzing what would be trimmed:")
            preview_result = trimmer.preview_analysis()
            
            if preview_result["success"]:
                logger.info("Preview completed successfully!")
            
            # Option 2: Actually trim all files with parallel processing
            logger.info("\n2. TRIMMING ALL FILES WITH PARALLEL PROCESSING:")
            trim_result = trimmer.trim_all_files_parallel()
            
            if trim_result["success"]:
                logger.info("✓ All files processed successfully!")
            
            # Option 3: Trim a single file (example)
            # logger.info("\n3. TRIMMING SINGLE FILE:")
            # single_result = trimmer.trim_single_file("DEFAULT_1_1.wav")
            # if single_result.success:
            #     logger.info(f"✓ Single file trimmed successfully: {single_result.time_saved:.2f}s saved")
            
        except Exception as e:
            logger.error(f"Error in main execution: {e}")

        """Main function with example usage"""
        # Configuration
        OUTPUT_FORMAT = "wav"
        DELETE_ORIGINAL_PARTS = True
        SILENCE_BETWEEN_PARTS = 0.1
        
        print("Fast Audio File Combiner")
        print("=" * 50)
        
        # Create combiner instance
        combiner = FastAudioCombiner(
            vid_id=VID_ID,
            output_format=OUTPUT_FORMAT,
            delete_original_parts=DELETE_ORIGINAL_PARTS,
            silence_between_parts=SILENCE_BETWEEN_PARTS,
            max_workers=None,  # Auto-detect
            log_level="INFO"
        )
        
        print(f"Video ID: {VID_ID}")
        print(f"Target folder: {VID_ID}_Audio")
        print(f"Output format: {OUTPUT_FORMAT}")
        print(f"Delete original parts: {DELETE_ORIGINAL_PARTS}")
        print(f"Silence between parts: {SILENCE_BETWEEN_PARTS}s")
        print(f"Max workers: {combiner.max_workers}")
        print("=" * 50)
        
        try:
            # Preview mode
            print("\n1. PREVIEW MODE:")
            preview_result = combiner.preview_combination()
            
            # Actually combine files
            print("\n" + "="*60)
            print("2. COMBINING FILES:")
            combine_result = combiner.combine_all_sessions()
            
            if combine_result['success']:
                print(f"\n🎉 SUCCESS! Combined {combine_result['sessions_combined']} sessions")
                print(f"⚡ Processing speed: {combine_result['stats']['total_duration_combined']/combine_result['processing_time']:.1f}x realtime")
            else:
                print(f"\n❌ FAILED: {combine_result.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error: {e}")
            logging.exception("Detailed error information:")

        """
        Main function to run the audio duration processor.
        """
        
        if not VID_ID:
            print("VID_ID cannot be empty!")
            return
        
        # Create processor instance and run
        processor = AudioDurationProcessor(VID_ID)
        success = processor.process_audio_files()
        
        if success:
            print("Processing completed successfully!")
        else:
            print("Processing failed. Check logs for details.")

        """
        Main function to run the optimized image generator
        """
        vid_id = VID_ID
        
        if not vid_id or vid_id == "YOUR_VID_ID_HERE":
            print("Please set the VID_ID at the top of this file!")
            return
        
        db_path = f"{vid_id}_longform.db"
        if not os.path.exists(db_path):
            print(f"Database not found: {db_path}")
            return
        
        generator = OptimizedImageGenerator(vid_id)
        asyncio.run(generator.process_all_rows_optimized())

        """
        Main function to run the audio length processor.
        """
        
        if not VID_ID:
            print("VID_ID cannot be empty!")
            return
        
        # Create processor instance and run
        processor = AudioLengthProcessor(VID_ID, log_level=logging.INFO)
        
        try:
            success = processor.process_audio_lengths()
            
            if success:
                print("Processing completed successfully!")
                
                # Show cache info
                cache_info = processor.get_cache_info()
                print(f"Cache info: {cache_info['cached_files']} files cached")
            else:
                print("Processing failed. Check logs for details.")
                
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            # Clean up cache
            processor.clear_cache()

        """
        Main function demonstrating the optimized video creator.
        """
        # Create optimized video creator instance
        creator = OptimizedVideoCreator(
            vid_id=VID_ID,
            output_width=1920,
            output_height=1080,
            fps=30,
            fade_duration=0.5,
            zoom_factor=1.2,
            max_workers=None  # Auto-detect optimal worker count
        )
        
        # Run the video creation process
        successful_videos, failed_videos = creator.run(parallel=True)
        
        # Additional processing or cleanup can be done here
        if successful_videos:
            print(f"\nCreated {len(successful_videos)} videos successfully!")
        
        if failed_videos:
            print(f"\nFailed to create {len(failed_videos)} videos. Consider running again or checking inputs.")

        """
        Main function that demonstrates usage of the AudioVideoCombiner class.
        """
        # Configuration
        MAX_WORKERS = 4  # Adjust based on your system
        TIMEOUT = 300    # 5 minutes per file
        
        # Create combiner instance
        combiner = AudioVideoCombiner(
            vid_id=VID_ID,
            max_workers=MAX_WORKERS,
            timeout=TIMEOUT
        )
        
        # Print configuration
        stats = combiner.get_processing_stats()
        print(f"Starting optimized audio-video combination")
        print(f"VID_ID: {stats['vid_id']}")
        print(f"Max workers: {stats['max_workers']}")
        print(f"Timeout: {stats['timeout']} seconds")
        print(f"Audio folder: {stats['audio_folder']}")
        print(f"Video folder: {stats['video_folder']}")
        print(f"Output folder: {stats['output_folder']}")
        print("-" * 60)
        
        # Start processing
        start_time = time.time()
        successful_results, failed_results = combiner.process_all_combinations()
        total_time = time.time() - start_time
        
        print("-" * 60)
        print(f"Audio-video combination process completed in {total_time:.1f} seconds!")
        print(f"Success rate: {len(successful_results)}/{len(successful_results) + len(failed_results)} "
            f"({100 * len(successful_results) / max(1, len(successful_results) + len(failed_results)):.1f}%)")
        
        """Main function to run the caption generator."""
        # --- CHOOSE YOUR MODE ---
        USE_PARALLEL = True  # Set to True for faster processing, False for sequential
        # ------------------------

        try:
            generator = VideoCaptionGenerator(vid_id=VID_ID, model_size="base")
            if USE_PARALLEL:
                generator.run_parallel()
            else:
                generator.run_sequential()
                
        except Exception as e:
            print(f"A fatal error occurred: {e}")
        finally:
            print("Script finished.")

        """Main function to run the caption generator."""
        
        # --- CHOOSE YOUR MODE ---
        USE_PARALLEL = True  # Set to True for faster processing, False for sequential
        # ------------------------

        try:
            generator = VideoCaptionGenerator(vid_id=VID_ID, model_size="base")
            if USE_PARALLEL:
                generator.run_parallel()
            else:
                generator.run_sequential()
                
        except Exception as e:
            print(f"A fatal error occurred: {e}")
        finally:
            print("Script finished.")

        processor = FastVideoConcatenator(VID_ID)
        processor.process()

        config = {
            'video_id': VID_ID,
            'video_file': f"{VID_ID}_combined/{VID_ID}_FULL_VIDEO.mp4",
            'database_path': f"{VID_ID}_longform.db",
            'openai_api_key': os.getenv('OPENAI_API_KEY', "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"),
            'total_shorts': 10,
            'whisper_model': "base",
            'table_name': "short_scripts"
        }
        
        try:
            extractor = VideoQuoteExtractor(**config)
            quotes = extractor.process_video()
            
            print(f"\nExtracted {len(quotes)} quotes:")
            for i, quote in enumerate(quotes, 1):
                print(f"\n{i}. {quote.start_time} - {quote.end_time}")
                print(f'   "{quote.quote}"')
        
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise

            """Main function to run the video clip extractor"""
        # Configuration
        MAX_WORKERS = 4  # Adjust based on your system capabilities
        ACCURATE_SEEK = True
        MIN_DURATION = 12.0  # Minimum clip duration in seconds
        
        # Create extractor and run
        extractor = VideoClipExtractor(
            vid_id=VID_ID,
            max_workers=MAX_WORKERS,
            accurate_seek=ACCURATE_SEEK,
            min_duration=MIN_DURATION
        )
        
        results = extractor.extract_all_clips()
        
        if results["total"] > 0:
            print(f"\nExtraction completed!")
            print(f"Success: {results['success']}/{results['total']} clips")
            print(f"Deleted (too short): {results['deleted_short']} clips")
            print(f"Time taken: {results['duration']:.1f} seconds")
        else:
            print("No clips were processed. Check the logs for errors.")

        """Main function demonstrating usage."""
        # Create configuration
        config = VideoConfig(
            vid_id=VID_ID,
            blur_strength=51,
            processing_method=ProcessingMethod.FFMPEG_ONLY,
            max_workers=1,  # Adjust based on your system
            ffmpeg_preset="fast",
            ffmpeg_crf=23
        )
        
        # Create processor
        processor = VideoProcessor(config)
        
        # Process videos
        stats = processor.process_batch()
        
        # Save configuration for future use
        processor.save_config(Path("video_processor_config.json"))
        
        return stats
    
    main()

if __name__ == "__main__":
    run_storyform("1", "world war 1", "1")
