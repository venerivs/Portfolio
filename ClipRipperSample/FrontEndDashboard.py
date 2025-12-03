import customtkinter as ctk
import psutil
import pynvml
import time
import urllib.request
import socket
import sys
import io
import threading
from queue import Queue
import subprocess
import os
import sqlite3
from datetime import datetime
from util.sf import run_storyform
from util.raaitah import run_reddit
# ADDED: Import the NOSLEEP automation function, aliasing it to avoid name conflicts
from util.ranosleep import run_reddit as run_reddit_horror 
import queue
import multiprocessing
import traceback
import schedule

class TerminalRedirect:
    """Redirect stdout and stderr to the terminal widget"""
    def __init__(self, terminal_widget):
        self.terminal_widget = terminal_widget
        self.queue = Queue()
        
    def write(self, text):
        if text.strip():  # Only add non-empty text
            self.queue.put(text)
        
    def flush(self):
        pass

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Automation Dashboard")
        self.geometry("1280x960")
        self.configure(fg_color="#202020")

        self.setup_threading()
        self.configure_grid()
        
        # CORRECT ORDER: Create the layout and its widgets first.
        self.create_layout()
        self.setup_database()
        self.setup_scheduler()

        # THEN, redirect the terminal output.
        self.setup_terminal_redirect()

        # Store program start time
        self.start_time = time.time()
        import datetime
        self.launch_datetime = datetime.datetime.now()

    def open_schedule_dialog(self):
        """Open a dialog to set the schedule for an existing account."""
        self.schedule_window = ctk.CTkToplevel(self)
        self.schedule_window.title("Set Automation Schedule")
        self.schedule_window.geometry("450x600")
        self.schedule_window.configure(fg_color="#202020")
        self.schedule_window.transient(self)
        self.schedule_window.grab_set()

        scrollable_frame = ctk.CTkScrollableFrame(self.schedule_window, fg_color="transparent")
        scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(scrollable_frame, text="Set Schedule for Account", font=("Arial", 18, "bold")).pack(pady=(10, 20))

        # --- Account Selection ---
        ctk.CTkLabel(scrollable_frame, text="Account:", font=("Arial", 12, "bold")).pack(anchor='w')
        nicknames = self.get_all_nicknames()
        self.schedule_nickname_dropdown = ctk.CTkComboBox(scrollable_frame, values=nicknames, state="readonly")
        self.schedule_nickname_dropdown.pack(fill='x', pady=(0, 15))

        # --- Automation Type Selection ---
        ctk.CTkLabel(scrollable_frame, text="Automation Type:", font=("Arial", 12, "bold")).pack(anchor='w')
        automation_types = ["Storyform", "Reddit - AITAH", "Reddit - NOSLEEP"]
        self.schedule_type_dropdown = ctk.CTkComboBox(scrollable_frame, values=automation_types, state="readonly")
        self.schedule_type_dropdown.pack(fill='x', pady=(0, 15))

        # --- Scheduling Section ---
        self.schedule_automation_active = ctk.CTkCheckBox(scrollable_frame, text="Enable Automation for this Account")
        self.schedule_automation_active.pack(anchor='w', pady=(10, 15))

        ctk.CTkLabel(scrollable_frame, text="Frequency:", font=("Arial", 12, "bold")).pack(anchor='w')
        self.schedule_frequency = ctk.CTkComboBox(scrollable_frame, values=["daily", "2x daily", "4x daily"], state="readonly")
        self.schedule_frequency.pack(fill='x', pady=(0, 15))

        ctk.CTkLabel(scrollable_frame, text="Times (comma-separated, 24h format):", font=("Arial", 12, "bold")).pack(anchor='w')
        self.schedule_times = ctk.CTkEntry(scrollable_frame, placeholder_text="e.g., 08:00 or 09:30,21:30")
        self.schedule_times.pack(fill='x', pady=(0, 20))

        # --- Error Label and Buttons ---
        self.schedule_error_label = ctk.CTkLabel(scrollable_frame, text="", text_color="#FF4444")
        self.schedule_error_label.pack(pady=5)

        button_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        button_frame.pack(pady=20)
        
        cancel_btn = ctk.CTkButton(button_frame, text="Cancel", command=self.schedule_window.destroy)
        cancel_btn.pack(side="left", padx=10)

        submit_btn = ctk.CTkButton(button_frame, text="Save Schedule", command=self.submit_schedule_update)
        submit_btn.pack(side="left", padx=10)

    def submit_schedule_update(self):
        """Validate and save the updated schedule to the database."""
        nickname = self.schedule_nickname_dropdown.get()
        automation_type = self.schedule_type_dropdown.get()
        is_active = 1 if self.schedule_automation_active.get() else 0
        frequency = self.schedule_frequency.get()
        times = self.schedule_times.get().strip()

        if not nickname or "Select" in nickname:
            self.schedule_error_label.configure(text="Please select an account.")
            return
        if not automation_type or "Select" in automation_type:
            self.schedule_error_label.configure(text="Please select an automation type.")
            return
        if is_active and not times:
            self.schedule_error_label.configure(text="Please enter schedule times if automation is active.")
            return

        # New database function to update the schedule
        self.update_account_schedule(nickname, is_active, frequency, times, automation_type)

        print(f"Schedule updated for '{nickname}'.")
        self.schedule_error_label.configure(text="")
        
        # Reload schedules and refresh UI
        self.load_schedules()
        self.refresh_database_display()
        self.schedule_window.destroy()

    def update_account_schedule(self, nickname, is_active, frequency, times, automation_type):
        """Update the scheduling details for a specific account in the database."""
        try:
            self.db_cursor.execute('''
                UPDATE credentials
                SET automation_active = ?, frequency = ?, times = ?, automation_type = ?
                WHERE nickname = ?
            ''', (is_active, frequency, times, automation_type, nickname))
            self.db_connection.commit()
        except sqlite3.Error as e:
            print(f"Error updating schedule for {nickname}: {e}")


    def setup_scheduler(self):
        """Initialize and run the scheduler in a separate thread."""
        print("Setting up automation scheduler...")
        self.load_schedules()

        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(1)

        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        print("Scheduler is running in the background.")

    def load_schedules(self):
        """Clear existing jobs and load all active schedules from the database."""
        schedule.clear()
        print("Loading schedules from database...")
        try:
            # This needs to be a function that gets all data, including the new column
            credentials = self.get_all_credentials() 
            active_jobs = 0
            for cred_row in credentials:
                cred = {
                    'id': cred_row[0], 'nickname': cred_row[1], 'channel_id': cred_row[2],
                    'api_key': cred_row[3], 'oauth_client_id': cred_row[4], 'client_secret': cred_row[5],
                    'automation_active': cred_row[6], 'frequency': cred_row[7], 'times': cred_row[8],
                    'automation_type': cred_row[9] # ADDED: Get the automation type
                }
                if cred['automation_active'] and cred['automation_type']: # Check that a type is set
                    self.schedule_job_for_account(cred)
                    active_jobs += 1
            print(f"Loaded {active_jobs} active schedule(s).")
        except Exception as e:
            print(f"Error loading schedules: {e}")


    def schedule_job_for_account(self, credential):
        """Schedules the automation job for a single account based on its settings."""
        nickname = credential['nickname']
        frequency = credential['frequency']
        times_str = credential.get('times', '')
        automation_type = credential.get('automation_type') # Get the type

        if not frequency or not times_str or not automation_type:
            print(f"Scheduling failed for '{nickname}': missing frequency, times, or automation type.")
            return

        times = [t.strip() for t in times_str.split(',')]

        # MODIFIED: The lambda now calls a generic runner with the nickname and type
        job_function = lambda n=nickname, at=automation_type: self.run_generic_scheduled_automation(n, at)
        
        job_tags = [nickname, automation_type] # Tag with both for display

        try:
            # ... (the schedule.every() logic remains exactly the same) ...
            if frequency == "daily":
                if len(times) >= 1:
                    schedule.every().day.at(times[0]).do(job_function).tag(*job_tags)
            elif frequency == "2x daily":
                if len(times) >= 2:
                    schedule.every().day.at(times[0]).do(job_function).tag(*job_tags)
                    schedule.every().day.at(times[1]).do(job_function).tag(*job_tags)
            elif frequency == "4x daily":
                if len(times) >= 4:
                    for t in times:
                        schedule.every().day.at(t).do(job_function).tag(*job_tags)

            print(f"Scheduled '{automation_type}' job for '{nickname}' at {', '.join(times)} ({frequency}).")
        except Exception as e:
            print(f"Could not schedule job for '{nickname}'. Error: {e}")

    def run_generic_scheduled_automation(self, nickname, automation_type):
        """The generic function executed by the scheduler for any automation type."""
        print(f"--- Running SCHEDULED automation for: {nickname} | Type: {automation_type} ---")

        if automation_type == "Storyform":
            # You might need a way to get a topic for storyform, or have a default one.
            # For now, we'll assume a default topic or that it handles it internally.
            topic = f"Scheduled Story for {nickname}" 
            vid_id_num = self.add_storyform_to_pending(topic)
            if vid_id_num:
                self.start_storyform_thread(vid_id_num, topic, 1) # Assuming chapter 1
                self.after(1000, self.refresh_database_display)
            else:
                print(f"Error creating database entry for Storyform run on account '{nickname}'.")

        elif "Reddit" in automation_type:
            # Map the automation type to the correct story type for the Reddit script
            story_type_map = {
                "Reddit - AITAH": "AITAH",
                "Reddit - NOSLEEP": "NOSLEEP"
            }
            story_type = story_type_map.get(automation_type)
            
            if not story_type:
                print(f"Error: Unknown Reddit automation type '{automation_type}'")
                return

            vid_id_num = self.add_reddit_to_pending() # This function needs to be generic now
            if vid_id_num:
                vid_to_use = f"V{vid_id_num}"
                print(f"Generated VID_ID '{vid_to_use}' for scheduled run.")
                self.start_reddit_thread_with_vid(vid_to_use, story_type, nickname)
                self.after(1000, self.refresh_database_display)
            else:
                print(f"Error creating database entry for Reddit run on account '{nickname}'.")

    def run_scheduled_automation(self, nickname):
        """The function that gets executed by the scheduler for a specific account."""
        print(f"--- Running SCHEDULED automation for: {nickname} ---")
        
        # For scheduled runs, you might want a default story type or add it to the DB
        story_type = "AITAH" 
        
        # Add a new entry to the pending uploads table
        vid_id_num = self.add_reddit_to_pending()
        if vid_id_num:
            vid_to_use = f"V{vid_id_num}"
            print(f"Generated VID_ID '{vid_to_use}' for scheduled run.")
            
            # Start the automation thread for the specific account
            self.start_reddit_thread_with_vid(vid_to_use, story_type, nickname)
            
            # Refresh the database display in the UI
            self.after(1000, self.refresh_database_display) # Use after to run on main thread
        else:
            print(f"Error creating database entry for scheduled run on account '{nickname}'.")

    
    def setup_threading(self):
        """Initialize threading with CPU core allocation"""
        # Get total CPU cores
        total_cores = multiprocessing.cpu_count()
        print(f"Total CPU cores detected: {total_cores}")
        
        # Allocate cores:
        # 1 core for usage bar
        # 1 core for terminal
        # 1 core for database
        # Remaining cores split between storyform and reddit automation
        
        self.core_allocation = {
            'usage_bar': 1,
            'terminal': 1,
            'database': 1,
            'storyform': max(1, (total_cores - 3) // 2),
            'reddit': max(1, (total_cores - 3) // 2)
        }
        
        print(f"Core allocation: {self.core_allocation}")
        
        # Thread pools for different components
        self.usage_executor = None
        self.terminal_executor = None
        self.database_executor = None
        self.storyform_executor = None
        self.reddit_executor = None
        
        # Queues for thread communication
        self.thread_queue = queue.Queue()
        self.terminal_queue = queue.Queue()
        self.database_queue = queue.Queue()
        
        # Active thread tracking
        self.active_threads = {
            'storyform': None,
            'reddit': None
        }
        
        # Start queue processor
        self.process_queues()

    def update_status(self, component, status):
        """Update component status display"""
        print(f"{component.capitalize()} status: {status}")
        # Add any UI updates for status changes here
    
    def update_progress(self, component, progress):
        """Update component progress display"""
        print(f"{component.capitalize()} progress: {progress}%")
        # Add any UI updates for progress changes here
    
    def set_thread_affinity(self, core_list):
        """Set CPU affinity for current thread (Linux/Windows)"""
        try:
            if hasattr(os, 'sched_setaffinity'):  # Linux
                os.sched_setaffinity(0, core_list)
            # Windows would need psutil.Process().cpu_affinity(core_list)
        except Exception as e:
            print(f"Could not set CPU affinity: {e}")
    
    def process_queues(self):
        """Process messages from all worker threads"""
        try:
            # Process terminal queue
            while not self.terminal_queue.empty():
                message = self.terminal_queue.get_nowait()
                self.handle_terminal_message(message)
            
            # Process database queue
            while not self.database_queue.empty():
                message = self.database_queue.get_nowait()
                self.handle_database_message(message)
            
            # Process main thread queue
            while not self.thread_queue.empty():
                message = self.thread_queue.get_nowait()
                self.handle_thread_message(message)
        except queue.Empty:
            pass
        
        # Schedule next check
        self.after(100, self.process_queues)
    
    def handle_terminal_message(self, message):
        """Handle messages from terminal thread"""
        if message['type'] == 'output':
            self.add_terminal_output(message['text'])
    
    def handle_database_message(self, message):
        """Handle messages from database thread"""
        if message['type'] == 'refresh':
            self.refresh_database_display()
        elif message['type'] == 'error':
            print(f"Database error: {message['error']}")
    
    def handle_thread_message(self, message):
        """Handle messages from other threads"""
        if message['type'] == 'status_update':
            self.update_status(message['component'], message['status'])
        elif message['type'] == 'progress':
            self.update_progress(message['component'], message['progress'])
        
        
    def configure_grid(self):
        """Configuring a 10x10 responsive grid"""
        for i in range(10):
            self.grid_rowconfigure(i, weight=1)
            self.grid_columnconfigure(i, weight=1)
            
    def create_layout(self):
        """Building the layout by adding layout functions"""
        self.create_header()
        self.create_colored_sections()
        self.create_terminal()
        self.create_usage_bar()

        
    def create_header(self):
        """Creating black header row with overall button and metric labels"""
        self.header_frame = ctk.CTkFrame(self, fg_color='#000000', height=60) # Black color
        self.header_frame.grid(row=0, column=0, columnspan=10, sticky='ew', padx=5, pady=5)
        self.header_frame.grid_propagate(False) # Maintain fixed height
        
        # Configure header frame grid with 4 columns for even spacing
        self.header_frame.grid_columnconfigure(0, weight=1)  # Overall button
        self.header_frame.grid_columnconfigure(1, weight=1)  # Views
        self.header_frame.grid_columnconfigure(2, weight=1)  # Likes
        self.header_frame.grid_columnconfigure(3, weight=1)  # Subscribers
        self.header_frame.grid_rowconfigure(0, weight=1)
        
        # Overall button at the far left
        self.overall_button = ctk.CTkButton(
            self.header_frame,
            text="Overall",
            font=("Arial", 16, "bold"),
            text_color="white",
            fg_color="#333333",
            hover_color="#555555",
            command=self.toggle_nickname_dropdown  # Add command to handle dropdown
        )
        self.overall_button.grid(row=0, column=0, sticky='w', padx=10)
        
        # Views label
        self.views_label = ctk.CTkLabel(
            self.header_frame,
            text="Views: Loading...",
            font=("Arial", 18, "bold"),
            text_color="white"
        )
        self.views_label.grid(row=0, column=1, sticky='w')
        
        # Likes label
        self.likes_label = ctk.CTkLabel(
            self.header_frame,
            text="Likes: Loading...",
            font=("Arial", 18, "bold"),
            text_color="white"
        )
        self.likes_label.grid(row=0, column=2, sticky='w')
        
        # Subscribers label
        self.subscribers_label = ctk.CTkLabel(
            self.header_frame,
            text="Subscribers: Loading...",
            font=("Arial", 18, "bold"),
            text_color="white"
        )
        self.subscribers_label.grid(row=0, column=3, sticky='w')
        
        # Initialize dropdown state
        self.dropdown_open = False
        self.dropdown_frame = None
        
        # Initialize metric tracking
        self.current_views = "0"
        self.current_likes = "0" 
        self.current_subscribers = "0"
        self.selected_nickname = "Overall"

    def get_all_nicknames(self):
        """Retrieve all nicknames from the credentials database"""
        try:
            self.db_cursor.execute('SELECT nickname FROM credentials ORDER BY nickname ASC')
            nicknames = [row[0] for row in self.db_cursor.fetchall()]
            return nicknames
        except sqlite3.Error as e:
            print(f"Error retrieving nicknames: {e}")
            return []

    def toggle_nickname_dropdown(self):
        """Toggle the dropdown menu for nickname selection"""
        if self.dropdown_open:
            self.close_dropdown()
        else:
            self.open_dropdown()

    def open_dropdown(self):
        """Open the nickname dropdown menu"""
        if self.dropdown_open:
            return
        
        # Get all nicknames from database
        nicknames = self.get_all_nicknames()
        
        if not nicknames:
            print("No nicknames found in database")
            return
        
        # Calculate dropdown height based on number of items (Overall + separator + nicknames)
        dropdown_height = (len(nicknames) + 1) * 27 + 10  # 27px per item + padding
        
        # Create dropdown frame as child of main window (not header_frame) for proper layering
        self.dropdown_frame = ctk.CTkFrame(
            self,  # Main window instead of header_frame
            fg_color="#2d2d2d",
            border_width=1,
            border_color="#555555",
            width=150,
            height=dropdown_height
        )
        
        # Get absolute position of the Overall button relative to main window
        button_x = self.overall_button.winfo_x() + self.header_frame.winfo_x() + 10  # Add header frame offset + padding
        button_y = self.overall_button.winfo_y() + self.overall_button.winfo_height() + self.header_frame.winfo_y() + 10  # Add header frame offset + spacing
        
        # Position dropdown and bring to front
        self.dropdown_frame.place(x=button_x, y=button_y)
        self.dropdown_frame.lift()  # Bring to front
        
        # Add "Overall" option at the top
        overall_btn = ctk.CTkButton(
            self.dropdown_frame,
            text="Overall",
            font=("Arial", 12),
            height=25,
            fg_color="transparent",
            text_color="white",
            hover_color="#404040",
            command=lambda: self.select_nickname("Overall")
        )
        overall_btn.pack(fill="x", padx=2, pady=1)
        
        # Add separator
        separator = ctk.CTkFrame(self.dropdown_frame, height=1, fg_color="#555555")
        separator.pack(fill="x", padx=5, pady=2)
        
        # Add nickname buttons
        for nickname in nicknames:
            nickname_btn = ctk.CTkButton(
                self.dropdown_frame,
                text=nickname,
                font=("Arial", 12),
                height=25,
                fg_color="transparent",
                text_color="white",
                hover_color="#404040",
                command=lambda n=nickname: self.select_nickname(n)
            )
            nickname_btn.pack(fill="x", padx=2, pady=1)
        
        self.dropdown_open = True
        
        # Bind click event to close dropdown when clicking outside
        self.bind_all("<Button-1>", self.on_click_outside)

    def close_dropdown(self):
        """Close the nickname dropdown menu"""
        if self.dropdown_frame:
            self.dropdown_frame.destroy()
            self.dropdown_frame = None
        
        self.dropdown_open = False
        
        # Unbind the click event
        self.unbind_all("<Button-1>")

    def on_click_outside(self, event):
        """Handle clicks outside the dropdown to close it"""
        if self.dropdown_frame and self.dropdown_open:
            # Get dropdown coordinates
            dropdown_x = self.dropdown_frame.winfo_x()
            dropdown_y = self.dropdown_frame.winfo_y()
            dropdown_width = self.dropdown_frame.winfo_width()
            dropdown_height = self.dropdown_frame.winfo_height()
            
            # Check if click is outside dropdown
            if not (dropdown_x <= event.x_root - self.winfo_rootx() <= dropdown_x + dropdown_width and
                    dropdown_y <= event.y_root - self.winfo_rooty() <= dropdown_y + dropdown_height):
                self.close_dropdown()

    def select_nickname(self, nickname):
        """Handle nickname selection from dropdown"""
        print(f"Selected nickname: {nickname}")
        
        # Update the Overall button text to show selection
        if nickname == "Overall":
            self.overall_button.configure(text="Overall")
        else:
            # Truncate long nicknames for display
            display_name = nickname[:10] + "..." if len(nickname) > 10 else nickname
            self.overall_button.configure(text=display_name)
        
        # Store the selected nickname for future use
        self.selected_nickname = nickname
        
        # Close the dropdown
        self.close_dropdown()
        
        # Here you can add logic to update the views, likes, subscribers data
        # based on the selected nickname
        self.update_metrics_for_nickname(nickname)

    def update_metrics_for_nickname(self, nickname):
        """Update the dashboard metrics based on selected nickname"""
        if nickname == "Overall":
            print("Fetching overall metrics for all accounts")
            self.fetch_overall_metrics()
        else:
            print(f"Fetching metrics for: {nickname}")
            self.fetch_channel_metrics(nickname)

    # In your App class (Code 1)

    def get_credential_by_nickname(self, nickname):
        """Get the credential data for a specific nickname"""
        try:
            self.db_cursor.execute('SELECT channel_id, api_key, oauth_client_id, client_secret FROM credentials WHERE nickname = ?', (nickname,))
            result = self.db_cursor.fetchone()
            if result:
                return {
                    'nickname': nickname, # <-- ADD THIS LINE
                    'channel_id': result[0],
                    'api_key': result[1],
                    'oauth_client_id': result[2],
                    'client_secret': result[3]
                }
            return None
        except sqlite3.Error as e:
            print(f"Error retrieving credential for {nickname}: {e}")
            return None

    def fetch_channel_metrics(self, nickname):
        """Fetch YouTube metrics for a specific channel"""
        credential = self.get_credential_by_nickname(nickname)
        if not credential:
            print(f"No credentials found for {nickname}")
            self.update_metric_displays("Error", "Error", "Error")
            return
        
        try:
            import requests
            
            # YouTube Data API v3 endpoint for channel statistics
            api_key = credential['api_key']
            channel_id = credential['channel_id']
            
            url = f"https://www.googleapis.com/youtube/v3/channels"
            params = {
                'part': 'statistics',
                'id': channel_id,
                'key': api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if response.status_code == 200 and 'items' in data and len(data['items']) > 0:
                stats = data['items'][0]['statistics']
                
                # Extract metrics
                view_count = int(stats.get('viewCount', 0))
                subscriber_count = int(stats.get('subscriberCount', 0))
                video_count = int(stats.get('videoCount', 0))
                
                # Format numbers for display
                views_formatted = self.format_number(view_count)
                subscribers_formatted = self.format_number(subscriber_count)
                
                # For likes, we need to fetch recent videos since channel stats don't include total likes
                total_likes = self.fetch_total_likes(channel_id, api_key)
                likes_formatted = self.format_number(total_likes)
                
                print(f"Successfully fetched metrics for {nickname}")
                print(f"Views: {views_formatted}, Subscribers: {subscribers_formatted}, Likes: {likes_formatted}")
                
                # Update the display
                self.update_metric_displays(views_formatted, likes_formatted, subscribers_formatted)
                
            else:
                error_msg = data.get('error', {}).get('message', 'Unknown API error')
                print(f"API Error for {nickname}: {error_msg}")
                self.update_metric_displays("API Error", "API Error", "API Error")
                
        except ImportError:
            print("Error: requests library not installed. Please install with: pip install requests")
            self.update_metric_displays("Install requests", "Install requests", "Install requests")
        except Exception as e:
            print(f"Error fetching metrics for {nickname}: {e}")
            self.update_metric_displays("Error", "Error", "Error")

    def fetch_total_likes(self, channel_id, api_key):
        """Fetch total likes across recent videos for a channel"""
        try:
            import requests
            
            # First get recent videos
            videos_url = "https://www.googleapis.com/youtube/v3/search"
            videos_params = {
                'part': 'id',
                'channelId': channel_id,
                'type': 'video',
                'order': 'date',
                'maxResults': 50,  # Get last 50 videos
                'key': api_key
            }
            
            videos_response = requests.get(videos_url, params=videos_params)
            videos_data = videos_response.json()
            
            if videos_response.status_code != 200 or 'items' not in videos_data:
                return 0
            
            # Extract video IDs
            video_ids = [item['id']['videoId'] for item in videos_data['items']]
            
            if not video_ids:
                return 0
            
            # Get statistics for these videos
            stats_url = "https://www.googleapis.com/youtube/v3/videos"
            stats_params = {
                'part': 'statistics',
                'id': ','.join(video_ids),
                'key': api_key
            }
            
            stats_response = requests.get(stats_url, params=stats_params)
            stats_data = stats_response.json()
            
            if stats_response.status_code != 200 or 'items' not in stats_data:
                return 0
            
            # Sum up likes from all videos
            total_likes = 0
            for video in stats_data['items']:
                likes = video.get('statistics', {}).get('likeCount', 0)
                if likes:
                    total_likes += int(likes)
            
            return total_likes
            
        except Exception as e:
            print(f"Error fetching likes: {e}")
            return 0

    def fetch_overall_metrics(self):
        """Fetch and aggregate metrics for all channels"""
        credentials = self.get_all_credentials()
        
        total_views = 0
        total_likes = 0
        total_subscribers = 0
        successful_fetches = 0
        
        for cred in credentials:
            nickname = cred[1]  # nickname is at index 1
            credential_data = self.get_credential_by_nickname(nickname)
            
            if not credential_data:
                continue
                
            try:
                import requests
                
                api_key = credential_data['api_key']
                channel_id = credential_data['channel_id']
                
                # Fetch channel statistics
                url = f"https://www.googleapis.com/youtube/v3/channels"
                params = {
                    'part': 'statistics',
                    'id': channel_id,
                    'key': api_key
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                if response.status_code == 200 and 'items' in data and len(data['items']) > 0:
                    stats = data['items'][0]['statistics']
                    
                    total_views += int(stats.get('viewCount', 0))
                    total_subscribers += int(stats.get('subscriberCount', 0))
                    
                    # Get likes for this channel
                    channel_likes = self.fetch_total_likes(channel_id, api_key)
                    total_likes += channel_likes
                    
                    successful_fetches += 1
                    print(f"Fetched data for {nickname}")
                else:
                    print(f"Failed to fetch data for {nickname}")
                    
            except Exception as e:
                print(f"Error fetching data for {nickname}: {e}")
        
        if successful_fetches > 0:
            # Format aggregated numbers
            views_formatted = self.format_number(total_views)
            likes_formatted = self.format_number(total_likes)
            subscribers_formatted = self.format_number(total_subscribers)
            
            print(f"Overall metrics - Views: {views_formatted}, Likes: {likes_formatted}, Subscribers: {subscribers_formatted}")
            self.update_metric_displays(views_formatted, likes_formatted, subscribers_formatted)
        else:
            print("No successful data fetches for overall metrics")
            self.update_metric_displays("No Data", "No Data", "No Data")

    def format_number(self, num):
        """Format large numbers with K, M, B suffixes"""
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.1f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}K"
        else:
            return str(num)

    def create_metric_displays(self):
        """Create or update the metric display labels in the header"""
        # You'll need to modify your create_header function to store references to the metric labels
        # so they can be updated. Add these lines after creating the labels in create_header():
        
        # Store references for updating
        self.current_views = "0"
        self.current_likes = "0" 
        self.current_subscribers = "0"

    def update_metric_displays(self, views, likes, subscribers):
        """Update the metric displays with new values"""
        # Update the label texts
        self.views_label.configure(text=f"Views: {views}")
        self.likes_label.configure(text=f"Likes: {likes}")
        self.subscribers_label.configure(text=f"Subscribers: {subscribers}")
        
        # Store current values
        self.current_views = views
        self.current_likes = likes
        self.current_subscribers = subscribers

    def get_next_vid_id(self):
        """Get the next auto-increment ID that will be used for VID_ID"""
        try:
            # Get the next auto-increment value
            self.db_cursor.execute("SELECT seq FROM sqlite_sequence WHERE name='pending_uploads'")
            result = self.db_cursor.fetchone()
            if result:
                return result[0] + 1
            else:
                return 1  # First entry
        except sqlite3.Error as e:
            print(f"Error getting next VID_ID: {e}")
            return 1  # Default to 1 if error
        
    def add_storyform_to_pending(self, topic):
        """Add storyform to pending uploads using threaded database operation"""
        try:
            self.db_cursor.execute('''
                INSERT INTO pending_uploads (account, title, uploaded)
                VALUES (?, ?, ?)
            ''', ("STORYFORM", topic, False))
            
            self.db_connection.commit()
            vid_id = self.db_cursor.lastrowid
            print(f"Storyform video added to pending uploads - VID_ID: {vid_id}, Topic: '{topic}'")
            return vid_id
            
        except sqlite3.Error as e:
            print(f"Error adding storyform to pending uploads: {e}")
            return None
                
    def create_colored_sections(self):
        """Creating colored sections in the specified grid positions"""
        
        # Reddit automation box - rows 1,2,3,4,5,6 columns 0,1,2,3,4 (expanded to absorb red section space)
        self.reddit_automation = ctk.CTkFrame(self, fg_color='#000000')
        self.reddit_automation.grid(row=1, column=0, columnspan=5, rowspan=6, sticky='nsew', padx=5, pady=5)
        self.reddit_automation.grid_columnconfigure((0, 1), weight=1)  # Two equal columns
        # MODIFIED: Adjusted row weights for better vertical centering
        self.reddit_automation.grid_rowconfigure(0, weight=0) # Title
        self.reddit_automation.grid_rowconfigure(1, weight=1) # Left side (button/status)
        self.reddit_automation.grid_rowconfigure(2, weight=1) # Spacer
        
        self.reddit_automation_label = ctk.CTkLabel(
            self.reddit_automation, 
            text="REDDIT AUTOMATION", 
            font=("Arial", 24, "bold"), 
            text_color="white"
        )
        # MODIFIED: Added padding
        self.reddit_automation_label.grid(row=0, column=0, columnspan=2, pady=(10, 5))
        
        # Create a frame for the button and status dot in the left half
        self.button_frame = ctk.CTkFrame(self.reddit_automation, fg_color='transparent')
        self.button_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_rowconfigure((0, 1), weight=1)
        
        # Initialize activation status
        self.is_activated = False
        
        # RUN button - now styled with dark gray background and white text
        self.run_button = ctk.CTkButton(
            self.button_frame,
            text="RUN",
            font=("Arial", 16, "bold"),
            width=120,
            height=40,
            fg_color="#323232",
            text_color="white",
            hover_color="#404040",
            command=self.toggle_activation
        )
        self.run_button.grid(row=0, column=0, pady=(0, 10))
        
        # Status frame for dot and label
        self.status_frame = ctk.CTkFrame(self.button_frame, fg_color='transparent')
        self.status_frame.grid(row=1, column=0, sticky='')
        
        # Status dot canvas
        self.status_canvas = ctk.CTkCanvas(self.status_frame, width=20, height=20, bg='#000000', highlightthickness=0)
        self.status_canvas.pack(side="left", padx=(0, 5))
        
        # Draw initial red dot
        self.status_dot = self.status_canvas.create_oval(2, 2, 18, 18, fill="#FF0000", outline="#FF0000")
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="INACTIVE",
            font=("Arial", 12, "bold"),
            text_color="white"
        )
        self.status_label.pack(side="left")
        
        # Right side controls frame
        self.controls_frame = ctk.CTkFrame(self.reddit_automation, fg_color='transparent')
        self.controls_frame.grid(row=1, column=1, sticky='nsew', padx=10, pady=10)
        self.controls_frame.grid_columnconfigure(0, weight=1)
        # MODIFIED: Added more rows for the new VID_ID dropdown
        self.controls_frame.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7), weight=1)
    
        # Story type dropdown
        self.story_type_label = ctk.CTkLabel(self.controls_frame, text="STORY TYPE:", font=("Arial", 14, "bold"), text_color="white")
        self.story_type_label.grid(row=0, column=0, sticky='w', pady=(0, 5))
        self.story_type_dropdown = ctk.CTkComboBox(self.controls_frame, values=["AITAH", "NOSLEEP"], font=("Arial", 12), fg_color="#323232", button_color="#404040", button_hover_color="#505050", text_color="white", dropdown_hover_color="#404040", state="readonly")
        self.story_type_dropdown.set("AITAH")
        self.story_type_dropdown.grid(row=1, column=0, sticky='ew', pady=(0, 15))
    
        # ADDED: Custom VID_ID Entry for Reddit
        self.reddit_vid_id_label = ctk.CTkLabel(
            self.controls_frame,
            text="CUSTOM VID_ID (OPTIONAL):",
            font=("Arial", 14, "bold"),
            text_color="white"
        )
        self.reddit_vid_id_label.grid(row=2, column=0, sticky='w', pady=(0, 5))
        self.reddit_vid_id_entry = ctk.CTkEntry(
            self.controls_frame,
            placeholder_text="e.g., V124 - Leave blank for auto",
            font=("Arial", 12),
            fg_color="#323232",
            text_color="white",
            placeholder_text_color="#888888",
            border_color="#404040",
            height=35
        )
        self.reddit_vid_id_entry.grid(row=3, column=0, sticky='ew', pady=(0, 15))
        
        # Time dropdown
        self.time_label = ctk.CTkLabel(self.controls_frame, text="TIME:", font=("Arial", 14, "bold"), text_color="white")
        # MODIFIED: Changed row from 2 to 4
        self.time_label.grid(row=4, column=0, sticky='w', pady=(0, 5))
        self.time_controls_frame = ctk.CTkFrame(self.controls_frame, fg_color='transparent')
        # MODIFIED: Changed row from 3 to 5
        self.time_controls_frame.grid(row=5, column=0, sticky='ew', pady=(0, 15))
        self.time_controls_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Initialize time dropdown storage
        self.time_dropdowns = []
        
        # Create initial time dropdown for daily (1 dropdown)
        self.create_time_dropdowns(1)
        
        # Post frequency label
        self.frequency_label = ctk.CTkLabel(self.controls_frame, text="POST FREQUENCY:", font=("Arial", 14, "bold"), text_color="white")
        # MODIFIED: Changed row from 4 to 6
        self.frequency_label.grid(row=6, column=0, sticky='w', pady=(0, 5))
        
        # Post frequency selection buttons frame
        self.frequency_frame = ctk.CTkFrame(self.controls_frame, fg_color='transparent')
        # MODIFIED: Changed row from 5 to 7
        self.frequency_frame.grid(row=7, column=0, sticky='ew')
        self.frequency_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Initialize frequency selection
        self.selected_frequency = "daily"
        
        # Frequency buttons
        self.daily_button = ctk.CTkButton(
            self.frequency_frame,
            text="Daily",
            width=70,
            height=30,
            font=("Arial", 10, "bold"),
            fg_color="#00FF00",  # Start selected (green)
            text_color="black",
            hover_color="#00CC00",
            command=lambda: self.select_frequency("daily")
        )
        self.daily_button.grid(row=0, column=0, padx=(0, 2), sticky='ew')
        
        self.twice_daily_button = ctk.CTkButton(
            self.frequency_frame,
            text="2x Daily",
            width=70,
            height=30,
            font=("Arial", 10, "bold"),
            fg_color="#323232",  # Unselected (dark gray)
            text_color="white",
            hover_color="#404040",
            command=lambda: self.select_frequency("2x daily")
        )
        self.twice_daily_button.grid(row=0, column=1, padx=2, sticky='ew')
        
        self.four_times_daily_button = ctk.CTkButton(
            self.frequency_frame,
            text="4x Daily",
            width=70,
            height=30,
            font=("Arial", 10, "bold"),
            fg_color="#323232",  # Unselected (dark gray)
            text_color="white",
            hover_color="#404040",
            command=lambda: self.select_frequency("4x daily")
        )
        self.four_times_daily_button.grid(row=0, column=2, padx=(2, 0), sticky='ew')
        
        # Storyform automation box - rows 1,2,3,4,5,6 columns 5,6,7,8,9 (combined green and yellow space)
        self.storyform_automation = ctk.CTkFrame(self, fg_color='#000000')  # Black background
        self.storyform_automation.grid(row=1, column=5, columnspan=5, rowspan=6, sticky='nsew', padx=5, pady=5)
        self.storyform_automation.grid_columnconfigure(0, weight=1)
        # Adjusted row weights for better spacing
        self.storyform_automation.grid_rowconfigure(0, weight=0) # Title
        self.storyform_automation.grid_rowconfigure(1, weight=1) # Topic
        self.storyform_automation.grid_rowconfigure(2, weight=1) # VID_ID
        self.storyform_automation.grid_rowconfigure(3, weight=1) # Chapter
        self.storyform_automation.grid_rowconfigure(4, weight=1) # Controls
        
        self.storyform_automation_label = ctk.CTkLabel(
            self.storyform_automation, 
            text="STORYFORM AUTOMATION", 
            font=("Arial", 24, "bold"), 
            text_color="white"
        )
        self.storyform_automation_label.grid(row=0, column=0, sticky='', pady=(10, 15))
        
        # Topic entry section
        self.storyform_topic_frame = ctk.CTkFrame(self.storyform_automation, fg_color='transparent')
        self.storyform_topic_frame.grid(row=1, column=0, sticky='nsew', padx=20, pady=(0, 10))
        self.storyform_topic_frame.grid_columnconfigure(0, weight=1)
        
        self.storyform_topic_label = ctk.CTkLabel(
            self.storyform_topic_frame,
            text="TOPIC:",
            font=("Arial", 14, "bold"),
            text_color="white"
        )
        self.storyform_topic_label.grid(row=0, column=0, sticky='w', pady=(0, 5))
        
        self.storyform_topic_entry = ctk.CTkEntry(
            self.storyform_topic_frame,
            placeholder_text="Enter topic here...",
            font=("Arial", 12),
            fg_color="#323232",
            text_color="white",
            placeholder_text_color="#888888",
            border_color="#404040",
            height=35
        )
        self.storyform_topic_entry.grid(row=1, column=0, sticky='ew')
    
        # ADDED: VID_ID entry section
        self.vid_id_frame = ctk.CTkFrame(self.storyform_automation, fg_color='transparent')
        self.vid_id_frame.grid(row=2, column=0, sticky='nsew', padx=20, pady=(0,10))
        self.vid_id_frame.grid_columnconfigure(0, weight=1)
    
        self.vid_id_label = ctk.CTkLabel(
            self.vid_id_frame,
            text="CUSTOM VID_ID (OPTIONAL):",
            font=("Arial", 14, "bold"),
            text_color="white"
        )
        self.vid_id_label.grid(row=0, column=0, sticky='w', pady=(0,5))
    
        self.vid_id_entry = ctk.CTkEntry(
            self.vid_id_frame,
            placeholder_text="e.g., V123 - Leave blank for auto",
            font=("Arial", 12),
            fg_color="#323232",
            text_color="white",
            placeholder_text_color="#888888",
            border_color="#404040",
            height=35
        )
        self.vid_id_entry.grid(row=1, column=0, sticky='ew')
        
        # Chapter selection section
        self.chapter_frame = ctk.CTkFrame(self.storyform_automation, fg_color='transparent')
        self.chapter_frame.grid(row=3, column=0, sticky='nsew', padx=20, pady=(0, 10))
        self.chapter_frame.grid_columnconfigure(0, weight=1)
        
        # Chapter label
        self.chapter_label = ctk.CTkLabel(
            self.chapter_frame,
            text="CHAPTER SELECTION:",
            font=("Arial", 14, "bold"),
            text_color="white"
        )
        self.chapter_label.grid(row=0, column=0, sticky='w', pady=(0, 10))
        
        # Chapter buttons container
        self.chapter_buttons_frame = ctk.CTkFrame(self.chapter_frame, fg_color='transparent')
        self.chapter_buttons_frame.grid(row=1, column=0, sticky='ew')
        
        # Configure grid for chapter buttons (6 columns, 2 rows for 0-10)
        for i in range(6):
            self.chapter_buttons_frame.grid_columnconfigure(i, weight=1)
        self.chapter_buttons_frame.grid_rowconfigure((0, 1), weight=1)
        
        # Initialize chapter selection
        self.selected_chapter = 1
        self.chapter_buttons = []
        
        # Create chapter buttons 1-10
        for i in range(1, 11):  # 1 to 10
            row = 0 if i <= 5 else 1
            col = (i - 1) if i <= 5 else (i - 6)
            
            # First button (1) starts selected
            fg_color = "#00FF00" if i == 1 else "#323232"
            text_color = "black" if i == 1 else "white"
            hover_color = "#00CC00" if i == 1 else "#404040"
            
            chapter_btn = ctk.CTkButton(
                self.chapter_buttons_frame,
                text=str(i),
                width=50,
                height=30,
                font=("Arial", 12, "bold"),
                fg_color=fg_color,
                text_color=text_color,
                hover_color=hover_color,
                command=lambda ch=i: self.select_chapter(ch)
            )
            chapter_btn.grid(row=row, column=col, padx=2, pady=2, sticky='ew')
            self.chapter_buttons.append(chapter_btn)
        
        # Run button and status section
        self.storyform_control_frame = ctk.CTkFrame(self.storyform_automation, fg_color='transparent')
        self.storyform_control_frame.grid(row=4, column=0, sticky='ew', padx=20, pady=(10, 20)) # Added more bottom padding
        self.storyform_control_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Run button (left side)
        self.storyform_run_button = ctk.CTkButton(
            self.storyform_control_frame,
            text="RUN",
            font=("Arial", 16, "bold"),
            width=120,
            height=40,
            fg_color="#323232",
            text_color="white",
            hover_color="#404040",
            command=lambda: self.run_storyform_automation()
        )
        self.storyform_run_button.grid(row=0, column=0, sticky='w', pady=10)
        
        # Status frame (right side) - matching reddit automation style
        self.storyform_status_frame = ctk.CTkFrame(self.storyform_control_frame, fg_color='#1a1a1a', height=40)
        self.storyform_status_frame.grid(row=0, column=1, sticky='e', pady=10)
        self.storyform_status_frame.grid_propagate(False)
        
        # Initialize storyform status
        self.storyform_is_active = False
        
        # Status dot (red for inactive, green for active)
        self.storyform_status_dot = ctk.CTkLabel(
            self.storyform_status_frame,
            text="●",
            font=("Arial", 20),
            text_color="#FF0000"  # Red for inactive
        )
        self.storyform_status_dot.pack(side="left", padx=(10, 5), pady=10)
        
        # Status label
        self.storyform_status_label = ctk.CTkLabel(
            self.storyform_status_frame,
            text="INACTIVE",
            font=("Arial", 12, "bold"),
            text_color="white"
        )
        self.storyform_status_label.pack(side="left", padx=(0, 10), pady=10)
        
        # Orange section (for layout testing)
        self.orange_section = ctk.CTkFrame(self, fg_color='#FF8C00')
        self.orange_section.grid(row=7, column=5, columnspan=5, rowspan=2, sticky='nsew', padx=5, pady=5)
    
    def select_chapter(self, chapter):
        """Handle chapter selection and update button colors"""
        self.selected_chapter = chapter
        
        # Reset all buttons to unselected state
        for i, btn in enumerate(self.chapter_buttons, 1):  # Start from 1
            btn.configure(fg_color="#323232", text_color="white", hover_color="#404040")
        
        # Set selected button to green (chapter is 1-10, but list index is 0-9)
        self.chapter_buttons[chapter - 1].configure(fg_color="#00FF00", text_color="black", hover_color="#00CC00")
        
        print(f"Chapter {chapter} selected")
    
    def run_storyform_automation(self):
        """Handle storyform automation run button with threading"""
        topic = self.storyform_topic_entry.get().strip()
        
        if not topic:
            print("Error: Please enter a topic before running storyform automation")
            return
        
        # Toggle active state
        self.storyform_is_active = not self.storyform_is_active
        
        if self.storyform_is_active:
            self.storyform_status_label.configure(text="ACTIVE")
            self.storyform_status_dot.configure(text_color="#00FF00")
            self.storyform_run_button.configure(text="STOP")
            print(f"Storyform automation started - Topic: '{topic}', Chapter: {self.selected_chapter}")
            
            # Add to pending uploads and get VID_ID
            vid_id = self.add_storyform_to_pending(topic)
            
            if vid_id:
                # Start storyform in dedicated thread
                self.start_storyform_thread(vid_id, topic, self.selected_chapter)
                
                # Refresh database display to show new entry
                self.refresh_database_display()
            else:
                print("Error: Failed to create database entry")
                # Reset status if database operation failed
                self.storyform_is_active = False
                self.storyform_status_label.configure(text="INACTIVE")
                self.storyform_status_dot.configure(text_color="#FF0000")
                self.storyform_run_button.configure(text="RUN")
        else:
            self.stop_storyform_thread()
            self.storyform_status_label.configure(text="INACTIVE")
            self.storyform_status_dot.configure(text_color="#FF0000")
            self.storyform_run_button.configure(text="RUN")
            print("Storyform automation stopped")

    def start_storyform_thread(self, vid_id, topic, chapter):
        """Start storyform automation in dedicated thread with CPU affinity"""
        def storyform_worker():
            try:
                # Set CPU affinity for storyform cores
                available_cores = list(range(3, 3 + self.core_allocation['storyform']))
                self.set_thread_affinity(available_cores)
                
                print(f"Storyform thread started on cores: {available_cores}")
                
                # Send status update
                self.thread_queue.put({
                    'type': 'status_update',
                    'component': 'storyform',
                    'status': 'running'
                })
                
                # Run the actual storyform function
                run_storyform(VID_ID=f"V{vid_id}", TOPIC=topic, CHAPTER_NUM=chapter)
                
                # Send completion status
                self.thread_queue.put({
                    'type': 'status_update',
                    'component': 'storyform',
                    'status': 'completed'
                })
                
            except Exception as e:
                print(f"Storyform thread error: {e}")
                self.thread_queue.put({
                    'type': 'status_update',
                    'component': 'storyform',
                    'status': 'error'
                })
        
        # Stop existing thread if running
        self.stop_storyform_thread()
        
        # Start new thread
        self.active_threads['storyform'] = threading.Thread(target=storyform_worker, daemon=True)
        self.active_threads['storyform'].start()

    def stop_storyform_thread(self):
        """Stop storyform thread"""
        if self.active_threads['storyform'] and self.active_threads['storyform'].is_alive():
            print("Stopping storyform thread...")
            # Note: Python threads can't be forcefully killed, the function would need to check for stop signals
            # For now, we just mark it as inactive
            self.active_threads['storyform'] = None

    def add_reddit_to_pending(self):
        """Add a new reddit video to pending uploads with reddit automation settings"""
        try:
            # Get current reddit settings
            settings = self.get_reddit_settings()
            
            # MODIFIED: Create a title based on settings, including story type
            title = f"Reddit Automation - {settings['story_type']} - {settings['frequency']} at {', '.join(settings['times'])}"
            
            self.db_cursor.execute('''
                INSERT INTO pending_uploads (account, title, uploaded)
                VALUES (?, ?, ?)
            ''', ("REDDIT", title, False))
            
            self.db_connection.commit()
            vid_id = self.db_cursor.lastrowid
            print(f"Reddit video added to pending uploads - VID_ID: {vid_id}, Settings: {settings}")
            return vid_id
            
        except sqlite3.Error as e:
            print(f"Error adding reddit to pending uploads: {e}")
            return None
        
    # MODIFIED: Added story_type parameter to decide which script to run
    # MODIFIED: Added story_type parameter to decide which script to run
    def start_reddit_thread_with_vid(self, vid_id, story_type, nickname):
        """Start reddit automation in dedicated thread with VID_ID, story type, and specific nickname."""
        
        credentials = self.get_credential_by_nickname(nickname)
        if not credentials:
            print(f"Run Error: Could not find credentials for '{nickname}'. Aborting.")
            return

        if not credentials.get('oauth_client_id') or not credentials.get('client_secret'):
            print(f"Run Warning: Credentials for '{nickname}' are incomplete. Video will be generated but not uploaded.")
            # Set to None to ensure the automation script handles this case
            credentials['oauth_client_id'] = None
            credentials['client_secret'] = None

        def reddit_worker():
            try:
                # Set CPU affinity for reddit cores
                start_core = 3 + self.core_allocation['storyform']
                available_cores = list(range(start_core, start_core + self.core_allocation['reddit']))
                self.set_thread_affinity(available_cores)

                print(f"Reddit thread for '{nickname}' started on cores: {available_cores}")

                self.thread_queue.put({
                    'type': 'status_update',
                    'component': 'reddit',
                    'status': f'running {story_type} for {nickname}'
                })
                
                # Pass credentials to the correct run function
                if story_type == "NOSLEEP":
                    print(f"Executing r/nosleep script for VID_ID: {vid_id} on account '{nickname}'")
                    run_reddit_horror(VID_ID=vid_id, credentials=credentials)
                else: # AITAH Stories
                    print(f"Executing AITAH script for VID_ID: {vid_id} on account '{nickname}'")
                    run_reddit(VID_ID=vid_id, credentials=credentials)

                self.thread_queue.put({
                    'type': 'status_update',
                    'component': 'reddit',
                    'status': f'completed for {nickname}'
                })

            except Exception as e:
                print(f"Reddit thread error for '{nickname}': {e}")
                traceback.print_exc()
                self.thread_queue.put({
                    'type': 'status_update',
                    'component': 'reddit',
                    'status': f'error for {nickname}'
                })

        # Stop existing manual thread if running
        self.stop_reddit_thread()

        # Start new thread
        self.active_threads['reddit'] = threading.Thread(target=reddit_worker, daemon=True)
        self.active_threads['reddit'].start()

        def reddit_worker():
            try:
                # Set CPU affinity for reddit cores
                start_core = 3 + self.core_allocation['storyform']
                available_cores = list(range(start_core, start_core + self.core_allocation['reddit']))
                self.set_thread_affinity(available_cores)

                print(f"Reddit thread started on cores: {available_cores}")

                self.thread_queue.put({
                    'type': 'status_update',
                    'component': 'reddit',
                    'status': f'running {story_type}'
                })

                # MODIFIED: Pass credentials to the correct run function
                if story_type == "NOSLEEP":
                    print(f"Executing r/nosleep script for VID_ID: {vid_id}")
                    # Pass credentials to run_reddit_horror as well
                    run_reddit_horror(VID_ID=vid_id, credentials=credentials)
                else: # AITAH Stories
                    print(f"Executing AITAH script for VID_ID: {vid_id}")
                    # Pass credentials to run_reddit
                    run_reddit(VID_ID=vid_id, credentials=credentials)

                self.thread_queue.put({
                    'type': 'status_update',
                    'component': 'reddit',
                    'status': 'completed'
                })

            except Exception as e:
                print(f"Reddit thread error: {e}")
                traceback.print_exc() # Print full traceback for easier debugging
                self.thread_queue.put({
                    'type': 'status_update',
                    'component': 'reddit',
                    'status': 'error'
                })

        # Stop existing thread if running
        self.stop_reddit_thread()

        # Start new thread
        self.active_threads['reddit'] = threading.Thread(target=reddit_worker, daemon=True)
        self.active_threads['reddit'].start()

    def toggle_activation(self):
        """Toggle the reddit automation status and run the appropriate function."""
        # This function is now for MANUAL runs only. Scheduling is handled separately.
        self.is_activated = not self.is_activated

        if self.is_activated:
            self.status_canvas.itemconfig(self.status_dot, fill="#00FF00", outline="#00FF00")
            self.status_label.configure(text="MANUAL RUN ACTIVE")
            
            story_type = self.story_type_dropdown.get()
            custom_vid_id_str = self.reddit_vid_id_entry.get().strip()
            nickname = self.selected_nickname

            if nickname == "Overall":
                print("Manual Run Error: Cannot run with 'Overall' selected. Please select a specific account.")
                self.is_activated = False
                self.status_canvas.itemconfig(self.status_dot, fill="#FF0000", outline="#FF0000")
                self.status_label.configure(text="INACTIVE")
                return

            vid_to_use = None
            if custom_vid_id_str:
                print(f"Manual run activated for {story_type} with custom VID_ID: {custom_vid_id_str}!")
                vid_to_use = custom_vid_id_str
            else:
                print(f"Manual run activated for {story_type}! Generating new VID_ID.")
                vid_id_num = self.add_reddit_to_pending()
                if vid_id_num:
                    vid_to_use = f"V{vid_id_num}"
                    self.refresh_database_display()
                else:
                    print("Error: Failed to create database entry for manual run.")
                    self.is_activated = False
                    self.status_canvas.itemconfig(self.status_dot, fill="#FF0000", outline="#FF0000")
                    self.status_label.configure(text="INACTIVE")
                    return
            
            if vid_to_use:
                # Pass the selected nickname to the thread starter
                self.start_reddit_thread_with_vid(vid_to_use, story_type, nickname)

        else:
            self.status_canvas.itemconfig(self.status_dot, fill="#FF0000", outline="#FF0000")
            self.status_label.configure(text="INACTIVE")
            print("Manual run deactivated!")
            self.stop_reddit_thread()


    def start_reddit_thread(self):
        """Start reddit automation in dedicated thread with CPU affinity"""
        def reddit_worker():
            try:
                # Set CPU affinity for reddit cores
                start_core = 3 + self.core_allocation['storyform']
                available_cores = list(range(start_core, start_core + self.core_allocation['reddit']))
                self.set_thread_affinity(available_cores)
                
                print(f"Reddit thread started on cores: {available_cores}")
                
                # Send status update
                self.thread_queue.put({
                    'type': 'status_update',
                    'component': 'reddit',
                    'status': 'running'
                })
                
                # Reddit automation loop
                while self.is_activated:
                    # Get current settings
                    settings = self.get_reddit_settings()
                    
                    # Your reddit automation logic here
                    print(f"Reddit automation running with settings: {settings}")
                    
                    # Sleep for a bit before next iteration
                    time.sleep(30)  # Adjust timing as needed
                
                # Send completion status
                self.thread_queue.put({
                    'type': 'status_update',
                    'component': 'reddit',
                    'status': 'stopped'
                })
                
            except Exception as e:
                print(f"Reddit thread error: {e}")
                self.thread_queue.put({
                    'type': 'status_update',
                    'component': 'reddit',
                    'status': 'error'
                })
        
        # Stop existing thread if running
        self.stop_reddit_thread()
        
        # Start new thread
        self.active_threads['reddit'] = threading.Thread(target=reddit_worker, daemon=True)
        self.active_threads['reddit'].start()
    
    def stop_reddit_thread(self):
        """Stop reddit thread"""
        if self.active_threads['reddit'] and self.active_threads['reddit'].is_alive():
            print("Stopping reddit thread...")
            self.active_threads['reddit'] = None
    
    def threaded_database_operation(self, operation, *args, **kwargs):
        """Run database operations in dedicated thread"""
        def db_worker():
            try:
                # Set CPU affinity for database core
                self.set_thread_affinity([1])  # Core 1 for database
                
                # Execute the database operation
                result = operation(*args, **kwargs)
                
                # Send result back to main thread
                self.database_queue.put({
                    'type': 'refresh',
                    'result': result
                })
                
            except Exception as e:
                self.database_queue.put({
                    'type': 'error',
                    'error': str(e)
                })
        
        # Run in background thread
        thread = threading.Thread(target=db_worker, daemon=True)
        thread.start()
    
        
    def select_frequency(self, frequency):
        """Handle frequency selection and update button colors"""
        self.selected_frequency = frequency
        
        # Reset all buttons to unselected state
        self.daily_button.configure(fg_color="#323232", text_color="white", hover_color="#404040")
        self.twice_daily_button.configure(fg_color="#323232", text_color="white", hover_color="#404040")
        self.four_times_daily_button.configure(fg_color="#323232", text_color="white", hover_color="#404040")
        
        # Set selected button to green
        if frequency == "daily":
            self.daily_button.configure(fg_color="#00FF00", text_color="black", hover_color="#00CC00")
        elif frequency == "2x daily":
            self.twice_daily_button.configure(fg_color="#00FF00", text_color="black", hover_color="#00CC00")
        elif frequency == "4x daily":
            self.four_times_daily_button.configure(fg_color="#00FF00", text_color="black", hover_color="#00CC00")
        
        print(f"Post frequency selected: {frequency}")
        
        # Update time dropdowns based on frequency
        frequency_to_count = {
            "daily": 1,
            "2x daily": 2,
            "4x daily": 4
        }
        self.create_time_dropdowns(frequency_to_count[frequency])
    
    def create_time_dropdowns(self, count):
        """Create the specified number of time dropdown menus with labels"""
        # Clear existing dropdowns and labels
        for widget in self.time_dropdowns:
            widget.destroy()
        self.time_dropdowns.clear()
        
        # Generate time options in 30-minute increments (military time)
        time_options = []
        for hour in range(24):
            for minute in [0, 30]:
                time_str = f"{hour:02d}:{minute:02d}"
                time_options.append(time_str)
        
        # Configure grid based on count
        if count == 1:
            # Single dropdown, full width
            self.time_controls_frame.grid_rowconfigure((0, 1), weight=1)
            self.time_controls_frame.grid_rowconfigure(2, weight=0)
        elif count == 2:
            # Two dropdowns side by side
            self.time_controls_frame.grid_rowconfigure((0, 1), weight=1)
            self.time_controls_frame.grid_rowconfigure(2, weight=0)
        elif count == 4:
            # Four dropdowns, 2x2 grid
            self.time_controls_frame.grid_rowconfigure((0, 1, 2, 3), weight=1)
        
        # Create dropdowns with labels based on count
        default_times = ["08:00", "20:00", "06:00", "18:00"]  # Default times for multiple selections
        time_labels = ["Time 1:", "Time 2:", "Time 3:", "Time 4:"]
        
        for i in range(count):
            # Calculate grid position for labels and dropdowns
            if count == 1:
                label_row, dropdown_row = 0, 1
                col = 0
                columnspan = 2
            elif count == 2:
                label_row, dropdown_row = 0, 1
                col = i
                columnspan = 1
            elif count == 4:
                label_row = (i // 2) * 2  # 0, 0, 2, 2
                dropdown_row = label_row + 1  # 1, 1, 3, 3
                col = i % 2
                columnspan = 1
            
            # Create label
            time_label = ctk.CTkLabel(
                self.time_controls_frame,
                text=time_labels[i] if count > 1 else "Time:",
                font=("Arial", 11, "bold"),
                text_color="white"
            )
            
            # Position label
            if count == 1:
                time_label.grid(row=label_row, column=col, columnspan=columnspan, sticky='w', padx=5, pady=(2, 0))
            else:
                time_label.grid(row=label_row, column=col, columnspan=columnspan, sticky='w', padx=2, pady=(2, 0))
            
            # Create dropdown
            time_dropdown = ctk.CTkComboBox(
                self.time_controls_frame,
                values=time_options,
                width=120 if count == 1 else 110,
                height=30,
                font=("Arial", 12),
                fg_color="#323232",
                button_color="#404040",
                button_hover_color="#505050",
                text_color="white",
                dropdown_hover_color="#404040",
                state="readonly"
            )
            
            # Position dropdown
            if count == 1:
                time_dropdown.grid(row=dropdown_row, column=col, columnspan=columnspan, sticky='ew', padx=5, pady=(0, 2))
            else:
                time_dropdown.grid(row=dropdown_row, column=col, columnspan=columnspan, sticky='ew', padx=2, pady=(0, 2))
            
            # Set default value
            if i < len(default_times):
                time_dropdown.set(default_times[i])
            else:
                time_dropdown.set("12:00")
            
            # Store both label and dropdown references
            self.time_dropdowns.append(time_label)
            self.time_dropdowns.append(time_dropdown)
    
    def get_reddit_settings(self):
        """Get current reddit automation settings for use in functions"""
        # Get all selected times (extract only dropdowns, skip labels)
        selected_times = []
        for i, widget in enumerate(self.time_dropdowns):
            # Only get values from dropdown widgets (every odd index after labels)
            if i % 2 == 1:  # Dropdown widgets are at odd indices
                selected_times.append(widget.get())
        
        # MODIFIED: Added story_type to the returned dictionary
        return {
            "story_type": self.story_type_dropdown.get(),
            "times": selected_times,
            "frequency": self.selected_frequency,
            "is_active": self.is_activated
        }
    
    def submit_storyform_topic(self):
        """Handle storyform topic submission"""
        topic = self.topic_entry.get().strip()
        
        if topic:
            print(f"Storyform topic submitted: {topic}")
            # Clear the entry field after submission
            self.topic_entry.delete(0, 'end')
        else:
            print("Error: Please enter a topic before submitting")
    
    def get_storyform_settings(self):
        """Get current storyform automation settings for use in functions"""
        return {
            "topic": self.topic_entry.get().strip()
        }
    
    def setup_database(self):
        """Initialize and migrate the database, creating all necessary tables."""
        try:
            # Create database connection
            self.db_connection = sqlite3.connect('storage.db', check_same_thread=False)
            self.db_cursor = self.db_connection.cursor()

            # --- Table Creation (run this first) ---
            # Use IF NOT EXISTS to prevent errors if tables are already there.

            # 1. Main credentials table with scheduling fields AND automation_type
            self.db_cursor.execute('''
                CREATE TABLE IF NOT EXISTS credentials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nickname TEXT NOT NULL UNIQUE,
                    api_key TEXT,
                    channel_id TEXT,
                    oauth_client_id TEXT,
                    client_secret TEXT,
                    automation_active INTEGER DEFAULT 0, -- 0 for false, 1 for true
                    frequency TEXT,
                    times TEXT, -- Comma-separated times, e.g., "08:00,20:00"
                    automation_type TEXT -- ADDED: To store 'Storyform', 'Reddit - AITAH', or 'Reddit - NOSLEEP'
                )
            ''')

            # This is a safe way to add the new column if the table already exists from a previous run
            try:
                self.db_cursor.execute('ALTER TABLE credentials ADD COLUMN automation_type TEXT')
                print("Added 'automation_type' column to credentials table.")
            except sqlite3.OperationalError:
                pass # Column already exists, which is fine, so we ignore the error.

            # 2. Table for creation logs, linked to credentials
            self.db_cursor.execute('''
                CREATE TABLE IF NOT EXISTS creation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    credential_id INTEGER NOT NULL,
                    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (credential_id) REFERENCES credentials (id) ON DELETE CASCADE
                )
            ''')
            
            # 3. Pending uploads table
            self.db_cursor.execute('''
                CREATE TABLE IF NOT EXISTS pending_uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    uploaded BOOLEAN DEFAULT FALSE
                )
            ''')

            # Commit the changes so the tables are created/updated
            self.db_connection.commit()
            print("Database 'storage.db' is up to date and all tables are present.")

        except sqlite3.Error as e:
            print(f"Database error during setup: {e}")
    
    def add_credential(self, nickname, channel_id, api_key, oauth_client_id, client_secret, automation_active, frequency, times):
        """Add a new credential to the database."""
        try:
            self.db_cursor.execute('''
                INSERT INTO credentials (nickname, channel_id, api_key, oauth_client_id, client_secret, automation_active, frequency, times)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (nickname, channel_id, api_key, oauth_client_id, client_secret, automation_active, frequency, times))
            
            cred_id = self.db_cursor.lastrowid
            
            self.db_cursor.execute('''
                INSERT INTO creation_logs (credential_id, created)
                VALUES (?, CURRENT_TIMESTAMP)
            ''', (cred_id,))

            self.db_connection.commit()
            print(f"Credential added for '{nickname}' with ID: {cred_id}")
            return cred_id
        except sqlite3.IntegrityError:
            print(f"Error: Nickname '{nickname}' already exists.")
            self.db_connection.rollback()
            return None
        except sqlite3.Error as e:
            print(f"Error adding credential: {e}")
            self.db_connection.rollback()
            return None
    
    def add_pending_upload(self, account, title):
        """Add a new pending upload to the database"""
        try:
            self.db_cursor.execute('''
                INSERT INTO pending_uploads (account, title)
                VALUES (?, ?)
            ''', (account, title))
            
            self.db_connection.commit()
            print(f"Pending upload added: '{title}' for account '{account}'")
            return self.db_cursor.lastrowid
            
        except sqlite3.Error as e:
            print(f"Error adding pending upload: {e}")
            return None
    
    def get_all_credentials(self):
        """Retrieve all credentials from the database."""
        try:
            # This now selects all columns directly from the single credentials table
            self.db_cursor.execute('''
                SELECT 
                    id, 
                    nickname, 
                    channel_id, 
                    api_key, 
                    oauth_client_id, 
                    client_secret, 
                    automation_active, 
                    frequency, 
                    times,
                    automation_type
                FROM 
                    credentials
                ORDER BY 
                    nickname ASC
            ''')
            return self.db_cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error retrieving credentials: {e}")
            return []
    
    def get_pending_uploads(self, uploaded=False):
        """Retrieve pending uploads from the database"""
        try:
            self.db_cursor.execute('''
                SELECT * FROM pending_uploads 
                WHERE uploaded = ? 
                ORDER BY created DESC
            ''', (uploaded,))
            return self.db_cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error retrieving pending uploads: {e}")
            return []
    
    def mark_upload_completed(self, upload_id):
        """Mark an upload as completed"""
        try:
            self.db_cursor.execute('''
                UPDATE pending_uploads 
                SET uploaded = TRUE 
                WHERE id = ?
            ''', (upload_id,))
            
            self.db_connection.commit()
            print(f"Upload ID {upload_id} marked as completed")
            return True
            
        except sqlite3.Error as e:
            print(f"Error updating upload status: {e}")
            return False
    
    def switch_tab(self, tab_name):
        """Switch between credentials and pending tabs"""
        self.current_tab = tab_name
        
        # Update tab button colors
        if tab_name == "credentials":
            self.credentials_tab_button.configure(fg_color="#00FF00", text_color="black", hover_color="#00CC00")
            self.pending_tab_button.configure(fg_color="#323232", text_color="white", hover_color="#404040")
        else:
            self.credentials_tab_button.configure(fg_color="#323232", text_color="white", hover_color="#404040")
            self.pending_tab_button.configure(fg_color="#00FF00", text_color="black", hover_color="#00CC00")
        
        # Refresh the display
        self.refresh_database_display()
        print(f"Switched to {tab_name} tab")
    
    def refresh_database_display(self):
        """Refresh the database display based on current tab"""
        # Temporarily enable editing to update content
        self.database_display.configure(state="normal")
        self.database_display.delete("1.0", "end")
        
        if self.current_tab == "credentials":
            self.display_credentials()
        else:
            self.display_pending_uploads()
        
        # Disable editing again to make it read-only
        self.database_display.configure(state="disabled")
    

    def display_credentials(self):
        """Display credentials table data with proper column alignment for all 10 columns."""
        try:
            credentials = self.get_all_credentials()
            
            # Header to include all new scheduling columns
            header = f"{'ID':<4} │ {'NICKNAME':<15} │ {'AUTO':<5} │ {'FREQUENCY':<10} │ {'TIMES':<18} │ {'TYPE':<18}\n"
            separator = "─" * 4 + "─┼─" + "─" * 15 + "─┼─" + "─" * 5 + "─┼─" + "─" * 10 + "─┼─" + "─" * 18 + "─┼─" + "─" * 18 + "\n"
            
            self.database_display.insert("end", "CREDENTIALS DATABASE\n")
            self.database_display.insert("end", "=" * 95 + "\n")
            self.database_display.insert("end", header)
            self.database_display.insert("end", separator)
            
            if credentials:
                for cred in credentials:
                    # Unpack all 10 values from the database row
                    (id_val, nickname, channel_id, api_key, oauth_id, 
                     oauth_secret, automation_active, frequency, times, automation_type) = cred
                    
                    # Truncate long values for display
                    nickname_display = (nickname[:12] + "...") if len(nickname) > 15 else nickname
                    active_display = "Yes" if automation_active else "No"
                    freq_display = frequency or "N/A"
                    times_display = (times[:15] + "...") if times and len(times) > 18 else (times or "N/A")
                    type_display = automation_type or "N/A"

                    # Format the line with the new columns
                    line = (f"{str(id_val):<4} │ {nickname_display:<15} │ {active_display:<5} │ "
                            f"{freq_display:<10} │ {times_display:<18} │ {type_display:<18}\n")
                    self.database_display.insert("end", line)
            else:
                self.database_display.insert("end", "No credentials found.\n")
                
        except Exception as e:
            # SAFER: Use a standard print(). It will go to the console before the
            # redirect and to the GUI terminal after. This avoids the AttributeError.
            print(f"Error displaying credentials: {e}")
            self.database_display.insert("end", f"Error loading credentials: {e}\n")
    

    def display_pending_uploads(self):
        """Display scheduled jobs and pending/completed uploads."""
        try:
            # --- Display Scheduled Jobs ---
            self.database_display.insert("end", "UPCOMING SCHEDULED VIDEOS\n")
            self.database_display.insert("end", "=" * 95 + "\n")
            
            header = f"{'ACCOUNT':<20} │ {'STORY TYPE':<15} │ {'NEXT RUN (Local Time)':<30}\n"
            separator = "─" * 20 + "─┼─" + "─" * 15 + "─┼─" + "─" * 30 + "\n"
            self.database_display.insert("end", header)
            self.database_display.insert("end", separator)

            if schedule.jobs:
                # Sort jobs by their next scheduled run time
                sorted_jobs = sorted(schedule.jobs, key=lambda j: j.next_run)
                for job in sorted_jobs:
                    # Extract nickname from the job's tags
                    nickname = list(job.tags)[0] if job.tags else "Unknown"
                    
                    # This is hardcoded in run_scheduled_automation
                    story_type = "AITAH" 
                    
                    # Format next run time for readability
                    next_run_time = job.next_run.strftime('%Y-%m-%d %H:%M:%S')

                    line = f"{nickname:<20} │ {story_type:<15} │ {next_run_time:<30}\n"
                    self.database_display.insert("end", line)
            else:
                self.database_display.insert("end", "No active schedules found.\n")

            self.database_display.insert("end", "\n\n") # Add space between sections

            # --- Display Upload History (from pending_uploads table) ---
            pending_uploads = self.get_pending_uploads(uploaded=False)
            completed_uploads = self.get_pending_uploads(uploaded=True)
            
            history_header = f"{'ID':<4} │ {'ACCOUNT':<18} │ {'TITLE':<38} │ {'CREATED':<19} │ {'STATUS':<8}\n"
            history_separator = "─" * 4 + "─┼─" + "─" * 18 + "─┼─" + "─" * 38 + "─┼─" + "─" * 19 + "─┼─" + "─" * 8 + "\n"
            
            self.database_display.insert("end", "UPLOAD & CREATION HISTORY\n")
            self.database_display.insert("end", "=" * 95 + "\n")
            self.database_display.insert("end", history_header)
            self.database_display.insert("end", history_separator)
            
            all_uploads = pending_uploads + completed_uploads
            
            if all_uploads:
                for upload in all_uploads:
                    id_val, account, title, created, uploaded = upload
                    account_display = (account[:15] + "...") if len(account) > 18 else account
                    title_display = (title[:35] + "...") if len(title) > 38 else title
                    created_display = created[:19] if created else "N/A"
                    status = "DONE" if uploaded else "PENDING"
                    
                    line = f"{str(id_val):<4} │ {account_display:<18} │ {title_display:<38} │ {created_display:<19} │ {status:<8}\n"
                    self.database_display.insert("end", line)
            else:
                self.database_display.insert("end", "No upload history found.\n")
                
        except Exception as e:
            self.database_display.insert("end", f"Error loading uploads: {e}\n")
    
    def handle_add_button(self):
        """Handle add button click based on current tab"""
        if self.current_tab == "credentials":
            self.add_credential_dialog()
        else:
            self.add_pending_upload_dialog()
    
    def handle_delete_button(self):
        """Handle delete button click based on current tab"""
        if self.current_tab == "credentials":
            self.delete_credential_dialog()
        else:
            self.delete_pending_upload_dialog()
    
    def delete_credential_dialog(self):
        """Open dialog to delete credential by ID"""
        # Create modal window
        self.delete_cred_window = ctk.CTkToplevel(self)
        self.delete_cred_window.title("Delete Credential")
        self.delete_cred_window.geometry("300x200")
        self.delete_cred_window.configure(fg_color="#202020")
        self.delete_cred_window.transient(self)
        self.delete_cred_window.grab_set()  # Make modal
        
        # Center the window
        self.delete_cred_window.after(100, lambda: self.delete_cred_window.lift())
        
        # Configure grid
        self.delete_cred_window.grid_columnconfigure(0, weight=1)
        self.delete_cred_window.grid_rowconfigure((0, 1, 2, 3), weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            self.delete_cred_window,
            text="Delete Credential",
            font=("Arial", 16, "bold"),
            text_color="white"
        )
        title_label.grid(row=0, column=0, pady=(20, 10))
        
        # Input frame
        input_frame = ctk.CTkFrame(self.delete_cred_window, fg_color="transparent")
        input_frame.grid(row=1, column=0, sticky='ew', padx=20, pady=10)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # ID field
        ctk.CTkLabel(input_frame, text="Credential ID:", font=("Arial", 12, "bold"), text_color="white").grid(row=0, column=0, sticky='w', pady=(0, 5))
        self.delete_cred_id_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Enter ID to delete...",
            font=("Arial", 12),
            fg_color="#323232",
            text_color="white",
            height=35
        )
        self.delete_cred_id_entry.grid(row=1, column=0, sticky='ew', pady=(0, 15))
        
        # Error label
        self.delete_cred_error_label = ctk.CTkLabel(
            self.delete_cred_window,
            text="",
            font=("Arial", 11),
            text_color="#FF4444"
        )
        self.delete_cred_error_label.grid(row=2, column=0, pady=5)
        
        # Button frame
        button_frame = ctk.CTkFrame(self.delete_cred_window, fg_color="transparent")
        button_frame.grid(row=3, column=0, pady=20)
        
        # Cancel button
        cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            font=("Arial", 12, "bold"),
            width=100,
            height=35,
            fg_color="#666666",
            text_color="white",
            hover_color="#555555",
            command=self.delete_cred_window.destroy
        )
        cancel_btn.pack(side="left", padx=(0, 10))
        
        # Delete button
        delete_btn = ctk.CTkButton(
            button_frame,
            text="Delete",
            font=("Arial", 12, "bold"),
            width=100,
            height=35,
            fg_color="#FF0000",
            text_color="white",
            hover_color="#CC0000",
            command=self.confirm_delete_credential
        )
        delete_btn.pack(side="left")
    
    def delete_pending_upload_dialog(self):
        """Open dialog to delete pending upload by ID"""
        # Create modal window
        self.delete_upload_window = ctk.CTkToplevel(self)
        self.delete_upload_window.title("Delete Pending Upload")
        self.delete_upload_window.geometry("300x200")
        self.delete_upload_window.configure(fg_color="#202020")
        self.delete_upload_window.transient(self)
        self.delete_upload_window.grab_set()  # Make modal
        
        # Center the window
        self.delete_upload_window.after(100, lambda: self.delete_upload_window.lift())
        
        # Configure grid
        self.delete_upload_window.grid_columnconfigure(0, weight=1)
        self.delete_upload_window.grid_rowconfigure((0, 1, 2, 3), weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            self.delete_upload_window,
            text="Delete Pending Upload",
            font=("Arial", 16, "bold"),
            text_color="white"
        )
        title_label.grid(row=0, column=0, pady=(20, 10))
        
        # Input frame
        input_frame = ctk.CTkFrame(self.delete_upload_window, fg_color="transparent")
        input_frame.grid(row=1, column=0, sticky='ew', padx=20, pady=10)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # ID field
        ctk.CTkLabel(input_frame, text="Upload ID:", font=("Arial", 12, "bold"), text_color="white").grid(row=0, column=0, sticky='w', pady=(0, 5))
        self.delete_upload_id_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Enter ID to delete...",
            font=("Arial", 12),
            fg_color="#323232",
            text_color="white",
            height=35
        )
        self.delete_upload_id_entry.grid(row=1, column=0, sticky='ew', pady=(0, 15))
        
        # Error label
        self.delete_upload_error_label = ctk.CTkLabel(
            self.delete_upload_window,
            text="",
            font=("Arial", 11),
            text_color="#FF4444"
        )
        self.delete_upload_error_label.grid(row=2, column=0, pady=5)
        
        # Button frame
        button_frame = ctk.CTkFrame(self.delete_upload_window, fg_color="transparent")
        button_frame.grid(row=3, column=0, pady=20)
        
        # Cancel button
        cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            font=("Arial", 12, "bold"),
            width=100,
            height=35,
            fg_color="#666666",
            text_color="white",
            hover_color="#555555",
            command=self.delete_upload_window.destroy
        )
        cancel_btn.pack(side="left", padx=(0, 10))
        
        # Delete button
        delete_btn = ctk.CTkButton(
            button_frame,
            text="Delete",
            font=("Arial", 12, "bold"),
            width=100,
            height=35,
            fg_color="#FF0000",
            text_color="white",
            hover_color="#CC0000",
            command=self.confirm_delete_upload
        )
        delete_btn.pack(side="left")
    
    def confirm_delete_credential(self):
        """Confirm and execute credential deletion"""
        id_text = self.delete_cred_id_entry.get().strip()
        
        # Validate ID input
        if not id_text:
            self.delete_cred_error_label.configure(text="Please enter a credential ID!")
            return
        
        try:
            cred_id = int(id_text)
        except ValueError:
            self.delete_cred_error_label.configure(text="ID must be a valid number!")
            return
        
        # Check if credential exists
        if not self.credential_exists(cred_id):
            self.delete_cred_error_label.configure(text="Credential ID not found!")
            return
        
        # Delete from database
        if self.delete_credential(cred_id):
            print(f"Successfully deleted credential ID: {cred_id}")
            self.refresh_database_display()
            self.delete_cred_window.destroy()
        else:
            self.delete_cred_error_label.configure(text="Error deleting credential!")
    
    def confirm_delete_upload(self):
        """Confirm and execute upload deletion"""
        id_text = self.delete_upload_id_entry.get().strip()
        
        # Validate ID input
        if not id_text:
            self.delete_upload_error_label.configure(text="Please enter an upload ID!")
            return
        
        try:
            upload_id = int(id_text)
        except ValueError:
            self.delete_upload_error_label.configure(text="ID must be a valid number!")
            return
        
        # Check if upload exists
        if not self.upload_exists(upload_id):
            self.delete_upload_error_label.configure(text="Upload ID not found!")
            return
        
        # Delete from database
        if self.delete_upload(upload_id):
            print(f"Successfully deleted upload ID: {upload_id}")
            self.refresh_database_display()
            self.delete_upload_window.destroy()
        else:
            self.delete_upload_error_label.configure(text="Error deleting upload!")
    
    def credential_exists(self, cred_id):
        """Check if credential ID exists in database"""
        try:
            self.db_cursor.execute('SELECT COUNT(*) FROM credentials WHERE id = ?', (cred_id,))
            return self.db_cursor.fetchone()[0] > 0
        except sqlite3.Error:
            return False
    
    def upload_exists(self, upload_id):
        """Check if upload ID exists in database"""
        try:
            self.db_cursor.execute('SELECT COUNT(*) FROM pending_uploads WHERE id = ?', (upload_id,))
            return self.db_cursor.fetchone()[0] > 0
        except sqlite3.Error:
            return False
    
    def delete_credential(self, cred_id):
        """Delete credential from all related tables."""
        try:
            # It's highly recommended to set up 'ON DELETE CASCADE' in your database schema.
            # If not, you must delete from child tables first.
            self.db_cursor.execute('DELETE FROM oauth_details WHERE credential_id = ?', (cred_id,))
            self.db_cursor.execute('DELETE FROM creation_logs WHERE credential_id = ?', (cred_id,))
            self.db_cursor.execute('DELETE FROM credentials WHERE id = ?', (cred_id,))
            
            self.db_connection.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error deleting credential: {e}")
            self.db_connection.rollback()
            return False
    
    def delete_upload(self, upload_id):
        """Delete upload from database"""
        try:
            self.db_cursor.execute('DELETE FROM pending_uploads WHERE id = ?', (upload_id,))
            self.db_connection.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error deleting upload: {e}")
            return False
    
    def add_credential_dialog(self):
        """Open dialog to add new credential with scheduling options."""
        # Create modal window
        self.add_cred_window = ctk.CTkToplevel(self)
        self.add_cred_window.title("Add Credential")
        self.add_cred_window.geometry("450x800") # Increased height for scheduling
        self.add_cred_window.configure(fg_color="#202020")
        self.add_cred_window.transient(self)
        self.add_cred_window.grab_set()

        # Center the window
        self.add_cred_window.after(100, lambda: self.add_cred_window.lift())

        # Main frame for scrolling
        scrollable_frame = ctk.CTkScrollableFrame(self.add_cred_window, fg_color="transparent")
        scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title_label = ctk.CTkLabel(scrollable_frame, text="Add New Credential", font=("Arial", 18, "bold"), text_color="white")
        title_label.pack(pady=(10, 20))

        # --- Input Fields ---
        # Nickname
        ctk.CTkLabel(scrollable_frame, text="Nickname:", font=("Arial", 12, "bold"), text_color="white").pack(anchor='w', pady=(0, 5))
        self.cred_nickname_entry = ctk.CTkEntry(scrollable_frame, placeholder_text="Enter nickname...", font=("Arial", 12), fg_color="#323232", text_color="white", height=35)
        self.cred_nickname_entry.pack(fill='x', pady=(0, 15))

        # Channel ID
        ctk.CTkLabel(scrollable_frame, text="Channel ID:", font=("Arial", 12, "bold"), text_color="white").pack(anchor='w', pady=(0, 5))
        self.cred_channel_entry = ctk.CTkEntry(scrollable_frame, placeholder_text="Enter channel ID...", font=("Arial", 12), fg_color="#323232", text_color="white", height=35)
        self.cred_channel_entry.pack(fill='x', pady=(0, 15))

        # API Key
        ctk.CTkLabel(scrollable_frame, text="API Key:", font=("Arial", 12, "bold"), text_color="white").pack(anchor='w', pady=(0, 5))
        self.cred_api_entry = ctk.CTkEntry(scrollable_frame, placeholder_text="Enter API key...", font=("Arial", 12), fg_color="#323232", text_color="white", height=35, show="*")
        self.cred_api_entry.pack(fill='x', pady=(0, 15))

        # OAuth Client ID
        ctk.CTkLabel(scrollable_frame, text="OAuth Client ID (Optional):", font=("Arial", 12, "bold"), text_color="white").pack(anchor='w', pady=(0, 5))
        self.cred_oauth_id_entry = ctk.CTkEntry(scrollable_frame, placeholder_text="Enter OAuth Client ID...", font=("Arial", 12), fg_color="#323232", text_color="white", height=35)
        self.cred_oauth_id_entry.pack(fill='x', pady=(0, 15))

        # Client Secret
        ctk.CTkLabel(scrollable_frame, text="Client Secret (Optional):", font=("Arial", 12, "bold"), text_color="white").pack(anchor='w', pady=(0, 5))
        self.cred_secret_entry = ctk.CTkEntry(scrollable_frame, placeholder_text="Enter Client Secret...", font=("Arial", 12), fg_color="#323232", text_color="white", height=35, show="*")
        self.cred_secret_entry.pack(fill='x', pady=(0, 15))

        # --- Scheduling Section ---
        ctk.CTkLabel(scrollable_frame, text="Automation Schedule", font=("Arial", 14, "bold"), text_color="white").pack(anchor='w', pady=(20, 10))
        
        self.cred_automation_active = ctk.CTkCheckBox(scrollable_frame, text="Enable Automation for this Account")
        self.cred_automation_active.pack(anchor='w', pady=(0, 15))

        ctk.CTkLabel(scrollable_frame, text="Frequency:", font=("Arial", 12, "bold"), text_color="white").pack(anchor='w', pady=(0, 5))
        self.cred_frequency = ctk.CTkComboBox(scrollable_frame, values=["daily", "2x daily", "4x daily"], state="readonly", fg_color="#323232", button_color="#404040")
        self.cred_frequency.set("daily")
        self.cred_frequency.pack(fill='x', pady=(0, 15))

        ctk.CTkLabel(scrollable_frame, text="Times (comma-separated, 24h format):", font=("Arial", 12, "bold"), text_color="white").pack(anchor='w', pady=(0, 5))
        self.cred_times = ctk.CTkEntry(scrollable_frame, placeholder_text="e.g., 08:00 or 09:30,21:30", fg_color="#323232", text_color="white", height=35)
        self.cred_times.pack(fill='x', pady=(0, 20))

        # --- Error Label and Buttons ---
        self.cred_error_label = ctk.CTkLabel(scrollable_frame, text="", font=("Arial", 11), text_color="#FF4444")
        self.cred_error_label.pack(pady=5)

        button_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        button_frame.pack(pady=20)

        cancel_btn = ctk.CTkButton(button_frame, text="Cancel", font=("Arial", 12, "bold"), width=100, height=35, fg_color="#666666", text_color="white", hover_color="#555555", command=self.add_cred_window.destroy)
        cancel_btn.pack(side="left", padx=(0, 10))

        submit_btn = ctk.CTkButton(button_frame, text="Add Credential", font=("Arial", 12, "bold"), width=120, height=35, fg_color="#00FF00", text_color="black", hover_color="#00CC00", command=self.submit_credential)
        submit_btn.pack(side="left")
    
    def submit_credential(self):
        """Submit new credential with validation, including schedule."""
        nickname = self.cred_nickname_entry.get().strip()
        channel_id = self.cred_channel_entry.get().strip()
        api_key = self.cred_api_entry.get().strip()
        oauth_client_id = self.cred_oauth_id_entry.get().strip()
        client_secret = self.cred_secret_entry.get().strip()
        
        # Get scheduling info
        automation_active = 1 if self.cred_automation_active.get() else 0
        frequency = self.cred_frequency.get()
        times = self.cred_times.get().strip()

        # --- Validation ---
        if not nickname or not channel_id or not api_key:
            self.cred_error_label.configure(text="Nickname, Channel ID, and API Key are required!")
            return
        
        if automation_active and not times:
            self.cred_error_label.configure(text="Please enter schedule times if automation is active.")
            return

        # Add to database
        if self.add_credential(nickname, channel_id, api_key, oauth_client_id, client_secret, automation_active, frequency, times):
            print(f"Successfully added credential: {nickname}")
            self.refresh_database_display()
            self.load_schedules() # Reload schedules to include the new one
            self.add_cred_window.destroy()
        else:
            self.cred_error_label.configure(text="Error adding credential. Nickname may already exist.")
    
    def add_pending_upload_dialog(self):
        """Open dialog to add new pending upload"""
        # Create modal window
        self.add_upload_window = ctk.CTkToplevel(self)
        self.add_upload_window.title("Add Pending Upload")
        self.add_upload_window.geometry("400x400")
        self.add_upload_window.configure(fg_color="#202020")
        self.add_upload_window.transient(self)
        self.add_upload_window.grab_set()  # Make modal
        
        # Center the window
        self.add_upload_window.after(100, lambda: self.add_upload_window.lift())
        
        # Configure grid
        self.add_upload_window.grid_columnconfigure(0, weight=1)
        self.add_upload_window.grid_rowconfigure((0, 1, 2, 3), weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            self.add_upload_window,
            text="Add Pending Upload",
            font=("Arial", 18, "bold"),
            text_color="white"
        )
        title_label.grid(row=0, column=0, pady=(20, 10))
        
        # Input frame
        input_frame = ctk.CTkFrame(self.add_upload_window, fg_color="transparent")
        input_frame.grid(row=1, column=0, sticky='ew', padx=20, pady=10)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Account field
        ctk.CTkLabel(input_frame, text="Account:", font=("Arial", 12, "bold"), text_color="white").grid(row=0, column=0, sticky='w', pady=(0, 5))
        self.upload_account_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Enter account name...",
            font=("Arial", 12),
            fg_color="#323232",
            text_color="white",
            height=35
        )
        self.upload_account_entry.grid(row=1, column=0, sticky='ew', pady=(0, 15))
        
        # Title field
        ctk.CTkLabel(input_frame, text="Title:", font=("Arial", 12, "bold"), text_color="white").grid(row=2, column=0, sticky='w', pady=(0, 5))
        self.upload_title_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Enter upload title...",
            font=("Arial", 12),
            fg_color="#323232",
            text_color="white",
            height=35
        )
        self.upload_title_entry.grid(row=3, column=0, sticky='ew', pady=(0, 15))
        
        # Error label
        self.upload_error_label = ctk.CTkLabel(
            self.add_upload_window,
            text="",
            font=("Arial", 11),
            text_color="#FF4444"
        )
        self.upload_error_label.grid(row=2, column=0, pady=5)
        
        # Button frame
        button_frame = ctk.CTkFrame(self.add_upload_window, fg_color="transparent")
        button_frame.grid(row=3, column=0, pady=20)
        
        # Cancel button
        cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            font=("Arial", 12, "bold"),
            width=100,
            height=35,
            fg_color="#666666",
            text_color="white",
            hover_color="#555555",
            command=self.add_upload_window.destroy
        )
        cancel_btn.pack(side="left", padx=(0, 10))
        
        # Submit button
        submit_btn = ctk.CTkButton(
            button_frame,
            text="Add Upload",
            font=("Arial", 12, "bold"),
            width=120,
            height=35,
            fg_color="#00FF00",
            text_color="black",
            hover_color="#00CC00",
            command=self.submit_pending_upload
        )
        submit_btn.pack(side="left")
    
    def submit_pending_upload(self):
        """Submit new pending upload with validation"""
        account = self.upload_account_entry.get().strip()
        title = self.upload_title_entry.get().strip()
        
        # Validate all fields are filled
        if not account or not title:
            self.upload_error_label.configure(text="All fields are required!")
            return
        
        # Additional validation
        if len(account) < 2:
            self.upload_error_label.configure(text="Account must be at least 2 characters!")
            return
        
        if len(title) < 3:
            self.upload_error_label.configure(text="Title must be at least 3 characters!")
            return
        
        # Add to database
        result = self.add_pending_upload(account, title)
        if result:
            print(f"Successfully added pending upload: {title}")
            self.refresh_database_display()
            self.add_upload_window.destroy()
        else:
            self.upload_error_label.configure(text="Error adding upload to database!")
        
    def create_terminal(self):
        """Creating terminal display with improved database viewer font"""
        self.terminal_frame = ctk.CTkFrame(self, fg_color='#1a1a1a')
        self.terminal_frame.grid(row=7, column=0, columnspan=5, rowspan=2, sticky='nsew', padx=5, pady=5)
        
        # Configure terminal frame grid
        self.terminal_frame.grid_rowconfigure(1, weight=1)
        self.terminal_frame.grid_columnconfigure(0, weight=1)
        
        # Terminal header
        self.terminal_header = ctk.CTkFrame(self.terminal_frame, fg_color='#2d2d2d', height=30)
        self.terminal_header.grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        self.terminal_header.grid_propagate(False)
        
        # Terminal title
        self.terminal_title = ctk.CTkLabel(
            self.terminal_header, 
            text="● Terminal Output", 
            font=("Consolas", 12, "bold"), 
            text_color="#00FF00"
        )
        self.terminal_title.pack(side="left", padx=10, pady=5)
        
        # Clear button
        self.clear_button = ctk.CTkButton(
            self.terminal_header,
            text="Clear",
            width=60,
            height=20,
            font=("Consolas", 10),
            command=self.clear_terminal
        )
        self.clear_button.pack(side="right", padx=10, pady=5)
        
        # Terminal text widget
        self.terminal_text = ctk.CTkTextbox(
            self.terminal_frame,
            font=("Consolas", 10),
            fg_color="#000000",
            text_color="#00FF00",
            wrap="word"
        )
        self.terminal_text.grid(row=1, column=0, sticky='nsew', padx=2, pady=(0, 2))
        
        # Add initial welcome message
        self.add_terminal_output("=== Clip Ripper Terminal ===\n")
        self.add_terminal_output("Terminal initialized. All output will be displayed here.\n\n")
        
        # Create database viewer with tabs (replaces orange section)
        self.database_viewer = ctk.CTkFrame(self, fg_color='#1a1a1a')
        self.database_viewer.grid(row=7, column=5, columnspan=5, rowspan=2, sticky='nsew', padx=5, pady=5)
        self.database_viewer.grid_columnconfigure(0, weight=1)
        self.database_viewer.grid_rowconfigure(1, weight=1)
        
        # Tab buttons frame
        self.tab_frame = ctk.CTkFrame(self.database_viewer, fg_color='#2d2d2d', height=35)
        self.tab_frame.grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        self.tab_frame.grid_propagate(False)
        self.tab_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Initialize current tab
        self.current_tab = "credentials"
        
        # Credentials tab button
        self.credentials_tab_button = ctk.CTkButton(
            self.tab_frame,
            text="CREDENTIALS",
            font=("Arial", 11, "bold"),
            height=25,
            fg_color="#00FF00",
            text_color="black",
            hover_color="#00CC00",
            command=lambda: self.switch_tab("credentials")
        )
        self.credentials_tab_button.grid(row=0, column=0, sticky='ew', padx=(5, 2), pady=5)
        
        # Pending tab button
        self.pending_tab_button = ctk.CTkButton(
            self.tab_frame,
            text="PENDING",
            font=("Arial", 11, "bold"),
            height=25,
            fg_color="#323232",
            text_color="white",
            hover_color="#404040",
            command=lambda: self.switch_tab("pending")
        )
        self.pending_tab_button.grid(row=0, column=1, sticky='ew', padx=(2, 5), pady=5)
        
        # Content frame for database displays
        self.content_frame = ctk.CTkFrame(self.database_viewer, fg_color='#000000')
        self.content_frame.grid(row=1, column=0, sticky='nsew', padx=2, pady=(0, 2))
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(1, weight=0) # For button frame
        
        # Create scrollable text widget for database content
        self.database_display = ctk.CTkTextbox(
            self.content_frame,
            font=("courier", 9),
            fg_color="#000000",
            text_color="#FFFFFF",
            wrap="none",
            state="disabled"
        )
        self.database_display.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Button frame for add/delete/schedule buttons
        self.button_frame = ctk.CTkFrame(self.content_frame, fg_color='transparent', height=40)
        self.button_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=(0, 5))
        self.button_frame.grid_propagate(False)
        self.button_frame.grid_columnconfigure(1, weight=1) # Middle space for the new button
        
        # Add button (bottom left)
        self.add_button = ctk.CTkButton(
            self.button_frame,
            text="ADD",
            font=("Arial", 12, "bold"),
            width=80,
            height=30,
            fg_color="#00FF00",
            text_color="black",
            hover_color="#00CC00",
            command=self.handle_add_button
        )
        self.add_button.grid(row=0, column=0, sticky='w', pady=5, padx=5)

        # NEW: Set Schedule button (centered)
        self.set_schedule_button = ctk.CTkButton(
            self.button_frame,
            text="SET SCHEDULE",
            font=("Arial", 12, "bold"),
            width=120,
            height=30,
            fg_color="#00BFFF", # A distinct blue color
            text_color="white",
            hover_color="#009ACD",
            command=self.open_schedule_dialog # This is the new function you will add
        )
        self.set_schedule_button.grid(row=0, column=1, pady=5)

        # Delete button (bottom right)
        self.delete_button = ctk.CTkButton(
            self.button_frame,
            text="DELETE",
            font=("Arial", 12, "bold"),
            width=80,
            height=30,
            fg_color="#FF0000",
            text_color="white",
            hover_color="#CC0000",
            command=self.handle_delete_button
        )
        self.delete_button.grid(row=0, column=2, sticky='e', pady=5, padx=5)
        
        # Load initial data
        self.refresh_database_display()
        
    def setup_terminal_redirect(self):
        """Setup stdout/stderr redirection to terminal with threading"""
        # Create redirector
        self.terminal_redirector = TerminalRedirect(self.terminal_text)
        
        # Store original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Redirect stdout and stderr
        sys.stdout = self.terminal_redirector
        sys.stderr = self.terminal_redirector
        
        # Start processing terminal output in dedicated thread
        self.start_terminal_thread()

    def start_terminal_thread(self):
        """Start terminal processing in dedicated thread"""
        def terminal_worker():
            try:
                # Set CPU affinity for terminal core
                self.set_thread_affinity([0])  # Core 0 for terminal
                
                while True:
                    try:
                        if not self.terminal_redirector.queue.empty():
                            text = self.terminal_redirector.queue.get_nowait()
                            self.terminal_queue.put({
                                'type': 'output',
                                'text': text
                            })
                    except queue.Empty:
                        pass
                    
                    time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                    
            except Exception as e:
                print(f"Terminal thread error: {e}")
        
        # Start terminal thread
        terminal_thread = threading.Thread(target=terminal_worker, daemon=True)
        terminal_thread.start()
        
    def process_terminal_output(self):
        """Process queued terminal output"""
        try:
            while not self.terminal_redirector.queue.empty():
                text = self.terminal_redirector.queue.get_nowait()
                self.add_terminal_output(text)
        except:
            pass
        
        # Schedule next check
        self.after(100, self.process_terminal_output)
        
    def add_terminal_output(self, text):
        """Add text to terminal with auto-scroll"""
        try:
            # Ensure text ends with newline for proper formatting
            if not text.endswith('\n'):
                text += '\n'
                
            # Add timestamp for new lines
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('['):
                    timestamp = time.strftime("%H:%M:%S")
                    lines[i] = f"[{timestamp}] {line}"
            
            formatted_text = '\n'.join(lines)
            
            # Insert text at end
            self.terminal_text.insert("end", formatted_text)
            
            # Auto-scroll to bottom
            self.terminal_text.see("end")
            
            # Limit terminal history (keep last 1000 lines)
            lines_count = int(self.terminal_text.index('end-1c').split('.')[0])
            if lines_count > 1000:
                self.terminal_text.delete("1.0", f"{lines_count-1000}.0")
                
        except Exception as e:
            # Fallback to original stdout to avoid infinite loop
            self.original_stdout.write(f"Terminal output error: {e}\n")
    
    def clear_terminal(self):
        """Clear terminal content"""
        self.terminal_text.delete("1.0", "end")
        self.add_terminal_output("Terminal cleared.\n")
    
    def run_command(self, command):
        """Run a command and display output in terminal"""
        try:
            self.add_terminal_output(f"$ {command}\n")
            
            # Run command in a separate thread to avoid blocking UI
            def execute_command():
                try:
                    result = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        cwd=os.getcwd()
                    )
                    
                    if result.stdout:
                        self.add_terminal_output(result.stdout)
                    if result.stderr:
                        self.add_terminal_output(f"ERROR: {result.stderr}")
                    
                    self.add_terminal_output(f"Command finished with exit code: {result.returncode}\n")
                    
                except Exception as e:
                    self.add_terminal_output(f"Command execution error: {str(e)}\n")
            
            # Run in background thread
            thread = threading.Thread(target=execute_command)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.add_terminal_output(f"Error running command: {str(e)}\n")
        
    def create_usage_bar(self):
        """Creating usage stats on bottom 10% of the screen"""
        self.usagebar = ctk.CTkFrame(self, fg_color='#191919')
        self.usagebar.grid(row=9, column=0, columnspan=10, sticky='nsew')
        
        # Usage bar responsive layout 2 x 6
        for i in range(6):
            self.usagebar.grid_columnconfigure(i, weight=1)
        self.usagebar.grid_rowconfigure((0, 1), weight=1)
        
        # CPU usage frame
        self.cpu_usage = ctk.CTkFrame(self.usagebar, fg_color='#323232')
        self.cpu_usage.grid(row=0, column=0, rowspan=2, sticky='nsew')
        
        # Initialize CPU gauge
        self.setup_cpu_gauge()
        
        # GPU usage frame
        self.gpu_usage = ctk.CTkFrame(self.usagebar, fg_color='#323232')
        self.gpu_usage.grid(row=0, column=1, rowspan=2, sticky='nsew')

        self.setup_gpu_gauge()
        
        # RAM usage frame
        self.ram_usage = ctk.CTkFrame(self.usagebar, fg_color='#323232')
        self.ram_usage.grid(row=0, column=2, rowspan=2, sticky='nsew')
        
        # Initialize RAM gauge
        self.setup_ram_gauge()
        
        # CPU2 usage frame (duplicate CPU gauge)
        self.cpu2_usage = ctk.CTkFrame(self.usagebar, fg_color="#323232")
        self.cpu2_usage.grid(row=0, column=3, rowspan=2, sticky='nsew')
        
        # Initialize second CPU gauge
        self.setup_cpu2_gauge()

        # Network usage frame
        self.network_usage = ctk.CTkFrame(self.usagebar, fg_color="#323232")
        self.network_usage.grid(row=0, column=4, rowspan=2, sticky='nsew')

        # Initialize network gauge
        self.setup_network_gauge()
        
    
        # System info frame
        self.system_info = ctk.CTkFrame(self.usagebar, fg_color="#323232")
        self.system_info.grid(row=0, column=5, rowspan=2, sticky='nsew')
        
        # Initialize system info display
        self.setup_system_info()
    
    def setup_cpu_gauge(self):
        """Initialize the CPU gauge canvas and text"""
        # Add CPU label at the top
        self.cpu_label = ctk.CTkLabel(self.cpu_usage, text="CPU", font=("Arial", 16, "bold"), text_color="white")
        self.cpu_label.pack(pady=(5, 0))
        
        # Create canvas that fills the remaining space
        self.cpu_canvas = ctk.CTkCanvas(self.cpu_usage, bg="black", highlightthickness=0)
        self.cpu_canvas.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        
        # Initialize text variable
        self.cpu_usage_text = None
        
        # Start updating after a short delay to ensure canvas is ready
        self.after(500, self.update_cpu_gauge)
    
    def get_usage_color(self, usage):
        """Color change based on usage level"""
        if usage < 25:
            return '#00FF00'
        elif usage > 25 and usage < 50:
            return '#FFFF00'
        elif usage > 50 and usage < 75:
            return "#FFA500"
        else:
            return '#FF0000'
    
    def update_cpu_gauge(self):
        """Update CPU gauge in dedicated thread"""
        def cpu_worker():
            try:
                # Set CPU affinity for usage bar core
                self.set_thread_affinity([2])  # Core 2 for usage bar
                
                # Get current CPU usage
                usage = psutil.cpu_percent(interval=0.1)
                
                # Send update to main thread
                self.after_idle(lambda: self._update_cpu_gauge_ui(usage))
                
            except Exception as e:
                print(f"CPU gauge error: {e}")
        
        # Run in background thread
        thread = threading.Thread(target=cpu_worker, daemon=True)
        thread.start()
        
        # Schedule next update
        self.after(1000, self.update_cpu_gauge)
    
    def _update_cpu_gauge_ui(self, usage):
        """Update CPU gauge UI elements (must run on main thread)"""
        try:
            # Get canvas dimensions
            canvas_width = self.cpu_canvas.winfo_width()
            canvas_height = self.cpu_canvas.winfo_height()
            
            # Skip if canvas isn't ready yet
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            # Calculate gauge size and position
            gauge_size = min(canvas_width, canvas_height) - 20
            if gauge_size < 50:
                gauge_size = 50
            
            center_x = canvas_width // 2
            center_y = canvas_height // 2
            
            x1 = center_x - gauge_size // 2
            y1 = center_y - gauge_size // 2
            x2 = center_x + gauge_size // 2
            y2 = center_y + gauge_size // 2
            
            # Clear previous gauge elements
            self.cpu_canvas.delete("gauge_arc")
            self.cpu_canvas.delete("gauge_bg")
            
            # Calculate arc and color
            extent = usage * 3.6
            color = self.get_usage_color(usage)
            line_width = max(gauge_size // 15, 5)
            
            # Draw gauge
            self.cpu_canvas.create_oval(x1, y1, x2, y2, outline="#333333", width=line_width, tags="gauge_bg")
            self.cpu_canvas.create_arc(x1, y1, x2, y2, start=90, extent=-extent, style="arc",
                                        outline=color, width=line_width, tags="gauge_arc")
            
            # Update text
            font_size = max(gauge_size // 8, 10)
            if self.cpu_usage_text is None:
                self.cpu_usage_text = self.cpu_canvas.create_text(center_x, center_y, text=f"{int(usage)}%", 
                                                                fill="white", font=("Arial", font_size), tags="gauge_text")
            else:
                self.cpu_canvas.itemconfig(self.cpu_usage_text, text=f"{int(usage)}%", font=("Arial", font_size))
                self.cpu_canvas.coords(self.cpu_usage_text, center_x, center_y)
            
        except Exception as e:
            print(f"Error updating CPU gauge UI: {e}")
    
    def setup_ram_gauge(self):
        """Initialize the RAM gauge canvas and text"""
        # Add RAM label at the top
        self.ram_label = ctk.CTkLabel(self.ram_usage, text="RAM", font=("Arial", 16, "bold"), text_color="white")
        self.ram_label.pack(pady=(5, 0))
        
        # Create canvas that fills the remaining space
        self.ram_canvas = ctk.CTkCanvas(self.ram_usage, bg="black", highlightthickness=0)
        self.ram_canvas.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        
        # Initialize text variable
        self.ram_usage_text = None
        
        # Start updating after a short delay to ensure canvas is ready
        self.after(600, self.update_ram_gauge)
    
    def update_ram_gauge(self):
        """Update the RAM usage gauge"""
        try:
            # Get current RAM usage
            ram = psutil.virtual_memory()
            usage = ram.percent
            
            # Get canvas dimensions
            canvas_width = self.ram_canvas.winfo_width()
            canvas_height = self.ram_canvas.winfo_height()
            
            # Skip if canvas isn't ready yet
            if canvas_width <= 1 or canvas_height <= 1:
                self.after(1000, self.update_ram_gauge)
                return
            
            # Calculate gauge size (use smaller dimension and add some padding)
            gauge_size = min(canvas_width, canvas_height) - 20
            if gauge_size < 50:  # Minimum size
                gauge_size = 50
            
            # Calculate center position
            center_x = canvas_width // 2
            center_y = canvas_height // 2
            
            # Calculate gauge coordinates
            x1 = center_x - gauge_size // 2
            y1 = center_y - gauge_size // 2
            x2 = center_x + gauge_size // 2
            y2 = center_y + gauge_size // 2
            
            # Clear previous gauge elements (but not text)
            self.ram_canvas.delete("ram_arc")
            self.ram_canvas.delete("ram_bg")
            
            # Calculate arc extent (360 degrees max)
            extent = usage * 3.6
            color = self.get_usage_color(usage)
            
            # Calculate line width based on gauge size
            line_width = max(gauge_size // 15, 5)
            
            # Draw the gauge background circle
            self.ram_canvas.create_oval(x1, y1, x2, y2, outline="#333333", width=line_width, tags="ram_bg")
            
            # Draw the usage arc
            self.ram_canvas.create_arc(x1, y1, x2, y2, start=90, extent=-extent, style="arc",
                                        outline=color, width=line_width, tags="ram_arc")
            
            # Create or update the percentage text
            font_size = max(gauge_size // 8, 10)
            if self.ram_usage_text is None:
                self.ram_usage_text = self.ram_canvas.create_text(center_x, center_y, text=f"{int(usage)}%", 
                                                                fill="white", font=("Arial", font_size), tags="ram_text")
            else:
                self.ram_canvas.itemconfig(self.ram_usage_text, text=f"{int(usage)}%", font=("Arial", font_size))
                self.ram_canvas.coords(self.ram_usage_text, center_x, center_y)
            
        except Exception as e:
            print(f"Error updating RAM gauge: {e}")
        
        # Schedule next update
        self.after(1000, self.update_ram_gauge)

    def get_gpu_usage(self):
        """Get GPU usage - works with NVIDIA cards via nvidia-ml-py"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except:
            # Return 0 if no NVIDIA GPU or nvidia-ml-py not installed
            return 0
    
    def setup_gpu_gauge(self):
        """Initialize the GPU gauge canvas and text"""
        # Add GPU label at the top
        self.gpu_label = ctk.CTkLabel(self.gpu_usage, text="GPU", font=("Arial", 16, "bold"), text_color="white")
        self.gpu_label.pack(pady=(5, 0))
        
        # Create canvas that fills the remaining space
        self.gpu_canvas = ctk.CTkCanvas(self.gpu_usage, bg="black", highlightthickness=0)
        self.gpu_canvas.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        
        # Initialize text variable
        self.gpu_usage_text = None
        
        # Start updating after a short delay to ensure canvas is ready
        self.after(700, self.update_gpu_gauge)
    
    def update_gpu_gauge(self):
        """Update the GPU usage gauge"""
        try:
            # Get current GPU usage
            usage = self.get_gpu_usage()
            
            # Get canvas dimensions
            canvas_width = self.gpu_canvas.winfo_width()
            canvas_height = self.gpu_canvas.winfo_height()
            
            # Skip if canvas isn't ready yet
            if canvas_width <= 1 or canvas_height <= 1:
                self.after(1000, self.update_gpu_gauge)
                return
            
            # Calculate gauge size (use smaller dimension and add some padding)
            gauge_size = min(canvas_width, canvas_height) - 20
            if gauge_size < 50:  # Minimum size
                gauge_size = 50
            
            # Calculate center position
            center_x = canvas_width // 2
            center_y = canvas_height // 2
            
            # Calculate gauge coordinates
            x1 = center_x - gauge_size // 2
            y1 = center_y - gauge_size // 2
            x2 = center_x + gauge_size // 2
            y2 = center_y + gauge_size // 2
            
            # Clear previous gauge elements (but not text)
            self.gpu_canvas.delete("gpu_arc")
            self.gpu_canvas.delete("gpu_bg")
            
            # Calculate arc extent (360 degrees max)
            extent = usage * 3.6
            color = self.get_usage_color(usage)
            
            # Calculate line width based on gauge size
            line_width = max(gauge_size // 15, 5)
            
            # Draw the gauge background circle
            self.gpu_canvas.create_oval(x1, y1, x2, y2, outline="#333333", width=line_width, tags="gpu_bg")
            
            # Draw the usage arc
            self.gpu_canvas.create_arc(x1, y1, x2, y2, start=90, extent=-extent, style="arc",
                                        outline=color, width=line_width, tags="gpu_arc")
            
            # Create or update the percentage text
            font_size = max(gauge_size // 8, 10)
            display_text = f"{int(usage)}%" if usage > 0 else "N/A"
            if self.gpu_usage_text is None:
                self.gpu_usage_text = self.gpu_canvas.create_text(center_x, center_y, text=display_text, 
                                                                fill="white", font=("Arial", font_size), tags="gpu_text")
            else:
                self.gpu_canvas.itemconfig(self.gpu_usage_text, text=display_text, font=("Arial", font_size))
                self.gpu_canvas.coords(self.gpu_usage_text, center_x, center_y)
            
        except Exception as e:
            print(f"Error updating GPU gauge: {e}")
        
        # Schedule next update
        self.after(1000, self.update_gpu_gauge)
    
    def setup_cpu2_gauge(self):
        """Initialize the second CPU gauge canvas and text"""
        # Add CPU2 label at the top
        self.cpu2_label = ctk.CTkLabel(self.cpu2_usage, text="CPU 2", font=("Arial", 16, "bold"), text_color="white")
        self.cpu2_label.pack(pady=(5, 0))
        
        # Create canvas that fills the remaining space
        self.cpu2_canvas = ctk.CTkCanvas(self.cpu2_usage, bg="black", highlightthickness=0)
        self.cpu2_canvas.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        
        # Initialize text variable
        self.cpu2_usage_text = None
        
        # Start updating after a short delay to ensure canvas is ready
        self.after(750, self.update_cpu2_gauge)
    
    def update_cpu2_gauge(self):
        """Update the second CPU usage gauge"""
        try:
            # Get current CPU usage
            usage = psutil.cpu_percent(interval=0.1)
            
            # Get canvas dimensions
            canvas_width = self.cpu2_canvas.winfo_width()
            canvas_height = self.cpu2_canvas.winfo_height()
            
            # Skip if canvas isn't ready yet
            if canvas_width <= 1 or canvas_height <= 1:
                self.after(1000, self.update_cpu2_gauge)
                return
            
            # Calculate gauge size (use smaller dimension and add some padding)
            gauge_size = min(canvas_width, canvas_height) - 20
            if gauge_size < 50:  # Minimum size
                gauge_size = 50
            
            # Calculate center position
            center_x = canvas_width // 2
            center_y = canvas_height // 2
            
            # Calculate gauge coordinates
            x1 = center_x - gauge_size // 2
            y1 = center_y - gauge_size // 2
            x2 = center_x + gauge_size // 2
            y2 = center_y + gauge_size // 2
            
            # Clear previous gauge elements (but not text)
            self.cpu2_canvas.delete("cpu2_arc")
            self.cpu2_canvas.delete("cpu2_bg")
            
            # Calculate arc extent (360 degrees max)
            extent = usage * 3.6
            color = self.get_usage_color(usage)
            
            # Calculate line width based on gauge size
            line_width = max(gauge_size // 15, 5)
            
            # Draw the gauge background circle
            self.cpu2_canvas.create_oval(x1, y1, x2, y2, outline="#333333", width=line_width, tags="cpu2_bg")
            
            # Draw the usage arc
            self.cpu2_canvas.create_arc(x1, y1, x2, y2, start=90, extent=-extent, style="arc",
                                        outline=color, width=line_width, tags="cpu2_arc")
            
            # Create or update the percentage text
            font_size = max(gauge_size // 8, 10)
            if self.cpu2_usage_text is None:
                self.cpu2_usage_text = self.cpu2_canvas.create_text(center_x, center_y, text=f"{int(usage)}%", 
                                                                    fill="white", font=("Arial", font_size), tags="cpu2_text")
            else:
                self.cpu2_canvas.itemconfig(self.cpu2_usage_text, text=f"{int(usage)}%", font=("Arial", font_size))
                self.cpu2_canvas.coords(self.cpu2_usage_text, center_x, center_y)
            
        except Exception as e:
            print(f"Error updating CPU2 gauge: {e}")
        
        # Schedule next update
        self.after(1000, self.update_cpu2_gauge)
    
    def setup_system_info(self):
        """Initialize the system info display"""
        # Add System Info label at the top
        self.sysinfo_label = ctk.CTkLabel(self.system_info, text="SYSTEM INFO", font=("Arial", 16, "bold"), text_color="white")
        self.sysinfo_label.pack(pady=(5, 0))
        
        # Create canvas that fills the remaining space
        self.sysinfo_canvas = ctk.CTkCanvas(self.system_info, bg="black", highlightthickness=0)
        self.sysinfo_canvas.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        
        # Start updating after a short delay to ensure canvas is ready
        self.after(900, self.update_system_info)
    
    def format_uptime(self, seconds):
        """Format uptime in a readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {seconds}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def update_system_info(self):
        """Update the system info display"""
        try:
            import datetime
            
            # Get canvas dimensions
            canvas_width = self.sysinfo_canvas.winfo_width()
            canvas_height = self.sysinfo_canvas.winfo_height()
            
            # Skip if canvas isn't ready yet
            if canvas_width <= 1 or canvas_height <= 1:
                self.after(1000, self.update_system_info)
                return
            
            # Clear previous text
            self.sysinfo_canvas.delete("sysinfo_text")
            
            # Calculate positions and font sizes
            center_x = canvas_width // 2
            font_size = max(canvas_width // 30, 8)
            value_font_size = max(canvas_width // 25, 9)
            
            # Calculate vertical positions (divide canvas into 6 sections)
            y_positions = [
                canvas_height // 6,
                canvas_height // 3,
                canvas_height // 2,
                (canvas_height * 2) // 3,
                (canvas_height * 5) // 6
            ]
            
            # 1. Launch time and date
            launch_str = self.launch_datetime.strftime("%Y-%m-%d %H:%M:%S")
            self.sysinfo_canvas.create_text(center_x, y_positions[0] - 8, text="Started:", 
                                            fill="#00BFFF", font=("Arial", font_size, "bold"), tags="sysinfo_text")
            self.sysinfo_canvas.create_text(center_x, y_positions[0] + 8, text=launch_str, 
                                            fill="white", font=("Arial", value_font_size), tags="sysinfo_text")
            
            # 2. Current time and date
            current_time = datetime.datetime.now()
            current_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            self.sysinfo_canvas.create_text(center_x, y_positions[2] - 8, text="Current:", 
                                            fill="#FFD700", font=("Arial", font_size, "bold"), tags="sysinfo_text")
            self.sysinfo_canvas.create_text(center_x, y_positions[2] + 8, text=current_str, 
                                            fill="white", font=("Arial", value_font_size), tags="sysinfo_text")
            
            # 3. Uptime
            uptime_seconds = time.time() - self.start_time
            uptime_str = self.format_uptime(uptime_seconds)
            self.sysinfo_canvas.create_text(center_x, y_positions[4] - 8, text="Uptime:", 
                                            fill="#00FF7F", font=("Arial", font_size, "bold"), tags="sysinfo_text")
            self.sysinfo_canvas.create_text(center_x, y_positions[4] + 8, text=uptime_str, 
                                            fill="white", font=("Arial", value_font_size), tags="sysinfo_text")
            
        except Exception as e:
            print(f"Error updating system info: {e}")
        
        # Schedule next update
        self.after(1000, self.update_system_info)

    def get_network_speed(self):
        """Get network speed in Mbps"""
        try:
            # Get network stats
            stats = psutil.net_io_counters()
            
            # If this is first run, store initial values
            if not hasattr(self, 'prev_bytes_sent'):
                self.prev_bytes_sent = stats.bytes_sent
                self.prev_bytes_recv = stats.bytes_recv
                self.prev_time = time.time()
                return 0, 0
            
            # Calculate time difference
            current_time = time.time()
            time_diff = current_time - self.prev_time
            
            if time_diff < 1.0:  # Avoid division by very small numbers
                return 0, 0
            
            # Calculate bytes per second
            bytes_sent_per_sec = (stats.bytes_sent - self.prev_bytes_sent) / time_diff
            bytes_recv_per_sec = (stats.bytes_recv - self.prev_bytes_recv) / time_diff
            
            # Convert to Mbps (megabits per second)
            upload_mbps = (bytes_sent_per_sec * 8) / (1024 * 1024)
            download_mbps = (bytes_recv_per_sec * 8) / (1024 * 1024)
            
            # Store current values for next calculation
            self.prev_bytes_sent = stats.bytes_sent
            self.prev_bytes_recv = stats.bytes_recv
            self.prev_time = current_time
            
            return upload_mbps, download_mbps
        except:
            return 0, 0
    
    def setup_network_gauge(self):
        """Initialize the Network gauge display using canvas like other gauges"""
        # Add Network label at the top
        self.network_label = ctk.CTkLabel(self.network_usage, text="NETWORK", font=("Arial", 16, "bold"), text_color="white")
        self.network_label.pack(pady=(5, 0))
        
        # Create canvas that fills the remaining space
        self.network_canvas = ctk.CTkCanvas(self.network_usage, bg="black", highlightthickness=0)
        self.network_canvas.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        
        # Initialize text variables
        self.network_upload_text = None
        self.network_download_text = None
        
        # Start updating after a short delay to ensure canvas is ready
        self.after(800, self.update_network_gauge)
    
    def check_youtube_status(self):
        """Check if YouTube is accessible"""
        try:
            # Set a short timeout to avoid blocking the UI
            socket.setdefaulttimeout(3)
            
            # Try to reach YouTube
            response = urllib.request.urlopen('https://www.youtube.com', timeout=3)
            return response.getcode() == 200
        except:
            return False
        
    def update_network_gauge(self):
        """Update the network speed display"""
        try:
            # Fixed: Call the correct method
            upload_mbps, download_mbps = self.get_network_speed()
            
            # Get canvas dimensions
            canvas_width = self.network_canvas.winfo_width()
            canvas_height = self.network_canvas.winfo_height()
            
            # Skip if canvas isn't ready yet
            if canvas_width <= 1 or canvas_height <= 1:
                self.after(1000, self.update_network_gauge)
                return
            
            # Clear previous text
            self.network_canvas.delete("network_text")
            
            # Calculate positions and font sizes
            center_x = canvas_width // 2
            sixth_y = canvas_height // 6
            half_y = canvas_height // 2
            five_sixth_y = (canvas_height * 5) // 6
            font_size = max(canvas_width // 25, 9)
            value_font_size = max(canvas_width // 20, 10)
            
            # Format speeds
            if upload_mbps >= 1:
                upload_text = f"{upload_mbps:.1f} Mbps"
            elif upload_mbps >= 0.001:
                upload_text = f"{upload_mbps*1000:.0f} Kbps"
            else:
                upload_text = "0.0 Mbps"
                
            if download_mbps >= 1:
                download_text = f"{download_mbps:.1f} Mbps"
            elif download_mbps >= 0.001:
                download_text = f"{download_mbps*1000:.0f} Kbps"
            else:
                download_text = "0.0 Mbps"
            
            # Draw upload info
            self.network_canvas.create_text(center_x, sixth_y - 8, text="↑ Upload", 
                                            fill="#00FF00", font=("Arial", font_size, "bold"), tags="network_text")
            self.network_canvas.create_text(center_x, sixth_y + 8, text=upload_text, 
                                            fill="white", font=("Arial", value_font_size), tags="network_text")
            
            # Draw download info
            self.network_canvas.create_text(center_x, half_y - 8, text="↓ Download", 
                                            fill="#0080FF", font=("Arial", font_size, "bold"), tags="network_text")
            self.network_canvas.create_text(center_x, half_y + 8, text=download_text, 
                                            fill="white", font=("Arial", value_font_size), tags="network_text")
            
            # Check YouTube status (only every 10th update to avoid too frequent checks)
            if not hasattr(self, 'youtube_check_counter'):
                self.youtube_check_counter = 0
                self.youtube_status = None
            
            self.youtube_check_counter += 1
            if self.youtube_check_counter >= 10 or self.youtube_status is None:
                self.youtube_status = self.check_youtube_status()
                self.youtube_check_counter = 0
            
            # Draw YouTube status
            status_color = "#00FF00" if self.youtube_status else "#FF0000"
            dot_radius = max(canvas_width // 40, 3)
            
            # Draw YouTube status dot and label
            dot_x = center_x - 40
            dot_y = five_sixth_y
            self.network_canvas.create_oval(dot_x - dot_radius, dot_y - dot_radius, 
                                            dot_x + dot_radius, dot_y + dot_radius, 
                                            fill=status_color, outline=status_color, tags="network_text")
            self.network_canvas.create_text(center_x + 10, dot_y, text="YouTube Status", 
                                            fill="white", font=("Arial", font_size), tags="network_text")
            
        except Exception as e:
            print(f"Error updating network gauge: {e}")
        
        # Schedule next update
        self.after(1000, self.update_network_gauge)

    def __del__(self):
        """Restore original stdout/stderr and close database when app is destroyed"""
        try:
            if hasattr(self, 'original_stdout'):
                sys.stdout = self.original_stdout
            if hasattr(self, 'original_stderr'):
                sys.stderr = self.original_stderr
            if hasattr(self, 'db_connection'):
                self.db_connection.close()
                print("Database connection closed")
        except:
            pass

# Run the app
if __name__ == "__main__":
    app = App()
    
    # Test the terminal with some sample output
    print("Clip Ripper Dashboard initialized successfully!")
    print("System monitoring active...")
    print("All print statements will now appear in the terminal above.")
    
    # You can also run commands programmatically
    # Uncomment the next line to test command execution
    # app.run_command("echo 'Hello from terminal!'")
    
    app.mainloop()