import os
import logging
from pathlib import Path
from typing import Optional
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pickle

logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

class YouTubeUploader:
    """
    Handles uploading videos to YouTube.
    """
    
    def __init__(self, credentials_file: str = "client_secrets.json"):
        self.credentials_file = credentials_file
        self.token_file = "token.pickle"
        self.youtube = None
        
    def authenticate(self):
        """Authenticate with YouTube API."""
        creds = None
        
        # Load saved credentials
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    logger.error(f"Credentials file not found: {self.credentials_file}")
                    logger.info("Please download client_secrets.json from Google Cloud Console")
                    return False
                    
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save credentials
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        self.youtube = build('youtube', 'v3', credentials=creds)
        logger.info("YouTube authentication successful")
        return True
    
    def upload_video(
        self,
        file_path: str,
        title: str,
        description: str,
        tags: list = None,
        category_id: str = "17",  # Sports
        privacy_status: str = "private"
    ) -> Optional[str]:
        """
        Upload a video to YouTube.
        
        Args:
            file_path: Path to video file
            title: Video title
            description: Video description
            tags: List of tags
            category_id: YouTube category (17 = Sports)
            privacy_status: 'public', 'private', or 'unlisted'
            
        Returns:
            Video ID if successful, None otherwise
        """
        if not self.youtube:
            if not self.authenticate():
                return None
        
        try:
            logger.info(f"Uploading video: {file_path}")
            
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'tags': tags or [],
                    'categoryId': category_id
                },
                'status': {
                    'privacyStatus': privacy_status,
                    'selfDeclaredMadeForKids': False
                }
            }
            
            media = MediaFileUpload(
                file_path,
                chunksize=-1,
                resumable=True,
                mimetype='video/mp4'
            )
            
            request = self.youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.info(f"Upload progress: {int(status.progress() * 100)}%")
            
            video_id = response['id']
            logger.info(f"✅ Upload complete! Video ID: {video_id}")
            logger.info(f"URL: https://www.youtube.com/watch?v={video_id}")
            
            return video_id
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return None
