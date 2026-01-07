"""WebDAV Manager - Handle WebDAV operations for video uploads"""
import asyncio
import aiohttp
import tempfile
import os
import time
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from ..core.database import Database
from ..core.models import WebDAVConfig, VideoRecord, UploadLog


class WebDAVManager:
    """Manager for WebDAV operations"""

    def __init__(self, db: Database):
        self.db = db
        self._config: Optional[WebDAVConfig] = None
        self._client = None

    async def get_config(self) -> WebDAVConfig:
        """Get WebDAV configuration from database"""
        await self.db.ensure_webdav_config_row()
        self._config = await self.db.get_webdav_config()
        return self._config

    async def update_config(self, **kwargs) -> WebDAVConfig:
        """Update WebDAV configuration"""
        await self.db.update_webdav_config(**kwargs)
        self._config = await self.db.get_webdav_config()
        return self._config

    def _get_client(self):
        """Get or create WebDAV client"""
        if not self._config or not self._config.webdav_enabled:
            return None
        
        try:
            from webdav3.client import Client
            options = {
                "webdav_hostname": self._config.webdav_url,
                "webdav_login": self._config.webdav_username,
                "webdav_password": self._config.webdav_password,
            }
            return Client(options)
        except ImportError:
            print("webdav3 library not installed. Run: pip install webdavclient3")
            return None
        except Exception as e:
            print(f"Failed to create WebDAV client: {e}")
            return None

    async def test_connection(self) -> dict:
        """Test WebDAV connection"""
        config = await self.get_config()
        if not config.webdav_enabled:
            return {"success": False, "message": "WebDAV is not enabled"}
        
        if not config.webdav_url:
            return {"success": False, "message": "WebDAV URL is not configured"}

        try:
            client = self._get_client()
            if not client:
                return {"success": False, "message": "Failed to create WebDAV client"}
            
            # Try to list the upload directory
            start_time = time.time()
            files = await asyncio.to_thread(client.list, config.webdav_upload_path or "/")
            duration = time.time() - start_time
            
            return {
                "success": True,
                "message": f"Connection successful. Found {len(files)} items.",
                "duration": round(duration, 2),
                "files_count": len(files)
            }
        except Exception as e:
            return {"success": False, "message": f"Connection failed: {str(e)}"}

    async def upload_video(self, video_url: str, task_id: str, token_id: int, 
                          watermark_free_url: str = None) -> dict:
        """Download video from URL and upload to WebDAV"""
        config = await self.get_config()
        if not config.webdav_enabled:
            return {"success": False, "message": "WebDAV is not enabled"}

        start_time = time.time()
        temp_file = None
        
        try:
            # Create video record
            record = VideoRecord(
                task_id=task_id,
                token_id=token_id,
                original_url=video_url,
                watermark_free_url=watermark_free_url,
                status="uploading"
            )
            record_id = await self.db.create_video_record(record)

            # Use watermark-free URL if available
            download_url = watermark_free_url or video_url
            
            # Download video to temp file
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download video: HTTP {response.status}")
                    
                    # Get file extension from URL or content-type
                    content_type = response.headers.get('content-type', '')
                    ext = '.mp4'
                    if 'webm' in content_type:
                        ext = '.webm'
                    elif 'mov' in content_type:
                        ext = '.mov'
                    
                    # Create temp file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                    file_size = 0
                    
                    async for chunk in response.content.iter_chunked(8192):
                        temp_file.write(chunk)
                        file_size += len(chunk)
                    
                    temp_file.close()

            # Generate remote path
            filename = f"{task_id}{ext}"
            upload_path = config.webdav_upload_path or "/video"
            remote_path = f"{upload_path}/{filename}"

            # Upload to WebDAV
            client = self._get_client()
            if not client:
                raise Exception("Failed to create WebDAV client")

            await asyncio.to_thread(client.upload_sync, remote_path=remote_path, local_path=temp_file.name)
            
            duration = time.time() - start_time
            
            # Update video record
            webdav_url = f"{config.webdav_url.rstrip('/')}{remote_path}"
            await self.db.update_video_record(
                record_id,
                webdav_path=remote_path,
                webdav_url=webdav_url,
                file_size=file_size,
                status="uploaded",
                uploaded_at=datetime.now()
            )

            # Create upload log
            log = UploadLog(
                video_record_id=record_id,
                operation="upload",
                status="success",
                message=f"Uploaded {filename} ({file_size} bytes)",
                duration=duration
            )
            await self.db.create_upload_log(log)

            return {
                "success": True,
                "message": "Video uploaded successfully",
                "record_id": record_id,
                "webdav_path": remote_path,
                "webdav_url": webdav_url,
                "file_size": file_size,
                "duration": round(duration, 2)
            }

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            # Update record if created
            if 'record_id' in locals():
                await self.db.update_video_record(
                    record_id,
                    status="failed",
                    error_message=error_msg
                )
                
                # Create error log
                log = UploadLog(
                    video_record_id=record_id,
                    operation="upload",
                    status="failed",
                    message=error_msg,
                    duration=duration
                )
                await self.db.create_upload_log(log)

            return {"success": False, "message": error_msg}
        
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    async def delete_video(self, record_id: int) -> dict:
        """Delete video from WebDAV server"""
        config = await self.get_config()
        if not config.webdav_enabled:
            return {"success": False, "message": "WebDAV is not enabled"}

        start_time = time.time()
        
        try:
            record = await self.db.get_video_record(record_id)
            if not record:
                return {"success": False, "message": "Video record not found"}
            
            if not record.webdav_path:
                return {"success": False, "message": "No WebDAV path for this record"}

            client = self._get_client()
            if not client:
                raise Exception("Failed to create WebDAV client")

            # Delete from WebDAV
            await asyncio.to_thread(client.clean, record.webdav_path)
            
            duration = time.time() - start_time
            
            # Update record
            await self.db.update_video_record(
                record_id,
                status="deleted",
                deleted_at=datetime.now()
            )

            # Create log
            log = UploadLog(
                video_record_id=record_id,
                operation="delete",
                status="success",
                message=f"Deleted {record.webdav_path}",
                duration=duration
            )
            await self.db.create_upload_log(log)

            return {
                "success": True,
                "message": "Video deleted successfully",
                "duration": round(duration, 2)
            }

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            # Create error log
            log = UploadLog(
                video_record_id=record_id,
                operation="delete",
                status="failed",
                message=error_msg,
                duration=duration
            )
            await self.db.create_upload_log(log)

            return {"success": False, "message": error_msg}

    async def delete_all_videos(self) -> dict:
        """Delete all videos from WebDAV server"""
        config = await self.get_config()
        if not config.webdav_enabled:
            return {"success": False, "message": "WebDAV is not enabled"}

        records = await self.db.get_all_video_records(status="uploaded")
        deleted = 0
        failed = 0
        
        for record in records:
            result = await self.delete_video(record.id)
            if result["success"]:
                deleted += 1
            else:
                failed += 1

        return {
            "success": True,
            "message": f"Deleted {deleted} videos, {failed} failed",
            "deleted": deleted,
            "failed": failed
        }

    async def auto_delete_old_videos(self) -> dict:
        """Auto delete videos older than configured days"""
        config = await self.get_config()
        if not config.webdav_enabled or not config.auto_delete_enabled:
            return {"success": False, "message": "Auto delete is not enabled"}

        records = await self.db.get_video_records_for_auto_delete(config.auto_delete_days)
        deleted = 0
        failed = 0
        
        for record in records:
            result = await self.delete_video(record.id)
            if result["success"]:
                deleted += 1
            else:
                failed += 1

        return {
            "success": True,
            "message": f"Auto deleted {deleted} videos, {failed} failed",
            "deleted": deleted,
            "failed": failed
        }

    async def list_webdav_files(self, path: str = None) -> dict:
        """List files on WebDAV server"""
        config = await self.get_config()
        if not config.webdav_enabled:
            return {"success": False, "message": "WebDAV is not enabled"}

        try:
            client = self._get_client()
            if not client:
                raise Exception("Failed to create WebDAV client")

            list_path = path or config.webdav_upload_path or "/"
            files = await asyncio.to_thread(client.list, list_path)
            
            return {
                "success": True,
                "path": list_path,
                "files": files,
                "count": len(files)
            }

        except Exception as e:
            return {"success": False, "message": str(e)}

    async def get_video_records(self, limit: int = 100, status: str = None) -> List[VideoRecord]:
        """Get video records"""
        return await self.db.get_all_video_records(limit, status)

    async def get_upload_logs(self, limit: int = 100) -> List[dict]:
        """Get upload logs"""
        return await self.db.get_upload_logs(limit)

    async def get_stats(self) -> dict:
        """Get video records statistics"""
        return await self.db.get_video_records_stats()

    async def clear_upload_logs(self):
        """Clear all upload logs"""
        await self.db.delete_all_upload_logs()

    async def clear_all_records(self):
        """Clear all video records and upload logs"""
        await self.db.delete_all_video_records()
