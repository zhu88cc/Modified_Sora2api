"""Sora-compatible API endpoints

Provides Sora-native API format for video and image generation.
"""
from fastapi import APIRouter
from ..services.generation_handler import GenerationHandler

router = APIRouter()

# Dependency injection
generation_handler: GenerationHandler = None


def set_generation_handler(handler: GenerationHandler):
    """Set generation handler instance"""
    global generation_handler
    generation_handler = handler


# Add Sora-specific endpoints here if needed
# Currently this module serves as a placeholder for future Sora-native API endpoints
