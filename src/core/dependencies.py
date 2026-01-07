"""FastAPI dependency injection module

This module provides a centralized dependency injection system for the application,
replacing global variables with proper FastAPI dependencies.
"""
from typing import Optional
from functools import lru_cache


class AppDependencies:
    """Container for application dependencies"""

    _instance: Optional['AppDependencies'] = None

    def __init__(self):
        self._db = None
        self._token_manager = None
        self._proxy_manager = None
        self._concurrency_manager = None
        self._load_balancer = None
        self._sora_client = None
        self._generation_handler = None

    @classmethod
    def get_instance(cls) -> 'AppDependencies':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def db(self):
        """Get database instance"""
        if self._db is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._db

    @property
    def token_manager(self):
        """Get token manager instance"""
        if self._token_manager is None:
            raise RuntimeError("TokenManager not initialized. Call initialize() first.")
        return self._token_manager

    @property
    def proxy_manager(self):
        """Get proxy manager instance"""
        if self._proxy_manager is None:
            raise RuntimeError("ProxyManager not initialized. Call initialize() first.")
        return self._proxy_manager

    @property
    def concurrency_manager(self):
        """Get concurrency manager instance"""
        if self._concurrency_manager is None:
            raise RuntimeError("ConcurrencyManager not initialized. Call initialize() first.")
        return self._concurrency_manager

    @property
    def load_balancer(self):
        """Get load balancer instance"""
        if self._load_balancer is None:
            raise RuntimeError("LoadBalancer not initialized. Call initialize() first.")
        return self._load_balancer

    @property
    def sora_client(self):
        """Get sora client instance"""
        if self._sora_client is None:
            raise RuntimeError("SoraClient not initialized. Call initialize() first.")
        return self._sora_client

    @property
    def generation_handler(self):
        """Get generation handler instance"""
        if self._generation_handler is None:
            raise RuntimeError("GenerationHandler not initialized. Call initialize() first.")
        return self._generation_handler

    def initialize(self, db, token_manager, proxy_manager, concurrency_manager,
                   load_balancer, sora_client, generation_handler):
        """Initialize all dependencies"""
        self._db = db
        self._token_manager = token_manager
        self._proxy_manager = proxy_manager
        self._concurrency_manager = concurrency_manager
        self._load_balancer = load_balancer
        self._sora_client = sora_client
        self._generation_handler = generation_handler


# Singleton accessor
def get_dependencies() -> AppDependencies:
    """Get the application dependencies container"""
    return AppDependencies.get_instance()


# FastAPI dependency functions
def get_db():
    """FastAPI dependency for database"""
    return get_dependencies().db


def get_token_manager():
    """FastAPI dependency for token manager"""
    return get_dependencies().token_manager


def get_proxy_manager():
    """FastAPI dependency for proxy manager"""
    return get_dependencies().proxy_manager


def get_concurrency_manager():
    """FastAPI dependency for concurrency manager"""
    return get_dependencies().concurrency_manager


def get_load_balancer():
    """FastAPI dependency for load balancer"""
    return get_dependencies().load_balancer


def get_sora_client():
    """FastAPI dependency for sora client"""
    return get_dependencies().sora_client


def get_generation_handler():
    """FastAPI dependency for generation handler"""
    return get_dependencies().generation_handler
