#!/usr/bin/env python3
"""
Configuration file for BioMapper Enhanced eDNA Analysis
======================================================

API Keys and Configuration Settings
"""

import os
from typing import Optional

class Config:
    """Configuration class for API keys and settings"""
    
    def __init__(self):
        # API Keys (can be set via environment variables or directly)
        self.NCBI_API_KEY: Optional[str] = os.getenv('NCBI_API_KEY') or "bef46ef0d9c7cccc3f5e275ab9f4c647da09"  # Your NCBI API key
        
        # Database URLs
        self.NCBI_BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
        
        # Analysis Parameters
        self.DENOISING_THRESHOLD = 0.01
        self.CLUSTERING_THRESHOLD = 0.97
        self.NOVELTY_THRESHOLD = 0.8
        self.QUALITY_THRESHOLD = 0.5
        
        # Processing Limits
        self.MAX_SEQUENCES_PER_BATCH = 25
        self.MAX_FILE_SIZE_MB = 100
        self.API_TIMEOUT_SECONDS = 30
        
    def set_ncbi_api_key(self, ncbi_key: str):
        """Set NCBI API key programmatically"""
        self.NCBI_API_KEY = ncbi_key
    
    def has_ncbi_api_key(self) -> bool:
        """Check if NCBI API key is configured"""
        return bool(self.NCBI_API_KEY)
    
    def get_config_summary(self) -> dict:
        """Get configuration summary (without exposing keys)"""
        return {
            "ncbi_configured": bool(self.NCBI_API_KEY),
            "denoising_threshold": self.DENOISING_THRESHOLD,
            "clustering_threshold": self.CLUSTERING_THRESHOLD,
            "max_batch_size": self.MAX_SEQUENCES_PER_BATCH
        }

# Global configuration instance
config = Config()

# Example usage - ADD YOUR NCBI API KEY HERE:
# config.set_ncbi_api_key("your_ncbi_api_key_here")

# OR set it directly:
# config.NCBI_API_KEY = "your_ncbi_api_key_here"