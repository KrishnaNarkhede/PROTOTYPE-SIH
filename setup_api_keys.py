#!/usr/bin/env python3
"""
API Keys Setup Script
====================

Run this script to easily configure your API keys
"""

from config import config

def setup_api_keys():
    """Interactive setup for NCBI API key"""
    print("🔑 BioMapper NCBI API Key Setup")
    print("=" * 40)
    
    # NCBI API Key
    ncbi_key = input("Enter your NCBI API key (or press Enter to use default): ").strip()
    if ncbi_key:
        config.NCBI_API_KEY = ncbi_key
        print("✅ NCBI API key configured")
    else:
        print("✅ Using default NCBI API key")
    
    # Show configuration status
    print("\n📊 Configuration Status:")
    print(f"NCBI: {'✅ Configured' if config.NCBI_API_KEY else '❌ Not configured'}")
    
    if config.has_ncbi_api_key():
        print("\n🚀 Ready to use NCBI database!")
    else:
        print("\n⚠️  No API key configured - using mock data")

if __name__ == "__main__":
    setup_api_keys()