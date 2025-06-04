"""
Secure credential management with multiple fallback options:
1. Keyring (secure storage) - primary
2. .env file - fallback 
3. Environment variables - final fallback
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def get_api_key(provider: str, workspace_path: Optional[Path] = None) -> Optional[str]:
    """
    Get API key for provider with fallback chain:
    1. Keyring (project-specific)
    2. Keyring (global)  
    3. .env file (project)
    4. Environment variable
    """
    workspace_path = workspace_path or Path.cwd()
    
    # Normalize provider names
    provider_map = {
        'google': 'gemini',  # Google and Gemini are the same
        'gemini': 'gemini',
        'anthropic': 'anthropic', 
        'openai': 'openai'
    }
    
    normalized_provider = provider_map.get(provider.lower(), provider.lower())
    
    # 1. Try keyring (project-specific)
    if KEYRING_AVAILABLE:
        try:
            service_name = f"agentic.{normalized_provider}.{workspace_path.name}"
            key = keyring.get_password(service_name, str(workspace_path))
            if key:
                return key
        except Exception:
            pass
    
    # 2. Try keyring (global)
    if KEYRING_AVAILABLE:
        try:
            service_name = f"agentic.{normalized_provider}"
            key = keyring.get_password(service_name, "global")
            if key:
                return key
        except Exception:
            pass
    
    # 3. Try .env file (project-specific)
    if DOTENV_AVAILABLE:
        env_file = workspace_path / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            key = _get_env_key(normalized_provider)
            if key:
                return key
    
    # 4. Try environment variables (current process)
    key = _get_env_key(normalized_provider)
    if key:
        return key
    
    return None


def _get_env_key(provider: str) -> Optional[str]:
    """Get API key from environment variables"""
    # Map providers to their common environment variable names
    env_var_map = {
        'gemini': ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
        'anthropic': ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY'],
        'openai': ['OPENAI_API_KEY'],
        'google': ['GOOGLE_API_KEY', 'GEMINI_API_KEY']
    }
    
    env_vars = env_var_map.get(provider, [f'{provider.upper()}_API_KEY'])
    
    for env_var in env_vars:
        key = os.getenv(env_var)
        if key:
            return key
    
    return None


def list_api_keys(workspace_path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """
    List all configured API keys with their sources
    Returns: {provider: {key: str, scope: str, source: str}}
    """
    workspace_path = workspace_path or Path.cwd()
    result = {}
    
    providers = ['anthropic', 'openai', 'gemini']
    
    for provider in providers:
        key_info = _get_key_with_source(provider, workspace_path)
        if key_info:
            result[provider] = key_info
    
    return result


def _get_key_with_source(provider: str, workspace_path: Path) -> Optional[Dict[str, str]]:
    """Get API key with information about its source"""
    
    # 1. Check keyring (project-specific)
    if KEYRING_AVAILABLE:
        try:
            service_name = f"agentic.{provider}.{workspace_path.name}"
            key = keyring.get_password(service_name, str(workspace_path))
            if key:
                return {
                    'key': key,
                    'scope': 'project',
                    'source': 'keyring'
                }
        except Exception:
            pass
    
    # 2. Check keyring (global)
    if KEYRING_AVAILABLE:
        try:
            service_name = f"agentic.{provider}"
            key = keyring.get_password(service_name, "global")
            if key:
                return {
                    'key': key,
                    'scope': 'global', 
                    'source': 'keyring'
                }
        except Exception:
            pass
    
    # 3. Check .env file
    if DOTENV_AVAILABLE:
        env_file = workspace_path / '.env'
        if env_file.exists():
            # Temporarily load just this .env file
            current_env = dict(os.environ)
            load_dotenv(env_file)
            key = _get_env_key(provider)
            # Restore original environment
            os.environ.clear()
            os.environ.update(current_env)
            
            if key:
                return {
                    'key': key,
                    'scope': 'project',
                    'source': '.env file'
                }
    
    # 4. Check environment variables
    key = _get_env_key(provider)
    if key:
        return {
            'key': key,
            'scope': 'environment',
            'source': 'env var'
        }
    
    return None


def create_env_template(workspace_path: Path) -> None:
    """Create a .env.example template with API key placeholders"""
    template_content = """# API Keys for Agentic
# Copy this file to .env and add your actual API keys

# Anthropic (Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# OpenAI (GPT models)  
OPENAI_API_KEY=your_openai_api_key_here

# Google/Gemini (Gemini models)
GOOGLE_API_KEY=your_google_api_key_here
# Alternative name for Gemini
GEMINI_API_KEY=your_google_api_key_here

# Note: Only one of GOOGLE_API_KEY or GEMINI_API_KEY is needed
# They both work for Gemini models
"""
    
    template_file = workspace_path / '.env.example'
    template_file.write_text(template_content)
    
    # Ensure .env is in .gitignore
    gitignore_file = workspace_path / '.gitignore'
    if gitignore_file.exists():
        gitignore_content = gitignore_file.read_text()
        if '.env' not in gitignore_content:
            gitignore_file.write_text(gitignore_content + '\n.env\n')
    else:
        gitignore_file.write_text('.env\n')


def validate_api_key(provider: str, key: str) -> Tuple[bool, str]:
    """
    Basic validation of API key format
    Returns: (is_valid, error_message)
    """
    if not key or not key.strip():
        return False, "API key cannot be empty"
    
    key = key.strip()
    
    # Basic format validation
    if provider.lower() == 'anthropic':
        if not key.startswith('sk-ant-'):
            return False, "Anthropic API keys should start with 'sk-ant-'"
        if len(key) < 20:
            return False, "Anthropic API key seems too short"
            
    elif provider.lower() == 'openai':
        if not key.startswith('sk-'):
            return False, "OpenAI API keys should start with 'sk-'"
        if len(key) < 20:
            return False, "OpenAI API key seems too short"
            
    elif provider.lower() in ['google', 'gemini']:
        # Google API keys can have various formats
        if len(key) < 20:
            return False, "Google API key seems too short"
    
    return True, ""


def setup_credential_storage() -> bool:
    """
    Setup credential storage on the system
    Returns: True if setup successful
    """
    if not KEYRING_AVAILABLE:
        return False
    
    try:
        # Test keyring functionality
        test_service = "agentic.test"
        test_key = "test_key"
        
        keyring.set_password(test_service, "test", test_key)
        retrieved = keyring.get_password(test_service, "test")
        keyring.delete_password(test_service, "test")
        
        return retrieved == test_key
        
    except Exception:
        return False 