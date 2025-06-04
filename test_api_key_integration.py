#!/usr/bin/env python3
"""
Test API Key Integration
Tests the complete API key management system with all fallback methods.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path for local testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic.utils.credentials import (
    get_api_key, 
    list_api_keys, 
    create_env_template,
    validate_api_key
)


def test_env_var_fallback():
    """Test environment variable fallback"""
    print("🧪 Testing environment variable fallback...")
    
    # Set temporary env var
    os.environ['GEMINI_API_KEY'] = 'test-env-var-key'
    
    try:
        key = get_api_key('gemini')
        assert key == 'test-env-var-key', f"Expected 'test-env-var-key', got '{key}'"
        print("✅ Environment variable retrieval works")
        
        # Test listing
        keys_info = list_api_keys()
        assert 'gemini' in keys_info, "Gemini key not found in listing"
        assert keys_info['gemini']['source'] == 'env var', "Wrong source detected"
        print("✅ Environment variable listing works")
        
    finally:
        # Clean up
        del os.environ['GEMINI_API_KEY']


def test_env_file_creation():
    """Test .env template creation"""
    print("🧪 Testing .env template creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create template
        create_env_template(temp_path)
        
        # Check files were created
        env_example = temp_path / '.env.example'
        gitignore = temp_path / '.gitignore'
        
        assert env_example.exists(), ".env.example not created"
        assert gitignore.exists(), ".gitignore not created"
        
        # Check content
        content = env_example.read_text()
        assert 'GEMINI_API_KEY' in content, "Gemini key not in template"
        assert 'ANTHROPIC_API_KEY' in content, "Anthropic key not in template"
        assert 'OPENAI_API_KEY' in content, "OpenAI key not in template"
        
        gitignore_content = gitignore.read_text()
        assert '.env' in gitignore_content, ".env not in .gitignore"
        
        print("✅ .env template creation works")


def test_api_key_validation():
    """Test API key validation"""
    print("🧪 Testing API key validation...")
    
    # Test valid keys
    valid_tests = [
        ('anthropic', 'sk-ant-1234567890abcdef1234567890'),
        ('openai', 'sk-1234567890abcdef1234567890'),
        ('gemini', 'AIzaSyD1234567890abcdef1234567890'),
    ]
    
    for provider, key in valid_tests:
        is_valid, _ = validate_api_key(provider, key)
        assert is_valid, f"{provider} key validation should pass: {key}"
    
    # Test invalid keys
    invalid_tests = [
        ('anthropic', 'invalid-key'),  # Wrong prefix
        ('openai', 'sk-'),  # Too short
        ('gemini', ''),  # Empty
    ]
    
    for provider, key in invalid_tests:
        is_valid, _ = validate_api_key(provider, key)
        assert not is_valid, f"{provider} key validation should fail: {key}"
    
    print("✅ API key validation works")


def test_provider_normalization():
    """Test provider name normalization"""
    print("🧪 Testing provider normalization...")
    
    # Set test env vars
    os.environ['GOOGLE_API_KEY'] = 'google-key-test'
    os.environ['ANTHROPIC_API_KEY'] = 'anthropic-key-test'
    
    try:
        # Test that google and gemini resolve to same key
        google_key = get_api_key('google')
        gemini_key = get_api_key('gemini')
        
        assert google_key == gemini_key, "Google and Gemini should resolve to same key"
        assert google_key == 'google-key-test', f"Expected 'google-key-test', got '{google_key}'"
        
        # Test anthropic
        anthropic_key = get_api_key('anthropic')
        assert anthropic_key == 'anthropic-key-test', f"Expected 'anthropic-key-test', got '{anthropic_key}'"
        
        print("✅ Provider normalization works")
        
    finally:
        # Clean up
        if 'GOOGLE_API_KEY' in os.environ:
            del os.environ['GOOGLE_API_KEY']
        if 'ANTHROPIC_API_KEY' in os.environ:
            del os.environ['ANTHROPIC_API_KEY']


def main():
    """Run all tests"""
    print("🚀 Testing Agentic API Key Management System\n")
    
    tests = [
        test_env_var_fallback,
        test_env_file_creation,
        test_api_key_validation,
        test_provider_normalization,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for test in tests:
        status = "✅ PASS" if test.__name__ not in [t.__name__ for t in tests if failed > 0] else "❌ FAIL"
        print(f"{status} {test.__name__}")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All API key management tests passed!")
        print("\nReady to use:")
        print("  • agentic keys set gemini --global")
        print("  • agentic keys env-template")
        print("  • export GEMINI_API_KEY='your-key'")
    else:
        print(f"\n💥 {failed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 