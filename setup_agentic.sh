#!/bin/bash

# 🔐 Agentic Setup Script
# Sets up authentication and dependencies for Agentic multi-agent system

set -e

echo "🚀 Setting up Agentic Authentication & Dependencies"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Check Node.js and NPM
print_status "Checking Node.js and NPM..."
if ! command_exists node; then
    print_error "Node.js not found. Please install Node.js first:"
    print_error "  macOS: brew install node"
    print_error "  Ubuntu: sudo apt install nodejs npm"
    print_error "  Or visit: https://nodejs.org"
    exit 1
fi

if ! command_exists npm; then
    print_error "NPM not found. Please install NPM first."
    exit 1
fi

print_success "Node.js $(node --version) and NPM $(npm --version) are installed"

# Step 2: Install Claude Code CLI
print_status "Installing Claude Code CLI..."
if command_exists claude; then
    print_warning "Claude Code CLI already installed: $(claude --version 2>/dev/null || echo 'version unknown')"
    echo -n "Upgrade to latest version? (y/N): "
    read -r upgrade
    if [[ $upgrade =~ ^[Yy]$ ]]; then
        npm install -g @anthropic-ai/claude-code
        print_success "Claude Code CLI upgraded"
    fi
else
    print_status "Installing Claude Code CLI globally..."
    npm install -g @anthropic-ai/claude-code
    print_success "Claude Code CLI installed successfully"
fi

# Step 3: Check Claude Code authentication
print_status "Checking Claude Code authentication..."
if claude --help >/dev/null 2>&1; then
    print_success "Claude Code CLI is working"
    
    # Try to check authentication status
    print_status "Testing Claude Code authentication..."
    echo "Testing authentication - this will open Claude if not logged in..."
    echo "Please follow the login prompts if they appear."
    echo ""
    
    # Create a temporary test to check auth
    if timeout 30s claude -p "test auth" >/dev/null 2>&1; then
        print_success "Claude Code is authenticated and working!"
    else
        print_warning "Claude Code may not be authenticated or timed out"
        print_warning "Please run 'claude' manually and follow login prompts"
        print_warning "You'll need a Claude Pro or Team subscription"
    fi
else
    print_error "Claude Code CLI installation failed"
    exit 1
fi

# Step 4: Check for Aider (optional enhancement)
print_status "Checking for Aider (optional enhancement)..."
if command_exists aider; then
    print_success "Aider is already installed: $(aider --version 2>/dev/null || echo 'version unknown')"
else
    echo -n "Install Aider for enhanced backend/frontend agents? (y/N): "
    read -r install_aider
    if [[ $install_aider =~ ^[Yy]$ ]]; then
        print_status "Installing Aider..."
        pip install aider-chat
        print_success "Aider installed successfully"
    else
        print_warning "Skipping Aider installation - some specialized agents won't work"
    fi
fi

# Step 5: API Key Setup (optional)
print_status "Setting up API keys (optional for enhanced features)..."

# Check existing API keys
ANTHROPIC_KEY_SET=false
OPENAI_KEY_SET=false

if [ -n "$ANTHROPIC_API_KEY" ]; then
    print_success "Anthropic API key is already set (${ANTHROPIC_API_KEY:0:10}...)"
    ANTHROPIC_KEY_SET=true
fi

if [ -n "$OPENAI_API_KEY" ]; then
    print_success "OpenAI API key is already set (${OPENAI_API_KEY:0:10}...)"
    OPENAI_KEY_SET=true
fi

# Prompt for missing keys
if [ "$ANTHROPIC_KEY_SET" = false ]; then
    echo ""
    print_warning "Anthropic API key not found"
    echo "This enables enhanced Aider agents for specialized development tasks"
    echo "Get your key at: https://console.anthropic.com/"
    echo -n "Enter Anthropic API key (or press Enter to skip): "
    read -r anthropic_key
    if [ -n "$anthropic_key" ]; then
        echo "export ANTHROPIC_API_KEY=\"$anthropic_key\"" >> ~/.bashrc
        echo "export ANTHROPIC_API_KEY=\"$anthropic_key\"" >> ~/.zshrc
        export ANTHROPIC_API_KEY="$anthropic_key"
        print_success "Anthropic API key added to shell profiles"
    fi
fi

if [ "$OPENAI_KEY_SET" = false ]; then
    echo ""
    print_warning "OpenAI API key not found"
    echo "This is completely optional - provides alternative model choices"
    echo "Get your key at: https://platform.openai.com/"
    echo -n "Enter OpenAI API key (or press Enter to skip): "
    read -r openai_key
    if [ -n "$openai_key" ]; then
        echo "export OPENAI_API_KEY=\"$openai_key\"" >> ~/.bashrc
        echo "export OPENAI_API_KEY=\"$openai_key\"" >> ~/.zshrc
        export OPENAI_API_KEY="$openai_key"
        print_success "OpenAI API key added to shell profiles"
    fi
fi

# Step 6: Create configuration directory
print_status "Setting up Agentic configuration..."
mkdir -p .agentic

# Create a configuration file
cat > .agentic/config.yml << EOF
# Agentic Configuration
version: "1.0.0"

# Primary authentication: Claude Code CLI (uses Claude Desktop login)
claude_code:
  enabled: true
  model: "sonnet"  # or "opus" for more advanced tasks
  default_tools:
    - "Edit"
    - "Bash(git *)"
    - "Write"

# Enhanced authentication: API keys (optional)
api:
  anthropic:
    api_key: "${ANTHROPIC_API_KEY:-}"
    model: "claude-3-5-sonnet"
  openai:
    api_key: "${OPENAI_API_KEY:-}"
    model: "gpt-4"

# Agent configuration
agents:
  claude_code:
    type: "claude_code"
    description: "Primary coding agent using Claude Code CLI"
    capabilities: ["coding", "refactoring", "analysis", "debugging"]
  
  backend:
    type: "aider_backend"
    description: "Backend development specialist using Aider"
    enabled: $([ -n "$ANTHROPIC_API_KEY" ] && echo "true" || echo "false")
  
  frontend:
    type: "aider_frontend" 
    description: "Frontend development specialist using Aider"
    enabled: $([ -n "$ANTHROPIC_API_KEY" ] && echo "true" || echo "false")
  
  testing:
    type: "aider_testing"
    description: "Testing and QA specialist using Aider"
    enabled: $([ -n "$ANTHROPIC_API_KEY" ] && echo "true" || echo "false")

# Project settings
project:
  language: "auto-detect"
  framework: "auto-detect"
  
logging:
  level: "INFO"
  file: ".agentic/logs/agentic.log"
EOF

print_success "Configuration file created at .agentic/config.yml"

# Create logs directory
mkdir -p .agentic/logs

# Step 7: Test the installation
print_status "Testing Agentic installation..."

# Test Python imports
if python3 -c "import agentic; print('✅ Agentic module imports successfully')" 2>/dev/null; then
    print_success "Agentic Python module is working"
else
    print_warning "Agentic Python module may not be installed properly"
    print_warning "Try: pip install -e ."
fi

# Test CLI
if command_exists agentic; then
    print_success "Agentic CLI is available"
    
    # Test basic CLI commands
    if agentic --help >/dev/null 2>&1; then
        print_success "Agentic CLI is working"
    else
        print_warning "Agentic CLI may have issues"
    fi
else
    print_warning "Agentic CLI not found in PATH"
    print_warning "Make sure to install with: pip install -e ."
fi

# Step 8: Summary
echo ""
print_success "🎉 Agentic setup complete!"
echo "========================================="
echo ""
echo "📋 Setup Summary:"
echo "  ✅ Claude Code CLI: Installed and ready"
echo "  ✅ Configuration: Created at .agentic/config.yml"

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "  ✅ Anthropic API: Configured for enhanced Aider agents"
else
    echo "  ⚠️  Anthropic API: Not configured (optional)"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    echo "  ✅ OpenAI API: Configured for alternative models"
else
    echo "  ⚠️  OpenAI API: Not configured (optional)"
fi

echo ""
echo "🚀 Quick Start:"
echo "  1. Make sure Claude Code is authenticated:"
echo "     claude"
echo "     # Follow login prompts if needed"
echo ""
echo "  2. Test Agentic:"
echo "     agentic --help"
echo "     agentic init"
echo "     agentic spawn claude_code"
echo ""
echo "  3. Start coding:"
echo "     agentic exec \"analyze this codebase structure\""
echo "     agentic exec \"refactor the authentication system\""
echo ""
echo "📚 Documentation:"
echo "  - Authentication guide: ./AUTHENTICATION_SETUP.md"
echo "  - Claude Code docs: https://docs.anthropic.com/en/docs/claude-code"
echo "  - Aider docs: https://aider.chat"
echo ""

if [ "$ANTHROPIC_KEY_SET" = false ] && [ "$OPENAI_KEY_SET" = false ]; then
    print_warning "Consider adding API keys later for enhanced capabilities"
    echo "  Re-run this script or manually set environment variables"
fi

print_success "Happy coding with Agentic! 🤖✨" 