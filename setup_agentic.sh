#!/bin/bash

echo "🤖 Agentic Setup & Configuration Script"
echo "========================================"

# Check if Aider is installed
if command -v aider &> /dev/null; then
    echo "✅ Aider is installed: $(which aider)"
else
    echo "❌ Aider not found. Installing..."
    pip install aider-chat
fi

# Check for API keys
echo ""
echo "🔑 Checking API Key Configuration..."

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  ANTHROPIC_API_KEY not set"
    echo "   Set it with: export ANTHROPIC_API_KEY='your-key-here'"
else
    echo "✅ ANTHROPIC_API_KEY is set (${ANTHROPIC_API_KEY:0:20}...)"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set"
    echo "   Set it with: export OPENAI_API_KEY='your-key-here'"
else
    echo "✅ OPENAI_API_KEY is set (${OPENAI_API_KEY:0:20}...)"
fi

# Test Aider with API key
echo ""
echo "🧪 Testing Aider Configuration..."

if [ ! -z "$ANTHROPIC_API_KEY" ]; then
    echo "Testing Aider with Anthropic API..."
    echo "hello world" | aider --anthropic-api-key "$ANTHROPIC_API_KEY" --model claude-3-5-sonnet --no-git --yes --dry-run --exit
    if [ $? -eq 0 ]; then
        echo "✅ Aider + Anthropic API works!"
    else
        echo "❌ Aider + Anthropic API test failed"
    fi
fi

# Create .agentic directory and config
echo ""
echo "📁 Setting up Agentic configuration..."

mkdir -p .agentic

cat > .agentic/config.yml << EOF
workspace_name: "$(basename $(pwd))"
workspace_path: "$(pwd)"
analysis_depth: standard
auto_spawn_agents: true
default_agents:
  - claude_code
max_concurrent_agents: 3
include_tests: true
include_docs: true
log_level: INFO
ignore_patterns:
  - "*.pyc"
  - "__pycache__"
  - ".git"
  - "node_modules"
  - "*.log"
  - ".DS_Store"
  - "*.tmp"
  - "*.temp"

# API Configuration (if environment variables are set)
EOF

if [ ! -z "$ANTHROPIC_API_KEY" ] || [ ! -z "$OPENAI_API_KEY" ]; then
    echo "api:" >> .agentic/config.yml
    if [ ! -z "$ANTHROPIC_API_KEY" ]; then
        echo "  anthropic:" >> .agentic/config.yml
        echo "    api_key: \"$ANTHROPIC_API_KEY\"" >> .agentic/config.yml
        echo "    model: \"claude-3-5-sonnet\"" >> .agentic/config.yml
    fi
    if [ ! -z "$OPENAI_API_KEY" ]; then
        echo "  openai:" >> .agentic/config.yml
        echo "    api_key: \"$OPENAI_API_KEY\"" >> .agentic/config.yml
        echo "    model: \"gpt-4\"" >> .agentic/config.yml
    fi
fi

echo "✅ Created .agentic/config.yml"

# Test direct Aider usage
echo ""
echo "🎯 Quick Test - Direct Aider Usage:"
echo "  aider --anthropic-api-key \$ANTHROPIC_API_KEY --model claude-3-5-sonnet"
echo ""
echo "🎯 Quick Test - Agentic Commands (once working):"
echo "  agentic status"
echo "  agentic exec \"explain this codebase\""
echo ""

echo "✅ Setup complete! Set your API keys and test the system." 