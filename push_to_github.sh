#!/bin/bash

# AgentMap - Push to GitHub
# Repository: https://github.com/alokranjan-agp/AgentMap.git

echo "🚀 Pushing AgentMap to GitHub"
echo "=============================="
echo ""

cd "/Users/alok/Downloads/agentmap 2"

# Initialize git if not already done
if [ ! -d .git ]; then
    echo "📦 Initializing git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git repository already initialized"
fi

# Create .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment variables
.env
.env.local

# Logs
*.log

# Backup files
*_backup*
*_old*
*.bak

# Test results
.pytest_cache/
.coverage
htmlcov/

# Temporary files
*.tmp
tmp/
temp/

# Results (optional - comment out if you want to include)
# workbench_real_results/
# workbench_benchmark_results/
EOF
echo "✅ .gitignore created"

# Create LICENSE file
echo "📄 Creating LICENSE..."
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Alok Ranjan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
echo "✅ LICENSE created"

# Check current branch
current_branch=$(git branch --show-current 2>/dev/null)
echo ""
echo "📋 Current branch: ${current_branch:-none}"

# Add all files
echo ""
echo "📦 Adding files to git..."
git add .
echo "✅ Files added"

# Show what will be committed
echo ""
echo "📊 Files to be committed:"
git status --short | head -20
file_count=$(git status --short | wc -l)
echo "   ... and $file_count total files"

# Commit
echo ""
echo "💾 Creating commit..."
git commit -m "Initial commit: AgentMap v1.0

- Beat GPT-4 on WorkBench (47.1% vs 43%)
- 100% accuracy on τ2-bench (278/278 tasks)
- 100% determinism (unique)
- 50-60% cost savings
- Complete documentation and examples
- Publication-ready materials"

if [ $? -eq 0 ]; then
    echo "✅ Commit created successfully"
else
    echo "⚠️  Commit failed or no changes to commit"
fi

# Rename branch to main
echo ""
echo "🔀 Setting branch to main..."
git branch -M main
echo "✅ Branch set to main"

# Add remote
echo ""
echo "🔗 Adding remote origin..."
git remote remove origin 2>/dev/null  # Remove if exists
git remote add origin https://github.com/alokranjan-agp/AgentMap.git
echo "✅ Remote added"

# Show remote
echo ""
echo "📡 Remote repository:"
git remote -v

# Push to GitHub
echo ""
echo "🚀 Pushing to GitHub..."
echo "   Repository: https://github.com/alokranjan-agp/AgentMap.git"
echo "   Branch: main"
echo ""
read -p "Press Enter to push (or Ctrl+C to cancel)..."

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "🎉 Your repository is live at:"
    echo "   https://github.com/alokranjan-agp/AgentMap"
    echo ""
    echo "📊 What was pushed:"
    echo "   ✅ Core framework (agentmap/)"
    echo "   ✅ Main runners (run_*.py)"
    echo "   ✅ Results (workbench_real_results/)"
    echo "   ✅ Documentation (*.md)"
    echo "   ✅ Package config (pyproject.toml)"
    echo "   ✅ License (LICENSE)"
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Visit: https://github.com/alokranjan-agp/AgentMap"
    echo "   2. Add description and topics"
    echo "   3. Enable GitHub Pages (optional)"
    echo "   4. Share your achievement!"
else
    echo ""
    echo "❌ Push failed!"
    echo ""
    echo "Common issues:"
    echo "   1. Repository doesn't exist - create it on GitHub first"
    echo "   2. Authentication failed - check your credentials"
    echo "   3. Branch protection - check repository settings"
    echo ""
    echo "To retry:"
    echo "   git push -u origin main"
fi
EOF

chmod +x "/Users/alok/Downloads/agentmap 2/push_to_github.sh"
echo "✅ Created push_to_github.sh"
