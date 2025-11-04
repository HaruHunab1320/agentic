"""
Language and Framework Selection System
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from agentic.models.task import ProjectContext
from agentic.utils.logging import LoggerMixin


class LanguageSelector(LoggerMixin):
    """Handles language and framework selection for code generation"""
    
    # Common language indicators in commands
    LANGUAGE_HINTS = {
        "python": ["python", "django", "flask", "fastapi", "pytest", "pip"],
        "javascript": ["javascript", "js", "node", "npm", "jest"],
        "typescript": ["typescript", "ts", "tsx", "angular", "nest"],
        "react": ["react", "jsx", "component", "hooks", "redux"],
        "vue": ["vue", "vuex", "nuxt", "composition api"],
        "java": ["java", "spring", "maven", "gradle", "junit"],
        "go": ["go", "golang", "gin", "echo", "gorilla"],
        "rust": ["rust", "cargo", "actix", "rocket", "tokio"],
        "ruby": ["ruby", "rails", "rspec", "bundler", "gem"],
        "php": ["php", "laravel", "symfony", "composer", "phpunit"],
        "csharp": ["c#", "csharp", ".net", "dotnet", "aspnet"],
        "swift": ["swift", "ios", "swiftui", "uikit", "xcode"],
        "kotlin": ["kotlin", "android", "ktor", "gradle"]
    }
    
    # Framework to language mapping
    FRAMEWORK_LANGUAGES = {
        "django": "python",
        "flask": "python", 
        "fastapi": "python",
        "express": "javascript",
        "nextjs": "javascript",
        "react": "javascript",
        "vue": "javascript",
        "angular": "typescript",
        "spring": "java",
        "rails": "ruby",
        "laravel": "php",
        "aspnet": "csharp"
    }
    
    # Common project types and their typical languages
    PROJECT_TYPE_LANGUAGES = {
        "web": ["javascript", "typescript", "python"],
        "api": ["python", "javascript", "go", "java"],
        "cli": ["python", "go", "rust"],
        "mobile": ["swift", "kotlin", "javascript"],
        "desktop": ["python", "csharp", "java"],
        "library": ["python", "javascript", "rust"]
    }
    
    def __init__(self):
        super().__init__()
        
    def detect_language_from_command(self, command: str) -> Optional[str]:
        """Detect language preference from command text"""
        command_lower = command.lower()
        
        # Check for explicit language mentions
        for language, indicators in self.LANGUAGE_HINTS.items():
            for indicator in indicators:
                # Look for word boundaries to avoid false matches
                pattern = r'\b' + re.escape(indicator) + r'\b'
                if re.search(pattern, command_lower):
                    self.logger.debug(f"Detected language '{language}' from indicator '{indicator}'")
                    return language if language not in ["react", "vue"] else "javascript"
        
        return None
    
    def detect_framework_from_command(self, command: str) -> Optional[str]:
        """Detect framework preference from command text"""
        command_lower = command.lower()
        
        # Check all language hints for framework matches
        for language, indicators in self.LANGUAGE_HINTS.items():
            if language in ["react", "vue", "angular"]:  # These are frameworks
                for indicator in indicators:
                    pattern = r'\b' + re.escape(indicator) + r'\b'
                    if re.search(pattern, command_lower):
                        return language
        
        # Check framework-specific terms
        framework_terms = {
            "express": ["express", "expressjs"],
            "nextjs": ["next.js", "nextjs", "next"],
            "django": ["django"],
            "flask": ["flask"],
            "fastapi": ["fastapi", "fast api"],
            "spring": ["spring boot", "spring"],
            "rails": ["rails", "ruby on rails"]
        }
        
        for framework, terms in framework_terms.items():
            for term in terms:
                if term in command_lower:
                    return framework
        
        return None
    
    def infer_language_from_context(self, command: str, project_context: Optional[ProjectContext]) -> Tuple[Optional[str], Optional[str]]:
        """
        Infer language and framework from command and project context
        
        Returns: (language, framework)
        """
        # First, try to detect from command
        command_language = self.detect_language_from_command(command)
        command_framework = self.detect_framework_from_command(command)
        
        # If framework is detected, get its language
        if command_framework and command_framework in self.FRAMEWORK_LANGUAGES:
            framework_language = self.FRAMEWORK_LANGUAGES[command_framework]
            if not command_language:
                command_language = framework_language
        
        # If we have explicit preferences from command, use them
        if command_language or command_framework:
            return command_language, command_framework
        
        # Otherwise, use project context
        if project_context:
            return project_context.primary_language, project_context.framework
        
        # No preference found
        return None, None
    
    def get_clarification_prompt(self, command: str, project_context: Optional[ProjectContext]) -> Optional[str]:
        """Generate a clarification prompt if language/framework is ambiguous"""
        # Check if we can infer language
        language, framework = self.infer_language_from_context(command, project_context)
        
        if language:
            return None  # No clarification needed
        
        # Determine what kind of project this might be
        command_lower = command.lower()
        
        # Check for project type hints
        if any(word in command_lower for word in ["web app", "website", "frontend", "ui"]):
            suggestions = ["React (TypeScript)", "Vue.js", "Angular", "Plain HTML/CSS/JS"]
        elif any(word in command_lower for word in ["api", "backend", "server", "rest"]):
            suggestions = ["Python (FastAPI)", "Node.js (Express)", "Go", "Java (Spring)"]
        elif any(word in command_lower for word in ["cli", "command line", "terminal"]):
            suggestions = ["Python", "Go", "Rust", "Node.js"]
        elif any(word in command_lower for word in ["mobile", "ios", "android"]):
            suggestions = ["React Native", "Swift (iOS)", "Kotlin (Android)", "Flutter"]
        else:
            # Generic suggestions based on project context
            if project_context and project_context.project_type:
                project_type = project_context.project_type
                if project_type in self.PROJECT_TYPE_LANGUAGES:
                    languages = self.PROJECT_TYPE_LANGUAGES[project_type]
                    suggestions = [lang.title() for lang in languages[:4]]
                else:
                    suggestions = ["Python", "JavaScript", "TypeScript", "Go"]
            else:
                suggestions = ["Python", "JavaScript", "TypeScript", "Go"]
        
        prompt = f"""
I need to know what language/framework you'd like to use for: "{command}"

Suggestions based on your request:
{chr(10).join(f"  - {s}" for s in suggestions)}

Or specify your preferred language/framework.
"""
        
        return prompt
    
    def parse_language_response(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse user's language/framework choice from response"""
        response_lower = response.lower().strip()
        
        # Direct language mapping
        language_map = {
            "python": "python",
            "javascript": "javascript",
            "js": "javascript",
            "typescript": "typescript", 
            "ts": "typescript",
            "go": "go",
            "golang": "go",
            "java": "java",
            "rust": "rust",
            "ruby": "ruby",
            "php": "php",
            "c#": "csharp",
            "csharp": "csharp",
            "swift": "swift",
            "kotlin": "kotlin"
        }
        
        # Framework mapping
        framework_map = {
            "react": ("javascript", "react"),
            "react native": ("javascript", "react-native"),
            "vue": ("javascript", "vue"),
            "vue.js": ("javascript", "vue"),
            "angular": ("typescript", "angular"),
            "express": ("javascript", "express"),
            "nextjs": ("javascript", "nextjs"),
            "next.js": ("javascript", "nextjs"),
            "django": ("python", "django"),
            "flask": ("python", "flask"),
            "fastapi": ("python", "fastapi"),
            "spring": ("java", "spring"),
            "rails": ("ruby", "rails"),
            "laravel": ("php", "laravel"),
            "flutter": ("dart", "flutter")
        }
        
        # Check for framework first (more specific)
        for framework_key, (lang, fw) in framework_map.items():
            if framework_key in response_lower:
                return lang, fw
        
        # Then check for language
        for lang_key, lang in language_map.items():
            if lang_key in response_lower:
                return lang, None
        
        # Check for special cases
        if "plain" in response_lower and "html" in response_lower:
            return "html", None
        
        # Default to None if can't parse
        return None, None
    
    def apply_to_task(self, task, language: Optional[str], framework: Optional[str]):
        """Apply language and framework preferences to a task"""
        if language:
            task.target_language = language
        if framework:
            task.target_framework = framework
        
        # Update command to include language context if not already present
        if language and language not in task.command.lower():
            # Prepend language context
            task.command = f"[Using {language}{f' with {framework}' if framework else ''}] {task.command}"