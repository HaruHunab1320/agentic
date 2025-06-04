"""
ML-Enhanced Intent Classifier for Phase 3 with learning capabilities
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Set, Tuple

from agentic.core.intent_classifier import IntentClassifier
from agentic.core.shared_memory_enhanced import SharedMemorySystem
from agentic.models.task import TaskIntent, TaskType
from agentic.utils.logging import LoggerMixin


class MLIntentClassifier(IntentClassifier, LoggerMixin):
    """Enhanced intent classifier with machine learning and pattern recognition"""
    
    def __init__(self, shared_memory: Optional[SharedMemorySystem] = None):
        super().__init__()
        self.shared_memory = shared_memory
        self.feature_weights = self._initialize_feature_weights()
        self.learned_patterns = {}
        self.classification_history = []
        self.confidence_threshold = 0.7
    
    async def analyze_intent(self, command: str, context: Optional[Dict] = None) -> TaskIntent:
        """Enhanced intent analysis with ML features and context"""
        self.logger.debug(f"Analyzing intent for: {command[:100]}...")
        
        # Extract features
        features = await self._extract_features(command, context or {})
        
        # Get predictions from different classifiers
        keyword_prediction = self._classify_with_keywords(features)
        pattern_prediction = await self._classify_with_patterns(features)
        context_prediction = await self._classify_with_context(features, context or {})
        
        # Combine predictions using weighted voting
        final_prediction = self._combine_predictions([
            (keyword_prediction, 0.4),
            (pattern_prediction, 0.35),
            (context_prediction, 0.25)
        ])
        
        # Get relevant context from shared memory
        if self.shared_memory:
            relevant_context = await self.shared_memory.get_relevant_context(command, limit=5)
            if relevant_context:
                final_prediction = self._adjust_prediction_with_memory(
                    final_prediction, relevant_context
                )
        
        # Create task intent with predictions
        task_intent = TaskIntent(
            task_type=final_prediction['task_type'],
            complexity_score=final_prediction['complexity_score'],
            estimated_duration=int(final_prediction['estimated_duration']),  # Convert to int to match model
            affected_areas=final_prediction['affected_areas'],
            requires_reasoning=final_prediction['requires_reasoning'],
            requires_coordination=final_prediction['requires_coordination'],
            file_patterns=final_prediction.get('file_patterns', [])
        )
        
        # Store classification for learning
        await self._store_classification(command, features, task_intent, context)
        
        return task_intent
    
    async def learn_from_feedback(self, command: str, predicted_intent: TaskIntent, 
                                actual_outcome: Dict, success: bool):
        """Learn from execution outcomes to improve future predictions"""
        self.logger.info(f"Learning from feedback: success={success}")
        
        # Extract features from the command that was classified
        features = await self._extract_features(command, {})
        
        # Calculate prediction accuracy
        accuracy_score = self._calculate_accuracy(predicted_intent, actual_outcome, success)
        
        # Update feature weights based on accuracy
        if accuracy_score < 0.5:  # Poor prediction
            await self._adjust_feature_weights(features, predicted_intent, actual_outcome, False)
        elif accuracy_score > 0.8:  # Good prediction
            await self._adjust_feature_weights(features, predicted_intent, actual_outcome, True)
        
        # Store learning pattern
        if self.shared_memory:
            learning_pattern = {
                'type': 'intent_classification',
                'command_features': features,
                'predicted_intent': predicted_intent.model_dump(),
                'actual_outcome': actual_outcome,
                'accuracy_score': accuracy_score,
                'success': success,
                'success_metrics': {
                    'accuracy': accuracy_score,
                    'success_rate': 1.0 if success else 0.0
                }
            }
            await self.shared_memory.learn_pattern(learning_pattern)
        
        # Update learned patterns
        pattern_key = self._get_pattern_key(features)
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                'successes': 0,
                'failures': 0,
                'preferred_task_type': predicted_intent.task_type,
                'average_complexity': predicted_intent.complexity_score
            }
        
        pattern = self.learned_patterns[pattern_key]
        if success:
            pattern['successes'] += 1
        else:
            pattern['failures'] += 1
        
        # Update preferred classification if this pattern consistently succeeds
        total_attempts = pattern['successes'] + pattern['failures']
        if total_attempts >= 5 and pattern['successes'] / total_attempts > 0.8:
            # This pattern is consistently successful, reinforce it
            pattern['preferred_task_type'] = predicted_intent.task_type
            pattern['average_complexity'] = (
                pattern['average_complexity'] * 0.7 + 
                predicted_intent.complexity_score * 0.3
            )
    
    async def _extract_features(self, command: str, context: Dict) -> Dict:
        """Extract comprehensive features from command and context"""
        features = {
            'command': command,
            'word_count': len(command.split()),
            'char_count': len(command),
            'has_technical_terms': self._has_technical_terms(command),
            'urgency_indicators': self._count_urgency_indicators(command),
            'complexity_indicators': self._count_complexity_indicators(command),
            'action_verbs': self._extract_action_verbs(command),
            'domain_keywords': self._extract_domain_keywords(command),
            'sentiment': self._analyze_sentiment(command),
            'question_indicators': self._count_question_indicators(command),
            'file_references': self._extract_file_references(command),
            'code_indicators': self._has_code_indicators(command)
        }
        
        # Add context features
        if context:
            features.update({
                'has_context': True,
                'context_size': len(str(context)),
                'project_type': context.get('project_type', 'unknown'),
                'recent_files_count': len(context.get('recent_files', [])),
                'recent_errors': len(context.get('recent_errors', [])),
                'active_agents': len(context.get('active_agents', []))
            })
        else:
            features['has_context'] = False
        
        return features
    
    def _classify_with_keywords(self, features: Dict) -> Dict:
        """Enhanced keyword-based classification"""
        command = features['command']
        command_lower = command.lower()
        command_words = set(re.findall(r'\b\w+\b', command_lower))
        
        # Use parent class logic but with enhanced scoring
        base_task_type = self._classify_task_type(command_words)
        
        # Calculate enhanced scores
        complexity_score = self._calculate_enhanced_complexity(features)
        
        # Determine affected areas with context
        affected_areas = self._identify_affected_areas_enhanced(features)
        
        # Enhanced reasoning requirement detection
        requires_reasoning = self._enhanced_reasoning_detection(features, base_task_type)
        
        # Enhanced coordination requirement detection
        requires_coordination = self._enhanced_coordination_detection(features, affected_areas)
        
        # Duration estimation with feature consideration
        estimated_duration = self._enhanced_duration_estimation(
            base_task_type, complexity_score, features
        )
        
        return {
            'task_type': base_task_type,
            'complexity_score': complexity_score,
            'estimated_duration': estimated_duration,
            'affected_areas': affected_areas,
            'requires_reasoning': requires_reasoning,
            'requires_coordination': requires_coordination,
            'confidence': 0.7  # Keyword-based baseline confidence
        }
    
    async def _classify_with_patterns(self, features: Dict) -> Dict:
        """Pattern-based classification using learned patterns"""
        pattern_key = self._get_pattern_key(features)
        
        # Check if we have learned patterns for similar commands
        if pattern_key in self.learned_patterns:
            pattern = self.learned_patterns[pattern_key]
            total_attempts = pattern['successes'] + pattern['failures']
            
            if total_attempts >= 3:  # Enough data to trust the pattern
                success_rate = pattern['successes'] / total_attempts
                
                return {
                    'task_type': pattern['preferred_task_type'],
                    'complexity_score': pattern['average_complexity'],
                    'estimated_duration': self._estimate_duration_from_pattern(pattern),
                    'affected_areas': self._get_areas_from_features(features),
                    'requires_reasoning': pattern['preferred_task_type'] in [TaskType.DEBUG, TaskType.EXPLAIN],
                    'requires_coordination': pattern['average_complexity'] > 0.7,
                    'confidence': success_rate * 0.9  # Confidence based on success rate
                }
        
        # No learned pattern, use shared memory patterns
        if self.shared_memory:
            suggestions = await self.shared_memory.suggest_based_on_patterns(features)
            if suggestions:
                best_suggestion = suggestions[0]
                pattern_data = best_suggestion['pattern']
                
                if pattern_data.get('type') == 'intent_classification':
                    predicted = pattern_data.get('predicted_intent', {})
                    return {
                        'task_type': TaskType(predicted.get('task_type', TaskType.IMPLEMENT)),
                        'complexity_score': predicted.get('complexity_score', 0.5),
                        'estimated_duration': predicted.get('estimated_duration', 30),
                        'affected_areas': predicted.get('affected_areas', ['general']),
                        'requires_reasoning': predicted.get('requires_reasoning', False),
                        'requires_coordination': predicted.get('requires_coordination', False),
                        'confidence': best_suggestion.get('relevance', 0.5)
                    }
        
        # Fallback to keyword-based classification
        return {
            'task_type': TaskType.IMPLEMENT,
            'complexity_score': 0.5,
            'estimated_duration': 30,
            'affected_areas': ['general'],
            'requires_reasoning': False,
            'requires_coordination': False,
            'confidence': 0.3  # Low confidence for fallback
        }
    
    async def _classify_with_context(self, features: Dict, context: Dict) -> Dict:
        """Context-aware classification"""
        base_prediction = {
            'task_type': TaskType.IMPLEMENT,
            'complexity_score': 0.5,
            'estimated_duration': 30,
            'affected_areas': ['general'],
            'requires_reasoning': False,
            'requires_coordination': False,
            'confidence': 0.5
        }
        
        if not features.get('has_context'):
            return base_prediction
        
        # Adjust based on context
        recent_errors = context.get('recent_errors', [])
        if recent_errors:
            base_prediction['task_type'] = TaskType.DEBUG
            base_prediction['requires_reasoning'] = True
            base_prediction['complexity_score'] = min(0.8, 0.5 + len(recent_errors) * 0.1)
        
        # Adjust for active agents
        active_agents = context.get('active_agents', [])
        if len(active_agents) > 1:
            base_prediction['requires_coordination'] = True
        
        # Adjust for project type
        project_type = context.get('project_type', 'unknown')
        if project_type in ['web_application', 'microservices']:
            base_prediction['complexity_score'] *= 1.2
            if len(base_prediction['affected_areas']) == 1 and base_prediction['affected_areas'][0] == 'general':
                base_prediction['affected_areas'] = ['frontend', 'backend']
        
        base_prediction['confidence'] = 0.6
        return base_prediction
    
    def _combine_predictions(self, predictions: List[Tuple[Dict, float]]) -> Dict:
        """Combine multiple predictions using weighted voting"""
        combined = {
            'task_type': TaskType.IMPLEMENT,
            'complexity_score': 0.0,
            'estimated_duration': 0,
            'affected_areas': [],
            'requires_reasoning': False,
            'requires_coordination': False,
            'confidence': 0.0
        }
        
        total_weight = sum(weight for _, weight in predictions)
        
        # Task type voting
        task_type_votes = {}
        for prediction, weight in predictions:
            task_type = prediction['task_type']
            if task_type not in task_type_votes:
                task_type_votes[task_type] = 0
            task_type_votes[task_type] += weight
        
        combined['task_type'] = max(task_type_votes.keys(), key=lambda x: task_type_votes[x])
        
        # Weighted averages for numeric values
        for key in ['complexity_score', 'estimated_duration']:
            weighted_sum = sum(pred[key] * weight for pred, weight in predictions)
            combined[key] = weighted_sum / total_weight
        
        # Boolean OR for boolean values
        combined['requires_reasoning'] = any(pred['requires_reasoning'] for pred, _ in predictions)
        combined['requires_coordination'] = any(pred['requires_coordination'] for pred, _ in predictions)
        
        # Union of affected areas
        all_areas = set()
        for prediction, _ in predictions:
            all_areas.update(prediction['affected_areas'])
        combined['affected_areas'] = list(all_areas)
        
        # Weighted confidence
        confidence_sum = sum(pred.get('confidence', 0.5) * weight for pred, weight in predictions)
        combined['confidence'] = confidence_sum / total_weight
        
        return combined
    
    def _adjust_prediction_with_memory(self, prediction: Dict, relevant_context: Dict) -> Dict:
        """Adjust prediction based on relevant historical context"""
        if not relevant_context:
            return prediction
        
        # Get the most relevant context entry
        most_relevant = next(iter(relevant_context.values()))
        context_data = most_relevant['context']
        
        # Adjust complexity if similar past contexts were complex
        if context_data.get('complexity_score', 0) > 0.7:
            prediction['complexity_score'] = max(
                prediction['complexity_score'], 
                context_data['complexity_score'] * 0.8
            )
        
        # Adjust coordination requirement
        if context_data.get('requires_coordination', False):
            prediction['requires_coordination'] = True
        
        # Boost confidence if we have relevant historical data
        prediction['confidence'] = min(1.0, prediction['confidence'] + 0.1)
        
        return prediction
    
    def _calculate_enhanced_complexity(self, features: Dict) -> float:
        """Enhanced complexity calculation using multiple features"""
        base_complexity = super()._calculate_complexity(
            features['command'], 
            set(features['command'].lower().split())
        )
        
        # Adjust based on additional features
        complexity_adjustments = 0.0
        
        # Technical terms increase complexity
        if features.get('has_technical_terms', False):
            complexity_adjustments += 0.2
        
        # Code indicators increase complexity
        if features.get('code_indicators', False):
            complexity_adjustments += 0.15
        
        # File references suggest implementation work
        file_refs = features.get('file_references', 0)
        if file_refs > 0:
            complexity_adjustments += min(0.3, file_refs * 0.1)
        
        # Context complexity
        if features.get('has_context') and features.get('recent_errors', 0) > 0:
            complexity_adjustments += 0.25
        
        # Multiple action verbs suggest complex multi-step tasks
        action_verbs = features.get('action_verbs', 0)
        if action_verbs > 2:
            complexity_adjustments += 0.2
        
        return min(1.0, base_complexity + complexity_adjustments)
    
    def _identify_affected_areas_enhanced(self, features: Dict) -> List[str]:
        """Enhanced affected areas identification"""
        command = features['command']
        command_words = set(command.lower().split())
        
        # Use parent class logic
        base_areas = self._identify_affected_areas(command_words)
        
        # Enhance based on additional features
        if features.get('code_indicators', False):
            if 'general' in base_areas:
                base_areas.remove('general')
                base_areas.extend(['backend', 'frontend'])
        
        # Project type context
        project_type = features.get('project_type', 'unknown')
        if project_type == 'web_application' and 'general' in base_areas:
            base_areas.remove('general')
            base_areas.extend(['frontend', 'backend'])
        
        return list(set(base_areas)) if base_areas else ['general']
    
    def _enhanced_reasoning_detection(self, features: Dict, task_type: TaskType) -> bool:
        """Enhanced detection of reasoning requirements"""
        command = features.get('command', '')
        command_lower = command.lower()
        
        # Check for reasoning keywords
        reasoning_keywords = {'why', 'how', 'explain', 'understand', 'analyze', 'debug', 'trace', 'reason'}
        command_words = set(command_lower.split())
        
        if reasoning_keywords.intersection(command_words):
            return True
        
        # Task type based reasoning
        if task_type in [TaskType.DEBUG, TaskType.EXPLAIN]:
            return True
        
        # Context-based reasoning indicators
        if features.get('has_context') and features.get('recent_errors', 0) > 0:
            return True
        
        # Complexity-based reasoning
        if features.get('complexity_indicators', 0) > 2:
            return True
        
        return False
    
    def _enhanced_coordination_detection(self, features: Dict, affected_areas: List[str]) -> bool:
        """Enhanced coordination requirement detection"""
        # Multiple areas always require coordination
        if len(affected_areas) > 1:
            return True
        
        # High complexity tasks
        if features.get('complexity_score', 0) > 0.7:
            return True
        
        # Multiple active agents
        if features.get('active_agents', 0) > 1:
            return True
        
        # Multiple file references
        if features.get('file_references', 0) > 3:
            return True
        
        return False
    
    def _enhanced_duration_estimation(self, task_type: TaskType, complexity_score: float, 
                                    features: Dict) -> int:
        """Enhanced duration estimation with feature consideration"""
        # Base duration from parent class
        word_count = features.get('word_count', 10)
        base_duration = self._estimate_duration(task_type, complexity_score, word_count)
        
        # Feature-based adjustments
        adjustments = 0
        
        # Complexity adjustments
        if features.get('complexity_indicators', 0) > 3:
            adjustments += 10
        
        # Technical term adjustments
        if features.get('has_technical_terms', False):
            adjustments += 5
        
        # Context size adjustments
        context_size = features.get('context_size', 0)
        if context_size > 1000:
            adjustments += 15
        elif context_size > 500:
            adjustments += 5
        
        # File reference adjustments
        file_refs = features.get('file_references', 0)
        if file_refs > 5:
            adjustments += 20
        elif file_refs > 2:
            adjustments += 10
        
        return max(5, int(base_duration + adjustments))
    
    # Helper methods for feature extraction
    
    def _has_technical_terms(self, command: str) -> bool:
        """Check if command contains technical terms"""
        technical_terms = {
            'algorithm', 'database', 'api', 'endpoint', 'middleware', 'framework',
            'library', 'dependency', 'architecture', 'deployment', 'scaling',
            'optimization', 'refactoring', 'debugging', 'testing', 'integration'
        }
        command_words = set(command.lower().split())
        return bool(command_words & technical_terms)
    
    def _count_urgency_indicators(self, command: str) -> int:
        """Count urgency indicators in command"""
        urgency_words = ['urgent', 'asap', 'immediately', 'quickly', 'fast', 'emergency', 'critical']
        command_lower = command.lower()
        return sum(1 for word in urgency_words if word in command_lower)
    
    def _count_complexity_indicators(self, command: str) -> int:
        """Count complexity indicators"""
        complexity_words = ['complex', 'difficult', 'challenging', 'multiple', 'various', 'comprehensive']
        command_lower = command.lower()
        return sum(1 for word in complexity_words if word in command_lower)
    
    def _extract_action_verbs(self, command: str) -> int:
        """Count action verbs in command"""
        action_verbs = {
            'create', 'build', 'implement', 'develop', 'design', 'write', 'generate',
            'add', 'update', 'modify', 'change', 'fix', 'debug', 'test', 'deploy',
            'configure', 'setup', 'install', 'remove', 'delete', 'refactor'
        }
        command_words = set(command.lower().split())
        return len(command_words & action_verbs)
    
    def _extract_domain_keywords(self, command: str) -> int:
        """Count domain-specific keywords"""
        domain_keywords = {
            'frontend', 'backend', 'database', 'api', 'ui', 'ux', 'server', 'client',
            'web', 'mobile', 'desktop', 'cloud', 'aws', 'docker', 'kubernetes'
        }
        command_words = set(command.lower().split())
        return len(command_words & domain_keywords)
    
    def _analyze_sentiment(self, command: str) -> str:
        """Simple sentiment analysis"""
        positive_words = {'good', 'great', 'excellent', 'perfect', 'awesome', 'nice'}
        negative_words = {'bad', 'terrible', 'awful', 'broken', 'wrong', 'failed', 'error'}
        
        command_words = set(command.lower().split())
        
        positive_count = len(command_words & positive_words)
        negative_count = len(command_words & negative_words)
        
        if negative_count > positive_count:
            return 'negative'
        elif positive_count > negative_count:
            return 'positive'
        else:
            return 'neutral'
    
    def _count_question_indicators(self, command: str) -> int:
        """Count question indicators"""
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
        command_lower = command.lower()
        question_count = sum(1 for word in question_words if word in command_lower)
        
        if command.strip().endswith('?'):
            question_count += 1
        
        return question_count
    
    def _extract_file_references(self, command: str) -> int:
        """Count file references in command"""
        # Look for file extensions and path patterns
        file_patterns = [
            r'\w+\.\w+',  # file.ext
            r'/\w+/',     # /path/
            r'\w+/\w+',   # path/file
        ]
        
        file_count = 0
        for pattern in file_patterns:
            file_count += len(re.findall(pattern, command))
        
        return file_count
    
    def _has_code_indicators(self, command: str) -> bool:
        """Check if command indicates code-related work"""
        code_indicators = {
            'function', 'method', 'class', 'variable', 'import', 'export',
            'component', 'module', 'package', 'library', 'framework',
            'api', 'endpoint', 'authentication', 'auth', 'bug', 'issue',
            'error', 'exception', 'debug', 'test', 'testing', 'unittest',
            'service', 'controller', 'model', 'view', 'route', 'handler',
            'middleware', 'database', 'query', 'schema', 'migration',
            'config', 'configuration', 'deploy', 'deployment'
        }
        command_words = set(command.lower().split())
        return bool(command_words & code_indicators)
    
    # Learning and pattern methods
    
    def _get_pattern_key(self, features: Dict) -> str:
        """Generate a pattern key for learning"""
        # Create a simplified feature signature
        key_features = [
            str(features.get('word_count', 0) // 5),  # Bucket word count
            str(features.get('has_technical_terms', False)),
            str(features.get('action_verbs', 0)),
            str(features.get('sentiment', 'neutral')),
            str(features.get('project_type', 'unknown'))
        ]
        return '|'.join(key_features)
    
    async def _store_classification(self, command: str, features: Dict, 
                                  intent: TaskIntent, context: Optional[Dict]):
        """Store classification for learning purposes"""
        classification_record = {
            'command': command,
            'features': features,
            'intent': intent.model_dump(),
            'context': context,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.classification_history.append(classification_record)
        
        # Keep only recent classifications in memory
        if len(self.classification_history) > 1000:
            self.classification_history = self.classification_history[-500:]
        
        # Store in shared memory if available
        if self.shared_memory:
            await self.shared_memory.store_context(
                f"classification_{len(self.classification_history)}",
                classification_record,
                agent_id="ml_intent_classifier",
                tags=['classification', 'intent', 'ml']
            )
    
    def _calculate_accuracy(self, predicted: TaskIntent, actual: Dict, success: bool) -> float:
        """Calculate prediction accuracy based on actual outcome"""
        accuracy = 0.0
        
        # Task type accuracy
        if success:
            accuracy += 0.4  # If task succeeded, task type was likely correct
        
        # Complexity accuracy (check if estimated vs actual duration is reasonable)
        estimated_duration = predicted.estimated_duration
        actual_duration = actual.get('duration', estimated_duration)
        
        if actual_duration > 0:
            duration_ratio = min(estimated_duration, actual_duration) / max(estimated_duration, actual_duration)
            accuracy += 0.3 * duration_ratio
        
        # Coordination accuracy
        actual_coordination = actual.get('required_coordination', False)
        if predicted.requires_coordination == actual_coordination:
            accuracy += 0.3
        
        return min(1.0, accuracy)
    
    async def _adjust_feature_weights(self, features: Dict, predicted: TaskIntent, 
                                    actual: Dict, positive_feedback: bool):
        """Adjust feature weights based on feedback"""
        adjustment_factor = 0.1 if positive_feedback else -0.05
        
        # Adjust weights for features that were present in this classification
        for feature_name, feature_value in features.items():
            if feature_name not in self.feature_weights:
                self.feature_weights[feature_name] = 1.0
            
            # Simple weight adjustment
            if feature_value:  # Feature was present
                self.feature_weights[feature_name] += adjustment_factor
                
        # Normalize weights to prevent them from getting too extreme
        for feature_name in self.feature_weights:
            self.feature_weights[feature_name] = max(0.1, min(2.0, self.feature_weights[feature_name]))
    
    def _initialize_feature_weights(self) -> Dict[str, float]:
        """Initialize feature weights"""
        return {
            'word_count': 1.0,
            'has_technical_terms': 1.2,
            'urgency_indicators': 0.8,
            'complexity_indicators': 1.3,
            'action_verbs': 1.1,
            'domain_keywords': 1.0,
            'question_indicators': 1.2,
            'file_references': 0.9,
            'code_indicators': 1.1,
            'has_context': 1.0
        }
    
    def _estimate_duration_from_pattern(self, pattern: Dict) -> int:
        """Estimate duration based on learned pattern"""
        complexity = pattern.get('average_complexity', 0.5)
        success_rate = pattern['successes'] / (pattern['successes'] + pattern['failures'])
        
        # Base duration adjusted for complexity and historical success
        base_duration = 30 + (complexity * 60)  # 30-90 minutes base
        
        # If this pattern historically fails, increase duration estimate
        if success_rate < 0.7:
            base_duration *= 1.5
        
        return int(base_duration)
    
    def _get_areas_from_features(self, features: Dict) -> List[str]:
        """Get affected areas from features"""
        areas = []
        
        if features.get('has_technical_terms') or features.get('code_indicators'):
            areas.extend(['backend', 'frontend'])
        
        project_type = features.get('project_type', 'unknown')
        if project_type == 'web_application':
            areas.extend(['frontend', 'backend'])
        
        return areas if areas else ['general']


# Import datetime for timestamp
from datetime import datetime 