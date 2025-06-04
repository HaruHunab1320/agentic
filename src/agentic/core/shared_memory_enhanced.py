"""
Enhanced Shared Memory System for Phase 3 with embeddings and learning capabilities
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from agentic.core.shared_memory import SharedMemory
from agentic.utils.logging import LoggerMixin
import aiosqlite


class SharedMemorySystem(SharedMemory):
    """Advanced shared memory with learning capabilities and semantic search"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        super().__init__()
        self.storage_path = storage_path or Path.home() / '.agentic' / 'memory'
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Enhanced memory components
        self.memory_db = None
        self.context_embeddings = {}
        self.pattern_memory = {}
        self.session_memory = {}
        self.knowledge_base = {}
        self.learning_data = []
        
        # Embedding simulation (in real implementation, would use sentence-transformers)
        self.embedding_cache = {}
    
    @property
    def db_path(self) -> str:
        """Get database path for direct access in tests"""
        return str(self.storage_path / "memory.db")
    
    async def initialize(self):
        """Initialize enhanced shared memory system"""
        await self._setup_storage()
        await self._load_persistent_memory()
        await self._initialize_embeddings()
        self.logger.info("Enhanced shared memory system initialized")
    
    async def _setup_storage(self):
        """Setup persistent storage with SQLite"""
        db_path = self.storage_path / "memory.db"
        self.memory_db = sqlite3.connect(str(db_path))
        
        # Create tables for different types of memory (using existing schema)
        cursor = self.memory_db.cursor()
        
        # Context storage table (existing schema)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                context TEXT NOT NULL,
                agent_id TEXT,
                timestamp REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                embedding_hash TEXT,
                tags TEXT
            )
        ''')
        
        # Pattern learning table (existing schema)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                success_metrics TEXT,
                usage_count INTEGER DEFAULT 0,
                last_used REAL,
                created_at REAL NOT NULL
            )
        ''')
        
        # Knowledge base table (existing schema)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                source_agent TEXT,
                confidence REAL DEFAULT 0.0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        ''')
        
        # Execution history table (existing schema)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT NOT NULL,
                context TEXT,
                agent_id TEXT,
                success INTEGER NOT NULL,
                duration REAL,
                outcome TEXT,
                lessons_learned TEXT,
                timestamp REAL NOT NULL
            )
        ''')
        
        self.memory_db.commit()
    
    async def _load_persistent_memory(self):
        """Load persistent memory from storage"""
        if not self.memory_db:
            return
        
        cursor = self.memory_db.cursor()
        
        try:
            # Load recent patterns
            cursor.execute('''
                SELECT pattern_type, pattern_data, success_metrics 
                FROM patterns 
                WHERE last_used > ? 
                ORDER BY usage_count DESC 
                LIMIT 100
            ''', (datetime.utcnow().timestamp() - 86400 * 30,))  # Last 30 days
            
            for row in cursor.fetchall():
                pattern_type, pattern_data, success_metrics = row
                self.pattern_memory[pattern_type] = {
                    'data': json.loads(pattern_data),
                    'success_metrics': json.loads(success_metrics) if success_metrics else {}
                }
        except sqlite3.OperationalError as e:
            self.logger.debug(f"No patterns table or schema mismatch: {e}")
        
        try:
            # Load knowledge base (using existing schema)
            cursor.execute('''
                SELECT topic, content, confidence 
                FROM knowledge 
                ORDER BY updated_at DESC 
                LIMIT 500
            ''')
            
            for row in cursor.fetchall():
                topic, content, confidence = row
                self.knowledge_base[topic] = {
                    'content': content,
                    'confidence': confidence
                }
        except sqlite3.OperationalError as e:
            self.logger.debug(f"No knowledge table or schema mismatch: {e}")
    
    async def _initialize_embeddings(self):
        """Initialize embedding system (simplified implementation)"""
        # In a real implementation, this would load a sentence transformer model
        # For now, we'll use a simple hash-based approach for demonstration
        self.logger.info("Embedding system initialized (simplified)")
    
    async def _generate_embedding_hash(self, text: str) -> str:
        """Generate a simple hash for text embedding (simplified implementation)"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    async def store_context(self, key: str, context: Dict, agent_id: str = None, tags: List[str] = None):
        """Store context with metadata and embeddings (using existing schema)"""
        context_json = json.dumps(context)
        embedding_hash = await self._generate_embedding_hash(context_json)
        tags_json = json.dumps(tags or [])
        
        # Store in database
        cursor = self.memory_db.cursor()
        cursor.execute('''
            INSERT INTO contexts (key, context, agent_id, timestamp, embedding_hash, tags)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (key, context_json, agent_id, datetime.utcnow().timestamp(), embedding_hash, tags_json))
        
        self.memory_db.commit()
        
        # Store in local cache for quick access
        self.context_embeddings[key] = {
            'context': context,
            'agent_id': agent_id,
            'timestamp': datetime.utcnow(),
            'embedding_hash': embedding_hash,
            'tags': tags or [],
            'access_count': 0
        }
        
        self.logger.debug(f"Stored context: {key}")
        return key  # Return key as ID for compatibility
    
    async def get_context(self, context_key: str) -> Optional[Dict]:
        """Get context by key (for compatibility with new interface)"""
        cursor = self.memory_db.cursor()
        cursor.execute('''
            SELECT key, context, agent_id, timestamp, tags
            FROM contexts WHERE key = ?
        ''', (context_key,))
        
        result = cursor.fetchone()
        if result:
            key, context_data, agent_id, timestamp, tags = result
            return {
                'id': key,
                'context_key': key,
                'context_data': json.loads(context_data),
                'metadata': {'agent_id': agent_id} if agent_id else {},
                'tags': json.loads(tags) if tags else [],
                'created_at': datetime.fromtimestamp(timestamp).isoformat(),
                'last_accessed': datetime.fromtimestamp(timestamp).isoformat()
            }
        return None
    
    async def get_relevant_context(self, query: str, limit: int = 10, 
                                 similarity_threshold: float = 0.5) -> Dict:
        """Get context relevant to query using semantic search"""
        query_hash = await self._generate_embedding_hash(query)
        
        # Search in database for similar contexts
        cursor = self.memory_db.cursor()
        cursor.execute('''
            SELECT key, context, agent_id, timestamp, embedding_hash, access_count
            FROM contexts
            WHERE timestamp > ?
            ORDER BY access_count DESC
            LIMIT ?
        ''', (datetime.utcnow().timestamp() - 86400 * 7, limit * 2))  # Last 7 days
        
        relevant_contexts = {}
        results = cursor.fetchall()
        
        for row in results:
            key, context_json, agent_id, timestamp, embedding_hash, access_count = row
            
            # Calculate similarity (simplified)
            similarity = self._calculate_similarity(query_hash, embedding_hash)
            
            if similarity >= similarity_threshold:
                context = json.loads(context_json)
                relevant_contexts[key] = {
                    'context': context,
                    'relevance': similarity,
                    'source_agent': agent_id,
                    'timestamp': datetime.fromtimestamp(timestamp),
                    'access_count': access_count
                }
                
                # Update access count
                cursor.execute('''
                    UPDATE contexts SET access_count = access_count + 1 WHERE key = ?
                ''', (key,))
                
                if len(relevant_contexts) >= limit:
                    break
        
        self.memory_db.commit()
        return dict(sorted(relevant_contexts.items(), 
                          key=lambda x: x[1]['relevance'], reverse=True))
    
    async def learn_pattern(self, pattern: Dict):
        """Learn a successful pattern for future use (using existing schema)"""
        pattern_type = pattern.get('type', 'unknown')
        
        # Store in database
        cursor = self.memory_db.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO patterns 
            (pattern_type, pattern_data, success_metrics, usage_count, last_used, created_at)
            VALUES (?, ?, ?, 1, ?, ?)
        ''', (
            pattern_type,
            json.dumps(pattern),
            json.dumps(pattern.get('success_metrics', {})),
            datetime.utcnow().timestamp(),
            datetime.utcnow().timestamp()
        ))
        
        self.memory_db.commit()
        
        # Update local cache
        self.pattern_memory[pattern_type] = {
            'data': pattern,
            'success_metrics': pattern.get('success_metrics', {}),
            'usage_count': 1,
            'last_used': datetime.utcnow()
        }
        
        self.logger.info(f"Learned pattern: {pattern_type}")
        return pattern_type  # Return pattern type as ID for compatibility
    
    async def get_patterns(self, pattern_type: str) -> List[Dict]:
        """Retrieve patterns by type from the database"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    "SELECT topic, content FROM knowledge WHERE topic LIKE ?",
                    (f"{pattern_type}_%",)
                )
                rows = await cursor.fetchall()
                
                patterns = []
                for row in rows:
                    try:
                        pattern_data = json.loads(row[1])
                        patterns.append({
                            'topic': row[0],
                            'data': pattern_data
                        })
                    except json.JSONDecodeError:
                        self.logger.warning(f"Invalid JSON in pattern {row[0]}")
                
                return patterns
        except Exception as e:
            self.logger.error(f"Error retrieving patterns: {e}")
            return []
    
    async def suggest_based_on_patterns(self, features: Dict) -> List[Dict]:
        """Suggest patterns based on provided features"""
        try:
            # Get all intent classification patterns
            patterns = await self.get_patterns("intent_classification")
            
            # Score patterns based on feature similarity
            suggestions = []
            for pattern in patterns:
                # Simple pattern matching - in production this would be more sophisticated
                relevance = self._calculate_pattern_relevance(features, pattern['data'])
                if relevance > 0.3:  # Threshold for relevance
                    suggestions.append({
                        'pattern': pattern['data'],
                        'relevance': relevance
                    })
            
            # Sort by relevance and return top matches
            suggestions.sort(key=lambda x: x['relevance'], reverse=True)
            return suggestions[:5]  # Top 5 suggestions
            
        except Exception as e:
            self.logger.error(f"Error suggesting patterns: {e}")
            return []
    
    def _calculate_pattern_relevance(self, features: Dict, pattern: Dict) -> float:
        """Calculate relevance score between features and pattern"""
        score = 0.0
        matches = 0
        total_features = 0
        
        # Compare common features
        for key in ['word_count', 'has_technical_terms', 'sentiment', 'project_type']:
            if key in features and key in pattern:
                total_features += 1
                if features[key] == pattern[key]:
                    matches += 1
                    score += 0.2
        
        # Bonus for action verb matches
        if features.get('action_verbs', 0) > 0 and pattern.get('action_verbs', 0) > 0:
            score += 0.1
        
        # Calculate base similarity
        if total_features > 0:
            similarity = matches / total_features
            score = max(score, similarity * 0.8)
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def share_knowledge(self, topic: str, content: str, source_agent: str, confidence: float = 0.8):
        """Share knowledge across agents (using existing schema)"""
        cursor = self.memory_db.cursor()
        
        # Check if knowledge already exists
        cursor.execute('SELECT id, confidence FROM knowledge WHERE topic = ?', (topic,))
        existing = cursor.fetchone()
        
        timestamp = datetime.utcnow().timestamp()
        
        if existing:
            # Update if new confidence is higher
            existing_id, existing_confidence = existing
            if confidence > existing_confidence:
                cursor.execute('''
                    UPDATE knowledge 
                    SET content = ?, source_agent = ?, confidence = ?, updated_at = ?
                    WHERE id = ?
                ''', (content, source_agent, confidence, timestamp, existing_id))
        else:
            # Insert new knowledge
            cursor.execute('''
                INSERT INTO knowledge (topic, content, source_agent, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (topic, content, source_agent, confidence, timestamp, timestamp))
        
        self.memory_db.commit()
        
        # Update local cache
        self.knowledge_base[topic] = {
            'content': content,
            'source_agent': source_agent,
            'confidence': confidence,
            'updated_at': datetime.utcnow()
        }
        
        self.logger.info(f"Shared knowledge: {topic} (confidence: {confidence})")
        return topic  # Return topic as ID for compatibility
    
    async def get_shared_knowledge(self, knowledge_type: str) -> List[Dict]:
        """Get shared knowledge by type"""
        cursor = self.memory_db.cursor()
        cursor.execute('''
            SELECT id, content, source_agent, confidence, created_at
            FROM knowledge WHERE topic LIKE ?
            ORDER BY confidence DESC, created_at DESC
        ''', (f"{knowledge_type}%",))
        
        knowledge_items = []
        for row in cursor.fetchall():
            knowledge_id, content, source_agent, confidence, created_at = row
            knowledge_items.append({
                'id': knowledge_id,
                'knowledge_data': content,
                'sharing_agent': source_agent,
                'confidence': confidence,
                'tags': [],  # Tags not stored in current schema
                'created_at': created_at
            })
        
        return knowledge_items
    
    async def search_knowledge(self, query: str) -> List[Dict]:
        """Search for knowledge using semantic search"""
        # Simple keyword-based search for now
        query_keywords = self._extract_keywords(query)
        
        cursor = self.memory_db.cursor()
        cursor.execute('''
            SELECT id, knowledge_type, knowledge_data, sharing_agent, confidence
            FROM knowledge
            ORDER BY confidence DESC
        ''')
        
        results = []
        for row in cursor.fetchall():
            knowledge_id, knowledge_type, knowledge_data, sharing_agent, confidence = row
            knowledge_content = json.loads(knowledge_data)
            
            # Simple relevance scoring based on keyword overlap
            content_text = str(knowledge_content).lower()
            relevance = sum(1 for keyword in query_keywords if keyword in content_text) / len(query_keywords) if query_keywords else 0
            
            if relevance > 0.1:  # Minimum relevance threshold
                results.append({
                    'id': knowledge_id,
                    'knowledge_type': knowledge_type,
                    'knowledge_data': knowledge_content,
                    'sharing_agent': sharing_agent,
                    'confidence': confidence,
                    'relevance': relevance
                })
        
        return sorted(results, key=lambda x: x['relevance'], reverse=True)
    
    async def store_execution(self, execution_key: str, execution_data: Dict, metrics: Dict = None):
        """Store execution result for learning"""
        cursor = self.memory_db.cursor()
        cursor.execute('''
            INSERT INTO executions (command, context, agent_id, success, duration, outcome, lessons_learned, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            execution_data.get('command', execution_key),
            execution_data.get('context', ''),
            execution_data.get('agent_id', ''),
            int(execution_data.get('success', True)),
            execution_data.get('duration', 0.0),
            json.dumps(execution_data),
            json.dumps(metrics or {}),
            datetime.utcnow().timestamp()
        ))
        
        self.memory_db.commit()
        return execution_key  # Return execution key as ID for compatibility
    
    async def get_statistics(self) -> Dict:
        """Get memory system statistics"""
        cursor = self.memory_db.cursor()
        
        # Context statistics
        cursor.execute('SELECT COUNT(*) FROM contexts')
        contexts_total = cursor.fetchone()[0]
        
        # Pattern statistics
        cursor.execute('SELECT COUNT(*) FROM patterns')
        patterns_total = cursor.fetchone()[0]
        
        # Knowledge statistics
        cursor.execute('SELECT COUNT(*) FROM knowledge')
        knowledge_total = cursor.fetchone()[0]
        
        # Execution statistics
        cursor.execute('SELECT COUNT(*) FROM executions')
        executions_total = cursor.fetchone()[0]
        
        return {
            'contexts': {
                'total': contexts_total
            },
            'patterns': {
                'total': patterns_total
            },
            'knowledge': {
                'total': knowledge_total
            },
            'executions': {
                'total': executions_total
            }
        }
    
    async def get_learning_insights(self) -> Dict:
        """Get insights from learning data"""
        cursor = self.memory_db.cursor()
        
        # Pattern insights
        cursor.execute('''
            SELECT pattern_type, COUNT(*), SUM(usage_count)
            FROM patterns
            GROUP BY pattern_type
            ORDER BY COUNT(*) DESC
        ''')
        
        pattern_insights = {
            'total_patterns': 0,
            'success_rate': 0.8,  # Simplified for now
            'command_patterns': {}
        }
        
        for row in cursor.fetchall():
            pattern_type, count, total_usage = row
            pattern_insights['total_patterns'] += count
            pattern_insights['command_patterns'][pattern_type] = {
                'count': count,
                'total_usage': total_usage or 0
            }
        
        # Execution insights - use correct column names
        cursor.execute('''
            SELECT agent_id, COUNT(*), AVG(success)
            FROM executions
            GROUP BY agent_id
            ORDER BY COUNT(*) DESC
            LIMIT 10
        ''')
        
        execution_insights = {
            'total_executions': 0,
            'agent_performance': {}
        }
        
        for row in cursor.fetchall():
            agent_id, count, avg_success = row
            execution_insights['total_executions'] += count
            execution_insights['agent_performance'][agent_id or 'unknown'] = {
                'executions': count,
                'success_rate': float(avg_success or 0.0)
            }
        
        # Knowledge insights
        cursor.execute('SELECT COUNT(*) FROM knowledge')
        knowledge_count = cursor.fetchone()[0]
        
        knowledge_insights = {
            'total_knowledge': knowledge_count,
            'coverage': knowledge_count / max(pattern_insights['total_patterns'], 1)
        }
        
        return {
            'pattern_insights': pattern_insights,
            'execution_insights': execution_insights,
            'knowledge_insights': knowledge_insights
        }
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Cleanup old data"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        cutoff_timestamp = cutoff_date.timestamp()
        
        cursor = self.memory_db.cursor()
        
        # Clean old contexts (use existing schema column names)
        cursor.execute('DELETE FROM contexts WHERE timestamp < ?', (cutoff_timestamp,))
        
        # Clean old patterns (keep more recent ones)
        pattern_cutoff = datetime.utcnow() - timedelta(days=days_to_keep * 2)
        cursor.execute('DELETE FROM patterns WHERE created_at < ?', (pattern_cutoff.timestamp(),))
        
        self.memory_db.commit()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text"""
        import re
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Simple stopword removal
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        return set(word for word in words if len(word) > 2 and word not in stopwords)
    
    def _simple_hash_embedding(self, text: str) -> tuple:
        """Simple hash-based embedding simulation"""
        import hashlib
        
        # Create multiple hash values to simulate embedding dimensions
        embedding = []
        for i in range(10):  # 10-dimensional embedding
            hash_input = f"{text}_{i}".encode('utf-8')
            hash_value = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
            # Normalize to [-1, 1] range
            normalized = (hash_value % 2000 - 1000) / 1000.0
            embedding.append(normalized)
        
        return tuple(embedding) 