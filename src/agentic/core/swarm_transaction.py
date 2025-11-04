"""
Swarm Transaction Manager for coordinating multi-agent operations

This module provides distributed transaction capabilities for multi-agent
swarms, ensuring all agents succeed together or rollback together.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable

from pydantic import BaseModel, Field

from agentic.core.change_tracker import ChangeTracker
from agentic.utils.logging import LoggerMixin


class TransactionPhase(str, Enum):
    """Phases of a swarm transaction"""
    PREPARING = "preparing"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMMITTING = "committing"
    ROLLING_BACK = "rolling_back"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentTransaction(BaseModel):
    """Tracks an individual agent's participation in a transaction"""
    agent_id: str
    agent_type: str
    changeset_id: Optional[str] = None
    status: str = "pending"  # pending, active, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    outputs: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    provides: List[str] = Field(default_factory=list)


class TransactionBarrier(BaseModel):
    """Synchronization barrier for coordinating agent phases"""
    name: str
    required_agents: Set[str]
    arrived_agents: Set[str] = Field(default_factory=set)
    shared_data: Dict[str, Any] = Field(default_factory=dict)
    completed: bool = False
    
    def agent_arrived(self, agent_id: str, data: Optional[Dict[str, Any]] = None):
        """Mark an agent as arrived at the barrier"""
        self.arrived_agents.add(agent_id)
        if data:
            self.shared_data[agent_id] = data
        
        if self.arrived_agents >= self.required_agents:
            self.completed = True
    
    def is_complete(self) -> bool:
        """Check if all required agents have arrived"""
        return self.completed or self.arrived_agents >= self.required_agents


class SwarmTransaction(BaseModel):
    """Represents a distributed transaction across multiple agents"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    phase: TransactionPhase = TransactionPhase.PREPARING
    agents: Dict[str, AgentTransaction] = Field(default_factory=dict)
    barriers: Dict[str, TransactionBarrier] = Field(default_factory=dict)
    shared_context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    success: bool = False
    rollback_on_failure: bool = True
    max_duration_seconds: int = 300  # 5 minutes default
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SwarmTransactionManager(LoggerMixin):
    """
    Manages distributed transactions across multiple agents in a swarm.
    
    Ensures ACID-like properties for multi-agent operations:
    - Atomicity: All agents succeed or all rollback
    - Consistency: Maintains valid state transitions
    - Isolation: Prevents interference between transactions
    - Durability: Persists transaction state
    """
    
    def __init__(self, change_tracker: ChangeTracker):
        super().__init__()
        self.change_tracker = change_tracker
        self.active_transactions: Dict[str, SwarmTransaction] = {}
        self.completed_transactions: Dict[str, SwarmTransaction] = {}
        self._transaction_locks: Dict[str, asyncio.Lock] = {}
        self._barrier_events: Dict[str, asyncio.Event] = {}
    
    async def begin_transaction(self, 
                               description: str,
                               agents: List[Dict[str, Any]],
                               rollback_on_failure: bool = True) -> SwarmTransaction:
        """Begin a new swarm transaction"""
        transaction = SwarmTransaction(
            description=description,
            rollback_on_failure=rollback_on_failure
        )
        
        # Initialize agent transactions
        for agent_info in agents:
            agent_tx = AgentTransaction(
                agent_id=agent_info['agent_id'],
                agent_type=agent_info.get('agent_type', 'unknown'),
                dependencies=agent_info.get('dependencies', []),
                provides=agent_info.get('provides', [])
            )
            transaction.agents[agent_tx.agent_id] = agent_tx
        
        # Create transaction lock
        self._transaction_locks[transaction.id] = asyncio.Lock()
        
        # Store transaction
        self.active_transactions[transaction.id] = transaction
        
        self.logger.info(f"Began swarm transaction {transaction.id} with {len(agents)} agents")
        
        return transaction
    
    async def register_agent_changeset(self, 
                                     transaction_id: str,
                                     agent_id: str,
                                     changeset_id: str):
        """Register an agent's changeset with the transaction"""
        if transaction_id not in self.active_transactions:
            raise ValueError(f"No active transaction {transaction_id}")
        
        transaction = self.active_transactions[transaction_id]
        if agent_id not in transaction.agents:
            raise ValueError(f"Agent {agent_id} not part of transaction")
        
        transaction.agents[agent_id].changeset_id = changeset_id
        transaction.agents[agent_id].status = "active"
        transaction.agents[agent_id].started_at = datetime.utcnow()
        
        self.logger.debug(f"Registered changeset {changeset_id} for agent {agent_id}")
    
    async def create_barrier(self,
                           transaction_id: str,
                           barrier_name: str,
                           required_agents: Optional[List[str]] = None) -> TransactionBarrier:
        """Create a synchronization barrier for agents"""
        if transaction_id not in self.active_transactions:
            raise ValueError(f"No active transaction {transaction_id}")
        
        transaction = self.active_transactions[transaction_id]
        
        # If no agents specified, require all agents
        if required_agents is None:
            required_agents = list(transaction.agents.keys())
        
        barrier = TransactionBarrier(
            name=barrier_name,
            required_agents=set(required_agents)
        )
        
        transaction.barriers[barrier_name] = barrier
        
        # Create event for async waiting
        event_key = f"{transaction_id}:{barrier_name}"
        self._barrier_events[event_key] = asyncio.Event()
        
        self.logger.info(f"Created barrier '{barrier_name}' requiring {len(required_agents)} agents")
        
        return barrier
    
    async def wait_at_barrier(self,
                            transaction_id: str,
                            barrier_name: str,
                            agent_id: str,
                            shared_data: Optional[Dict[str, Any]] = None,
                            timeout: Optional[float] = None) -> Dict[str, Any]:
        """Have an agent wait at a barrier until all required agents arrive"""
        if transaction_id not in self.active_transactions:
            raise ValueError(f"No active transaction {transaction_id}")
        
        transaction = self.active_transactions[transaction_id]
        if barrier_name not in transaction.barriers:
            raise ValueError(f"No barrier '{barrier_name}' in transaction")
        
        barrier = transaction.barriers[barrier_name]
        
        # Mark agent as arrived
        barrier.agent_arrived(agent_id, shared_data)
        
        self.logger.debug(f"Agent {agent_id} arrived at barrier '{barrier_name}' "
                         f"({len(barrier.arrived_agents)}/{len(barrier.required_agents)})")
        
        # Check if barrier is complete
        if barrier.is_complete():
            event_key = f"{transaction_id}:{barrier_name}"
            if event_key in self._barrier_events:
                self._barrier_events[event_key].set()
        
        # Wait for all agents
        event_key = f"{transaction_id}:{barrier_name}"
        event = self._barrier_events.get(event_key)
        if event:
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Barrier '{barrier_name}' timed out after {timeout}s")
        
        self.logger.info(f"All agents passed barrier '{barrier_name}'")
        
        # Return shared data from all agents
        return barrier.shared_data
    
    async def update_phase(self, transaction_id: str, phase: TransactionPhase):
        """Update the transaction phase"""
        if transaction_id not in self.active_transactions:
            raise ValueError(f"No active transaction {transaction_id}")
        
        async with self._transaction_locks[transaction_id]:
            transaction = self.active_transactions[transaction_id]
            old_phase = transaction.phase
            transaction.phase = phase
            
            self.logger.info(f"Transaction {transaction_id} phase: {old_phase} -> {phase}")
    
    async def share_context(self,
                          transaction_id: str,
                          key: str,
                          value: Any,
                          agent_id: Optional[str] = None):
        """Share context data between agents in a transaction"""
        if transaction_id not in self.active_transactions:
            raise ValueError(f"No active transaction {transaction_id}")
        
        transaction = self.active_transactions[transaction_id]
        
        # Store with agent attribution if provided
        if agent_id:
            context_key = f"{agent_id}:{key}"
        else:
            context_key = key
        
        transaction.shared_context[context_key] = value
        
        self.logger.debug(f"Shared context '{context_key}' in transaction")
    
    async def get_shared_context(self,
                               transaction_id: str,
                               key: str,
                               agent_id: Optional[str] = None) -> Any:
        """Get shared context data from the transaction"""
        if transaction_id not in self.active_transactions:
            raise ValueError(f"No active transaction {transaction_id}")
        
        transaction = self.active_transactions[transaction_id]
        
        # Try agent-specific key first if agent_id provided
        if agent_id:
            context_key = f"{agent_id}:{key}"
            if context_key in transaction.shared_context:
                return transaction.shared_context[context_key]
        
        # Fall back to global key
        return transaction.shared_context.get(key)
    
    async def mark_agent_complete(self,
                                transaction_id: str,
                                agent_id: str,
                                outputs: Optional[Dict[str, Any]] = None):
        """Mark an agent as completed within the transaction"""
        if transaction_id not in self.active_transactions:
            raise ValueError(f"No active transaction {transaction_id}")
        
        transaction = self.active_transactions[transaction_id]
        if agent_id not in transaction.agents:
            raise ValueError(f"Agent {agent_id} not part of transaction")
        
        agent_tx = transaction.agents[agent_id]
        agent_tx.status = "completed"
        agent_tx.completed_at = datetime.utcnow()
        if outputs:
            agent_tx.outputs = outputs
        
        self.logger.info(f"Agent {agent_id} completed in transaction {transaction_id}")
        
        # Check if all agents are complete
        await self._check_transaction_completion(transaction_id)
    
    async def mark_agent_failed(self,
                              transaction_id: str,
                              agent_id: str,
                              error: str):
        """Mark an agent as failed within the transaction"""
        if transaction_id not in self.active_transactions:
            raise ValueError(f"No active transaction {transaction_id}")
        
        transaction = self.active_transactions[transaction_id]
        if agent_id not in transaction.agents:
            raise ValueError(f"Agent {agent_id} not part of transaction")
        
        agent_tx = transaction.agents[agent_id]
        agent_tx.status = "failed"
        agent_tx.completed_at = datetime.utcnow()
        agent_tx.error = error
        
        self.logger.error(f"Agent {agent_id} failed in transaction {transaction_id}: {error}")
        
        # Trigger rollback if configured
        if transaction.rollback_on_failure:
            await self._initiate_rollback(transaction_id)
    
    async def _check_transaction_completion(self, transaction_id: str):
        """Check if all agents have completed and finalize transaction"""
        transaction = self.active_transactions[transaction_id]
        
        # Check agent statuses
        all_complete = all(
            agent.status in ["completed", "failed"]
            for agent in transaction.agents.values()
        )
        
        if not all_complete:
            return
        
        # Check if any failed
        any_failed = any(
            agent.status == "failed"
            for agent in transaction.agents.values()
        )
        
        if any_failed and transaction.rollback_on_failure:
            await self._initiate_rollback(transaction_id)
        else:
            await self._commit_transaction(transaction_id)
    
    async def _initiate_rollback(self, transaction_id: str):
        """Initiate rollback of all changes in the transaction"""
        transaction = self.active_transactions[transaction_id]
        
        if transaction.phase == TransactionPhase.ROLLING_BACK:
            return  # Already rolling back
        
        self.logger.warning(f"Initiating rollback for transaction {transaction_id}")
        
        await self.update_phase(transaction_id, TransactionPhase.ROLLING_BACK)
        
        # Rollback each agent's changes
        rollback_errors = []
        for agent_id, agent_tx in transaction.agents.items():
            if agent_tx.changeset_id and agent_tx.status != "failed":
                try:
                    self.change_tracker.rollback_changeset(agent_tx.changeset_id)
                    self.logger.info(f"Rolled back changes for agent {agent_id}")
                except Exception as e:
                    rollback_errors.append(f"Agent {agent_id}: {e}")
                    self.logger.error(f"Failed to rollback agent {agent_id}: {e}")
        
        # Update transaction status
        transaction.phase = TransactionPhase.FAILED
        transaction.completed_at = datetime.utcnow()
        transaction.success = False
        
        if rollback_errors:
            transaction.metadata['rollback_errors'] = rollback_errors
        
        # Move to completed
        self._finalize_transaction(transaction_id)
    
    async def _commit_transaction(self, transaction_id: str):
        """Commit all changes in the transaction"""
        transaction = self.active_transactions[transaction_id]
        
        self.logger.info(f"Committing transaction {transaction_id}")
        
        await self.update_phase(transaction_id, TransactionPhase.COMMITTING)
        
        # Commit each agent's changes
        commit_errors = []
        for agent_id, agent_tx in transaction.agents.items():
            if agent_tx.changeset_id and agent_tx.status == "completed":
                try:
                    self.change_tracker.commit_changeset(agent_tx.changeset_id)
                    self.logger.debug(f"Committed changes for agent {agent_id}")
                except Exception as e:
                    commit_errors.append(f"Agent {agent_id}: {e}")
                    self.logger.error(f"Failed to commit agent {agent_id}: {e}")
        
        # Update transaction status
        if commit_errors:
            # Partial failure - try to rollback
            await self._initiate_rollback(transaction_id)
        else:
            transaction.phase = TransactionPhase.COMPLETED
            transaction.completed_at = datetime.utcnow()
            transaction.success = True
            
            # Create checkpoint
            checkpoint_id = self.change_tracker.create_checkpoint(
                name=f"Transaction {transaction_id[:8]}",
                description=transaction.description,
                changeset_ids=[
                    agent.changeset_id 
                    for agent in transaction.agents.values() 
                    if agent.changeset_id
                ]
            )
            transaction.metadata['checkpoint_id'] = checkpoint_id
            
            self._finalize_transaction(transaction_id)
    
    def _finalize_transaction(self, transaction_id: str):
        """Move transaction to completed state"""
        if transaction_id in self.active_transactions:
            transaction = self.active_transactions.pop(transaction_id)
            self.completed_transactions[transaction_id] = transaction
            
            # Clean up locks and events
            if transaction_id in self._transaction_locks:
                del self._transaction_locks[transaction_id]
            
            # Clean up barrier events
            for barrier_name in transaction.barriers:
                event_key = f"{transaction_id}:{barrier_name}"
                if event_key in self._barrier_events:
                    del self._barrier_events[event_key]
            
            self.logger.info(f"Transaction {transaction_id} finalized: "
                           f"success={transaction.success}, "
                           f"duration={(transaction.completed_at - transaction.created_at).total_seconds()}s")
    
    async def execute_with_transaction(self,
                                     description: str,
                                     agents: List[Dict[str, Any]],
                                     execution_func: Callable,
                                     rollback_on_failure: bool = True) -> SwarmTransaction:
        """
        Execute a function within a transaction context.
        
        This is a convenience method that handles the transaction lifecycle.
        """
        transaction = await self.begin_transaction(
            description=description,
            agents=agents,
            rollback_on_failure=rollback_on_failure
        )
        
        try:
            # Execute the function with transaction context
            await execution_func(transaction)
            
            # Wait for completion
            timeout = transaction.max_duration_seconds
            start_time = asyncio.get_event_loop().time()
            
            while transaction.id in self.active_transactions:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    raise TimeoutError(f"Transaction timed out after {timeout}s")
                await asyncio.sleep(0.1)
            
            return self.completed_transactions[transaction.id]
            
        except Exception as e:
            self.logger.error(f"Transaction execution failed: {e}")
            if transaction.id in self.active_transactions:
                await self._initiate_rollback(transaction.id)
            raise
    
    def get_transaction_summary(self, transaction_id: str) -> Dict[str, Any]:
        """Get a summary of a transaction"""
        if transaction_id in self.active_transactions:
            transaction = self.active_transactions[transaction_id]
        elif transaction_id in self.completed_transactions:
            transaction = self.completed_transactions[transaction_id]
        else:
            raise ValueError(f"No transaction with id {transaction_id}")
        
        agent_summary = {}
        for agent_id, agent_tx in transaction.agents.items():
            agent_summary[agent_id] = {
                'type': agent_tx.agent_type,
                'status': agent_tx.status,
                'duration': (
                    (agent_tx.completed_at - agent_tx.started_at).total_seconds()
                    if agent_tx.started_at and agent_tx.completed_at else None
                ),
                'error': agent_tx.error
            }
        
        return {
            'id': transaction.id,
            'description': transaction.description,
            'phase': transaction.phase,
            'success': transaction.success,
            'created_at': transaction.created_at,
            'completed_at': transaction.completed_at,
            'duration': (
                (transaction.completed_at - transaction.created_at).total_seconds()
                if transaction.completed_at else None
            ),
            'agents': agent_summary,
            'barriers_completed': [
                name for name, barrier in transaction.barriers.items()
                if barrier.is_complete()
            ],
            'metadata': transaction.metadata
        }