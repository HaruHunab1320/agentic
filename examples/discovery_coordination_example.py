#!/usr/bin/env python3
"""
Example showing how discoveries can trigger intelligent coordination between agents
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agentic.core.agent_registry import AgentRegistry
from src.agentic.core.coordination_engine import CoordinationEngine
from src.agentic.core.shared_memory import SharedMemory
from src.agentic.models.agent import AgentConfig, AgentType, DiscoveryType
from src.agentic.models.task import Task, TaskIntent


class IntelligentCoordinator:
    """Example of intelligent coordination based on discoveries"""
    
    def __init__(self, coordination_engine, agent_registry, shared_memory):
        self.coordination_engine = coordination_engine
        self.agent_registry = agent_registry
        self.shared_memory = shared_memory
        self.pending_follow_ups = []
        
        # Register discovery handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register handlers for different discovery types"""
        self.coordination_engine.register_discovery_handler(
            DiscoveryType.API_READY, 
            self._handle_api_ready
        )
        
        self.coordination_engine.register_discovery_handler(
            DiscoveryType.TEST_NEEDED,
            self._handle_test_needed
        )
        
        self.coordination_engine.register_discovery_handler(
            DiscoveryType.BUG_FOUND,
            self._handle_bug_found
        )
        
        self.coordination_engine.register_discovery_handler(
            DiscoveryType.REFACTOR_OPPORTUNITY,
            self._handle_refactor_opportunity
        )
    
    def _handle_api_ready(self, discovery, agent_id):
        """When an API is ready, queue test creation"""
        print(f"\n🚀 API Ready: {discovery.description}")
        
        # Extract endpoint info from context
        endpoint = discovery.context.get("endpoint", "/api/unknown")
        
        # Queue a follow-up task to create tests
        follow_up = Task(
            command=f"Create comprehensive tests for the new API endpoint at {endpoint}",
            intent=TaskIntent(
                primary_goal="Create API tests",
                affected_areas=["testing"],
                requires_coordination=False
            )
        )
        
        self.pending_follow_ups.append(follow_up)
        print(f"   → Queued test creation task for {endpoint}")
    
    def _handle_test_needed(self, discovery, agent_id):
        """When tests are needed, queue test creation with proper agent"""
        print(f"\n🧪 Test Needed: {discovery.description}")
        
        file_path = discovery.file_path or "unknown file"
        
        # Queue a follow-up task
        follow_up = Task(
            command=f"Create unit tests for {file_path} with good coverage",
            intent=TaskIntent(
                primary_goal="Create unit tests",
                affected_areas=["testing"],
                requires_coordination=False
            )
        )
        
        self.pending_follow_ups.append(follow_up)
        print(f"   → Queued test creation for {file_path}")
    
    def _handle_bug_found(self, discovery, agent_id):
        """When a bug is found, queue fix with appropriate priority"""
        print(f"\n🐛 Bug Found: {discovery.description}")
        
        # High severity bugs get immediate attention
        if discovery.severity in ["error", "critical"]:
            follow_up = Task(
                command=f"Fix the bug: {discovery.description}",
                intent=TaskIntent(
                    primary_goal="Fix bug",
                    affected_areas=["backend", "frontend"],  # Depends on bug location
                    requires_coordination=False,
                    complexity_score=0.7  # Higher priority
                )
            )
            
            self.pending_follow_ups.insert(0, follow_up)  # Add to front
            print(f"   → High priority fix queued!")
        else:
            print(f"   → Low severity, logged for later")
    
    def _handle_refactor_opportunity(self, discovery, agent_id):
        """When refactoring is suggested, evaluate and potentially queue"""
        print(f"\n♻️  Refactor Opportunity: {discovery.description}")
        
        # Only refactor if it's significant
        if "duplication" in discovery.description.lower():
            follow_up = Task(
                command=f"Refactor to eliminate code duplication: {discovery.description}",
                intent=TaskIntent(
                    primary_goal="Refactor code",
                    affected_areas=["backend", "frontend"],
                    requires_coordination=True,  # May affect multiple files
                    complexity_score=0.5
                )
            )
            
            self.pending_follow_ups.append(follow_up)
            print(f"   → Refactoring task queued")
    
    async def process_follow_ups(self):
        """Process any pending follow-up tasks"""
        if not self.pending_follow_ups:
            return
        
        print(f"\n📋 Processing {len(self.pending_follow_ups)} follow-up tasks...")
        
        # Execute follow-up tasks
        result = await self.coordination_engine.execute_coordinated_tasks(
            self.pending_follow_ups
        )
        
        print(f"\n✅ Follow-up tasks completed!")
        print(f"   Successful: {len(result.completed_tasks)}")
        print(f"   Failed: {len(result.failed_tasks)}")
        
        self.pending_follow_ups.clear()


async def main():
    """Run the intelligent coordination example"""
    print("\n🤖 Intelligent Agent Coordination Example\n")
    
    # Set up components
    workspace = Path.cwd()
    agent_registry = AgentRegistry(workspace)
    shared_memory = SharedMemory()
    coordination_engine = CoordinationEngine(agent_registry, shared_memory)
    
    # Create intelligent coordinator
    coordinator = IntelligentCoordinator(
        coordination_engine,
        agent_registry,
        shared_memory
    )
    
    # Initial tasks that will generate discoveries
    initial_tasks = [
        Task(
            command="Create a user authentication API with login and register endpoints",
            intent=TaskIntent(
                primary_goal="Create auth API",
                affected_areas=["backend", "api"],
                requires_coordination=False
            )
        ),
        Task(
            command="Analyze the existing codebase for missing tests and potential issues",
            intent=TaskIntent(
                primary_goal="Code analysis",
                affected_areas=["code_review"],
                requires_coordination=False
            )
        )
    ]
    
    print("📋 Phase 1: Initial tasks...")
    
    try:
        # Execute initial tasks
        result = await coordination_engine.execute_coordinated_tasks(initial_tasks)
        
        print(f"\n✅ Initial phase completed!")
        print(f"   Discoveries made: {len(coordination_engine.get_discoveries())}")
        
        # Wait a moment for discoveries to be processed
        await asyncio.sleep(1)
        
        # Process any follow-up tasks generated by discoveries
        await coordinator.process_follow_ups()
        
        # Show final summary
        print("\n📊 Final Summary:")
        all_discoveries = coordination_engine.get_discoveries()
        
        # Group by type
        by_type = {}
        for discovery in all_discoveries:
            if discovery.type not in by_type:
                by_type[discovery.type] = []
            by_type[discovery.type].append(discovery)
        
        for discovery_type, discoveries in by_type.items():
            print(f"\n   {discovery_type.value}: {len(discoveries)} total")
        
        # Show coordination insights
        print("\n💡 Coordination Insights:")
        print(f"   Total tasks executed: {len(initial_tasks) + len(coordinator.pending_follow_ups)}")
        print(f"   Automatic follow-ups generated: {len(coordinator.pending_follow_ups)}")
        print(f"   Cross-agent discoveries shared: {len(all_discoveries)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        await agent_registry.terminate_all_agents()


if __name__ == "__main__":
    asyncio.run(main())