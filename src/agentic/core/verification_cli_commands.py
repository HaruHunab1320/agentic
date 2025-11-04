"""
CLI commands for verification system

Provides commands to interact with the verification loop system.
"""

import asyncio
from pathlib import Path

from agentic.core.verification_coordinator import VerificationCoordinator
from agentic.utils.logging import LoggerMixin


class VerificationCLI(LoggerMixin):
    """CLI commands for verification system"""
    
    def __init__(self, workspace_path: Path):
        super().__init__()
        self.workspace_path = workspace_path
        self.verification_coordinator = VerificationCoordinator(workspace_path)
    
    async def verify(self, project_type: str = "auto") -> None:
        """Run standalone verification"""
        self.logger.info("Running system verification...")
        
        result = await self.verification_coordinator.verify_system(project_type)
        
        # Display results
        print("\n" + "="*60)
        print("Verification Results")
        print("="*60)
        
        print(f"\nOverall Status: {'✓ PASSED' if result.success else '✗ FAILED'}")
        
        # Test results
        if result.test_results:
            print("\nTest Results:")
            for test_type, test_result in result.test_results.items():
                status = "✓" if test_result.passed else "✗"
                print(f"  {status} {test_type}: ", end="")
                if test_result.total_tests > 0:
                    print(f"{test_result.passed_tests}/{test_result.total_tests} passed")
                else:
                    print(test_result.output[:50])
        
        # System health
        if result.system_health:
            print("\nSystem Health:")
            for check, healthy in result.system_health.items():
                status = "✓" if healthy else "✗"
                print(f"  {status} {check}")
        
        # Failures
        if result.failures:
            print(f"\n⚠️  {len(result.failures)} failures detected")
        
        # Suggestions
        if result.suggestions:
            print("\nSuggested Actions:")
            for i, suggestion in enumerate(result.suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        print("\n" + "="*60)
    
    async def analyze_quality(self) -> None:
        """Analyze code quality metrics"""
        from agentic.core.intelligent_coordinator_with_verification import VerificationLoopController
        
        controller = VerificationLoopController(self.workspace_path)
        
        # Run verification to get metrics
        result = await controller.verification_coordinator.verify_system()
        metrics = controller._calculate_metrics(result)
        
        print("\n" + "="*60)
        print("Code Quality Analysis")
        print("="*60)
        
        # Display metrics
        print(f"\nTest Pass Rate:    {metrics.test_pass_rate:.1%}")
        print(f"Build Success:     {'✓ Yes' if metrics.build_success else '✗ No'}")
        print(f"System Health:     {metrics.system_health_score:.1%}")
        print(f"Overall Quality:   {metrics.overall_quality:.2f}/1.0")
        
        # Quality bar graph
        bar_length = int(metrics.overall_quality * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"\nQuality Score:     [{bar}] {metrics.overall_quality:.1%}")
        
        # Recommendations
        print("\nRecommendations:")
        if metrics.test_pass_rate < 1.0:
            print("  • Fix failing tests to improve quality")
        if not metrics.build_success:
            print("  • Resolve build errors")
        if metrics.system_health_score < 1.0:
            print("  • Address system health issues")
        if metrics.overall_quality < 0.8:
            print("  • Consider running 'agentic fix' to auto-fix issues")
        
        print("\n" + "="*60)
    
    async def fix(self, max_iterations: int = 3) -> None:
        """Run automatic fix iterations"""
        from agentic.core.agent_registry import AgentRegistry
        from agentic.core.shared_memory import SharedMemory
        from agentic.core.intelligent_coordinator_with_verification import IntelligentCoordinatorWithVerification
        
        # Set up components
        shared_memory = SharedMemory()
        agent_registry = AgentRegistry(self.workspace_path, shared_memory)
        
        # Create coordinator
        coordinator = IntelligentCoordinatorWithVerification(
            agent_registry,
            shared_memory,
            self.workspace_path
        )
        
        # Set max iterations
        coordinator.verification_controller.max_verification_iterations = max_iterations
        
        print(f"\nStarting automatic fix process (max {max_iterations} iterations)...")
        
        # Run verification and fix loop
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Run verification
            success, metrics, fix_tasks = await coordinator.verification_controller.run_verification_loop(
                coordinator.current_phase
            )
            
            print(f"Quality Score: {metrics.overall_quality:.2f}")
            print(f"Test Pass Rate: {metrics.test_pass_rate:.1%}")
            
            if success:
                print("\n✓ All checks passed!")
                break
            
            if not fix_tasks:
                print("\n⚠️  No automatic fixes available")
                break
            
            print(f"\nGenerating {len(fix_tasks)} fix tasks...")
            
            # Execute fix tasks
            for fix_task in fix_tasks:
                print(f"  Executing: {fix_task.command[:60]}...")
                results = await coordinator._execute_tasks_with_feedback([fix_task])
                
                for task_id, (result, _) in results.items():
                    if result.status == "completed":
                        print(f"    ✓ {task_id} completed")
                    else:
                        print(f"    ✗ {task_id} failed: {result.error}")
            
            # Check if we should continue
            if not coordinator.verification_controller.should_continue_fixing():
                print("\n⚠️  Stopping - quality not improving")
                break
        
        # Final verification
        print("\n--- Final Verification ---")
        final_result = await coordinator.verification_controller.verification_coordinator.verify_system()
        final_metrics = coordinator.verification_controller._calculate_metrics(final_result)
        
        print(f"\nFinal Quality Score: {final_metrics.overall_quality:.2f}")
        print(f"Improvement: {'+' if final_metrics.overall_quality > metrics.overall_quality else ''}{final_metrics.overall_quality - metrics.overall_quality:.2f}")
        
        # Cleanup
        await agent_registry.cleanup()


async def main():
    """Main entry point for verification CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic Verification System")
    parser.add_argument("command", choices=["verify", "analyze", "fix"],
                       help="Command to run")
    parser.add_argument("--project-type", default="auto",
                       help="Project type (auto, python, npm, etc.)")
    parser.add_argument("--max-iterations", type=int, default=3,
                       help="Maximum fix iterations for 'fix' command")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(),
                       help="Workspace path")
    
    args = parser.parse_args()
    
    cli = VerificationCLI(args.workspace)
    
    if args.command == "verify":
        await cli.verify(args.project_type)
    elif args.command == "analyze":
        await cli.analyze_quality()
    elif args.command == "fix":
        await cli.fix(args.max_iterations)


if __name__ == "__main__":
    asyncio.run(main())