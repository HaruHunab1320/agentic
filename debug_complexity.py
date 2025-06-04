import asyncio
from src.agentic.core.hierarchical_agents import TaskAnalyzer
from src.agentic.models.task import Task, TaskIntent, TaskType

async def test():
    analyzer = TaskAnalyzer()
    intent = TaskIntent(
        task_type=TaskType.DEBUG,
        complexity_score=0.2,
        estimated_duration=15,
        affected_areas=['frontend'],
        requires_reasoning=False,
        requires_coordination=False,
        file_patterns=['**/*.css', '**/*.js']
    )
    task = Task(
        id='simple_task_001',
        command='Fix button styling in user profile',
        intent=intent
    )
    analysis = await analyzer.analyze_task(task)
    print(f'Complexity score: {analysis.complexity_score}')
    print(f'Complexity level: {analysis.complexity_level}')
    print(f'Required domains: {analysis.required_domains}')

if __name__ == "__main__":
    asyncio.run(test()) 