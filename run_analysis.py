"""Run Apple financial analysis with full process output."""
import asyncio
import json
import sys
from datetime import datetime

# Fix Windows encoding and buffering
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    # Force unbuffered output
    import functools
    print = functools.partial(print, flush=True)

# IMPORTANT: Import built-in tools to register them with the global registry
# These tools use @tool decorator which registers them on import
from agent_engine.tools.builtin import (
    code_execute,
    list_directory,
    read_file,
    web_search,
    write_file,
)
# Ensure they are "used" to prevent linter warnings
_ = (code_execute, list_directory, read_file, web_search, write_file)

from agent_engine.agents.graph import AgentOrchestrator


def format_message(msg):
    """Format a message for display."""
    if hasattr(msg, 'content'):
        return msg.content
    return str(msg)


async def main():
    request = (
        "Analyze Apple Inc. financial reports from 2019 to 2024. "
        "Include: revenue and profit trends, business segment performance, "
        "key financial metrics changes, major challenges and opportunities, "
        "and future outlook. Use web search to get the latest financial data. "
        "Please respond in Chinese."
    )
    
    print("=" * 80)
    print("苹果公司财报分析任务")
    print("=" * 80)
    print(f"任务请求: {request}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    orchestrator = AgentOrchestrator(checkpointer=None)
    
    print("\n开始流式执行任务...\n")
    
    step_count = 0
    async for event in orchestrator.stream(
        user_request=request,
        task_id="apple_analysis_stream",
        config={"recursion_limit": 150},
    ):
        step_count += 1
        print(f"\n{'='*80}")
        print(f"步骤 {step_count} - 时间: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)
        
        for node_name, node_output in event.items():
            print(f"\n>>> 节点: {node_name}")
            print("-" * 40)
            
            if isinstance(node_output, dict):
                # 显示状态
                if 'status' in node_output:
                    print(f"状态: {node_output['status']}")
                
                # 显示子任务
                if 'subtasks' in node_output and node_output['subtasks']:
                    print(f"\n子任务列表 ({len(node_output['subtasks'])} 个):")
                    for i, st in enumerate(node_output['subtasks'], 1):
                        desc = st.get('description', '')[:100]
                        status = st.get('status', 'unknown')
                        print(f"  {i}. [{status}] {desc}...")
                
                # 显示当前子任务索引
                if 'current_subtask_index' in node_output:
                    print(f"\n当前子任务索引: {node_output['current_subtask_index']}")
                
                # 显示执行结果
                if 'execution_results' in node_output and node_output['execution_results']:
                    print(f"\n执行结果 ({len(node_output['execution_results'])} 个):")
                    for i, result in enumerate(node_output['execution_results'][-3:], 1):
                        if isinstance(result, dict):
                            result_str = json.dumps(result, ensure_ascii=False, indent=2)[:500]
                        else:
                            result_str = str(result)[:500]
                        print(f"  结果 {i}: {result_str}...")
                
                # 显示消息
                if 'messages' in node_output and node_output['messages']:
                    print(f"\n消息 ({len(node_output['messages'])} 条):")
                    for msg in node_output['messages'][-2:]:
                        content = format_message(msg)[:800]
                        print(f"  - {content}")
                
                # 显示评论反馈
                if 'critic_feedback' in node_output and node_output['critic_feedback']:
                    print(f"\n评估反馈:")
                    feedback = node_output['critic_feedback']
                    print(f"  完成度: {feedback.get('is_complete', 'N/A')}")
                    print(f"  正确性: {feedback.get('is_correct', 'N/A')}")
                    if feedback.get('suggestions'):
                        print(f"  建议: {feedback.get('suggestions')[:200]}")
                
                # 显示度量
                if 'metrics' in node_output:
                    metrics = node_output['metrics']
                    print(f"\n度量信息:")
                    print(f"  总 tokens: {metrics.get('total_tokens', 0)}")
                    print(f"  输入 tokens: {metrics.get('input_tokens', 0)}")
                    print(f"  输出 tokens: {metrics.get('output_tokens', 0)}")
                    print(f"  步骤数: {metrics.get('step_count', 0)}")
                    print(f"  工具调用: {metrics.get('tool_call_count', 0)}")
                
                # 显示错误
                if 'error' in node_output and node_output['error']:
                    print(f"\n错误: {node_output['error']}")
            else:
                print(f"输出: {str(node_output)[:500]}")
    
    print("\n" + "=" * 80)
    print(f"任务完成 - 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总步骤数: {step_count}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
