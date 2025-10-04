"""
CLI commands for LLM determinism demo.
"""

import os
import json
from .llm_demo import (
    build_llm_demo_map,
    llm_demo_stages,
    execute_llm_query,
    call_llm_with_plan
)
from .planner_ao import AOPlanner
from .router import CostLens


def cmd_llm_plan(args):
    """Generate a plan for an LLM query."""
    os.makedirs(args.out, exist_ok=True)
    
    result = execute_llm_query(args.question, use_cache=not args.no_cache)
    
    # Save the plan
    plan_file = os.path.join(args.out, "llm_plan.json")
    with open(plan_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Question: {args.question}")
    print(f"Question Hash: {result.get('question_hash', 'N/A')}")
    print(f"Map Version: {result.get('map_version', 'N/A')}")
    
    if 'plan' in result:
        plan = result['plan']
        if 'steps' in plan:
            print(f"\nPlanned Actions:")
            for i, step in enumerate(plan['steps'], 1):
                action_id = step.get('action_id', 'unknown')
                cost = step.get('cost', 0)
                latency = step.get('latency_ms', 0)
                print(f"  {i}. {action_id} (cost: ${cost:.4f}, latency: {latency}ms)")
    
    print(f"\nWrote {plan_file}")


def cmd_llm_execute(args):
    """Execute an LLM query using a saved plan."""
    os.makedirs(args.out, exist_ok=True)
    
    # Load the plan
    with open(args.plan) as f:
        plan_data = json.load(f)
    
    question = plan_data.get('question', args.question)
    plan_steps = plan_data.get('plan', {}).get('steps', [])
    
    print(f"Executing plan for question: {question}")
    print(f"Number of steps: {len(plan_steps)}\n")
    
    # Execute the LLM call
    result = call_llm_with_plan(plan_steps, question)
    
    # Save the result
    result_file = os.path.join(args.out, "llm_result.json")
    with open(result_file, "w") as f:
        json.dump({
            "question": question,
            "plan_file": args.plan,
            "result": result
        }, f, indent=2)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Model: {result.get('model', 'N/A')}")
        print(f"Action: {result.get('action_id', 'N/A')}")
        print(f"\nAnswer:\n{result.get('answer', 'N/A')}\n")
        
        if 'usage' in result:
            usage = result['usage']
            print(f"Token Usage:")
            print(f"  Prompt: {usage.get('prompt_tokens', 0)}")
            print(f"  Completion: {usage.get('completion_tokens', 0)}")
            print(f"  Total: {usage.get('total_tokens', 0)}")
    
    print(f"\nWrote {result_file}")


def cmd_llm_compare(args):
    """Compare multiple executions of the same question for determinism."""
    os.makedirs(args.out, exist_ok=True)
    
    question = args.question
    num_runs = args.runs
    use_cache = getattr(args, 'cache', True)
    use_majority_vote = getattr(args, 'majority_vote', False)
    
    print(f"Running {num_runs} executions of the same question...")
    print(f"Question: {question}")
    if not use_cache:
        print("Cache: DISABLED")
    if use_majority_vote:
        print("Strategy: MAJORITY VOTE (5 calls per run)")
    print()
    
    # First, create a plan
    plan_result = execute_llm_query(question)
    
    # Check for errors
    if 'error' in plan_result:
        print(f"Error: {plan_result['error']}")
        return
    
    plan_steps = plan_result.get('plan', {}).get('steps', [])
    
    # If majority vote is requested, modify the plan to include it
    if use_majority_vote and plan_steps:
        # Insert majority_vote action into the plan
        majority_vote_step = {
            "action_id": "retry.majority_vote",
            "inputs": {"question": question, "num_calls": 5},  
            "cost": 0.010,
            "latency_ms": 2500
        }
        # Add it before the LLM call step
        plan_steps.insert(1, majority_vote_step)
    
    if not plan_steps:
        print("Error: Could not generate plan - no steps found")
        print(f"Plan result: {plan_result}")
        return
    
    # Execute multiple times
    results = []
    answers = []
    cache_hits = 0
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...", end=" ")
        result = call_llm_with_plan(plan_steps, question, use_cache=use_cache)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            answer = result.get('answer', '')
            answers.append(answer)
            results.append(result)
            from_cache = result.get('from_cache', False)
            cache_indicator = " [CACHED]" if from_cache else ""
            if from_cache:
                cache_hits += 1
            print(f"✓ ({result.get('usage', {}).get('total_tokens', 0)} tokens){cache_indicator}")
    
    # Analyze determinism
    unique_answers = list(set(answers))
    is_deterministic = len(unique_answers) == 1
    
    comparison = {
        "question": question,
        "num_runs": num_runs,
        "successful_runs": len(results),
        "unique_answers": len(unique_answers),
        "is_deterministic": is_deterministic,
        "plan": plan_result.get('plan'),
        "results": results,
    }
    
    # Save comparison
    comparison_file = os.path.join(args.out, "llm_comparison.json")
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Determinism Analysis:")
    print(f"  Total runs: {num_runs}")
    print(f"  Successful: {len(results)}")
    print(f"  Cache hits: {cache_hits}")
    print(f"  Unique answers: {len(unique_answers)}")
    print(f"  Deterministic: {'✓ YES' if is_deterministic else '✗ NO'}")
    
    if not is_deterministic:
        print(f"\nFound {len(unique_answers)} different answers:")
        for i, answer in enumerate(unique_answers, 1):
            print(f"\n  Answer {i} ({answers.count(answer)} times):")
            print(f"  {answer[:200]}{'...' if len(answer) > 200 else ''}")
    
    print(f"\nWrote {comparison_file}")


def cmd_llm_deterministic(args):
    """
    Achieve determinism through controlled execution timing and consensus.
    
    This is the most aggressive determinism strategy that doesn't use caching.
    It makes multiple sequential calls with controlled timing and retries until
    consensus is achieved.
    """
    import os
    from .deterministic_executor import (
        DeterministicExecutor,
        DeterministicConfig,
        create_deterministic_llm_call
    )
    from openai import OpenAI
    
    os.makedirs(args.out, exist_ok=True)
    
    question = args.question
    num_attempts = args.attempts
    max_iterations = args.max_iterations
    
    print(f"Deterministic Execution Strategy")
    print(f"Question: {question}")
    print(f"Attempts per iteration: {num_attempts}")
    print(f"Max iterations: {max_iterations}")
    print(f"Strategy: Controlled timing + Consensus validation + Adaptive retry")
    print()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment")
        return
    
    # Create executor with configuration
    config = DeterministicConfig(
        min_delay_between_calls=2.0,  # 2 seconds between calls
        consensus_threshold=max(3, num_attempts // 2 + 1),  # Majority
        use_response_fingerprinting=True
    )
    
    executor = DeterministicExecutor(config)
    
    # Create OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Create the LLM call function
    llm_call_fn = create_deterministic_llm_call(
        client=client,
        model="gpt-3.5-turbo",
        question=question
    )
    
    # Execute with retry until consensus
    print("Starting deterministic execution...\n")
    result = executor.execute_with_retry_until_consensus(
        llm_call_fn=llm_call_fn,
        max_iterations=max_iterations,
        attempts_per_iteration=num_attempts
    )
    
    # Save results
    output_file = os.path.join(args.out, "deterministic_result.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    
    # Display summary
    metadata = result.get('deterministic_metadata', {})
    if metadata:
        print(f"\nDeterminism Summary:")
        print(f"  Total attempts: {metadata.get('num_attempts', 0)}")
        print(f"  Unique responses: {metadata.get('unique_responses', 0)}")
        print(f"  Consensus: {metadata.get('consensus_count', 0)}/{metadata.get('num_attempts', 0)} ({metadata.get('consensus_percentage', 0):.1f}%)")
        print(f"  Deterministic: {'✓ YES' if metadata.get('has_consensus', False) else '✗ NO'}")
        
        if metadata.get('has_consensus', False):
            print(f"\n✓ Successfully achieved determinism without caching!")
        else:
            print(f"\n⚠ Could not achieve consensus. Consider:")
            print(f"  - Increasing --attempts (currently {num_attempts})")
            print(f"  - Increasing --max-iterations (currently {max_iterations})")
            print(f"  - Using simpler, more factual questions")
            print(f"  - Or using caching for perfect determinism")
    
    print()


def _extend_parser_for_llm_demo(sub):
    """Add LLM demo subcommands to the parser."""
    
    # llm-plan command
    lp = sub.add_parser("llm-plan", help="Generate a plan for an LLM query")
    lp.add_argument("--question", required=True, help="Question to ask the LLM")
    lp.add_argument("--out", required=True, help="Output directory")
    lp.add_argument("--no-cache", action="store_true", help="Disable caching")
    lp.set_defaults(func=cmd_llm_plan)
    
    # llm-execute command
    le = sub.add_parser("llm-execute", help="Execute an LLM query using a saved plan")
    le.add_argument("--plan", required=True, help="Path to plan JSON file")
    le.add_argument("--question", default="", help="Override question from plan")
    le.add_argument("--out", required=True, help="Output directory")
    le.set_defaults(func=cmd_llm_execute)
    
    # llm-compare command
    lc = sub.add_parser("llm-compare", help="Compare multiple executions for determinism")
    lc.add_argument("--question", required=True, help="Question to ask the LLM")
    lc.add_argument("--runs", type=int, default=5, help="Number of runs (default: 5)")
    lc.add_argument("--out", required=True, help="Output directory")
    lc.add_argument("--no-cache", action="store_false", dest="cache", help="Disable caching")
    lc.add_argument("--majority-vote", action="store_true", help="Use majority vote strategy")
    lc.set_defaults(func=cmd_llm_compare, cache=True)  # Set default cache=True

    # llm-deterministic command - New aggressive determinism strategy
    ld = sub.add_parser("llm-deterministic", help="Achieve determinism through controlled execution")
    ld.add_argument("--question", required=True, help="Question to ask the LLM")
    ld.add_argument("--out", required=True, help="Output directory")
    ld.add_argument("--attempts", type=int, default=7, help="Number of attempts per iteration (default: 7)")
    ld.add_argument("--max-iterations", type=int, default=3, help="Maximum iterations to achieve consensus (default: 3)")
    ld.set_defaults(func=cmd_llm_deterministic)
