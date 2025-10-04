"""
LLM Determinism Demo for AgentMap

Implements multiple strategies to achieve LLM determinism:
1. Caching - Perfect determinism through response storage
2. Majority Vote with Semantic Similarity - Consensus-based determinism
3. Batch-Size Control - Inspired by "Defeating Nondeterminism in LLM Inference"
   (https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

Key Insight: LLM non-determinism is primarily caused by batch-invariance violations
in GPU kernels. When using external APIs, we can't control kernel implementation,
but we can:
- Use consistent batch sizes (batch_size=1 for maximum determinism)
- Implement semantic similarity matching
- Use majority voting across multiple calls
- Cache responses for perfect reproducibility
"""

from __future__ import annotations
import os
import json
import hashlib
from typing import Dict, Any, List, Optional
from .ontology import AgentMap, Tile, Action, Policy
from .planner_ao import AOPlanner, Stage
from .router import CostLens

try:
    from openai import OpenAI
    HAS_LLM = True
except ImportError as e:
    HAS_LLM = False
    OpenAI = None
    import sys
    print(f"Warning: Could not import openai: {e}", file=sys.stderr)
    print(f"Python path: {sys.executable}", file=sys.stderr)

def build_llm_demo_map(ablate_policies=False, ablate_tiles=False) -> AgentMap:
    """
    Build a demo map for LLM determinism with multiple providers and strategies.
    
    Tiles:
    - openai_tile: GPT models with different temperature settings
    - retry_tile: Retry strategies with consensus voting
    - cache_tile: Response caching based on question hash
    
    Policies:
    - temperature_zero: Ensures temperature=0 for determinism
    - max_tokens_limit: Limits response length
    - question_hash_match: Validates cache hits
    """
    
    # OpenAI Tile - Different GPT models with deterministic settings
    openai_tile = Tile(
        id="openai",
        actions=[
            Action(
                id="openai.gpt4_turbo",
                inputs={"question": "string", "temperature": "number", "max_tokens": "number"},
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "minLength": 1},
                        "temperature": {"type": "number"},
                        "max_tokens": {"type": "number"}
                    },
                    "required": ["question"]
                },
                cost_base=0.01,      # $0.01 per call (GPT-4 Turbo pricing)
                latency_ms=800,      # ~800ms average
                risk=0.02,           # 2% failure rate
                policy_ids=["temperature_zero", "max_tokens_limit"]
            ),
            Action(
                id="openai.gpt35_turbo",
                inputs={"question": "string", "temperature": "number", "max_tokens": "number"},
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "minLength": 1},
                        "temperature": {"type": "number"},
                        "max_tokens": {"type": "number"}
                    },
                    "required": ["question"]
                },
                cost_base=0.002,     # $0.002 per call (GPT-3.5 Turbo pricing)
                latency_ms=500,      # ~500ms average
                risk=0.03,           # 3% failure rate
                policy_ids=["temperature_zero", "max_tokens_limit"]
            ),
        ]
    )
    
    # Retry Tile - Consensus and retry strategies
    retry_tile = Tile(
        id="retry",
        actions=[
            Action(
                id="retry.majority_vote",
                inputs={"question": "string", "num_calls": "number"},
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "minLength": 1},
                        "num_calls": {"type": "number"}
                    },
                    "required": ["question"]
                },
                cost_base=0.006,     # 3x GPT-3.5 calls
                latency_ms=1500,     # Parallel calls
                risk=0.001,          # Very low risk with consensus
                policy_ids=["temperature_zero"]
            ),
            Action(
                id="retry.exponential_backoff",
                inputs={"question": "string", "max_retries": "number"},
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "minLength": 1},
                        "max_retries": {"type": "number"}
                    },
                    "required": ["question"]
                },
                cost_base=0.004,     # 2x average retries
                latency_ms=1200,     # With backoff delays
                risk=0.005,          # Low risk with retries
                policy_ids=["temperature_zero", "max_tokens_limit"]
            ),
        ]
    )
    
    # Cache Tile - Response caching for determinism
    cache_tile = Tile(
        id="cache",
        actions=[
            Action(
                id="cache.lookup_or_call",
                inputs={"question": "string", "question_hash": "string"},
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "minLength": 1},
                        "question_hash": {"type": "string"}
                    },
                    "required": ["question", "question_hash"]
                },
                cost_base=0.0001,    # Nearly free for cache hits
                latency_ms=50,       # Very fast cache lookup
                risk=0.01,           # Small risk of cache miss
                policy_ids=["question_hash_match"]
            ),
            Action(
                id="cache.invalidate_and_refresh",
                inputs={"question": "string", "question_hash": "string"},
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "minLength": 1},
                        "question_hash": {"type": "string"}
                    },
                    "required": ["question", "question_hash"]
                },
                cost_base=0.002,     # Cache miss + GPT-3.5 call
                latency_ms=600,      # Lookup + LLM call
                risk=0.03,
                policy_ids=["question_hash_match", "temperature_zero"]
            ),
        ]
    )
    
    # Policies for determinism
    policies = []
    if not ablate_policies:
        policies = [
            Policy(
                id="temperature_zero",
                expr="temperature == 0"
            ),
            Policy(
                id="max_tokens_limit",
                expr="max_tokens <= 4096"
            ),
            Policy(
                id="question_hash_match",
                expr="question_hash != ''"
            ),
        ]
    
    tiles = [openai_tile, retry_tile, cache_tile]
    
    if ablate_tiles:
        # Flatten into one tile (no composition)
        all_actions = []
        for t in tiles:
            all_actions.extend(t.actions)
        tiles = [Tile(id="flat", actions=all_actions)]
    
    return AgentMap(tiles=tiles, policies=policies)


def llm_demo_stages() -> List[Stage]:
    """
    Define the stages for LLM determinism workflow.
    
    Workflow:
    1. cache_check: Try to get cached response
    2. llm_call: If cache miss, call LLM with deterministic settings
    3. retry_strategy: If needed, use retry/consensus for reliability
    """
    return [
        Stage(
            name="cache_check",
            candidates=["cache.lookup_or_call", "cache.invalidate_and_refresh"]
        ),
        Stage(
            name="llm_call",
            candidates=["openai.gpt4_turbo", "openai.gpt35_turbo"]
        ),
        Stage(
            name="retry_strategy",
            candidates=["retry.majority_vote", "retry.exponential_backoff"]
        ),
    ]


def execute_llm_query(question: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Execute an LLM query using the AgentMap planner.
    
    Args:
        question: The question to ask the LLM
        use_cache: Whether to use caching
        
    Returns:
        Dictionary with plan and execution details
    """
    if not HAS_LLM:
        return {
            "error": "openai not installed. Run: pip install openai",
            "question": question
        }
    
    try:
        # Build the map and planner
        amap = build_llm_demo_map()
        planner = AOPlanner(amap, CostLens(weights=dict(latency=0.3, price=0.7)))
        
        # Create context with question hash for caching
        question_hash = hashlib.sha256(question.encode('utf-8')).hexdigest()
        
        context = {
            "question": question,
            "question_hash": question_hash,
            "temperature": 0,
            "max_tokens": 1024,
            "num_calls": 3,
            "max_retries": 3,
        }
        
        # Plan the workflow
        stages = llm_demo_stages()
        plan = planner.plan(goal="deterministic_llm_query", stages=stages, context=context)
        
        # Serialize the plan
        if hasattr(plan, 'model_dump'):
            plan_dict = plan.model_dump()
        elif hasattr(plan, 'dict'):
            plan_dict = plan.dict()
        else:
            plan_dict = {"steps": []}
        
        return {
            "question": question,
            "question_hash": question_hash,
            "plan": plan_dict,
            "map_version": amap.map_version(),
        }
    except Exception as e:
        return {
            "error": f"Planning failed: {str(e)}",
            "question": question
        }


def call_llm_with_plan(plan_steps: List[Dict[str, Any]], question: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Actually execute the LLM calls based on the plan.
    
    Args:
        plan_steps: The planned steps from AgentMap
        question: The original question
        use_cache: Whether to use caching (default: True)
        
    Returns:
        LLM response and metadata
    """
    if not HAS_LLM:
        return {"error": "openai not installed"}
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set in environment"}
    
    # Check for cached response first (only if caching is enabled)
    question_hash = hashlib.sha256(question.encode('utf-8')).hexdigest()
    cache_file = f".llm_cache/{question_hash}.json"
    
    if use_cache:
        # If cache action is in the plan, try to use it
        for step in plan_steps:
            if "cache" in step.get("action_id", ""):
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            cached = json.load(f)
                            cached["from_cache"] = True
                            return cached
                    except Exception:
                        pass
                break
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Check if retry strategy (majority vote) is in the plan
        use_majority_vote = False
        num_calls = 3  # Default
        for step in plan_steps:
            if "majority_vote" in step.get("action_id", ""):
                use_majority_vote = True
                num_calls = step.get("inputs", {}).get("num_calls", 3)
                break
        
        # Execute based on the first LLM action in the plan
        for step in plan_steps:
            action_id = step.get("action_id", "")
            
            if "gpt" in action_id:
                model = "gpt-4-turbo-preview" if "gpt4" in action_id else "gpt-3.5-turbo"
                
                if use_majority_vote:
                    # Make multiple calls and use majority vote for determinism
                    responses = []
                    total_tokens = 0
                    
                    for i in range(num_calls):
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant. Provide consistent, deterministic answers."},
                                {"role": "user", "content": question}
                            ],
                            temperature=0,
                            max_tokens=1024,
                            seed=42,
                        )
                        
                        answer = response.choices[0].message.content
                        responses.append(answer)
                        total_tokens += response.usage.total_tokens
                    
                    # Find most common response (majority vote)
                    response_groups = []
                    similarity_threshold = 0.85  # 85% similarity
                    
                    for response in responses:
                        # Find if this response is similar to any existing group
                        matched = False
                        for group in response_groups:
                            # Simple similarity: compare first 200 chars
                            sample_resp = response[:200]
                            sample_group = group[0][:200]
                            
                            # Count matching words
                            words_resp = set(sample_resp.lower().split())
                            words_group = set(sample_group.lower().split())
                            if len(words_resp) > 0:
                                similarity = len(words_resp & words_group) / len(words_resp | words_group)
                                if similarity >= similarity_threshold:
                                    group.append(response)
                                    matched = True
                                    break
                        
                        if not matched:
                            response_groups.append([response])
                    
                    # Find largest group (most common semantic response)
                    largest_group = max(response_groups, key=len)
                    most_common_answer = largest_group[0]  # Take first from largest group
                    count = len(largest_group)
                    
                    result = {
                        "answer": most_common_answer,
                        "model": model,
                        "action_id": "retry.majority_vote",
                        "finish_reason": "majority_vote",
                        "usage": {
                            "prompt_tokens": 0,  # Approximate
                            "completion_tokens": 0,
                            "total_tokens": total_tokens,
                        },
                        "from_cache": False,
                        "majority_vote": {
                            "num_calls": num_calls,
                            "consensus_count": count,
                            "unique_responses": len(response_groups),
                            "semantic_groups": len(response_groups),
                            "all_responses": [group[0] for group in response_groups]  # One from each group
                        }
                    }
                else:
                    # Single call with seed for best-effort determinism
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant. Provide consistent, deterministic answers."},
                            {"role": "user", "content": question}
                        ],
                        temperature=0,
                        max_tokens=1024,
                        seed=42,
                    )
                    
                    result = {
                        "answer": response.choices[0].message.content,
                        "model": model,
                        "action_id": action_id,
                        "finish_reason": response.choices[0].finish_reason,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        },
                        "from_cache": False
                    }
                
                # Cache the response for future deterministic retrieval (only if caching is enabled)
                if use_cache:
                    os.makedirs(".llm_cache", exist_ok=True)
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump(result, f, indent=2)
                    except Exception:
                        pass
                
                return result
        
        return {"error": "No LLM action found in plan"}
        
    except Exception as e:
        return {"error": str(e)}
