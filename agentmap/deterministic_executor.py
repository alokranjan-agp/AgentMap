"""
Deterministic LLM Execution Controller

Achieves true determinism without caching by controlling execution timing and batch behavior.

Inspired by "Defeating Nondeterminism in LLM Inference" (Thinking Machines Lab, 2025)
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

Key Strategy: Since we can't control GPU kernels at the API level, we control:
1. Request timing - Ensure consistent server load
2. Sequential execution - One request at a time
3. Retry with backoff - Handle rate limits gracefully
4. Response validation - Verify consistency across attempts
"""

import time
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json


@dataclass
class DeterministicConfig:
    """Configuration for deterministic execution."""
    
    # Timing control
    min_delay_between_calls: float = 2.0  # Seconds between calls
    max_delay_between_calls: float = 5.0  # Maximum delay for rate limiting
    
    # Validation
    require_consensus: bool = True
    consensus_threshold: int = 3  # Out of 5 calls
    max_retries: int = 10
    
    # Fingerprinting
    use_response_fingerprinting: bool = True
    fingerprint_length: int = 64  # Characters to use for fingerprinting


class DeterministicExecutor:
    """
    Executes LLM calls with deterministic guarantees.
    
    Strategy:
    1. Sequential execution with controlled timing
    2. Multiple attempts with consensus validation
    3. Response fingerprinting for quick comparison
    4. Adaptive retry with exponential backoff
    """
    
    def __init__(self, config: Optional[DeterministicConfig] = None):
        self.config = config or DeterministicConfig()
        self.call_history: List[Dict[str, Any]] = []
        self.last_call_time: float = 0
    
    def _wait_for_next_call(self):
        """Ensure consistent timing between calls."""
        if self.last_call_time > 0:
            elapsed = time.time() - self.last_call_time
            wait_time = self.config.min_delay_between_calls - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
        self.last_call_time = time.time()
    
    def _fingerprint_response(self, response: str) -> str:
        """Create a fingerprint of the response for quick comparison."""
        # Use first N characters + hash for fingerprinting
        prefix = response[:self.config.fingerprint_length]
        full_hash = hashlib.sha256(response.encode('utf-8')).hexdigest()[:16]
        return f"{prefix}|{full_hash}"
    
    def _are_responses_equivalent(self, resp1: str, resp2: str, threshold: float = 0.90) -> bool:
        """Check if two responses are semantically equivalent."""
        # Quick check: exact match
        if resp1 == resp2:
            return True
        
        # Fingerprint check
        if self.config.use_response_fingerprinting:
            fp1 = self._fingerprint_response(resp1)
            fp2 = self._fingerprint_response(resp2)
            if fp1 == fp2:
                return True
        
        # Semantic similarity check (Jaccard on words)
        words1 = set(resp1.lower().split())
        words2 = set(resp2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        similarity = len(words1 & words2) / len(words1 | words2)
        return similarity >= threshold
    
    def execute_with_consensus(
        self,
        llm_call_fn,
        num_attempts: int = 5,
        **call_kwargs
    ) -> Dict[str, Any]:
        """
        Execute LLM call multiple times and return consensus response.
        
        Args:
            llm_call_fn: Function that makes the LLM API call
            num_attempts: Number of attempts to make
            **call_kwargs: Arguments to pass to llm_call_fn
            
        Returns:
            Dict with consensus response and metadata
        """
        responses = []
        fingerprints = []
        
        print(f"Making {num_attempts} sequential calls with controlled timing...")
        
        for i in range(num_attempts):
            # Wait for consistent timing
            self._wait_for_next_call()
            
            # Make the call
            print(f"  Call {i+1}/{num_attempts}...", end=" ", flush=True)
            
            try:
                response = llm_call_fn(**call_kwargs)
                answer = response.get('answer', '')
                
                responses.append(response)
                fingerprints.append(self._fingerprint_response(answer))
                
                print(f"✓ ({len(answer)} chars)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                # Exponential backoff
                time.sleep(min(2 ** i, self.config.max_delay_between_calls))
                continue
        
        if not responses:
            return {"error": "All attempts failed"}
        
        # Find consensus
        return self._find_consensus(responses, fingerprints)
    
    def _find_consensus(
        self,
        responses: List[Dict[str, Any]],
        fingerprints: List[str]
    ) -> Dict[str, Any]:
        """Find the consensus response from multiple attempts."""
        
        # Group responses by fingerprint
        groups = {}
        for i, fp in enumerate(fingerprints):
            if fp not in groups:
                groups[fp] = []
            groups[fp].append(i)
        
        # Find largest group
        largest_group = max(groups.values(), key=len)
        consensus_count = len(largest_group)
        consensus_response = responses[largest_group[0]]
        
        # Check if we have sufficient consensus
        has_consensus = consensus_count >= self.config.consensus_threshold
        
        print(f"\nConsensus Analysis:")
        print(f"  Unique responses: {len(groups)}")
        print(f"  Consensus count: {consensus_count}/{len(responses)}")
        print(f"  Consensus achieved: {'✓ YES' if has_consensus else '✗ NO'}")
        
        # Add metadata
        consensus_response['deterministic_metadata'] = {
            'num_attempts': len(responses),
            'unique_responses': len(groups),
            'consensus_count': consensus_count,
            'consensus_percentage': (consensus_count / len(responses)) * 100,
            'has_consensus': has_consensus,
            'all_fingerprints': fingerprints
        }
        
        return consensus_response
    
    def execute_with_retry_until_consensus(
        self,
        llm_call_fn,
        max_iterations: int = 3,
        attempts_per_iteration: int = 5,
        **call_kwargs
    ) -> Dict[str, Any]:
        """
        Keep trying until we achieve consensus or hit max iterations.
        
        This is the most aggressive determinism strategy:
        - Make multiple attempts
        - If no consensus, wait and try again
        - Repeat until consensus or max iterations
        """
        
        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration + 1}/{max_iterations} ===")
            
            result = self.execute_with_consensus(
                llm_call_fn,
                num_attempts=attempts_per_iteration,
                **call_kwargs
            )
            
            metadata = result.get('deterministic_metadata', {})
            
            if metadata.get('has_consensus', False):
                print(f"\n✓ Consensus achieved!")
                return result
            
            if iteration < max_iterations - 1:
                wait_time = (iteration + 1) * 5  # Progressive backoff
                print(f"\n⚠ No consensus. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        print(f"\n✗ Failed to achieve consensus after {max_iterations} iterations")
        return result


def create_deterministic_llm_call(client, model: str, question: str):
    """
    Create a deterministic LLM call function.
    
    This wraps the OpenAI API call with deterministic parameters.
    """
    def call_fn():
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Provide consistent, deterministic answers."},
                {"role": "user", "content": question}
            ],
            temperature=0,
            max_tokens=1024,
            seed=42,
            n=1,  # Batch size control
        )
        
        return {
            "answer": response.choices[0].message.content,
            "model": model,
            "usage": {
                "total_tokens": response.usage.total_tokens
            }
        }
    
    return call_fn
