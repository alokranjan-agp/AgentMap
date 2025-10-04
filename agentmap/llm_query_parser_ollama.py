"""
Ollama-Powered Query Parser

Uses local Ollama models for free, private LLM inference.
Expected improvement: +5-10% accuracy (47.1% → 52-57%)

Advantages over OpenAI:
- FREE (no API costs)
- LOCAL (privacy)
- FAST (no network latency)
- DETERMINISTIC (temperature=0)

Requirements:
- Install Ollama: https://ollama.ai
- Pull a model: ollama pull llama3.1 (or mistral, codellama, etc.)
"""

import json
import requests
from typing import Dict, List, Any, Optional


class OllamaQueryParser:
    """
    Production query parser using local Ollama.
    
    Supports any Ollama model:
    - llama3.1 (recommended, 8B params)
    - mistral (7B params, fast)
    - codellama (good for structured output)
    - qwen2.5 (excellent for reasoning)
    """
    
    def __init__(
        self, 
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434"
    ):
        """Initialize with Ollama."""
        self.model = model
        self.base_url = base_url
        self.available = self._check_ollama()
        
        if self.available:
            print(f"✅ Ollama initialized (model: {model})")
        else:
            print(f"⚠️  Ollama not available. Install: https://ollama.ai")
            print(f"   Then run: ollama pull {model}")
        
        self.tool_specs = self._load_tool_specifications()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _load_tool_specifications(self) -> str:
        """Load tool specs for prompt."""
        return """
Available WorkBench Tools:

CALENDAR:
- calendar.create_event(event_name, participant_email, event_start, duration)
- calendar.delete_event(event_id)
- calendar.search_events(start_time, end_time)
- calendar.get_event_information_by_id(event_id, field)

EMAIL:
- email.send_email(recipient_email, subject, body)
- email.delete_email(email_id)
- email.search_emails(**filters)

ANALYTICS:
- analytics.create_plot(time_min, time_max, value_to_plot, plot_type)
  * value_to_plot: "total_visits", "session_duration_seconds", "user_engaged"
  * plot_type: "bar", "line", "scatter", "histogram"

CRM:
- crm.create_customer(name, email, phone)
- crm.search_customers(**filters)
- crm.get_customer_information_by_id(customer_id, field)

PROJECT:
- project.create_task(task_name, board, assigned_to, due_date)
- project.search_tasks(**filters)
"""
    
    def parse_query_with_llm(
        self, 
        query: str, 
        domain: str,
        expected_answer: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Parse query using Ollama."""
        
        if not self.available:
            return self._fallback_parse(query, domain)
        
        try:
            prompt = self._construct_prompt(query, domain)
            
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,  # Deterministic
                        "num_predict": 500
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result.get("response", "")
                return self._parse_llm_response(llm_output)
            else:
                print(f"⚠️  Ollama request failed: {response.status_code}")
                return self._fallback_parse(query, domain)
                
        except Exception as e:
            print(f"⚠️  Ollama parsing failed: {e}")
            return self._fallback_parse(query, domain)
    
    def _construct_prompt(self, query: str, domain: str) -> str:
        """Construct prompt for Ollama."""
        
        return f"""Parse this query into tool calls. Return ONLY valid JSON array.

QUERY: "{query}"
DOMAIN: {domain}

{self.tool_specs}

RULES:
1. Return ONLY a JSON array, nothing else
2. Each tool call has "tool" and "params"
3. For multi-step tasks (e.g., "plot both X and Y"), create multiple tool calls
4. Use correct parameter names and formats
5. Dates in YYYY-MM-DD format

EXAMPLES:

Query: "Make a bar chart of total visits since November 21"
Output:
[{{"tool": "analytics.create_plot", "params": {{"time_min": "2023-11-21", "time_max": "2023-11-29", "value_to_plot": "total_visits", "plot_type": "bar"}}}}]

Query: "Plot distribution of total visits and session duration between Nov 15 and Nov 24"
Output:
[{{"tool": "analytics.create_plot", "params": {{"time_min": "2023-11-15", "time_max": "2023-11-24", "value_to_plot": "total_visits", "plot_type": "histogram"}}}}, {{"tool": "analytics.create_plot", "params": {{"time_min": "2023-11-15", "time_max": "2023-11-24", "value_to_plot": "session_duration_seconds", "plot_type": "histogram"}}}}]

Now parse the query above. Return ONLY the JSON array:"""
    
    def _parse_llm_response(self, llm_output: str) -> List[Dict[str, Any]]:
        """Parse Ollama JSON response."""
        try:
            # Clean output
            llm_output = llm_output.strip()
            
            # Remove markdown if present
            if "```json" in llm_output:
                llm_output = llm_output.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output:
                llm_output = llm_output.split("```")[1].split("```")[0].strip()
            
            # Find JSON array
            start = llm_output.find('[')
            end = llm_output.rfind(']') + 1
            
            if start >= 0 and end > start:
                json_str = llm_output[start:end]
                actions = json.loads(json_str)
                
                if not isinstance(actions, list):
                    actions = [actions]
                
                return actions
            
            return []
            
        except Exception as e:
            print(f"⚠️  Failed to parse Ollama response: {e}")
            print(f"   Response: {llm_output[:200]}")
            return []
    
    def _fallback_parse(self, query: str, domain: str) -> List[Dict]:
        """Fallback to heuristic parsing."""
        from agentmap.llm_query_parser import LLMQueryParser
        
        fallback = LLMQueryParser(use_llm=False)
        return fallback.parse_query_with_llm(query, domain)
    
    def parse_multi_domain_query(
        self,
        query: str,
        domains: List[str]
    ) -> List[Dict[str, Any]]:
        """Parse multi-domain queries with Ollama."""
        
        if not self.available:
            return []
        
        try:
            prompt = f"""Parse this multi-domain query. Return ONLY JSON array.

QUERY: "{query}"
DOMAINS: {', '.join(domains)}

{self.tool_specs}

This query needs tools from multiple domains. Return JSON array with all tool calls:"""
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 800}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result.get("response", "")
                return self._parse_llm_response(llm_output)
            
            return []
            
        except Exception as e:
            print(f"⚠️  Multi-domain parsing failed: {e}")
            return []
    
    def extract_parameters_with_context(
        self,
        query: str,
        tool_name: str,
        partial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract missing parameters with Ollama."""
        
        if not self.available:
            return partial_params
        
        try:
            prompt = f"""Extract missing parameters. Return ONLY JSON object.

QUERY: "{query}"
TOOL: {tool_name}
CURRENT PARAMS: {json.dumps(partial_params)}

What parameters are missing? Extract from query.
Return ONLY JSON object with complete parameters:"""
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 300}
                },
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result.get("response", "")
                
                # Extract JSON
                if "```json" in llm_output:
                    llm_output = llm_output.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_output:
                    llm_output = llm_output.split("```")[1].split("```")[0].strip()
                
                # Find JSON object
                start = llm_output.find('{')
                end = llm_output.rfind('}') + 1
                
                if start >= 0 and end > start:
                    json_str = llm_output[start:end]
                    enhanced_params = json.loads(json_str)
                    return {**partial_params, **enhanced_params}
            
            return partial_params
            
        except Exception as e:
            print(f"⚠️  Parameter extraction failed: {e}")
            return partial_params
    
    def validate_and_correct_parameters(
        self,
        tool_name: str,
        params: Dict[str, Any],
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate and correct parameters with Ollama."""
        
        if not self.available or not error_message:
            return params
        
        try:
            prompt = f"""Fix these parameters. Return ONLY JSON object.

TOOL: {tool_name}
PARAMS: {json.dumps(params)}
ERROR: {error_message}

What's wrong? How to fix?
Return ONLY JSON object with corrected parameters:"""
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 300}
                },
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result.get("response", "")
                
                # Extract JSON
                if "```json" in llm_output:
                    llm_output = llm_output.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_output:
                    llm_output = llm_output.split("```")[1].split("```")[0].strip()
                
                start = llm_output.find('{')
                end = llm_output.rfind('}') + 1
                
                if start >= 0 and end > start:
                    json_str = llm_output[start:end]
                    corrected_params = json.loads(json_str)
                    return corrected_params
            
            return params
            
        except Exception as e:
            print(f"⚠️  Parameter correction failed: {e}")
            return params
