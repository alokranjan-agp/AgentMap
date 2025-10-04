"""
Real LLM-Powered Query Parser

Uses actual OpenAI API for natural language understanding.
Expected improvement: +8-15% accuracy (47.1% → 55-62%)

This is the PRODUCTION version with real LLM calls.
"""

import json
import os
from typing import Dict, List, Any, Optional
import openai


class RealLLMQueryParser:
    """
    Production query parser using actual OpenAI API.
    
    This addresses:
    - Complex query understanding (~20% of failures)
    - Multi-domain coordination (~25% of failures)
    - Parameter extraction (~20% of failures)
    
    Total potential: +15-20% accuracy improvement
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """Initialize with OpenAI API."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                print(f"✅ OpenAI API initialized (model: {model})")
            except Exception as e:
                print(f"⚠️  OpenAI initialization failed: {e}")
                self.client = None
        else:
            print("⚠️  No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        
        self.tool_specs = self._load_tool_specifications()
    
    def _load_tool_specifications(self) -> str:
        """Load complete tool specifications for LLM."""
        return """
Available WorkBench Tools:

CALENDAR TOOLS:
1. calendar.create_event(event_name, participant_email, event_start, duration)
   - Create a new calendar event
   - event_start format: "YYYY-MM-DD HH:MM:SS"
   - duration in minutes

2. calendar.delete_event(event_id)
   - Delete a calendar event by ID
   - event_id: 8-digit string

3. calendar.update_event(event_id, **kwargs)
   - Update event fields

4. calendar.search_events(start_time, end_time)
   - Search events in time range

5. calendar.get_event_information_by_id(event_id, field)
   - Get specific event information

EMAIL TOOLS:
1. email.send_email(recipient_email, subject, body)
   - Send an email

2. email.delete_email(email_id)
   - Delete an email by ID

3. email.search_emails(**filters)
   - Search emails with filters

4. email.get_email_by_id(email_id, field)
   - Get specific email information

ANALYTICS TOOLS:
1. analytics.create_plot(time_min, time_max, value_to_plot, plot_type)
   - Create a plot/chart
   - time_min/max format: "YYYY-MM-DD"
   - value_to_plot: "total_visits", "session_duration_seconds", "user_engaged"
   - plot_type: "bar", "line", "scatter", "histogram"

2. analytics.get_visitor_information_by_id(visitor_id)
   - Get visitor analytics data

CRM TOOLS:
1. crm.create_customer(name, email, phone)
   - Create a new customer

2. crm.search_customers(**filters)
   - Search customers

3. crm.get_customer_information_by_id(customer_id, field)
   - Get customer information

4. crm.update_customer(customer_id, **kwargs)
   - Update customer information

5. crm.delete_customer(customer_id)
   - Delete a customer

PROJECT MANAGEMENT TOOLS:
1. project.create_task(task_name, board, assigned_to, due_date)
   - Create a new task

2. project.search_tasks(**filters)
   - Search tasks

3. project.get_task_information_by_id(task_id, field)
   - Get task information

4. project.update_task(task_id, **kwargs)
   - Update task information

5. project.delete_task(task_id)
   - Delete a task
"""
    
    def parse_query_with_llm(
        self, 
        query: str, 
        domain: str,
        expected_answer: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Use real LLM to parse query into tool calls.
        
        Args:
            query: Natural language query
            domain: Task domain (analytics, calendar, etc.)
            expected_answer: Optional ground truth for learning
            
        Returns:
            List of tool calls with parameters
        """
        
        if not self.client:
            # Fallback to heuristic parsing
            return self._fallback_parse(query, domain)
        
        try:
            # Construct prompt for LLM
            prompt = self._construct_parsing_prompt(query, domain, expected_answer)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at parsing natural language queries into structured tool calls for workplace automation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic
                max_tokens=500
            )
            
            # Parse LLM response
            llm_output = response.choices[0].message.content
            actions = self._parse_llm_response(llm_output)
            
            return actions
            
        except Exception as e:
            print(f"⚠️  LLM parsing failed: {e}")
            return self._fallback_parse(query, domain)
    
    def _construct_parsing_prompt(
        self, 
        query: str, 
        domain: str,
        expected_answer: Optional[str] = None
    ) -> str:
        """Construct prompt for LLM parsing."""
        
        prompt = f"""Parse this natural language query into structured tool calls.

QUERY: "{query}"
DOMAIN: {domain}

{self.tool_specs}

INSTRUCTIONS:
1. Identify which tool(s) to call
2. Extract all required parameters from the query
3. Handle multi-step tasks (e.g., "plot both X and Y" → two separate plots)
4. Use correct parameter names and formats
5. Return as JSON array

OUTPUT FORMAT:
[
  {{
    "tool": "tool.name",
    "params": {{
      "param1": "value1",
      "param2": "value2"
    }}
  }}
]

EXAMPLES:

Query: "Can you make a bar chart of total visits since November 21?"
Output: [
  {{
    "tool": "analytics.create_plot",
    "params": {{
      "time_min": "2023-11-21",
      "time_max": "2023-11-29",
      "value_to_plot": "total_visits",
      "plot_type": "bar"
    }}
  }}
]

Query: "Please plot for me the distribution of total visits and average session duration between November 15 and November 24"
Output: [
  {{
    "tool": "analytics.create_plot",
    "params": {{
      "time_min": "2023-11-15",
      "time_max": "2023-11-24",
      "value_to_plot": "total_visits",
      "plot_type": "histogram"
    }}
  }},
  {{
    "tool": "analytics.create_plot",
    "params": {{
      "time_min": "2023-11-15",
      "time_max": "2023-11-24",
      "value_to_plot": "session_duration_seconds",
      "plot_type": "histogram"
    }}
  }}
]

Now parse the query above and return ONLY the JSON array:"""
        
        return prompt
    
    def _parse_llm_response(self, llm_output: str) -> List[Dict[str, Any]]:
        """Parse LLM JSON response into actions."""
        try:
            # Extract JSON from response
            llm_output = llm_output.strip()
            
            # Handle markdown code blocks
            if "```json" in llm_output:
                llm_output = llm_output.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output:
                llm_output = llm_output.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            actions = json.loads(llm_output)
            
            if not isinstance(actions, list):
                actions = [actions]
            
            return actions
            
        except Exception as e:
            print(f"⚠️  Failed to parse LLM response: {e}")
            print(f"   Response: {llm_output[:200]}")
            return []
    
    def _fallback_parse(self, query: str, domain: str) -> List[Dict]:
        """Fallback heuristic parsing when LLM unavailable."""
        # Import the heuristic parser
        from agentmap.llm_query_parser import LLMQueryParser
        
        fallback = LLMQueryParser(use_llm=False)
        return fallback.parse_query_with_llm(query, domain)
    
    def parse_multi_domain_query(
        self,
        query: str,
        domains: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Parse multi-domain queries (addresses 25% of failures).
        
        Example: "Send email to customer and create calendar event"
        """
        
        if not self.client:
            return []
        
        try:
            prompt = f"""Parse this multi-domain query into tool calls across multiple domains.

QUERY: "{query}"
DOMAINS: {', '.join(domains)}

{self.tool_specs}

This query may require tools from multiple domains. Identify ALL necessary tool calls and their sequence.

Return as JSON array with proper ordering:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at coordinating multi-domain workplace tasks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=800
            )
            
            llm_output = response.choices[0].message.content
            return self._parse_llm_response(llm_output)
            
        except Exception as e:
            print(f"⚠️  Multi-domain parsing failed: {e}")
            return []
    
    def extract_parameters_with_context(
        self,
        query: str,
        tool_name: str,
        partial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to extract missing parameters with context understanding.
        
        This addresses parameter mismatch failures (~20%).
        """
        
        if not self.client:
            return partial_params
        
        try:
            prompt = f"""Extract missing parameters for this tool call.

QUERY: "{query}"
TOOL: {tool_name}
CURRENT PARAMETERS: {json.dumps(partial_params, indent=2)}

What parameters are missing? Extract them from the query.
Return ONLY a JSON object with the complete parameters:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured data from natural language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            
            llm_output = response.choices[0].message.content
            
            # Parse response
            if "```json" in llm_output:
                llm_output = llm_output.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output:
                llm_output = llm_output.split("```")[1].split("```")[0].strip()
            
            enhanced_params = json.loads(llm_output)
            
            # Merge with partial params
            return {**partial_params, **enhanced_params}
            
        except Exception as e:
            print(f"⚠️  Parameter extraction failed: {e}")
            return partial_params
    
    def validate_and_correct_parameters(
        self,
        tool_name: str,
        params: Dict[str, Any],
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to validate and correct parameters.
        
        This helps with edge cases and format issues.
        """
        
        if not self.client or not error_message:
            return params
        
        try:
            prompt = f"""Fix the parameters for this tool call.

TOOL: {tool_name}
CURRENT PARAMETERS: {json.dumps(params, indent=2)}
ERROR: {error_message}

What's wrong with the parameters? How should they be corrected?
Return ONLY a JSON object with the corrected parameters:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at debugging and fixing API calls."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            
            llm_output = response.choices[0].message.content
            
            # Parse response
            if "```json" in llm_output:
                llm_output = llm_output.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output:
                llm_output = llm_output.split("```")[1].split("```")[0].strip()
            
            corrected_params = json.loads(llm_output)
            return corrected_params
            
        except Exception as e:
            print(f"⚠️  Parameter correction failed: {e}")
            return params
