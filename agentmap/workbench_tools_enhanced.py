"""
Enhanced WorkBench Tools with Real Integration

This version integrates with the actual WorkBench tool implementations
to achieve better accuracy on the benchmark.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# Add WorkBench to path
WORKBENCH_PATH = Path("/Users/alok/Downloads/WorkBench")
sys.path.insert(0, str(WORKBENCH_PATH))


class EnhancedWorkBenchToolkit:
    """
    Enhanced toolkit that uses actual WorkBench tool implementations.
    This should achieve much better accuracy than the basic version.
    """
    
    def __init__(self):
        """Initialize with actual WorkBench tools."""
        self.workbench_path = WORKBENCH_PATH
        self.tools_available = False
        
        # Try to load WorkBench tools
        try:
            # Change to WorkBench directory for data loading
            original_cwd = os.getcwd()
            os.chdir(self.workbench_path)
            
            # Import tools (they load data from relative paths)
            from src.tools import calendar
            from src.tools import email
            from src.tools import analytics
            from src.tools import customer_relationship_manager as crm
            from src.tools import project_management as project
            
            # Store tool modules
            self.calendar = calendar
            self.email = email
            self.analytics = analytics
            self.crm = crm
            self.project = project
            
            self.tools_available = True
            print("✅ Successfully loaded WorkBench tools")
            
        except Exception as e:
            print(f"⚠️  Could not load WorkBench tools: {e}")
            self.tools_available = False
            
        finally:
            # Change back to original directory
            os.chdir(original_cwd)
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool using actual WorkBench implementations.
        
        Args:
            tool_name: Tool name (e.g., 'calendar.delete_event')
            **kwargs: Tool parameters
            
        Returns:
            Execution result
        """
        if not self.tools_available:
            return {
                'success': False,
                'tool': tool_name,
                'error': 'WorkBench tools not available'
            }
        
        try:
            # Execute in WorkBench directory context
            original_cwd = os.getcwd()
            os.chdir(self.workbench_path)
            
            try:
                result = self._execute_workbench_tool(tool_name, **kwargs)
                
                return {
                    'success': True,
                    'tool': tool_name,
                    'result': result
                }
            finally:
                os.chdir(original_cwd)
            
        except Exception as e:
            return {
                'success': False,
                'tool': tool_name,
                'error': str(e)
            }
    
    def _execute_workbench_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute actual WorkBench tool."""
        
        # Calendar tools
        if tool_name == 'calendar.delete_event':
            return self.calendar.delete_event.func(event_id=kwargs.get('event_id'))
        elif tool_name == 'calendar.create_event':
            return self.calendar.create_event.func(
                event_name=kwargs.get('event_name') or kwargs.get('name'),
                participant_email=kwargs.get('participant_email') or kwargs.get('participant'),
                event_start=kwargs.get('event_start') or kwargs.get('start_time'),
                duration=kwargs.get('duration')
            )
        elif tool_name == 'calendar.update_event':
            return self.calendar.update_event.func(
                event_id=kwargs.get('event_id'),
                **kwargs
            )
        elif tool_name == 'calendar.search_events':
            return self.calendar.search_events.func(
                start_time=kwargs.get('start_time'),
                end_time=kwargs.get('end_time')
            )
        elif tool_name == 'calendar.get_event_information_by_id':
            return self.calendar.get_event_information_by_id.func(
                event_id=kwargs.get('event_id'),
                field=kwargs.get('field')
            )
        
        # Email tools
        elif tool_name == 'email.send' or tool_name == 'email.send_email':
            return self.email.send_email.func(
                recipient_email=kwargs.get('recipient_email') or kwargs.get('recipient'),
                subject=kwargs.get('subject'),
                body=kwargs.get('body')
            )
        elif tool_name == 'email.delete' or tool_name == 'email.delete_email':
            return self.email.delete_email.func(
                email_id=kwargs.get('email_id')
            )
        elif tool_name == 'email.search' or tool_name == 'email.search_emails':
            return self.email.search_emails.func(**kwargs)
        elif tool_name == 'email.get_email_by_id':
            return self.email.get_email_by_id.func(
                email_id=kwargs.get('email_id'),
                field=kwargs.get('field')
            )
        
        # Analytics tools
        elif tool_name == 'analytics.create_plot':
            return self.analytics.create_plot.func(
                time_min=kwargs.get('time_min'),
                time_max=kwargs.get('time_max'),
                value_to_plot=kwargs.get('value_to_plot'),
                plot_type=kwargs.get('plot_type')
            )
        elif tool_name == 'analytics.get_visitor_information_by_id':
            return self.analytics.get_visitor_information_by_id.func(
                visitor_id=kwargs.get('visitor_id')
            )
        elif tool_name == 'analytics.get_page_views':
            return self.analytics.get_page_views.func(
                date_range=kwargs.get('date_range')
            )
        elif tool_name == 'analytics.compare_periods':
            return self.analytics.compare_periods.func(
                period1=kwargs.get('period1'),
                period2=kwargs.get('period2')
            )
        
        # CRM tools
        elif tool_name == 'crm.create_customer':
            return self.crm.create_customer.func(
                name=kwargs.get('name') or kwargs.get('customer_name'),
                email=kwargs.get('email') or kwargs.get('customer_email'),
                phone=kwargs.get('phone') or kwargs.get('customer_phone')
            )
        elif tool_name == 'crm.search_customers':
            return self.crm.search_customers.func(**kwargs)
        elif tool_name == 'crm.get_customer_information_by_id':
            return self.crm.get_customer_information_by_id.func(
                customer_id=kwargs.get('customer_id'),
                field=kwargs.get('field')
            )
        elif tool_name == 'crm.update_customer':
            return self.crm.update_customer.func(
                customer_id=kwargs.get('customer_id'),
                **kwargs
            )
        elif tool_name == 'crm.delete_customer':
            return self.crm.delete_customer.func(
                customer_id=kwargs.get('customer_id')
            )
        
        # Project tools
        elif tool_name == 'project.create_task':
            return self.project.create_task.func(
                task_name=kwargs.get('task_name') or kwargs.get('name'),
                board=kwargs.get('board'),
                assigned_to=kwargs.get('assigned_to'),
                due_date=kwargs.get('due_date')
            )
        elif tool_name == 'project.search_tasks':
            return self.project.search_tasks.func(**kwargs)
        elif tool_name == 'project.get_task_information_by_id':
            return self.project.get_task_information_by_id.func(
                task_id=kwargs.get('task_id'),
                field=kwargs.get('field')
            )
        elif tool_name == 'project.update_task':
            return self.project.update_task.func(
                task_id=kwargs.get('task_id'),
                **kwargs
            )
        elif tool_name == 'project.delete_task':
            return self.project.delete_task.func(
                task_id=kwargs.get('task_id')
            )
        
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def get_available_tools(self) -> List[str]:
        """Get list of all available tools."""
        return [
            'calendar.search_events',
            'calendar.create_event',
            'calendar.update_event',
            'calendar.delete_event',
            'calendar.check_availability',
            'calendar.find_free_slot',
            'email.search',
            'email.send',
            'email.reply',
            'email.forward',
            'email.delete',
            'analytics.get_page_views',
            'analytics.get_visitor_count',
            'analytics.get_traffic_sources',
            'analytics.get_engagement',
            'analytics.compare_periods',
            'crm.search_customers',
            'crm.create_customer',
            'crm.update_customer',
            'crm.get_activity',
            'crm.schedule_followup',
            'project.search_tasks',
            'project.create_task',
            'project.update_task',
            'project.move_task',
            'project.assign_task',
        ]
