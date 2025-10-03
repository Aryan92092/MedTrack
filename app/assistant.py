import os
import json
import pandas as pd
from datetime import datetime
from flask import current_app
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import re

class AIAssistant:
    def __init__(self):
        # Initialize Gemini API (supports GEMINI_API_KEY or GOOGLE_API_KEY)
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        # Fallback: read from instance/gemini.key if env not set
        if not api_key:
            try:
                instance_key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'instance', 'gemini.key')
                if os.path.exists(instance_key_path):
                    with open(instance_key_path, 'r', encoding='utf-8') as f:
                        api_key = f.read().strip()
            except Exception:
                api_key = None
        if api_key:
            # Standard Gemini configuration
            genai.configure(api_key=api_key)
            # Allow override; default to a broadly available model
            model_name = (os.getenv('GEMINI_MODEL') or 'gemini-1.5-flash').strip()
            # Normalize accidental resource prefix
            if model_name.startswith('models/'):
                model_name = model_name.split('/', 1)[1]
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None
        self.conversation_history = {}
        
    def get_user_history(self, user_id: str) -> List[Dict]:
        """Get conversation history for a user"""
        return self.conversation_history.get(user_id, [])
    
    def add_to_history(self, user_id: str, role: str, content: str):
        """Add message to user's conversation history"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 20 messages to manage memory
        if len(self.conversation_history[user_id]) > 20:
            self.conversation_history[user_id] = self.conversation_history[user_id][-20:]
    
    def analyze_csv_data(self, csv_content: str, filename: str = "uploaded_file.csv") -> Dict[str, Any]:
        """Analyze CSV data and provide insights"""
        try:
            # Read CSV from string
            df = pd.read_csv(pd.StringIO(csv_content))
            
            analysis = {
                'filename': filename,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'basic_stats': {},
                'sample_data': df.head(5).to_dict('records'),
                'insights': []
            }
            
            # Generate basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                analysis['basic_stats'] = df[numeric_cols].describe().to_dict()
            
            # Generate insights based on data
            insights = []
            
            # Check for medicine-related data
            if any(col.lower() in ['name', 'medicine', 'drug'] for col in df.columns):
                insights.append("This appears to be medicine/inventory data")
                
            if 'expiry_date' in df.columns or 'expiry' in df.columns:
                insights.append("Contains expiry date information - important for inventory management")
                
            if 'quantity' in df.columns or 'stock' in df.columns:
                insights.append("Contains quantity/stock information")
                
            # Check for missing values
            missing_cols = [col for col, count in analysis['missing_values'].items() if count > 0]
            if missing_cols:
                insights.append(f"Missing values found in: {', '.join(missing_cols)}")
            
            # Check for duplicates
            if df.duplicated().any():
                insights.append("Duplicate rows detected")
            
            analysis['insights'] = insights
            
            return analysis
            
        except Exception as e:
            return {
                'error': f"Error analyzing CSV: {str(e)}",
                'filename': filename
            }
    
    def get_stock_insights(self, question: str) -> str:
        """Provide insights about stock/inventory management"""
        stock_keywords = {
            'inventory': 'Inventory management involves tracking stock levels, monitoring expiry dates, and optimizing reorder points.',
            'expiry': 'Expiry management is crucial for pharmaceutical inventory. Monitor items expiring within 30-90 days and implement FIFO (First In, First Out) rotation.',
            'reorder': 'Set minimum stock levels based on consumption patterns. Consider lead times and safety stock to prevent stockouts.',
            'analytics': 'Use consumption analytics to identify trends, seasonal patterns, and optimize inventory levels.',
            'cost': 'Track purchase costs, selling prices, and profit margins. Monitor total inventory value and identify high-value items.',
            'alerts': 'Set up automated alerts for low stock, expiring items, and unusual consumption patterns.',
            'fifo': 'FIFO (First In, First Out) ensures older stock is used first, reducing waste and maintaining product quality.',
            'safety': 'Maintain safety stock levels to handle unexpected demand spikes or supply delays.',
            'turnover': 'Inventory turnover ratio indicates how quickly stock is sold. Higher turnover generally means better efficiency.'
        }
        
        question_lower = question.lower()
        for keyword, insight in stock_keywords.items():
            if keyword in question_lower:
                return insight
        
        return "I can help with inventory management, stock optimization, expiry tracking, and analytics. What specific aspect would you like to know about?"
    
    def _format_messages_for_gemini(self, messages: List[Dict]) -> str:
        """Convert OpenAI-style messages to Gemini prompt format"""
        prompt_parts = []
        
        for message in messages:
            role = message['role']
            content = message['content']
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"
    
    def generate_response(self, user_id: str, message: str, csv_data: Optional[str] = None, csv_filename: Optional[str] = None) -> str:
        """Generate AI response based on user message and context"""
        try:
            # Add user message to history
            self.add_to_history(user_id, 'user', message)
            
            # Get conversation history
            history = self.get_user_history(user_id)
            
            # Prepare system message
            system_message = """You are an AI assistant for MediTrack, a medicine inventory management system. 
            You help users with:
            1. Stock and inventory management questions
            2. CSV data analysis and insights
            3. Medicine inventory optimization
            4. Expiry date management
            5. Analytics and reporting
            
            Be helpful, concise, and professional. Focus on practical advice for inventory management."""
            
            # Prepare messages for OpenAI
            messages = [{"role": "system", "content": system_message}]
            
            # Add conversation history
            for msg in history[-10:]:  # Last 10 messages
                messages.append({"role": msg['role'], "content": msg['content']})
            
            # If CSV data is provided, analyze it first
            csv_analysis = None
            if csv_data:
                csv_analysis = self.analyze_csv_data(csv_data, csv_filename or "uploaded_file.csv")
                
                # Add CSV context to the conversation
                csv_context = f"User has uploaded a CSV file with {csv_analysis.get('shape', 'unknown')} rows and columns: {', '.join(csv_analysis.get('columns', []))}. "
                if csv_analysis.get('insights'):
                    csv_context += f"Key insights: {'; '.join(csv_analysis['insights'])}. "
                
                messages.append({"role": "assistant", "content": csv_context})
            
            # Check if it's a stock-related question
            if any(keyword in message.lower() for keyword in ['stock', 'inventory', 'expiry', 'reorder', 'analytics', 'cost', 'turnover']):
                stock_insight = self.get_stock_insights(message)
                messages.append({"role": "assistant", "content": stock_insight})
            
            # Generate response using Gemini
            if self.model is None:
                return "Gemini API key not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY and restart the app."
            
            # Convert messages to Gemini format
            prompt = self._format_messages_for_gemini(messages)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.7
                )
            )
            
            ai_response = response.text
            
            # Add AI response to history
            self.add_to_history(user_id, 'assistant', ai_response)
            
            return ai_response
            
        except Exception as e:
            # Fallback response if Gemini fails
            error_response = f"I apologize, but I'm having trouble processing your request right now. Error: {str(e)}"
            self.add_to_history(user_id, 'assistant', error_response)
            return error_response
    
    def get_quick_answers(self) -> Dict[str, str]:
        """Get quick answer templates for common questions"""
        return {
            "How to manage expiring medicines?": "Monitor expiry dates regularly, implement FIFO rotation, set alerts for items expiring within 30-90 days, and consider discounting near-expiry items.",
            "What is optimal stock level?": "Optimal stock level = (Average daily consumption × Lead time) + Safety stock. Monitor consumption patterns and adjust based on seasonal trends.",
            "How to reduce inventory costs?": "Implement just-in-time ordering, negotiate better supplier terms, reduce waste through better expiry management, and optimize reorder points.",
            "What analytics are important?": "Track consumption trends, inventory turnover, expiry rates, profit margins, and identify fast/slow-moving items for better decision making.",
            "How to handle low stock alerts?": "Set minimum stock levels based on consumption patterns, maintain supplier relationships for quick reorders, and consider safety stock for critical items."
        }

# Global instance
ai_assistant = AIAssistant()
