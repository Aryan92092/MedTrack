# AI Assistant Setup Guide

## Overview
The AI Assistant has been successfully integrated into your MediTrack application! It provides intelligent help with inventory management, stock optimization, and CSV data analysis.

## Features
- **Stock & Inventory Management**: Get expert advice on inventory optimization, expiry management, and stock levels
- **CSV Data Analysis**: Upload CSV files and get intelligent insights about your data
- **Quick Questions**: Pre-built answers for common inventory management questions
- **Conversation History**: Maintains context across chat sessions
- **Responsive Design**: Works on desktop and mobile devices

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Gemini API Key
You need a Google Gemini API key to use the AI assistant:

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an account or sign in with your Google account
3. Click "Create API Key" to generate a new key
4. Copy the generated API key
5. Set the environment variable (either works):

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
# or
$env:GOOGLE_API_KEY="your-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your-api-key-here
rem or
set GOOGLE_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="your-api-key-here"
# or
export GOOGLE_API_KEY="your-api-key-here"
```

### 3. Run the Application
```bash
python app.py
```

## Usage

### Accessing the AI Assistant
- Look for the green AI assistant button in the bottom-right corner of any page
- Click the button to open the chat interface
- The assistant is available on all pages of the application

### Chat Features
- **Send Messages**: Type your question and press Enter or click Send
- **Quick Questions**: Use the pre-built question buttons for common queries
- **CSV Analysis**: Click the CSV upload button to analyze your data files
- **Clear Chat**: Use the clear button to reset conversation history

### Sample Questions
- "How do I manage expiring medicines?"
- "What's the optimal stock level for my inventory?"
- "How can I reduce inventory costs?"
- "What analytics should I track?"
- "Upload this CSV and analyze my inventory data"

### CSV Analysis
1. Click the CSV upload button in the chat
2. Select a CSV file from your computer
3. Ask questions about the data:
   - "What insights can you provide about this data?"
   - "Are there any missing values or issues?"
   - "What recommendations do you have for inventory management?"

## Technical Details

### Files Added/Modified
- `app/assistant.py` - AI assistant backend logic (using Gemini API)
- `app/main/routes.py` - Added AI chat routes
- `app/templates/ai_assistant.html` - Chat widget HTML
- `app/templates/base.html` - Integrated AI widget
- `app/static/styles.css` - AI widget styling
- `app/static/ai_assistant.js` - Chat functionality
- `requirements.txt` - Added Google Generative AI dependency

### API Endpoints
- `POST /ai/chat` - Send chat message
- `POST /ai/chat-with-csv` - Send message with CSV context
- `GET /ai/quick-answers` - Get quick answer templates
- `GET /ai/history` - Get chat history
- `POST /ai/clear-history` - Clear chat history

### Security Notes
- All AI routes require user authentication
- Conversation history is stored in memory (resets on server restart)
- CSV data is processed in memory and not stored permanently
- Gemini API calls are made server-side for security

## Troubleshooting

### Common Issues

1. **"API Key Issue" or "API key expired"**
   - **Root Cause**: Your Gemini API key has expired or is invalid
   - **Solution**: 
     - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
     - Generate a new API key
     - Update your environment variable: `$env:GEMINI_API_KEY="your-new-api-key"`
     - Restart the application
   - **Prevention**: Set up API key rotation or monitor usage in Google AI Studio

2. **"Usage Limit Reached"**
   - **Root Cause**: You've exceeded the daily quota for Gemini API
   - **Solution**: Wait until the next day or upgrade your API plan
   - **Check**: Visit Google AI Studio dashboard to see usage statistics

3. **"AI Assistant not responding" (Generic)**
   - Check if GEMINI_API_KEY is set correctly
   - Verify your Gemini API key is valid and has quota
   - Check browser console for JavaScript errors
   - Ensure you're logged in to the application

4. **"CSV upload not working"**
   - Ensure the file is a valid CSV format
   - Check file size (should be reasonable for processing)
   - Verify the CSV has proper headers

5. **"Styling issues"**
   - Clear browser cache
   - Check if all CSS files are loading properly
   - Verify the AI widget is included in base.html

### Getting Help
- Check the browser console for error messages
- Verify all dependencies are installed
- Ensure the Flask application is running properly
- Test with simple questions first before complex queries

## Customization

### Modifying Quick Answers
Edit the `get_quick_answers()` method in `app/assistant.py` to customize the quick question buttons.

### Changing AI Model
By default the app uses `gemini-1.5-flash`.
To change it, set `GEMINI_MODEL` in your environment or `.env`, for example:
```
GEMINI_MODEL=models/gemini-2.5-pro
```

### Styling Customization
Modify the CSS classes in `app/static/styles.css` starting from line 1402 to customize the appearance.

## Cost Considerations
- Gemini API usage is charged per token (generous free tier available)
- Gemini Pro is cost-effective for most use cases
- Monitor your Google AI Studio usage dashboard for costs
- Consider implementing rate limiting for production use

Enjoy your new AI assistant! 🚀
