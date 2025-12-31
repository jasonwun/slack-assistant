import os
import re
import json
from typing import Dict, Any
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Slack app
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def clean_slack_message(text: str) -> str:
    """
    Remove Slack mention prefixes (e.g., <@U1234567890>) from the message.
    
    Args:
        text: Raw Slack message text
        
    Returns:
        Cleaned message text without mention prefixes
    """
    # Remove Slack user mentions in format <@USER_ID>
    cleaned = re.sub(r'<@[A-Z0-9]+>', '', text)
    # Remove any extra whitespace and strip
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def analyze_intent(text: str) -> Dict[str, Any]:
    """
    Analyze user message intent using OpenAI GPT-4o.
    
    Args:
        text: Cleaned message text to analyze
        
    Returns:
        Dictionary containing category, priority, is_actionable, summary, 
        reasoning, and suggested_action
    """
    system_prompt = """You are an intelligent assistant that analyzes Slack messages to categorize user intent and determine appropriate next steps.

Your task is to analyze the incoming message and return a structured JSON response with the following fields:
- category: One of "FEATURE_REQUEST", "TECHNICAL_QUERY", "ADMIN_URGENT", or "GENERAL_INFO"
- priority: An integer from 1 (lowest) to 5 (highest)
- is_actionable: A boolean indicating if the message requires action
- summary: A concise English summary of the message (max 100 words)
- reasoning: Your reasoning in English for the categorization and priority assignment (max 200 words)
- suggested_action: One of "SEARCH_CODE", "CREATE_JIRA", or "NONE"

Category Guidelines:
- FEATURE_REQUEST: User is requesting a new feature, enhancement, or improvement
- TECHNICAL_QUERY: User has a technical question, needs help debugging, or wants to understand how something works
- ADMIN_URGENT: Administrative tasks requiring immediate attention (e.g., access issues, critical bugs, security concerns)
- GENERAL_INFO: General information sharing, casual conversation, or non-actionable messages

Priority Guidelines:
- 5: Critical/urgent issues requiring immediate attention
- 4: High priority items that should be addressed soon
- 3: Medium priority, normal workflow items
- 2: Low priority, can be deferred
- 1: Very low priority, informational only

Suggested Action Guidelines:
- SEARCH_CODE: When the message relates to understanding or finding code
- CREATE_JIRA: When the message is a feature request or bug report that should be tracked
- NONE: When no specific action is needed

Return ONLY valid JSON, no additional text or markdown formatting."""

    user_prompt = f"Analyze the following Slack message and provide the structured JSON response:\n\n{text}"

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3  # Lower temperature for more consistent categorization
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        
        # Validate required fields
        required_fields = ["category", "priority", "is_actionable", "summary", "reasoning", "suggested_action"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate category
        valid_categories = ["FEATURE_REQUEST", "TECHNICAL_QUERY", "ADMIN_URGENT", "GENERAL_INFO"]
        if result["category"] not in valid_categories:
            raise ValueError(f"Invalid category: {result['category']}")
        
        # Validate priority range
        if not isinstance(result["priority"], int) or result["priority"] < 1 or result["priority"] > 5:
            raise ValueError(f"Invalid priority: {result['priority']}")
        
        # Validate suggested_action
        valid_actions = ["SEARCH_CODE", "CREATE_JIRA", "NONE"]
        if result["suggested_action"] not in valid_actions:
            raise ValueError(f"Invalid suggested_action: {result['suggested_action']}")
        
        return result
        
    except json.JSONDecodeError as e:
        return {
            "category": "GENERAL_INFO",
            "priority": 1,
            "is_actionable": False,
            "summary": "Failed to parse AI response",
            "reasoning": f"JSON parsing error: {str(e)}",
            "suggested_action": "NONE"
        }
    except Exception as e:
        return {
            "category": "GENERAL_INFO",
            "priority": 1,
            "is_actionable": False,
            "summary": "Error during intent analysis",
            "reasoning": f"Error: {str(e)}",
            "suggested_action": "NONE"
        }


def format_intent_summary(analysis: Dict[str, Any]) -> str:
    """
    Format the intent analysis result into a readable Slack message.
    
    Args:
        analysis: The intent analysis dictionary
        
    Returns:
        Formatted string for Slack
    """
    priority_emoji = {
        5: "🔴",
        4: "🟠",
        3: "🟡",
        2: "🟢",
        1: "⚪"
    }
    
    action_emoji = {
        "SEARCH_CODE": "🔍",
        "CREATE_JIRA": "📋",
        "NONE": "✅"
    }
    
    emoji = priority_emoji.get(analysis["priority"], "⚪")
    action_emoji_str = action_emoji.get(analysis["suggested_action"], "✅")
    
    return f"""*Intent Analysis Result*

*Category:* `{analysis["category"]}`
*Priority:* {emoji} {analysis["priority"]}/5
*Actionable:* {'Yes' if analysis["is_actionable"] else 'No'}
*Suggested Action:* {action_emoji_str} {analysis["suggested_action"]}

*Summary:*
{analysis["summary"]}

*Reasoning:*
{analysis["reasoning"]}

*Raw JSON:*
```json
{json.dumps(analysis, indent=2)}
```"""


@app.event("app_mention")
def handle_mention(event, say):
    """
    Handle @mention events in Slack.
    Cleans the message, analyzes intent, and posts the result.
    """
    # Get the raw message text
    raw_text = event.get("text", "")
    
    # Clean the message (remove mention prefix)
    cleaned_text = clean_slack_message(raw_text)
    
    # If message is empty after cleaning, inform the user
    if not cleaned_text:
        say("I received your mention, but the message appears to be empty. Please include your question or request!")
        return
    
    # Analyze the intent
    analysis = analyze_intent(cleaned_text)
    
    # Format and post the result
    summary = format_intent_summary(analysis)
    say(summary)


@app.event("message")
def handle_message_events(body, logger):
    """
    Handle other message events (optional, for logging).
    """
    logger.info(body)


def main():
    """
    Main entry point for the Slack bot.
    """
    # Verify required environment variables
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please set them in your .env file or environment."
        )
    
    # Start the Socket Mode handler
    handler = SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
    print("🤖 GhostAssistant is running in Socket Mode...")
    handler.start()


if __name__ == "__main__":
    main()
