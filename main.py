import os
import re
import json
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from datetime import datetime
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Slack app
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# Conversation state tracking
# Key: thread_ts, Value: conversation state dict
conversation_states: Dict[str, Dict[str, Any]] = {}

# Users to monitor for mentions (comma-separated user IDs or usernames)
# Example: "U1234567890,U9876543210" or "jundayin,anotheruser"
MONITORED_USERS = os.environ.get("MONITORED_USERS", "").split(",") if os.environ.get("MONITORED_USERS") else []
MONITORED_USERS = [u.strip() for u in MONITORED_USERS if u.strip()]

if not MONITORED_USERS:
    print("⚠️  No MONITORED_USERS configured. Set MONITORED_USERS environment variable (comma-separated user IDs or usernames)")
else:
    print(f"✅ Monitoring mentions for users: {', '.join(MONITORED_USERS)}")

# AI Provider enum
class AIProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"

# Initialize AI clients
openai_client: Optional[OpenAI] = None
claude_client: Optional[Anthropic] = None
current_provider: Optional[AIProvider] = None

# Determine which AI provider to use
if os.environ.get("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    current_provider = AIProvider.OPENAI
    print("✅ Using OpenAI as AI provider")

if os.environ.get("ANTHROPIC_API_KEY"):
    claude_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    if current_provider is None:
        current_provider = AIProvider.CLAUDE
        print("✅ Using Claude as AI provider")
    else:
        print("✅ Claude API key also available (OpenAI is primary)")

if current_provider is None:
    raise ValueError("At least one of OPENAI_API_KEY or ANTHROPIC_API_KEY must be set")


def extract_user_mentions(text: str) -> List[str]:
    """
    Extract user IDs from Slack mention format (e.g., <@U1234567890>).
    
    Args:
        text: Raw Slack message text
        
    Returns:
        List of user IDs mentioned in the message
    """
    mentions = re.findall(r'<@([A-Z0-9]+)>', text)
    return mentions


def clean_slack_message(text: str, remove_mentions: Optional[List[str]] = None) -> str:
    """
    Remove Slack mention prefixes (e.g., <@U1234567890>) from the message.
    
    Args:
        text: Raw Slack message text
        remove_mentions: Optional list of user IDs to remove (if None, removes all)
        
    Returns:
        Cleaned message text without mention prefixes
    """
    if remove_mentions:
        # Remove specific mentions
        for user_id in remove_mentions:
            text = re.sub(rf'<@{user_id}>', '', text)
    else:
        # Remove all user mentions
        text = re.sub(r'<@[A-Z0-9]+>', '', text)
    
    # Remove any extra whitespace and strip
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned


def get_user_id_from_username(username: str, client) -> Optional[str]:
    """
    Get user ID from username.
    
    Args:
        username: Slack username (without @)
        client: Slack client
        
    Returns:
        User ID if found, None otherwise
    """
    try:
        # List all users and find by name
        result = client.users_list()
        if result.get("ok"):
            members = result.get("members", [])
            for member in members:
                member_name = member.get("name", "")
                real_name = member.get("real_name", "").lower()
                display_name = member.get("profile", {}).get("display_name", "").lower()
                username_lower = username.lower()
                
                if (member_name == username or 
                    real_name == username_lower or
                    display_name == username_lower or
                    real_name.startswith(username_lower) or
                    display_name.startswith(username_lower)):
                    return member.get("id")
    except Exception as e:
        print(f"Error looking up user {username}: {e}")
    
    return None


def is_monitored_user_mentioned(event: Dict[str, Any], client) -> Tuple[bool, Optional[str]]:
    """
    Check if any monitored user is mentioned in the message.
    
    Args:
        event: Slack event
        client: Slack client
        
    Returns:
        Tuple of (is_mentioned: bool, mentioned_user_id: Optional[str])
    """
    if not MONITORED_USERS:
        return False, None
    
    text = event.get("text", "")
    mentioned_user_ids = extract_user_mentions(text)
    
    if not mentioned_user_ids:
        return False, None
    
    # Resolve usernames to user IDs if needed (cache this for performance)
    # For now, check directly if mentioned IDs match
    # In production, you might want to cache the username->ID mapping
    monitored_user_ids = []
    for user_identifier in MONITORED_USERS:
        if user_identifier.startswith("U") and len(user_identifier) > 1:
            # Already a user ID
            monitored_user_ids.append(user_identifier)
        else:
            # Try to resolve username to user ID
            user_id = get_user_id_from_username(user_identifier, client)
            if user_id:
                monitored_user_ids.append(user_id)
            else:
                print(f"⚠️  Could not resolve username '{user_identifier}' to user ID. Make sure to use user ID format (U1234567890) in MONITORED_USERS")
    
    # Check if any monitored user is mentioned
    for mentioned_id in mentioned_user_ids:
        if mentioned_id in monitored_user_ids:
            return True, mentioned_id
    
    return False, None


def analyze_intent(text: str) -> Dict[str, Any]:
    """
    Analyze user message intent using the configured AI provider.
    
    Args:
        text: Cleaned message text to analyze
        
    Returns:
        Dictionary containing category, priority, is_actionable, summary, 
        reasoning, suggested_action, and is_investigation
    """
    system_prompt = """You are an intelligent assistant that analyzes Slack messages to categorize user intent and determine appropriate next steps.

Your task is to analyze the incoming message and return a structured JSON response with the following fields:
- category: One of "FEATURE_REQUEST", "TECHNICAL_QUERY", "ADMIN_URGENT", "INVESTIGATION", or "GENERAL_INFO"
- priority: One of "BLOCKER", "CRITICAL", "HIGH", "MEDIUM", "LOW", or "LOWEST" (using Agile priority definitions)
- priority_confidence: A float from 0.0 to 1.0 indicating confidence in the priority assignment (1.0 = very confident, <0.7 = uncertain)
- is_actionable: A boolean indicating if the message requires action
- is_investigation: A boolean indicating if this is an ad-hoc investigation request that requires data retrieval
- summary: A concise English summary of the message (max 100 words)
- reasoning: Your reasoning in English for the categorization and priority assignment (max 200 words)
- suggested_action: One of "INVOKE_MCP", "SEARCH_CODE", "CREATE_JIRA", or "NONE"

Category Guidelines:
- FEATURE_REQUEST: User is requesting a new feature, enhancement, or improvement
- TECHNICAL_QUERY: User has a technical question, needs help debugging, or wants to understand how something works
- ADMIN_URGENT: Administrative tasks requiring immediate attention (e.g., access issues, critical bugs, security concerns)
- INVESTIGATION: Ad-hoc requests to investigate issues, analyze data, check logs, or gather information from external sources
- GENERAL_INFO: General information sharing, casual conversation, or non-actionable messages

Priority Guidelines (Agile Definitions):
- BLOCKER: Blocks development or testing work, cannot proceed without resolution. System is down or completely unusable.
- CRITICAL: Major functionality is broken, significant impact on users or business. Requires immediate attention but work can continue around it.
- HIGH: Important issue that should be addressed soon. Affects functionality but has workarounds or limited impact.
- MEDIUM: Normal priority, standard workflow item. Should be addressed in regular course of work.
- LOW: Minor issue or enhancement, can be deferred. Nice to have but not urgent.
- LOWEST: Very low priority, informational only, or trivial issues that may never be addressed.

Investigation Detection:
- Set is_investigation to true if the message asks to investigate, check, analyze, look into, or gather data about something
- Common patterns: "can you investigate", "check why", "analyze the issue", "look into", "what happened with", "pull data from"

Suggested Action Guidelines:
- INVOKE_MCP: When the message is an investigation request that requires data from GitHub, Azure AppInsights, or ADX
- SEARCH_CODE: When the message relates to understanding or finding code (but not investigation)
- CREATE_JIRA: When the message is a feature request or bug report that should be tracked
- NONE: When no specific action is needed

Return ONLY valid JSON, no additional text or markdown formatting."""

    user_prompt = f"Analyze the following Slack message and provide the structured JSON response:\n\n{text}"

    try:
        if current_provider == AIProvider.OPENAI and openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            result = json.loads(response.choices[0].message.content)
        elif current_provider == AIProvider.CLAUDE and claude_client:
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            # Claude returns text, need to extract JSON
            content = response.content[0].text
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in Claude response")
        else:
            raise ValueError("No AI provider available")
        
        # Validate required fields
        required_fields = ["category", "priority", "priority_confidence", "is_actionable", "is_investigation", "summary", "reasoning", "suggested_action"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        # Ensure priority_confidence is a float between 0 and 1
        if "priority_confidence" in result:
            result["priority_confidence"] = float(result["priority_confidence"])
            if result["priority_confidence"] < 0.0 or result["priority_confidence"] > 1.0:
                result["priority_confidence"] = 0.5  # Default to medium confidence if invalid
        
        # Validate category
        valid_categories = ["FEATURE_REQUEST", "TECHNICAL_QUERY", "ADMIN_URGENT", "INVESTIGATION", "GENERAL_INFO"]
        if result["category"] not in valid_categories:
            raise ValueError(f"Invalid category: {result['category']}")
        
        # Validate priority (Agile labels)
        valid_priorities = ["BLOCKER", "CRITICAL", "HIGH", "MEDIUM", "LOW", "LOWEST"]
        if result["priority"] not in valid_priorities:
            raise ValueError(f"Invalid priority: {result['priority']}. Must be one of {valid_priorities}")
        
        # Validate suggested_action
        valid_actions = ["INVOKE_MCP", "SEARCH_CODE", "CREATE_JIRA", "NONE"]
        if result["suggested_action"] not in valid_actions:
            raise ValueError(f"Invalid suggested_action: {result['suggested_action']}")
        
        return result
        
    except json.JSONDecodeError as e:
        return {
            "category": "GENERAL_INFO",
            "priority": "LOWEST",
            "priority_confidence": 0.5,
            "is_actionable": False,
            "is_investigation": False,
            "summary": "Failed to parse AI response",
            "reasoning": f"JSON parsing error: {str(e)}",
            "suggested_action": "NONE"
        }
    except Exception as e:
        return {
            "category": "GENERAL_INFO",
            "priority": "LOWEST",
            "priority_confidence": 0.5,
            "is_actionable": False,
            "is_investigation": False,
            "summary": "Error during intent analysis",
            "reasoning": f"Error: {str(e)}",
            "suggested_action": "NONE"
        }


def invoke_mcp_servers(investigation_query: str) -> Dict[str, Any]:
    """
    Invoke MCP servers to pull data from GitHub, Azure AppInsights, and ADX.
    
    Args:
        investigation_query: The investigation query from the user
        
    Returns:
        Dictionary containing results from each MCP server
    """
    results = {
        "github": None,
        "appinsights": None,
        "adx": None,
        "error": None
    }
    
    # TODO: Implement actual MCP server client integration
    # This is a placeholder structure for MCP server integration
    # In production, you would:
    # 1. Connect to MCP servers via stdio or HTTP transport
    # 2. Call appropriate tools/resources from each server
    # 3. Aggregate and format the results
    
    try:
        # Placeholder for GitHub MCP server
        # Example: Search repositories, get commit history, check issues/PRs
        github_mcp_url = os.environ.get("GITHUB_MCP_SERVER_URL")
        if github_mcp_url:
            # In production: Use MCP client to call GitHub server
            results["github"] = {
                "status": "connected",
                "data": f"GitHub data for: {investigation_query}",
                "note": "MCP server integration pending"
            }
        
        # Placeholder for Azure AppInsights MCP server
        # Example: Query logs, metrics, traces
        appinsights_mcp_url = os.environ.get("APPINSIGHTS_MCP_SERVER_URL")
        if appinsights_mcp_url:
            # In production: Use MCP client to call AppInsights server
            results["appinsights"] = {
                "status": "connected",
                "data": f"AppInsights data for: {investigation_query}",
                "note": "MCP server integration pending"
            }
        
        # Placeholder for Azure Data Explorer (ADX) MCP server
        # Example: Query KQL, get analytics data
        adx_mcp_url = os.environ.get("ADX_MCP_SERVER_URL")
        if adx_mcp_url:
            # In production: Use MCP client to call ADX server
            results["adx"] = {
                "status": "connected",
                "data": f"ADX data for: {investigation_query}",
                "note": "MCP server integration pending"
            }
        
        if not any([github_mcp_url, appinsights_mcp_url, adx_mcp_url]):
            results["error"] = "No MCP server URLs configured. Set GITHUB_MCP_SERVER_URL, APPINSIGHTS_MCP_SERVER_URL, or ADX_MCP_SERVER_URL"
        
    except Exception as e:
        results["error"] = f"Error invoking MCP servers: {str(e)}"
    
    return results


def generate_response_with_context(
    original_message: str,
    intent_analysis: Dict[str, Any],
    mcp_results: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a comprehensive response using AI with context from intent analysis and MCP results.
    
    Args:
        original_message: The original user message
        intent_analysis: The intent analysis result
        mcp_results: Optional results from MCP servers
        
    Returns:
        Generated response text
    """
    system_prompt = """You are a helpful AI assistant in Slack. Provide clear, concise, and actionable responses.
If you have data from MCP servers (GitHub, AppInsights, ADX), incorporate it into your response.
Be professional but friendly."""

    user_prompt = f"""Original user message: {original_message}

Intent Analysis:
- Category: {intent_analysis.get('category')}
- Priority: {intent_analysis.get('priority')}
- Summary: {intent_analysis.get('summary')}
- Reasoning: {intent_analysis.get('reasoning')}
"""

    if mcp_results:
        user_prompt += f"\nMCP Server Results:\n{json.dumps(mcp_results, indent=2)}\n"
        user_prompt += "\nPlease provide a comprehensive response based on the investigation data above."

    try:
        if current_provider == AIProvider.OPENAI and openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        elif current_provider == AIProvider.CLAUDE and claude_client:
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text
        else:
            return "I apologize, but I'm unable to generate a response at this time."
    except Exception as e:
        return f"I encountered an error while generating a response: {str(e)}"


def can_draw_conclusion(mcp_results: Optional[Dict[str, Any]], investigation_query: str) -> Tuple[bool, str]:
    """
    Determine if enough data has been gathered to draw a conclusion.
    
    Args:
        mcp_results: Results from MCP servers
        investigation_query: The original investigation query
        
    Returns:
        Tuple of (can_conclude: bool, reasoning: str)
    """
    if not mcp_results:
        return False, "No data gathered from MCP servers"
    
    if mcp_results.get("error"):
        return False, f"Error gathering data: {mcp_results['error']}"
    
    # Check if we have meaningful data from at least one source
    has_data = any([
        mcp_results.get("github") and mcp_results["github"].get("status") == "connected",
        mcp_results.get("appinsights") and mcp_results["appinsights"].get("status") == "connected",
        mcp_results.get("adx") and mcp_results["adx"].get("status") == "connected"
    ])
    
    if has_data:
        return True, "Sufficient data gathered from MCP servers to draw conclusion"
    else:
        return False, "Insufficient data gathered from MCP servers"


def check_user_acceptance(conversation_history: List[Dict[str, str]], conclusion: str) -> Tuple[bool, str]:
    """
    Check if the user accepts the assistant's conclusion.
    
    Args:
        conversation_history: List of messages in the thread
        conclusion: The conclusion drawn by the assistant
        
    Returns:
        Tuple of (accepted: bool, reasoning: str)
    """
    if not conversation_history or len(conversation_history) < 2:
        return False, "Not enough conversation history"
    
    system_prompt = """You are analyzing a conversation to determine if the user accepts the assistant's conclusion.
Return ONLY a JSON object with two fields:
- "accepted": boolean indicating if the user accepts the conclusion
- "reasoning": string explaining why

Look for indicators of acceptance: "yes", "correct", "that's right", "agreed", "thanks", "solved", "resolved", etc.
Look for indicators of rejection: "no", "wrong", "that's not right", "disagree", "still an issue", "not resolved", etc."""

    recent_messages = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
    conversation_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent_messages])
    
    user_prompt = f"""Assistant's conclusion: {conclusion}

Recent conversation:
{conversation_text}

Has the user accepted this conclusion?"""

    try:
        if current_provider == AIProvider.OPENAI and openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            result = json.loads(response.choices[0].message.content)
        elif current_provider == AIProvider.CLAUDE and claude_client:
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=512,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            content = response.content[0].text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                return False, "Could not parse response"
        else:
            return False, "No AI provider available"
        
        accepted = result.get("accepted", False)
        reasoning = result.get("reasoning", "No reasoning provided")
        return accepted, reasoning
    except Exception as e:
        print(f"Error checking user acceptance: {e}")
        return False, f"Error: {str(e)}"


def check_if_resolved(conversation_history: List[Dict[str, str]]) -> bool:
    """
    Check if an issue has been resolved based on conversation history.
    
    Args:
        conversation_history: List of messages in the thread
        
    Returns:
        True if the issue appears to be resolved, False otherwise
    """
    if not conversation_history or len(conversation_history) < 2:
        return False
    
    # Use AI to determine if the issue is resolved
    system_prompt = """You are analyzing a conversation thread to determine if an issue has been resolved.
Return ONLY a JSON object with a single field "resolved" (boolean) indicating if the issue is resolved.
Look for indicators like: "fixed", "resolved", "working now", "issue is closed", "thanks", "solved", etc."""

    # Get last few messages for context
    recent_messages = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
    conversation_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent_messages])
    
    user_prompt = f"Analyze this conversation and determine if the issue is resolved:\n\n{conversation_text}"

    try:
        if current_provider == AIProvider.OPENAI and openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            result = json.loads(response.choices[0].message.content)
        elif current_provider == AIProvider.CLAUDE and claude_client:
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=512,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            content = response.content[0].text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                return False
        else:
            return False
        
        return result.get("resolved", False)
    except Exception as e:
        print(f"Error checking resolution status: {e}")
        return False


def format_intent_summary(analysis: Dict[str, Any]) -> str:
    """
    Format the intent analysis result into a readable Slack message.
    
    Args:
        analysis: The intent analysis dictionary
        
    Returns:
        Formatted string for Slack
    """
    priority_emoji = {
        "BLOCKER": "🚫",
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢",
        "LOWEST": "⚪"
    }
    
    action_emoji = {
        "INVOKE_MCP": "🔍",
        "SEARCH_CODE": "🔍",
        "CREATE_JIRA": "📋",
        "NONE": "✅"
    }
    
    priority = analysis.get("priority", "LOWEST")
    emoji = priority_emoji.get(priority, "⚪")
    action_emoji_str = action_emoji.get(analysis["suggested_action"], "✅")
    investigation_badge = "🔬 Investigation" if analysis.get("is_investigation", False) else ""
    
    return f"""*Intent Analysis Result* {investigation_badge}

*Category:* `{analysis["category"]}`
*Priority:* {emoji} {priority}
*Actionable:* {'Yes' if analysis["is_actionable"] else 'No'}
*Suggested Action:* {action_emoji_str} {analysis["suggested_action"]}

*Summary:*
{analysis["summary"]}

*Reasoning:*
{analysis["reasoning"]}"""


@app.event("message")
def handle_message(event: Dict[str, Any], say, client, ack):
    """
    Handle all messages - routes to either:
    1. New messages mentioning monitored users (handle_user_mention logic)
    2. Follow-up messages in threads (handle_thread_followup logic)
    """
    # Acknowledge the event immediately
    ack()
    
    # Skip bot messages
    if event.get("bot_id"):
        return
    
    thread_ts = event.get("thread_ts")
    
    # If it's a thread message and we're tracking it, handle as follow-up
    if thread_ts and thread_ts in conversation_states:
        handle_thread_followup(event, say, client)
        return
    
    # Otherwise, check if it's a new message mentioning a monitored user
    if not thread_ts:  # Only process top-level messages (not thread replies)
        handle_user_mention(event, say, client)


def handle_user_mention(event: Dict[str, Any], say, client):
    """
    Handle messages that mention monitored users.
    The bot listens for mentions of configured users (e.g., @jundayin) and chimes in on their behalf.
    
    Implements the interaction flow:
    1. Check if a monitored user is mentioned
    2. Notify the mentioned user
    3. Analyze intent
    4. Ask about ETA if priority is unclear
    5. Invoke MCP servers for investigations
    6. Generate response
    7. Only ack if conclusion is drawn and accepted
    """
    
    # Check if any monitored user is mentioned
    is_mentioned, mentioned_user_id = is_monitored_user_mentioned(event, client)
    
    if not is_mentioned or not mentioned_user_id:
        return  # No monitored user mentioned, ignore
    
    print(f"Detected mention of monitored user {mentioned_user_id} in message: {event.get('ts')}")
    
    # Get the raw message text
    raw_text = event.get("text", "")
    
    # Clean the message (remove the mention of the monitored user)
    cleaned_text = clean_slack_message(raw_text, remove_mentions=[mentioned_user_id])
    
    # Get the thread timestamp to reply in the same thread
    thread_ts = event.get("ts")
    channel = event.get("channel")
    message_sender = event.get("user")  # Person who sent the message mentioning the user
    
    # If message is empty after cleaning, inform the user
    if not cleaned_text:
        say("I noticed <@{mentioned_user_id}> was mentioned, but the message appears to be empty. Please include your question or request!", thread_ts=thread_ts)
        return
    
    # Initialize conversation state
    conversation_states[thread_ts] = {
        "monitored_user": mentioned_user_id,  # User being monitored/helped
        "message_sender": message_sender,  # Person who sent the message
        "original_message": cleaned_text,
        "channel": channel,
        "thread_ts": thread_ts,
        "state": "analyzing",
        "awaiting_eta": False,
        "conclusion_drawn": False,
        "conclusion_accepted": False,
        "created_at": datetime.now().isoformat()
    }
    
    # Step 1: Notify the monitored user that assistant is here on their behalf
    try:
        user_info = client.users_info(user=mentioned_user_id)
        user_name = user_info.get("user", {}).get("real_name", "there")
    except:
        user_name = "there"
    
    say(f"👋 Hi <@{mentioned_user_id}>! I noticed you were mentioned. I'm here to help on your behalf. Let me analyze this message...", thread_ts=thread_ts)
    
    try:
        # Step 2: Analyze the intent
        analysis = analyze_intent(cleaned_text)
        
        # Step 3: Check if priority is unclear (confidence < 0.7)
        priority_confidence = analysis.get("priority_confidence", 1.0)
        if priority_confidence < 0.7 and analysis.get("is_actionable", False):
            # Ask user about ETA/priority
            conversation_states[thread_ts]["awaiting_eta"] = True
            conversation_states[thread_ts]["state"] = "awaiting_eta"
            
            say(
                f"I've analyzed your message, but I'm not entirely certain about the priority level.\n\n"
                f"*My assessment:* {analysis.get('priority')} (confidence: {priority_confidence:.0%})\n\n"
                f"Could you help me understand the urgency? What's the expected timeline/ETA for this?\n"
                f"This will help me prioritize and determine the best course of action.",
                thread_ts=thread_ts
            )
            # Store analysis for later use
            conversation_states[thread_ts]["analysis"] = analysis
            return
        
        # Post intent analysis
        summary = format_intent_summary(analysis)
        say(summary, thread_ts=thread_ts)
        
        # Step 4: If it's an investigation request, invoke MCP servers
        mcp_results = None
        if analysis.get("is_investigation", False) or analysis.get("suggested_action") == "INVOKE_MCP":
            say("🔍 Gathering data from external sources...", thread_ts=thread_ts)
            mcp_results = invoke_mcp_servers(cleaned_text)
            
            if mcp_results.get("error"):
                say(f"⚠️ MCP Server Error: {mcp_results['error']}", thread_ts=thread_ts)
        
        # Step 5: Generate comprehensive response
        response = generate_response_with_context(
            cleaned_text,
            analysis,
            mcp_results
        )
        
        say(response, thread_ts=thread_ts)
        
        # Step 6: Check if we can draw a conclusion
        can_conclude, conclusion_reason = can_draw_conclusion(mcp_results, cleaned_text)
        
        if can_conclude and analysis.get("is_investigation", False):
            conversation_states[thread_ts]["conclusion_drawn"] = True
            conversation_states[thread_ts]["conclusion"] = response
            conversation_states[thread_ts]["state"] = "awaiting_confirmation"
            
            say(
                f"✅ *Conclusion:* Based on the data gathered, I believe I have enough information to address your request.\n\n"
                f"Please let me know if this conclusion is correct or if you need me to investigate further.",
                thread_ts=thread_ts
            )
        else:
            # Can't draw conclusion or not an investigation - leave open
            conversation_states[thread_ts]["state"] = "open"
            if not can_conclude:
                say("⚠️ I wasn't able to gather enough data to draw a definitive conclusion. The message will remain open for further investigation.", thread_ts=thread_ts)
        
        # Store state for follow-up
        conversation_states[thread_ts]["analysis"] = analysis
        conversation_states[thread_ts]["mcp_results"] = mcp_results
        
    except Exception as e:
        error_msg = f"❌ I encountered an error while processing your request: {str(e)}"
        say(error_msg, thread_ts=thread_ts)
        print(f"Error in handle_mention: {e}")
        conversation_states[thread_ts]["state"] = "error"


def handle_thread_followup(event: Dict[str, Any], say, client):
    """
    Handle follow-up messages in threads where the bot is active.
    Listens for follow-up responses including:
    - ETA/priority clarifications
    - Acceptance/rejection of conclusions
    - General conversation monitoring
    """
    thread_ts = event.get("thread_ts")
    if not thread_ts:
        return  # Not a thread message, ignore
    
    # Check if this is a thread we're tracking
    if thread_ts not in conversation_states:
        return  # Not tracking this thread
    
    channel = event.get("channel")
    user = event.get("user")
    text = event.get("text", "")
    state = conversation_states[thread_ts]
    
    # Get conversation history from the thread
    try:
        result = client.conversations_replies(channel=channel, ts=thread_ts)
        messages = result.get("messages", [])
        
        # Build conversation history
        conversation_history = []
        for msg in messages:
            # Skip bot messages for conversation history
            if msg.get("bot_id"):
                continue
            conversation_history.append({
                "role": "user",
                "content": msg.get("text", "")
            })
        
        # Handle different conversation states
        if state["state"] == "awaiting_eta" and state["awaiting_eta"]:
            # User is responding to ETA question
            # Re-analyze with the ETA information
            original_message = state["original_message"]
            enhanced_message = f"{original_message}\n\nUser provided ETA context: {text}"
            
            say("🤔 Re-analyzing with the timeline information you provided...", thread_ts=thread_ts)
            
            analysis = analyze_intent(enhanced_message)
            summary = format_intent_summary(analysis)
            say(summary, thread_ts=thread_ts)
            
            # Continue with investigation if needed
            mcp_results = None
            if analysis.get("is_investigation", False) or analysis.get("suggested_action") == "INVOKE_MCP":
                say("🔍 Gathering data from external sources...", thread_ts=thread_ts)
                mcp_results = invoke_mcp_servers(enhanced_message)
                
                if mcp_results.get("error"):
                    say(f"⚠️ MCP Server Error: {mcp_results['error']}", thread_ts=thread_ts)
            
            response = generate_response_with_context(
                enhanced_message,
                analysis,
                mcp_results
            )
            
            say(response, thread_ts=thread_ts)
            
            # Check if conclusion can be drawn
            can_conclude, _ = can_draw_conclusion(mcp_results, enhanced_message)
            if can_conclude and analysis.get("is_investigation", False):
                state["conclusion_drawn"] = True
                state["conclusion"] = response
                state["state"] = "awaiting_confirmation"
                say(
                    f"✅ *Conclusion:* Based on the data gathered, I believe I have enough information to address your request.\n\n"
                    f"Please let me know if this conclusion is correct or if you need me to investigate further.",
                    thread_ts=thread_ts
                )
            else:
                state["state"] = "open"
            
            state["awaiting_eta"] = False
            state["analysis"] = analysis
            state["mcp_results"] = mcp_results
        
        elif state["state"] == "awaiting_confirmation" and state["conclusion_drawn"]:
            # Check if user accepts the conclusion
            accepted, reasoning = check_user_acceptance(conversation_history, state.get("conclusion", ""))
            
            if accepted:
                # User accepts - acknowledge the original message
                state["conclusion_accepted"] = True
                state["state"] = "resolved"
                
                try:
                    # Add reaction to acknowledge (no notification)
                    client.reactions_add(
                        channel=channel,
                        timestamp=thread_ts,
                        name="white_check_mark"
                    )
                    say("✅ Great! I've acknowledged the original message. The issue is resolved.", thread_ts=thread_ts)
                except Exception as e:
                    print(f"Error adding reaction: {e}")
                    say("✅ Great! The issue is resolved.", thread_ts=thread_ts)
            else:
                # User doesn't accept - leave open
                state["state"] = "open"
                say("I understand. I'll leave this open for further investigation. Let me know if you need anything else.", thread_ts=thread_ts)
        
        else:
            # General conversation monitoring - check if issue is resolved naturally
            if len(conversation_history) >= 2:
                is_resolved = check_if_resolved(conversation_history)
                
                if is_resolved and state["state"] != "resolved":
                    # Issue resolved naturally - acknowledge
                    state["state"] = "resolved"
                    try:
                        client.reactions_add(
                            channel=channel,
                            timestamp=thread_ts,
                            name="white_check_mark"
                        )
                        say("✅ Issue appears to be resolved. I've acknowledged the original message.", thread_ts=thread_ts)
                    except Exception as e:
                        print(f"Error adding reaction: {e}")
                        say("✅ Issue appears to be resolved.", thread_ts=thread_ts)
    
    except Exception as e:
        print(f"Error in handle_message: {e}")


def main():
    """
    Main entry point for the Slack bot.
    """
    # Verify required environment variables
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please set them in your .env file or environment."
        )
    
    # Verify at least one AI provider is configured
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError(
            "At least one of OPENAI_API_KEY or ANTHROPIC_API_KEY must be set"
        )
    
    # Start the Socket Mode handler
    handler = SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
    print("🤖 Slack Assistant is running in Socket Mode...")
    print(f"   AI Provider: {current_provider.value if current_provider else 'None'}")
    handler.start()


if __name__ == "__main__":
    main()
