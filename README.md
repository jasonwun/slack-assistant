# Slack AI Assistant

An intelligent Slack bot that reads messages mentioning the user, understands and categorizes intent, and can invoke MCP servers to pull data from GitHub, Azure AppInsights, and ADX for investigation requests.

## Features

1. **Message Monitoring**: Automatically reads messages that mention the bot
2. **Intent Analysis**: Understands and categorizes user intent using AI
3. **MCP Server Integration**: Invokes MCP servers for ad-hoc investigation requests to pull data from:
   - GitHub repositories
   - Azure Application Insights
   - Azure Data Explorer (ADX)
4. **Threaded Conversations**: All replies are kept within the same thread as the original message
5. **Auto-Acknowledgment**: Automatically acknowledges messages on behalf of the user when issues are confirmed resolved
6. **Multi-AI Support**: Supports both OpenAI and Claude API keys
7. **Concurrent Processing**: Handles multiple threads simultaneously using async processing

## Setup

### Prerequisites

- Python 3.14 or higher
- Slack workspace with bot app created
- At least one AI API key (OpenAI or Anthropic Claude)

### Installation

1. Install dependencies:
```bash
uv sync
# or
pip install -e .
```

2. Create a `.env` file in the project root:
```env
# Required: Slack tokens
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token

# Required: At least one AI provider
OPENAI_API_KEY=sk-your-openai-key
# OR
ANTHROPIC_API_KEY=sk-ant-your-claude-key

# Required: Users to monitor for mentions (comma-separated)
# Use Slack user IDs (U1234567890) or usernames (jundayin)
MONITORED_USERS=U1234567890,jundayin

# Optional: MCP Server URLs (for investigation requests)
GITHUB_MCP_SERVER_URL=http://localhost:8001
APPINSIGHTS_MCP_SERVER_URL=http://localhost:8002
ADX_MCP_SERVER_URL=http://localhost:8003
```

### Slack App Configuration

1. Create a new Slack app at https://api.slack.com/apps
2. Enable Socket Mode
3. Add the following OAuth scopes:
   - `channels:history` - To read channel messages
   - `groups:history` - To read private channel messages (if needed)
   - `im:history` - To read direct messages (if needed)
   - `mpim:history` - To read group direct messages (if needed)
   - `chat:write` - To post messages
   - `reactions:write` - To add reactions (for acknowledgments)
   - `channels:read` - To read channel information
   - `users:read` - To read user information (for username resolution)
4. Subscribe to the `message.channels` event (and `message.groups`, `message.im`, `message.mpim` if needed)
5. Install the app to your workspace
6. Copy the Bot Token and App Token to your `.env` file
7. Find the user ID(s) you want to monitor:
   - Right-click on a user in Slack → "View profile" → Copy the user ID from the URL
   - Or use the format `U1234567890` (starts with U followed by alphanumeric characters)
   - Add to `MONITORED_USERS` in your `.env` file

## Usage

Run the bot:
```bash
python main.py
```

The bot will:
- Listen for messages mentioning configured users (e.g., @jundayin)
- Chime in on behalf of the mentioned user
- Analyze the intent of each message
- For investigation requests, invoke MCP servers to gather data
- Respond in the same thread
- Monitor conversations and acknowledge when issues are resolved

**Note:** The bot monitors mentions of users specified in `MONITORED_USERS`, not mentions of the bot itself. When someone mentions a monitored user (e.g., "@jundayin can you investigate this?"), the bot will automatically chime in to help.

## Intent Categories

- **FEATURE_REQUEST**: User is requesting a new feature or enhancement
- **TECHNICAL_QUERY**: Technical questions or debugging requests
- **ADMIN_URGENT**: Administrative tasks requiring immediate attention
- **INVESTIGATION**: Ad-hoc requests to investigate issues or gather data
- **GENERAL_INFO**: General information sharing or casual conversation

## MCP Server Integration

The bot includes a structure for MCP (Model Context Protocol) server integration. To fully implement:

1. Set up MCP servers for GitHub, Azure AppInsights, and ADX
2. Configure the server URLs in your `.env` file
3. Update the `invoke_mcp_servers()` function in `main.py` to use an MCP client library

The current implementation includes placeholder logic that can be extended with actual MCP client calls.

## Architecture

- **Async Processing**: All handlers are async to support concurrent thread processing
- **AI Provider Abstraction**: Supports both OpenAI and Claude with automatic fallback
- **Thread Management**: Maintains conversation context within Slack threads
- **Resolution Detection**: Uses AI to detect when issues are resolved and acknowledges automatically

## Development

The codebase is structured to be easily extensible:
- Add new intent categories in `analyze_intent()`
- Extend MCP server integration in `invoke_mcp_servers()`
- Customize resolution detection in `check_if_resolved()`

## License

MIT

