# White Agent - OpenAI-Powered Shopping Assistant

White agent that uses GPT-4 to predict grocery shopping baskets based on user purchase history. Communicates via the A2A protocol and interacts with the e-commerce API to search products and build shopping carts.

## Overview

This white agent is evaluated by the green agent. It receives user shopping history, analyzes patterns using GPT-4, searches for products via the e-commerce API, and builds a predicted shopping basket.

## Prerequisites

- Python 3.10+
- OpenAI API key (required)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Running Locally

Start the white agent server:
```bash
bash run.sh
```

Or run directly:
```bash
export ROLE=white
python my_white_agent.py
```

The agent will start on `http://0.0.0.0:9002` by default.

### Configuration

Set these environment variables to customize behavior:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `HOST`: Host to bind to (default: `0.0.0.0`)
- `AGENT_PORT`: Port to run on (default: `9002`)

## How It Works

1. **Receives prompt** from green agent with user shopping history
2. **Analyzes patterns** using GPT-4 to identify frequently purchased items
3. **Searches products** via the e-commerce API
4. **Builds cart** by adding predicted items
5. **Signals completion** with `##READY_FOR_CHECKOUT##`

### Agent Strategy

Strategies to improve predictions:

- **Pattern recognition**: Identifies frequently reordered items
- **Category awareness**: Considers product departments and aisles
- **Smart search**: Searches for high-confidence items first
- **API efficiency**: Batches cart additions when possible

## Testing

### Test with Green Agent

1. Start the white agent (this directory):
```bash
bash run.sh
```

2. Start the green agent (in a separate terminal):
```bash
cd ../green_agent
bash run.sh
```

3. Send a test request:
```bash
cd ../green_agent
python quick_test.py
```

### Run Unit Tests

```bash
python -m pytest tests/ -v
```

### Quick Manual Test

```bash
python test_my_agent.py
```

## API Endpoints

The white agent exposes standard A2A endpoints:

- `GET /.well-known/agent-card.json` - Agent metadata
- `POST /` - JSON-RPC message endpoint

## Deployment

### Using Docker

Build and run with Docker:
```bash
docker build -t my-white-agent .
docker run -p 9002:9002 -e OPENAI_API_KEY=$OPENAI_API_KEY my-white-agent
```

### Cloud Deployment

Deploy to Railway, Google Cloud Run, or other platforms:
1. Set the `OPENAI_API_KEY` environment variable
2. Configure the correct port (default: 9002)
3. Expose the service publicly or to the green agent

## Project Structure

```
white_agent/
├── my_white_agent.py           # Main white agent implementation
├── run.sh                      # Startup script
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── test_my_agent.py            # Manual test script
└── tests/
    └── test_white_agent_integration.py  # Integration tests
```

## Troubleshooting

### OpenAI API Key Not Found
Make sure you've exported your API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### Connection Refused
Ensure the agent is running and the port (9002) is accessible.

### Low F1 Scores
- Check that the agent is correctly parsing the user history
- Verify API search results are returning relevant products
- Review GPT-4 reasoning in the logs

## Contributing

See the main repository README for contribution guidelines.

## License

See main repository for license information.
