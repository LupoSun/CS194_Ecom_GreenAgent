# Rate Limit Crash Fix

## Problem Summary

When the OpenAI API rate limit was hit during evaluation, the entire evaluation would crash. This was happening because:

### Root Cause

1. **White Agent** (`real_white_agent/my_white_agent.py`, line 271-282):
   - When a rate limit error occurred, the white agent caught the exception
   - Sent an error message back: `"OpenAI API Error: {e}"`
   - **Immediately returned**, terminating its execution

2. **Green Agent** (`green_agent_demo/green_main_A2A.py`, line 764-809):
   - Received the error message as normal text
   - Printed: `"Turn 1: OpenAI API Error: ..."`
   - Didn't find the `##READY_FOR_CHECKOUT##` signal
   - Tried to continue the conversation by sending another message
   - **But the white agent had already terminated**, causing the next message to fail and crash

## Solution Implemented

### White Agent Changes

Added intelligent retry logic with exponential backoff:

1. **Rate Limit Detection**: Detect rate limit errors (429 errors) vs other errors
2. **Automatic Retry**: Retry up to 3 times with exponential backoff
3. **Wait Time Parsing**: Extract the recommended wait time from the error message (e.g., "try again in 1.312s")
4. **Graceful Failure**: If max retries exceeded, send the `##READY_FOR_CHECKOUT##` completion signal so the green agent doesn't hang
5. **Silent Retries**: Don't send intermediate retry messages to avoid confusing the conversation flow

### Green Agent Changes

Added better error detection:

1. **Import asyncio**: Added `asyncio` to imports for potential async operations
2. **Error Awareness**: Detect when completion signal is sent after an error
3. **Warning Logs**: Log warnings when white agent completes with errors, indicating cart may be incomplete

## Code Changes

### real_white_agent/my_white_agent.py

```python
# Added retry logic in the main chat loop
max_retries = 3
retry_count = 0

while retry_count < max_retries:
    try:
        completion = self.client.chat.completions.create(...)
        break  # Success
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            # Extract wait time and retry
            wait_time = extract_from_error_or_exponential_backoff()
            await asyncio.sleep(wait_time)
            retry_count += 1
        else:
            # Non-rate-limit error - send completion signal and exit
            await event_queue.enqueue_event(COMPLETION_SIGNAL)
            return
```

### green_agent_demo/green_main_A2A.py

```python
# Added import
import asyncio

# Enhanced error detection in interaction loop
if COMPLETION_SIGNAL in response_text:
    if "OpenAI API Error" in response_text:
        print("⚠️ Warning: White agent completed with errors. Cart may be incomplete.")
    completion_received = True
    break
```

## Benefits

1. **Resilience**: Automatically handles transient rate limit errors
2. **No Hangs**: Always sends completion signal, even on failure
3. **Better Logging**: Clear indication when errors occur
4. **Exponential Backoff**: Respects rate limits by waiting appropriately
5. **Graceful Degradation**: Returns partial results rather than crashing

## Testing

To test the fix:

1. Run an evaluation with a rate-limited OpenAI API key
2. Observe the retry messages in logs: `"[MyWhiteAgent] Rate limit hit (attempt 1/3). Waiting 1.3s..."`
3. Verify the evaluation completes (possibly with empty/partial cart) instead of crashing
4. Check green agent logs for warning: `"⚠️ Warning: White agent completed with errors"`

## Future Improvements

1. Make max_retries configurable via environment variable
2. Add exponential backoff multiplier configuration
3. Implement circuit breaker pattern for repeated failures
4. Add metrics/telemetry for retry rates
5. Consider using alternative rate limit handling strategies (e.g., token bucket)

