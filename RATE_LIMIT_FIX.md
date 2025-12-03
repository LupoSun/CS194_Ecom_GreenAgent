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

Added intelligent error handling and user-skipping in benchmark mode:

1. **Import asyncio**: Added `asyncio` to imports for async sleep operations
2. **Error Detection**: Detect when completion signal is sent after an error
3. **Smart Skip Logic**: In benchmark mode, when an OpenAI API error is detected:
   - Pause for 1 second (to respect rate limits)
   - Skip to the next user (don't waste time on failed users)
   - Track skipped users and reasons
4. **Enhanced Reporting**: Benchmark summary now shows:
   - Number of users tested successfully
   - Number of users skipped due to errors
   - Details of which users were skipped and why

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

# Enhanced error detection - raises exception to skip user
if COMPLETION_SIGNAL in response_text:
    if "OpenAI API Error" in response_text or "Error after" in response_text:
        print("⚠️ Warning: White agent completed with errors. Skipping user.")
        raise ValueError(f"White agent failed: {error_snippet}")
    completion_received = True
    break

# In benchmark loop - smart skip logic
skipped_users = []
try:
    # ... run assessment ...
except Exception as e:
    error_msg = str(e)
    if "OpenAI API Error" in error_msg or "White agent failed" in error_msg:
        print(f"⚠️ OpenAI API Error detected: {error_msg[:100]}")
        print(f"Pausing 1 second and skipping to next user...")
        skipped_users.append({"user_id": user_id, "reason": error_msg[:100]})
        await asyncio.sleep(1)
    continue
```

## What Happens Now

### Scenario 1: Rate limit with successful retry
```
[MyWhiteAgent] Rate limit hit (attempt 1/3). Waiting 1.3s...
[MyWhiteAgent] Successful retry, continuing...
[Green Agent] ✅ Completion signal received
[Green Agent] F1=0.723
```

### Scenario 2: Rate limit exceeded in benchmark mode
```
[MyWhiteAgent] OpenAI API Error after 3 retries: Rate limit exceeded
[Green Agent] ⚠️ Warning: White agent completed with errors. Skipping user.
[Green Agent] ⚠️ OpenAI API Error detected: White agent failed...
[Green Agent] Pausing 1 second and skipping to next user...
[Green Agent] [2/10] User 54321
```

### Scenario 3: Benchmark completion with skipped users
```
Benchmark Complete ✅ (Mode: WHITE AGENT)

Tested 8 users successfully
Skipped 2 users due to errors

Average Metrics:
- F1 Score: 0.687
- Precision: 0.712
- Recall: 0.665
- Blended F1: 0.701

Skipped users:
  User 12345: White agent failed: OpenAI API Error after 3 retries
  User 67890: White agent failed: Rate limit exceeded
```

## Benefits

1. **Resilience**: Automatically handles transient rate limit errors with retry logic
2. **No Hangs**: Always sends completion signal, even on failure
3. **Better Logging**: Clear indication when errors occur
4. **Exponential Backoff**: Respects rate limits by waiting appropriately
5. **Smart Skipping**: In benchmark mode, skips failed users and continues with others
6. **Transparent Reporting**: Shows which users were skipped and why
7. **Graceful Degradation**: Continues evaluation rather than crashing

## Testing

To test the fix:

1. **Run a benchmark evaluation** with a rate-limited OpenAI API key
2. **Observe retry logic**: Look for messages like `"[MyWhiteAgent] Rate limit hit (attempt 1/3). Waiting 1.3s..."`
3. **Verify skip behavior**: When max retries exceeded, check for:
   - `"⚠️ OpenAI API Error detected: ..."`
   - `"Pausing 1 second and skipping to next user..."`
   - Benchmark continues with next user
4. **Check final summary**: 
   - Shows "Tested X users successfully"
   - Shows "Skipped Y users due to errors"
   - Lists which users were skipped and why
5. **Verify no crashes**: Benchmark completes even with multiple rate limit errors

## Future Improvements

1. Make max_retries configurable via environment variable
2. Add exponential backoff multiplier configuration
3. Implement circuit breaker pattern for repeated failures
4. Add metrics/telemetry for retry rates
5. Consider using alternative rate limit handling strategies (e.g., token bucket)

