import asyncio
import random
import time
from collections import deque

class GroqRateLimitException(Exception):
    def __init__(self, message, retry_after=1.5, limit_type="RPM"):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after
        self.limit_type = limit_type
        self.headers = {
            "x-ratelimit-limit-requests": "60",
            "x-ratelimit-remaining-requests": "0",
            "retry-after": str(round(retry_after, 2))
        }

class GroqServerException(Exception):
    pass

class CalibratedStandInModel:
    def __init__(
        self,
        rpm=60,
        tpm=8000,
        rpd=100,
        max_in_flight=4,
        base_ttft_ms=180.0,
        tokens_per_sec=250.0,
        overload_penalty_ms=120.0,
        failure_rate=0.03
    ):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.max_in_flight = max_in_flight
        self.base_ttft_ms = base_ttft_ms
        self.tokens_per_sec = tokens_per_sec
        self.overload_penalty_ms = overload_penalty_ms
        self.failure_rate = failure_rate

        self.request_timestamps = deque()
        self.daily_requests_used = 0
        self.in_flight = 0
        self.peak_queue = 0
        self.lock = asyncio.Lock()

    def reset_stats(self):
        self.request_timestamps.clear()
        self.in_flight = 0
        self.peak_queue = 0
        self.daily_requests_used = 0

    async def generate_stream(self, prompt, simulated_answer=None, max_tokens=150):
        async with self.lock:
            now = time.monotonic()
            
            # Prune 60-second sliding window
            while self.request_timestamps and now - self.request_timestamps[0] >= 60.0:
                self.request_timestamps.popleft()

            # Check daily budget limit
            if self.daily_requests_used >= self.rpd:
                raise GroqRateLimitException(
                    "Daily quota exceeded: 100 requests per day limit reached. Budget exhausted.",
                    retry_after=3600.0,
                    limit_type="RPD"
                )

            # Check RPM limit
            if len(self.request_timestamps) >= self.rpm:
                oldest = self.request_timestamps[0]
                retry_after = max(0.5, 60.0 - (now - oldest))
                raise GroqRateLimitException(
                    f"Rate limit reached for RPM: limit {self.rpm}, used {len(self.request_timestamps)}. Please try again in {retry_after:.1f}s.",
                    retry_after=retry_after,
                    limit_type="RPM"
                )

            # Check concurrency limit
            if self.in_flight >= self.max_in_flight:
                raise GroqRateLimitException(
                    f"Concurrency limit reached: {self.in_flight} requests in flight.",
                    retry_after=random.uniform(0.3, 0.8),
                    limit_type="CONCURRENCY"
                )

            self.request_timestamps.append(now)
            self.daily_requests_used += 1
            self.in_flight += 1
            current_active = self.in_flight
            if current_active > self.peak_queue:
                self.peak_queue = current_active

        try:
            # Simulate failure
            if random.random() < self.failure_rate:
                # Failure happens after a brief delay
                await asyncio.sleep(0.08)
                raise GroqServerException("Internal Server Error: model backend timed out.")

            # Calculate TTFT with load penalty
            jitter = random.gauss(0, 25.0)
            overload = max(0, current_active - 1) * self.overload_penalty_ms
            effective_ttft_ms = max(40.0, self.base_ttft_ms + overload + jitter)
            
            # Sleep until first token
            await asyncio.sleep(effective_ttft_ms / 1000.0)
            yield ("__FIRST_TOKEN__", effective_ttft_ms)

            # Generate tokens
            if simulated_answer is None:
                simulated_answer = (
                    "Based on the provided passages, the model applies attention to dynamically "
                    "weight relationships across representations [Attention Is All You Need, p. 4]."
                )
            
            words = simulated_answer.split()
            token_delay = 1.0 / self.tokens_per_sec
            for word in words:
                await asyncio.sleep(token_delay)
                yield (word + " ", None)

        finally:
            async with self.lock:
                self.in_flight -= 1

if __name__ == "__main__":
    async def test():
        model = CalibratedStandInModel(rpm=5, max_in_flight=2)
        print("Testing stand-in streaming...")
        t0 = time.perf_counter()
        async for chunk, ttft in model.generate_stream("Test prompt"):
            if ttft is not None:
                print(f"TTFT received: {ttft:.1f} ms")
            else:
                print(chunk, end="", flush=True)
        print(f"\nTotal elapsed: {(time.perf_counter() - t0)*1000:.1f} ms")
    asyncio.run(test())
