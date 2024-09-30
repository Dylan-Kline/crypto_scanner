import asyncio
from collections import deque

class AsyncCappedList:

    def __init__(self, max_len):
        self.deque = deque(max_len=max_len)
        self.lock = asyncio.Lock()

    async def append(self, item):
        async with self.lock:
            self.deque.append(item)

    async def get_list(self):
        async with self.lock:
            return list(self.deque)