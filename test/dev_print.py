import asyncio 
async def agen():

    for i in range(5):
        yield i
        # if i == 3:
        #     raise ValueError("Exception!")

async def awaitable_to_coro(aw):
    return await aw

async def main():
    aaitter = aiter(agen())
    while True:
        i = await anext(aaitter)
        print(i)

if __name__ == "__main__":
    asyncio.run(main())