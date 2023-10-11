from dotenv import load_dotenv

from flows.refactor import RefactorBot

load_dotenv()

RefactorBot(None).run()
