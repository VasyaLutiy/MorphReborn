from dotenv import load_dotenv

from flows.morph import MorphBot

load_dotenv()

MorphBot(None).run()
