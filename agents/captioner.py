import logging
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import LLMChain

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Captioner:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.4)

        prompt_template = """
You are an expert in translating sign language glosses into natural, grammatically correct English sentences. Glosses are concise, often in all caps, with words in a specific order. Your task is to interpret the glosses and generate a single, natural English sentence that conveys the most logical meaning. If the gloss implies a question, use question phrasing. If it implies a statement, use declarative phrasing. Ensure proper capitalization and punctuation. Only output the translated sentence, nothing else.

Examples:
- WHAT FAVOURITE FOOD → What is your favorite food?
- ME WANT FOOD → I want some food.
- MOTHER DRIVE WHERE → Where is mother driving?
- YESTERDAY WEATHER HOW → How was the weather yesterday?
- HELP ME PLEASE → Please help me.

Now translate the following gloss into a natural English sentence:
{glosses}
→"""
        prompt = PromptTemplate.from_template(prompt_template)
        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def preprocess_gloss(self, glosses: str) -> str:
        return glosses.strip().upper()

    def postprocess_caption(self, caption: str) -> str:
        caption = caption.strip()
        caption = caption.split('\n')[0].split('-')[0].strip()
        caption = re.sub(r"→.*$", "", caption).strip()
        if caption:
            caption = caption[0].upper() + caption[1:]
        if caption and caption[-1] not in ".!?":
            caption += "." if not caption.endswith("?") else ""
        return caption

    def caption(self, glosses: str) -> str:
        try:
            glosses = self.preprocess_gloss(glosses)
            logger.info(f"Processing gloss: {glosses}")
            raw_caption = self.chain.run({"glosses": glosses})
            caption = self.postprocess_caption(raw_caption)
            logger.info(f"Generated caption: {caption}")
            return caption
        except Exception as e:
            logger.error(f"Error processing gloss '{glosses}': {e}")
            return "Translation failed."
