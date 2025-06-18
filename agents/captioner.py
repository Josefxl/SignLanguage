import logging
import re
import torch
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Captioner:
    def __init__(self, model_name="google/flan-t5-xl"):
        logger.info(f"Loading model {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )

            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=150,
                num_beams=4,
                device=device
            )

            self.llm = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        prompt_template = """
You are an expert in translating sign language glosses into natural, grammatically correct English sentences. Glosses are concise, often in all caps, with words in a specific order. Your task is to interpret the glosses and generate a single, natural English sentence that conveys the most logical meaning. If the gloss implies a question, use question phrasing. If it implies a statement, use declarative phrasing. Ensure proper capitalization and punctuation. Only output the translated sentence, nothing else.

Examples:
- WHAT FAVOURITE FOOD → What is your favorite food?
- MOTHER DRIVE WHERE → Where is mother driving?
- YESTERDAY WEATHER HOW → How was the weather yesterday?
- BOY FALL DOWN → The boy fell down.
- HOMEWORK FINISH YOU → Did you finish your homework?
- SISTER SCHOOL TODAY → Did your sister go to school today?
- GOOD YOU MORNING → Good morning to you!
- FLY COUNTRY TOMORROW → Are you flying to another country tomorrow?
- FATHER HOME NOT → Father is not at home.
- HELP ME PLEASE → Please help me.
- TRIP HOW WEEK → How was your trip this week?
- BOOK READ YOU → Did you read the book?

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
    def generate_hint(self, phrase):
        basic_hints = {
            "I want to go to the store.": "Use signs like 'I WANT GO STORE'",
            "What is your name?": "Try signing 'YOUR NAME WHAT'",
            "I am hungry.": "Sign 'ME HUNGRY'",
            "Please help me.": "Use 'PLEASE HELP ME'",
            "Where do you live?": "Sign 'YOU LIVE WHERE'",
        }
        return basic_hints.get(phrase, "Break it down to simple signs like SUBJECT + VERB + OBJECT")
