from typing import Optional, List
from groq import Groq                     # 🔁 CHANGED (was OpenAI)
from agents.deals import ScrapedDeal, DealSelection
from agents.agent import Agent
import os
import json


class ScannerAgent(Agent):
    MODEL = "llama-3.1-8b-instant"        # 🔁 CHANGED MODEL

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.

Respond strictly in JSON with no explanation using EXACTLY this format:

{
  "deals": [
    {
      "product_description": "string",
      "price": number,
      "url": "string"
    }
  ]
}

You must include the fields:
- product_description
- price
- url

Do not use title.
Do not omit url.

If the price of a deal isn't clear, do not include that deal in your response.
Most important is that you respond with the 5 deals that have the most detailed product description with price.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" — this is not the actual price.
"""


    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
    You should rephrase the description to be a summary of the product itself, not the terms of the deal.
    Remember to respond with a short paragraph of text in the product_description field for each of the 5 items that you select.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    
    Deals:
    
    """

    USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        """
        Set up this instance by initializing Groq
        """
        self.log("Scanner Agent is initializing")
        self.openai = Groq(api_key=os.getenv("GROQ_API_KEY"))   # 🔁 CHANGED CLIENT
        self.log("Scanner Agent is ready")

    def fetch_deals(self, memory) -> List[ScrapedDeal]:
        self.log("Scanner Agent is about to fetch deals from RSS feed")
        urls = [opp.deal.url for opp in memory]
        scraped = ScrapedDeal.fetch()
        result = [scrape for scrape in scraped if scrape.url not in urls]
        self.log(f"Scanner Agent received {len(result)} deals not already scraped")
        return result

    def make_user_prompt(self, scraped) -> str:
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += "\n\n".join([scrape.describe() for scrape in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

# --- implementation/scanner_agent.py ---

    def scan(self, memory: List[str] = []) -> Optional[DealSelection]:
        scraped = self.fetch_deals(memory)
        if scraped:
            user_prompt = self.make_user_prompt(scraped)
            self.log("Scanner Agent is calling Groq")

            result = self.openai.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            # 1. Get raw content
            content = result.choices[0].message.content.strip()

            # 2. 🛠️ CLEAN MARKDOWN: Groq/Llama often wraps JSON in ```json ... ```
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            try:
                # 3. Parse JSON
                data = json.loads(content)

                # 4. 🛠️ SCHEMA HEALING: Handle if LLM returned a dict or a raw list
                if isinstance(data, dict) and "deals" in data:
                    deals_list = data["deals"]
                elif isinstance(data, list):
                    deals_list = data
                else:
                    deals_list = []

                # 5. Validate through Pydantic
                selection = DealSelection(deals=deals_list)
                selection.deals = [deal for deal in selection.deals if deal.price > 0]

                self.log(f"Scanner Agent received {len(selection.deals)} selected deals")
                return selection

            except (json.JSONDecodeError, Exception) as e:
                self.log(f"Scanner Agent failed to parse response: {str(e)}")
                # Log a snippet of the content to see what went wrong
                self.log(f"Raw content snippet: {content[:100]}...")
                return None
        return None

    def test_scan(self, memory: List[str] = []) -> Optional[DealSelection]:
        results = {
            "deals": [
                {
                    "product_description": "The Hisense R6 Series 55R6030N is a 55-inch 4K UHD Roku Smart TV...",
                    "price": 178,
                    "url": "https://www.dealnews.com/products/Hisense/Hisense-R6-Series-55-R6030-N-55-4-K-UHD-Roku-Smart-TV/484824.html?iref=rss-c142",
                },
                {
                    "product_description": "The Poly Studio P21 is a 21.5-inch LED personal meeting display...",
                    "price": 30,
                    "url": "https://www.dealnews.com/products/Poly-Studio-P21-21-5-1080-p-LED-Personal-Meeting-Display/378335.html?iref=rss-c39",
                },
                {
                    "product_description": "The Lenovo IdeaPad Slim 5 laptop is powered by a 7th generation AMD Ryzen 5...",
                    "price": 446,
                    "url": "https://www.dealnews.com/products/Lenovo/Lenovo-Idea-Pad-Slim-5-7-th-Gen-Ryzen-5-16-Touch-Laptop/485068.html?iref=rss-c39",
                },
                {
                    "product_description": "The Dell G15 gaming laptop is equipped with a 6th-generation AMD Ryzen 5...",
                    "price": 650,
                    "url": "https://www.dealnews.com/products/Dell/Dell-G15-Ryzen-5-15-6-Gaming-Laptop-w-Nvidia-RTX-3050/485067.html?iref=rss-c39",
                },
            ]
        }
        return DealSelection(**results)
