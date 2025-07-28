import os
from dotenv import load_dotenv

# ✅ Load env variables early
load_dotenv()

# ✅ Set USER_AGENT fallback if not present
os.environ.setdefault("USER_AGENT", "coldemailgen/1.0")
print("USER_AGENT:", os.getenv("USER_AGENT"))

# ✅ Hugging Face login (if token is available)
if os.getenv("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.getenv("HF_TOKEN"))
    print("🔐 Hugging Face login successful.")

# ✅ Import LLM-related stuff after env setup
from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


class Chain:
    def __init__(self):
        self.llm = ChatTogether(
            model="meta-llama/Llama-3-70b-chat-hf",
            together_api_key=os.getenv("TOGETHER_API_KEY")
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The following text has been scraped from a website's careers page. 
            Your task is to identify each job posting and organize it into JSON format with these fields:

            - `role`: Job title or position.
            - `experience`: Required years or type of experience.
            - `skills`: Key skills or qualifications needed.
            - `description`: Brief summary of job responsibilities and expectations.

            Only return valid JSON, without any additional text or explanations. If a specific field is missing for a job posting, exclude that field from the JSON for that entry.

            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})

        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")

        return res if isinstance(res, list) else [res]
    
    def write_mail(self, job, links):
        # Normalize links to list of cleaned strings
        clean_links = set()
        if isinstance(links, list):
            for link in links:
                if isinstance(link, dict) and "links" in link:
                    clean_links.add(link["links"].strip())
                elif isinstance(link, str):
                    clean_links.add(link.strip())

        link_list = list(clean_links)[0] if clean_links else ""

        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Aditya, a Campaign Operations Associate based in Chennai, writing a professional cold email for the above job opportunity.

            Structure the email as follows:

            - Subject line (short and relevant)
            - A warm and professional greeting
            - A clear and concise self-introduction
            - Relevant skills and experience aligned to the job description
            - Mention of the portfolio link: {link_list}
            - A polite and enthusiastic closing line

            The tone should be professional yet human, confident but humble. Use short paragraphs and natural language. Avoid sounding robotic or overly formal. DO NOT include any explanations or markdown, just plain text email.

            ### FORMATTED EMAIL:
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": str(job),
            "link_list": link_list
        })

        return res.content


if __name__ == "__main__":
    print("TOGETHER_API_KEY:", os.getenv("TOGETHER_API_KEY"))
