
import os
from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

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
        # Remove duplicate links
        if isinstance(links, list):
            links = list(set([meta['links'] for meta in links if 'links' in meta]))
        link_list = "\n".join(links)

        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Aditya, a Campaign Operations Associate from Chennai. 
            Your job is to write a cold email to the client regarding the job mentioned above describing your capability
            in fulfilling their needs.
            Also add your portfolio links: {link_list}
            Do not provide a preamble.

            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": str(job),
            "link_list": link_list
        })
        return res.content


if __name__ == "__main__":
    print(os.getenv("TOGETHER_API_KEY"))
