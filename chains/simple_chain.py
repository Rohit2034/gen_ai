from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser



load_dotenv()

prompt  = PromptTemplate(
    template="Geneerare 5 interesting facts about {topic}",
    input_variables=["topic"]

)
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)

parser = StructuredOutputParser()
chain = prompt | model | parser 
result = chain.invoke({"topic":"black hole"})

print(result)