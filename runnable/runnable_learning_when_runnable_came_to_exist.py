from abc import ABC,abstractmethod


class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass
import random

class NakliLlm(Runnable):
    def __init__(self):
        print("LLM created")

    def invoke(self,prompt):
        response_list=[
            'Delhi is the capital of India',
            'Mumbai is the financial capital of India',
            'Kolkata is the cultural capital of India',
        ]

        return random.choice(response_list)
    
    def predict(self,prompt):
        response_list=[
            'Delhi is the capital of India',
            'Mumbai is the financial capital of India',
            'Kolkata is the cultural capital of India',
        ]

        return random.choice(response_list)


class Nakliprompt(Runnable):
    def __init__(self,template,input_variables):
        self.template = template
        self.input_variables = input_variables
    def invoke(self,input_dict):
        return self.template.formate(**input_dict)
    def format(self,input_dict):
        return self.template.format(**input_dict)
class NakliStrOutputParser(Runnable):
        def __init__(self):
            pass
        def invoke(self,input_data):
            return input_data['response']

class RunnableConnector(Runnable):
    def __init__(self,runnable_list):
        self.runnable_list = runnable_list
    def invoke(self,input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
            return input_data 
             
new_template = Nakliprompt(
    template='write a {length} poem about {topic}',
    input_variables=['length','topic']
) 

parser =NakliStrOutputParser()
llm = NakliLlm()
chain= RunnableConnector([new_template,llm,parser])

chain.invoke({'length':'long', 'topic':'india'})