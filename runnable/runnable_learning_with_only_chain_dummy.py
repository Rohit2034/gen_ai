import random

class NakliLlm:
    def __init__(self):
        print("LLM created")
    def predict(self,prompt):
        response_list=[
            'Delhi is the capital of India',
            'Mumbai is the financial capital of India',
            'Kolkata is the cultural capital of India',
        ]

        return random.choice(response_list)
llm = NakliLlm()
llm.predict("What is the capital of India?")   



class Nakliprompt:
    def __init__(self,template,input_variables):
        self.template = template
        self.input_variables = input_variables
    def format(self,**input):
        return self.template.format(**input)
    

template = Nakliprompt(
    template="Write a  {length} poem about{country}?",
    input_variables=["length", "country"]
)

print(template.format(length="short", country="India"))


prompt = template.format(length="short", country="India")
llm = NakliLlm()

llm.predict(prompt)


class NakliLlmChain:
    def __init__(self,llm,prompt):
        self.llm = llm
        self.prompt =prompt

    def run(self,input_data):
       final_prompt =  self.prompt.format(**input_data)
       result = self.llm.predict(final_prompt)
       return result['response']

template2 = Nakliprompt(
    template = "what is the {length} of the {country} ?",
    input_variables=["length", "country"]
)
    
llm = NakliLlm()
chain = NakliLlmChain(llm,template)
chain.run({'length':'short','topic':'india'})

