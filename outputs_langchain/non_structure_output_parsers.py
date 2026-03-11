# output parsers in langchain help to convert raw llm responses into structured fromats like json,CSV,pydentic models and more . they ensure consistency , validation and ease of use in applications. In this file we will see how to use output parsers in langchain to convert raw llm responses into structured formats like json,CSV,pydentic models and more . they ensure consistency , validation and ease of use in applications.

# string output parser -: the stringoutput parser is the default output parser in langchain. it simply returns the raw llm response as a string. it is useful when you want to get the raw response from the llm without any processing.




# json output parser -> llm answer to json only problem is we can't enforce schema on the output    
# structured output user-> is an parser in langchain that helps extract srructured json data from llm responses based on predefined schema. it ensures that the output from the llm adheres to the specified structure, making it easier to work with the data in downstream applications. structured output parsers can be defined using pydentic models, typed dicts or json schema. they provide a way to validate and parse the llm responses into a consistent format, which can be especially useful when dealing with complex data or when you want to ensure that the output meets certain criteria.  no data validation is done in json output parser but in structured output parser we can do data validation using pydentic models or json schema. structured output parsers can also handle nested structures and complex data types, making them a powerful tool for extracting meaningful information from llm responses.
# pydentic output parser is a structers parser in langchain that uses pydentic models to define the schema for the output. it allows you to define a pydentic model with fields and their types, and then use that model to parse the llm response. the pydentic output parser will validate the llm response against the defined model and return an instance of the model with the parsed data. this is useful for ensuring that the output from the llm adheres to a specific structure and for easily working with the parsed data in your application.


