from __future__ import annotations
import anthropic
import instructor
import json
import openai
import os
import pandas as pd
import pydantic
import typing

class dimension(pydantic.BaseModel):
    dimension_name: str
    description: str
    children: list[dimension]

class response(pydantic.BaseModel):
    dimensions: list[dimension]


class generate_hierarchy:

    # initialize class variables
    def __init__(
            self,
            top_level:str,
            max_iterations:int,
            client:typing.Literal['openai', 'anthropic'],
            top_levels_separate:bool = False,
            ) -> None:
        """Initialize the hierarchy generation class.
        
        Args:
            top_level (str): The top level categories or existing taxonomy for which to 
            generate the hierarchy. This must match the name for the input domains and 
            prompt files.
            max_iterations (int): The maximum number of iterations of the API call. This
            is the maximum number of times the LLM will subdivide the model. However,
            the process will stop early if the LLM returns an identical result to its 
            previous iteration.
            client (str): The LLM to use. Options are 'openai' and 'anthropic'. OpenAI 
            will use GPT-4o and Anthropic will use Claude 3.5 Sonnet v2.
            top_levels_separate (bool): Whether or not to run the API call separately
            for each top level domain. This is recommended for use with Anthropic Claude
            3.5 Sonnet v2 to avoid hitting the token limit. The results of each 
            domain will be combined into a single hierarchy.

        """
        with open('../input_domains/' + str(top_level) + '_domains.json') as f:
            self.input = json.load(f)
        if top_levels_separate:
            with open('../prompts/' + str(top_level) + '_separate_prompt.txt') as f:
                self.prompt = f.read()
        else:
            with open('../prompts/' + str(top_level) + '_prompt.txt') as f:
                self.prompt = f.read()
        self.max_iterations = max_iterations
        self.top_level = top_level
        self.client = client
        self.top_levels_separate = top_levels_separate
        self.results = []
        self.dimension_names = []
        self.dimension_descriptions = []

    def api_call(
            self,
            prompt:str,
            input:list,
            domain_name: typing.Optional[str]
            ) -> typing.Tuple[list, int]:
        """Calls the LLM API to generate the hierarchy.
        
        Args:
        prompt: The prompt to use for the LLM. This is set using the top level domains
        on initializtion.
        input: The hierarchical model to pass to LLM. For the first pass, this is set
        using the top level domains.
        domain_name: Optional domain name for use when top levels domains are being
        passed separately.

        Returns:
        current_model: The final model returned by the LLM.
        iterations_completed: The number of times the LLM API was called.
        """
        domain_name = domain_name
        max_iterations = self.max_iterations
        input = input
        # openai gpt-4o
        if self.client == 'openai':
            for i in range(1, self.max_iterations + 1):
                print(i)
                client=instructor.from_openai(openai.OpenAI())
                message = client.chat.completions.create(
                    model='gpt-4o',
                    response_model=response,
                    messages=[
                        {'role':'user',
                        'content':eval(prompt)},
                    ]
                )
                iterations_completed = i
                current_model = [x.model_dump() for x in message.dimensions]
                if input == current_model:
                    print('finished on iteration ', i-1)
                    break
                if iterations_completed == self.max_iterations:
                    print('completed all ', self.max_iterations, ' iterations')
                input = current_model
        # anthropic claude 3.5 sonnet
        if self.client == 'anthropic':
            region = 'us-west-2'
            for i in range(1, self.max_iterations + 1):
                print(i)
                client = instructor.from_anthropic(anthropic.AnthropicBedrock(
                    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID'),
                    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY'),
                    aws_region=region,
                ))

                message = client.messages.create(
                    model='anthropic.claude-3-5-sonnet-20241022-v2:0',
                    response_model=response,
                    max_tokens=80000,
                    messages=[
                        {'role':'user',
                        'content':eval(prompt)},
                        ],
                    )
                iterations_completed = i 
                current_model = [x.model_dump() for x in message.dimensions]
                if input == current_model:
                    print('finished on iteration ', i-1)
                    break
                if iterations_completed == self.max_iterations:
                    print('completed all ', self.max_iterations, ' iterations')
                input = current_model

        return current_model, iterations_completed

    def parse_node(
            self,
            node: dict,
            hierarchy: list) -> None:
        """Parses the nested model generated by the LLM into a table format."""
        if len(node['children']) == 0:
            self.results.append(hierarchy + [node['dimension_name']])
            if node['dimension_name'] not in self.dimension_names:
                self.dimension_names.append(node['dimension_name'])
            if node['description'] not in self.dimension_descriptions:
                self.dimension_descriptions.append(node['description'])   
        else:
            self.results.append(hierarchy + [node['dimension_name']])
            for child in node['children']:
                child_hierarchy = hierarchy.copy()
                child_hierarchy.append(node['dimension_name'])
                if node['dimension_name'] not in self.dimension_names:
                    self.dimension_names.append(node['dimension_name'])
                if node['description'] not in self.dimension_descriptions:
                    self.dimension_descriptions.append(node['description']) 
                self.parse_node(child, child_hierarchy)

    def run(self) -> typing.Tuple[list, pd.DataFrame]:
        """Runs the hierarchy generation process.

        Runs the API call function, parses the output, and saves the results to a JSON 
        and CSV file.

        Returns:
        model: The final hierarchy generated by the LLM in a nested format.
        result_df: The final hierarchy generated by the LLM in a dataframe.
        """
        if self.top_levels_separate:
            model = []
            iterations_completed = 0
            for domain in self.input:
                domain_name = domain['dimension_name']
                domain_model, domain_iterations_completed = self.api_call(self.prompt, domain, domain_name)
                model.append(domain_model[0])
                if domain_iterations_completed > iterations_completed:
                    iterations_completed = domain_iterations_completed
        else:
            model, iterations_completed = self.api_call(self.prompt,self.input)
        for domain in model:
            self.parse_node(domain, []) 
        description_dict = dict(zip(self.dimension_names, self.dimension_descriptions))
        for row in self.results:
            dimension = row[-1]
            description = description_dict[dimension]
            if len(row) < (iterations_completed + 1):
                row.extend([pd.NA] * ((iterations_completed + 1) - len(row)))
            row.append(description)
        result_df = pd.DataFrame(self.results)
        result_df.columns = [f'level_{i}' for i in range(1, len(result_df.columns))] + ['description']
        with open('../results/' + str(self.top_level) + '_' + str(self.client) + '_hierarchy_max' + str(self.max_iterations) + '.json', 'w') as f:
            json.dump(model, f)
        result_df = result_df.dropna(how='all', axis=1).fillna('')
        result_df.to_csv('../results/' + str(self.top_level) + '_' + str(self.client) + '_hierarchy_max' + str(self.max_iterations) + '.csv', index=False)
        return model, result_df
    
    