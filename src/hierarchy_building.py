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

    # call api to generate hierarchy
    def api_call(self, prompt, input, domain_name: typing.Optional[str]):
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

    # parse model to format as table
    def parse_node(self, node, hierarchy):
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

    # generate and save the hierarchy 
    def run(self):
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
        # save model as json and csv
        with open('../results/' + str(self.top_level) + '_' + str(self.client) + '_hierarchy_max' + str(self.max_iterations) + '.json', 'w') as f:
            json.dump(model, f)
        result_df = result_df.dropna(how='all', axis=1).fillna('')
        result_df.to_csv('../results/' + str(self.top_level) + '_' + str(self.client) + '_hierarchy_max' + str(self.max_iterations) + '.csv', index=False)
        return model, result_df
    
    