from RoleRAG.rolerag.pipelines.prompts.prompt_manager import PromptTemplate


PROMPTS = {
    "hipporag_ner": PromptTemplate(
        system_prompt="",
        user_prompt="""Your task is to extract named entities from the given paragraph.
Return ONLY one valid JSON object. Do not include markdown fences or explanations.
The JSON schema is: {"named_entities": ["entity 1", "entity 2"]}.

Please reference the following example:

[Example]

Paragraph:

Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.

Here is the example entity list:

{"named_entities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]}

Now respond with only the entity list JSON object.

Paragraph:
```
__USER_INPUT__
```""",
    ),
    "hipporag_openie": PromptTemplate(
        system_prompt="",
        user_prompt="""Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
Return ONLY one valid JSON object. Do not include markdown fences or explanations.
The JSON schema is: {"triples": [["source", "relation", "target"]]}.

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

# Here is an example for your reference:

[Example]

Paragraph:

Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.

Here is the example entity list:

{"named_entities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]}

Based on the entity above, the triple list should be:

{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"],
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}

Now convert the paragraph into triples and respond with only the JSON object.
Paragraph:
```
__PASSAGE__
```

__NAMED_ENTITY_JSON__""",
    ),
    "hipporag_qa": PromptTemplate(
        system_prompt='As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations, the answer without any additional words, labels, explanations, or formatting.',
        user_prompt="{reference}\n\nQuestion: {question}\nThought: ",
    ),
}
