from RoleRAG.rolerag.pipelines.prompts.prompt_manager import PromptTemplate


PROMPTS = {
    "perqa_type_system": PromptTemplate(
        system_prompt="""Given a question, determine what type the question belongs to. Types include:
1. Bridge: A bridge question involves two or more facts that are connected by an intermediate entity (usually an associative link). The bridge question requires finding the intermediary entity, then using it to answer the question.
2. Comparison: A comparison question involves comparing two or more independent facts. The comparison question requires analyzing and comparing the differences or similarities between different facts to draw a conclusion.

Examples:
Question: What language was used by Renana Jhabvala's mother?
Output:
```json
{"Response": "Bridge"}
```

Question: Which film came out first, The Love Route or Engal Aasan?
Output:
```json
{"Response": "Comparison"}
```

NOTE: Always respond with the JSON object. The question should be preferentially judged as "Bridge" or "Comparison".
Now it's your turn!""",
        user_prompt="Question: {question}\nOutput: ",
    ),
    "perqa_bridge_system": PromptTemplate(
        system_prompt="""Given a bridge question, split it into smaller, independent, and individual subqueries.
A bridge question involves two or more facts that are connected by an intermediate entity (usually an associative link). The bridge question requires finding the intermediary entity, then using it to answer the question.
For the subquery generation, input a tag "<A>" where the answer of the parent query should come to make the query complete. Specifically,
    1. Subquery is NOT allowed to ask open-ended question. For example, for question "What language was used by Renana Jhabvala's mother?", it is NOT allowed to decompose and ask "Who is Renana Jhabvala?".
    2. Each subquery is a simple fact question, not a question that requires reasoning. For example, "Who lives longer, <A1> or <A2>?" is NOT allowed.

Examples:
Question: What language was used by Renana Jhabvala's mother?
Result:
```json
{
    "Response": {
        "Q1": ["Who was Renana Jhabvala's mother?", "<A1>"],
        "Q2": ["What language was used by <A1>?", "<A2>"]
    }
}
```

Question: Who is Sobe (Sister Of Saint Anne)'s child-in-law?
Result:
```json
{
    "Response": {
        "Q1": ["Who is the child of Sobe (Sister Of Saint Anne)?", "<A1>"],
        "Q2": ["Who is the spouse of <A1>?", "<A2>"]
    }
}
```

Question: What followed the last person to live in Versailles in the country that became allies with America after the battle of Saratoga?
Result:
```json
{
    "Response": {
        "Q1": ["Who became allies with America after the Battle of Saratoga?", "<A1>"],
        "Q2": ["Who was the last person to live in Versailles?", "<A2>"],
        "Q3": ["What followed <A2> in <A1>?", "<A3>"]
    }
}
```

NOTE: Always respond with the JSON object.
Now it's your turn!""",
        user_prompt="Question: {question}\nResult: ",
    ),
    "perqa_comparison_system": PromptTemplate(
        system_prompt="""Given a comparison question, split it into smaller, independent, and individual subqueries.
A comparison question involves comparing two or more independent facts. The comparison question requires analyzing and comparing the differences or similarities between different facts to draw a conclusion.
For the subquery generation, input a tag "<A>" where the answer of the parent query should come to make the query complete. Specifically,
    1. Subquery is NOT allowed to ask open-ended question. For example, for question "What language was used by Renana Jhabvala's mother?", it is NOT allowed to decompose and ask "Who is Renana Jhabvala?".
    2. Each subquery is a simple fact question, not a question that requires reasoning. For example, "Who lives longer, <A1> or <A2>?" is NOT allowed.

Examples:
Question: Are Harper High School (Chicago) and Santa Sabina College located in the same country?
Result:
```json
{
    "Response": {
        "Q1": ["What country is Santa Sabina College located in?", "<A1>"],
        "Q2": ["What country is Harper High School (Chicago) located in?", "<A2>"]
    }
}
```

Question: Who lived longer, Constance Keys or Anthony De Jasay?
Result:
```json
{
    "Response": {
        "Q1": ["When was Constance Keys born?", "<A1>"],
        "Q2": ["When did Constance Keys die?", "<A2>"],
        "Q3": ["When was Anthony De Jasay born?", "<A3>"],
        "Q4": ["When did Anthony De Jasay die?", "<A4>"]
    }
}
```

Question: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
Result:
```json
{
    "Response": {
        "Q1": ["Who was the director of the film A Flame in My Heart?", "<A1>"],
        "Q2": ["Who was the director of the film Butcher, Baker, Nightmare Maker?", "<A2>"],
        "Q3": ["When was <A1> born?", "<A3>"],
        "Q4": ["When was <A2> born?", "<A4>"]
    }
}
```

NOTE: Always respond with the JSON object.
Now it's your turn!""",
        user_prompt="Question: {question}\nResult: ",
    ),
    "perqa_rag_qa": PromptTemplate(
        system_prompt="""You are a question answering system. Use the retrievals while generating the answers and keep the answers grounded in the retrievals.
Generate a JSON with a single key "Response" and a value that is a short phrase or a few words. In JSON, put every value as a string always, not float.

Examples:
Query: In which state is Hertfordshire located?
Retrievals:
    1. Hertfordshire: Hertfordshire is the county immediately north of London and is part of the East of England region, a mainly statistical unit. A significant minority of the population across all districts are City of London commuters. To the east is Essex, to the west is Buckinghamshire and to the north are Bedfordshire and Cambridgeshire.
    2. Hertfordshire: Hertfordshire is an administrative and historic county situated in the East of England, which is part of the United Kingdom. Within England, Hertfordshire serves as an administrative division and is bordered by Greater London to the south, with its landscape largely encompassed by the London Basin.
    3. Hertfordshire Chain Walk: The Hertfordshire Chain Walk is located in Hertfordshire, England, and consists of 15 linked circular walks. These walks, each of which is between 4.25 and 9 miles, make up a total distance of 87 miles. The tracks pass through villages in East Hertfordshire close to London, the Icknield Way and the Cambridgeshire border.
    4. University of Hertfordshire: Campus is the university's Learning Resource Centre, a combined library and computer centre. The University of Hertfordshire Students' Union is headquartered at College Lane campus. The College Lane campus is also the location of Hertfordshire International College, which is part of the Navitas group, providing a direct pathway for international students to the University.
    5. Moor Park, Hertfordshire: Moor Park, Hertfordshire Moor Park is a private residential estate in the Three Rivers District of Hertfordshire, England. Located approximately...""",
        user_prompt="Query: {question}\nRetrievals:\n{paragraphs}\nAnswer: ",
    ),
    "perqa_bridge_aggregate": PromptTemplate(
        system_prompt="""You are a question answering system. Use the evidence while generating the answer and keep the answer grounded in the evidence. Each piece of evidence is represented as "Question >> Answer", where ">>" means "the Answer to the Question is...".
Generate a JSON with a single key "Response" and a value that is a short phrase or a few words. In JSON, put every value as a string always, not float.

Examples:
Question: When was the baseball team winning the world series in 2015 baseball created?
Evidence:
    1. Who won the world series in 2015 baseball? >> Kansas City Royals
    2. When was Kansas City Royals created? >> 1969
Answer:
```json
{"Response": "1969"}
```""",
        user_prompt="Question: {question}\nEvidence:\n{paragraphs}\nAnswer: ",
    ),
    "perqa_comparison_aggregate": PromptTemplate(
        system_prompt="""You are a question answering system. Use the evidence while generating the answer and keep the answer grounded in the evidence. Each piece of evidence is represented as "Question >> Answer", where ">>" means "the Answer to the Question is...".
Generate a JSON with a single key "Response" and a value that is a short phrase or a few words. In JSON, put every value as a string always, not float.

Examples:
Question: Which film came out first, The Love Route or Engal Aasan?
Evidence:
    1. When was The Love Route released? >> February 25, 1915
    2. When was Engal Aasan released? >> July 2009
Answer:
```json
{"Response": "The Love Route"}
```""",
        user_prompt="Question: {question}\nEvidence:\n{paragraphs}\nAnswer: ",
    ),
}
