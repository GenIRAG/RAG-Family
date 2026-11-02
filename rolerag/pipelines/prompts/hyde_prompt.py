from RoleRAG.rolerag.pipelines.prompts.prompt_manager import PromptTemplate


PROMPTS = {
    "hyde_web_search": PromptTemplate(
        system_prompt="",
        user_prompt="Please write a passage to answer the question.\nQuestion: {query}\nPassage:",
    ),
    "hyde_scifact": PromptTemplate(
        system_prompt="",
        user_prompt="Please write a scientific paper passage to support/refute the claim.\nClaim: {query}\nPassage:",
    ),
    "hyde_arguana": PromptTemplate(
        system_prompt="",
        user_prompt="Please write a counter argument for the passage.\nPassage: {query}\nCounter Argument:",
    ),
    "hyde_trec_covid": PromptTemplate(
        system_prompt="",
        user_prompt="Please write a scientific paper passage to answer the question.\nQuestion: {query}\nPassage:",
    ),
    "hyde_fiqa": PromptTemplate(
        system_prompt="",
        user_prompt="Please write a financial article passage to answer the question.\nQuestion: {query}\nPassage:",
    ),
    "hyde_dbpedia_entity": PromptTemplate(
        system_prompt="",
        user_prompt="Please write a passage to answer the question.\nQuestion: {query}\nPassage: ",
    ),
    "hyde_trec_news": PromptTemplate(
        system_prompt="",
        user_prompt="Please write a news passage about the topic.\nTopic: {query}\nPassage:",
    ),
    "hyde_mr_tydi": PromptTemplate(
        system_prompt="",
        user_prompt="Please write a passage in {language} to answer the question in detail.\nQuestion: {query}\nPassage:",
    ),
}
