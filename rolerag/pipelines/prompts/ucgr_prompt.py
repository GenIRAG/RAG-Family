from RoleRAG.rolerag.pipelines.prompts.prompt_manager import PromptTemplate


PROMPTS = {
    "ucgr_retrieval": PromptTemplate(
        system_prompt="",
        user_prompt="Given a query, your task is provide a concise Wikipedia Page title and Section title to answer this query.\n\nQuery: {question}\nResponse:",
    ),
}
