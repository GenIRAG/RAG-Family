from RoleRAG.rolerag.pipelines.prompts.prompt_manager import PromptTemplate


PROMPTS = {
    "naive_rag": PromptTemplate(
        system_prompt="",
        user_prompt="Answer the question based on the given documents.Output only the answer, no explanations.\n\nDocuments:\n{reference}\n\nQuestion: {question}",
    ),
    "reference_only": PromptTemplate(
        system_prompt="",
        user_prompt="Here are the relevant documents:\n\n{reference}",
    ),
    "no_reference": PromptTemplate(
        system_prompt="You are a helpful assistant.",
        user_prompt="Question: {question}",
    ),
}
