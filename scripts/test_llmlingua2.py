import os

os.environ["HF_ENDPOINT"] = "YOUR_MIRROR_URL"

from llmlingua import PromptCompressor


prompt = """
Item 15, report from City Manager Recommendation to adopt three resolutions. First, to join the Victory Pace program. Second, to join the California first program. And number three, consenting to to inclusion of certain properties within the jurisdiction in the California Hero program. It was emotion, motion, a second and public comment. CNN. Please cast your vote. Oh. Was your public comment? Yeah. Please come forward. I thank you, Mr. Mayor. Thank you. Members of the council. My name is Alex Mitchell. I represent the hero program. Just wanted to let you know that the hero program. Has been in California for the last three and a half years
"""

llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    device_map="cpu",
    use_llmlingua2=True,
)
compressed_prompt = llm_lingua.compress_prompt(
    prompt,
    rate=0.33,
)

print(compressed_prompt)
