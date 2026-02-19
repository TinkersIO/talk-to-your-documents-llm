import asyncio

class QueryExpander:
    def __init__(self, agent):
        self.agent = agent

    async def expand(self, query: str):
        expansion_prompt = f"""
You are expanding a user query to improve document retrieval.
Generate EXACTLY 3 alternative search queries.
Do NOT number them.
Do NOT add bullet points.
Each query must be on a new line.

Add:
- synonyms
- related technical terms
- alternative phrasings

Return ONLY the queries.
Do NOT explain

Original Question:
{query}
"""

        try:
            expanded = await self.agent.ainvoke({
                "input": [
                    {"role": "system", "content": "You are a search query expansion assistant."},
                    {"role": "user", "content": expansion_prompt}
                ]
            })

            if isinstance(expanded, dict):
                expanded_queries = expanded.get("output", "").split("\n")
            else:
                expanded_queries = str(expanded).split("\n")

            expanded_queries = [q.strip() for q in expanded_queries if q.strip()]
            return [query] + expanded_queries

        except:
            return [query]
