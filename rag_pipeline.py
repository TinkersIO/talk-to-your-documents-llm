class RAGPipeline:
    def __init__(self, expander, retriever, context_builder, agent_service):
        self.expander = expander
        self.retriever = retriever
        self.context_builder = context_builder
        self.agent_service = agent_service

    async def run(self, query: str):
        expanded_queries = await self.expander.expand(query)
        docs = self.retriever.retrieve(expanded_queries)
        context = self.context_builder.build(docs)
        answer = await self.agent_service.answer(query, context)
        return answer
