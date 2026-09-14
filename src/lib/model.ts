import { ChatOllama } from '@langchain/ollama';

export const getAgentModel = (model = 'glm-5.2:cloud') => {
    return new ChatOllama({
        model,
        temperature: 0.5,
        baseUrl: process.env.OLLAMA_HOST,
        headers: {
            Authorization: `Bearer ${process.env.OLLAMA_API_KEY}`,
        },
    });
};
