// This is a Simple Multi-AI Agent Pipeline for research and writing summary
import {
    StateGraph,
    StateSchema,
    MessagesValue,
    GraphNode,
    START,
    END,
    MemorySaver,
    InMemoryStore,
} from '@langchain/langgraph';
import { AIMessage, SystemMessage } from '@langchain/core/messages';

import { getAgentModel } from '@/lib/model';
import { getSearchResults, scrapeWebsite } from '@/lib/ai/tools';
import { ToolNode } from '@langchain/langgraph/prebuilt';

const State = new StateSchema({
    messages: MessagesValue,
});

const checkpointer = new MemorySaver();
const store = new InMemoryStore();

const model = getAgentModel();

const llmCall: GraphNode<typeof State> = async (state) => {
    const response = await model
        .bindTools([getSearchResults, scrapeWebsite])
        .invoke(
            [
                new SystemMessage(
                    'You are a helpful assistant and you are a multi-agent system that can use tools to perform tasks. You are also a writer and you are tasked with writing a summary of the information provided to you. You are not allowed to hallucinate or make up information. Also improve the quality of the information provided to you, and make accurate query to the tools.',
                ),
                ...state.messages,
            ],
            { outputVersion: 'v1', recursionLimit: 3 },
        );
    console.log('Response: ', response);
    return {
        messages: [response],
    };
};

const toolNodes = new ToolNode([getSearchResults, scrapeWebsite]);

function routeAfterLlmCall(state: typeof State.State) {
    const lastMessage = state.messages.at(-1);
    if (
        AIMessage.isInstance(lastMessage) &&
        (lastMessage?.tool_calls?.length ?? 0) > 0
    ) {
        return 'toolNodes';
    }
    return END;
}

export const graph = new StateGraph(State)
    .addNode('llmCall', llmCall)
    .addNode('toolNodes', toolNodes)
    .addEdge(START, 'llmCall')
    .addConditionalEdges('llmCall', routeAfterLlmCall, ['toolNodes', END])
    .addEdge('toolNodes', 'llmCall')
    .compile({ checkpointer, store });
