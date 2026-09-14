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
import { getSearchResults, scrapeWebsite, writeSummary } from '@/lib/ai/tools';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import z from 'zod';

const agentNodes = [
    'llmCall',
    'researchNode',
    'writerNode',
    'toolNodes',
    'end',
] as const;

const State = new StateSchema({
    messages: MessagesValue,
    nextAgent: z.enum(agentNodes),
});

const checkpointer = new MemorySaver();
const store = new InMemoryStore();

const model = getAgentModel();
const toolNodes = new ToolNode([getSearchResults, scrapeWebsite, writeSummary]);

const llmCall: GraphNode<typeof State> = async (state) => {
    const response = await model.invoke(
        [
            new SystemMessage(
                'Tou are the main agent and you are responsible for the overall task and you are the one who will decide what to do next. You have down the line research agent and tool agent to help you with the task. You are not allowed to hallucinate or make up information. ',
            ),
            ...state.messages,
        ],
        { outputVersion: 'v1' },
    );
    console.log('LLM Response: ', response);
    return {
        messages: [response],
        nextAgent: 'researchNode',
    };
};

const researchNode: GraphNode<typeof State> = async (state) => {
    const response = await model
        .bindTools([getSearchResults, scrapeWebsite])
        .invoke(
            [
                new SystemMessage(
                    'You are a helpful research assistant and you are responsible doing the research for the task. You have the tools to help you with the research. You can use the tools to get information from the web. in case you need to scrape the website, you can use the scrapeWebsite tool. in case you need to search the web, you can use the getSearchResults tool. You are not allowed to hallucinate or make up information. Then you will pass the information to the writer agent to write the summary of the information provided to you.',
                ),
                ...state.messages,
            ],
            { outputVersion: 'v1' },
        );
    console.log('Research Response: ', response);
    return {
        messages: [response],
        nextAgent: 'writerNode',
    };
};

const writerNode: GraphNode<typeof State> = async (state) => {
    const response = await model
        .bindTools([writeSummary])
        .invoke(
            [
                new SystemMessage(
                    'You are a helpful technical writer and you are responsible writing the summary of the information provided to you. You have the tools to help you with the writing. You can use the tools to write the summary of the information provided to you. You are not allowed to hallucinate or make up information. Then you will pass the summary to the main agent to end the task.',
                ),
                ...state.messages,
            ],
            { outputVersion: 'v1' },
        );
    console.log('Writer Response: ', response);
    return {
        messages: [response],
        nextAgent: 'end',
    };
};

function routeAfterLlmCall(state: typeof State.State) {
    if (state.nextAgent === 'researchNode') {
        return 'researchNode';
    }
    if (state.nextAgent === 'writerNode') {
        return 'writerNode';
    }
    if (state.nextAgent === 'end') {
        return END;
    }
    return 'llmCall';
}

export const workflow = new StateGraph(State)
    .addNode('llmCall', llmCall)
    .addNode('researchNode', researchNode)
    .addNode('writerNode', writerNode)
    .addEdge(START, 'llmCall')
    .addConditionalEdges('llmCall', routeAfterLlmCall, ['researchNode', END])
    .addEdge('llmCall', 'researchNode')
    .addEdge('researchNode', 'writerNode')
    .addEdge('writerNode', END)
    .compile({ checkpointer, store });

export const graph = workflow;
