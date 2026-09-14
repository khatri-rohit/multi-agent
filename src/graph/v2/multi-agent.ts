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
import {
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    isAIMessage,
} from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import z from 'zod';

import { getAgentModel } from '@/lib/model';
import { getSearchResults, scrapeWebsite } from '@/lib/ai/tools';

const agentNodes = [
    'supervisor',
    'planning',
    'research',
    'analysis',
    'writer',
    'end',
] as const;

const State = new StateSchema({
    nextAgent: z.enum(agentNodes),
    planningData: z.string(),
    researchData: z.string(),
    analysisData: z.string(),
    finalReport: z.string(),
    taskCompleted: z.boolean(),
    currentTask: z.string(),
    messages: MessagesValue,
});

const checkpointer = new MemorySaver();
const store = new InMemoryStore();
const model = getAgentModel();

const researchTools = [getSearchResults, scrapeWebsite];
const researchToolsNode = new ToolNode(researchTools, {
    handleToolErrors: true,
});

const supervisorPrompt = ChatPromptTemplate.fromMessages([
    new SystemMessage(`
You are a supervisor of a team of agents. 
1. Planning agent - Responsible for planning the task and assigning it to the appropriate agent.
2. Research agent - Responsible for researching the topics given by the planning agent using some given tools like getSearchResults and scrapeWebsite, your not allowed to invent any facts and information on your own you'll always use facts for research wityh accuracy and precision.
3. Analysis agent - Responsible for analyzing the research data and make sure the data is complete and accurate and is not allowed to make up any information.
4. Writer agent - Responsible for writing the final report based on the analysis data with context and proper formatting and structure.
5. End agent - Responsible for ending the task and returning the final report.

Based on the current state of the conversation, you need to decide the next agent to be called and the task to be assigned to the agent. If the task is completed, response with 'DONE'.

Current state of the conversation:
- Has planning data: {planningData}
- Has research data: {researchData}
- Has analysis data: {analysisData}
- Has final report: {finalReport}

Respond with ONLY the agent name (planning, research, analysis, writer, end) and the task to be assigned to the agent.
`),
    new HumanMessage('{task}'),
]);

const supervisorAgent: GraphNode<typeof State> = async (state) => {
    const messages = state.messages;
    let task = (messages.at(-1)?.content as string) ?? 'No Task';

    const planningData = state.planningData;
    const hasResearchData = !!state.researchData;
    const hasAnalysisData = !!state.analysisData;
    const hasFinalReport = !!state.finalReport;

    const chain = await supervisorPrompt.invoke({
        task: task,
        planningData: planningData,
        researchData: hasResearchData,
        analysisData: hasAnalysisData,
        finalReport: hasFinalReport,
    });

    console.log(
        'Supervisor: ',
        chain.messages.map((message) => message.content).join('\n'),
    );
    const decision = (chain.messages.at(-1)?.content as string)
        .toLowerCase()
        .trim();

    let nextAgent = '';

    let suppervisorMsg = '';

    if (decision.includes('done') || hasFinalReport) {
        nextAgent = 'end';
        task = 'No Task';
        suppervisorMsg = 'Supervisor: All tasks completed! Great work team';
    } else if (decision.includes('planning')) {
        nextAgent = 'planning';
        task = decision.split(' ')[1];
        suppervisorMsg = 'Supervisor: Planning task assigned to planning agent';
    } else if (decision.includes('research')) {
        nextAgent = 'research';
        task = decision.split(' ')[1];
        suppervisorMsg = 'Supervisor: Research task assigned to research agent';
    } else if (decision.includes('analysis')) {
        nextAgent = 'analysis';
        task = decision.split(' ')[1];
        suppervisorMsg = 'Supervisor: Analysis task assigned to analysis agent';
    } else if (decision.includes('writer')) {
        nextAgent = 'writer';
        task = decision.split(' ')[1];
        suppervisorMsg = 'Supervisor: Writer task assigned to writer agent';
    } else if (decision.includes('end')) {
        nextAgent = 'end';
        task = 'No Task';
        suppervisorMsg = 'Supervisor: End task assigned to end agent';
    }

    return {
        messages: [
            new AIMessage({ content: suppervisorMsg }),
            new HumanMessage({ content: task }),
        ],
        nextAgent: nextAgent,
        currentTask: task,
        suppervisorMsg: suppervisorMsg,
    };
};

const planningAgent: GraphNode<typeof State> = async (state) => {
    const task = state.currentTask;

    const response = await model.invoke([
        new HumanMessage({
            content: `
            You are a specailized planning agent that is responsible for planning the task: ${task}.
            How to complete the task in the best way possible, your planning should be detailed and specific which gonna help the research agent to complete the task in the best way possible.
            You are not allowed to make up any information and your planning should be based on the task, the research agent have access to some tools like getSearchResults and scrapeWebsite to help them to complete the task.        `,
        }),
    ]);
    console.log('Planning: ', response.content);

    return {
        messages: [new AIMessage({ content: response.content })],
        planningData: response.content,
        nextAgent: 'supervisor',
    };
};

const researchAgent: GraphNode<typeof State> = async (state) => {
    const task = state.currentTask;
    const planningData = state.planningData;

    const response = await model.bindTools(researchTools).invoke([
        new HumanMessage({
            content: `
            You are a research agent that is responsible for researching the task: ${task}.
            You are using the following tools to research the task: ${researchTools.map((tool) => tool.name).join(', ')}.
            You are not allowed to make up any information and you need to use the tools provided to you to research the task.
            Your research should be based on the planning data: ${planningData}.
            Your research should Include the following:
            1. Key facts and background information
            2. Current trends and developments
            3. Important statistics or data that are relevant to the task
            4. Notable examples or case studies if has any
            
            Be concise but thorough.
            `,
        }),
    ]);
    console.log('Research: ', response.content);
    const researchData = `Research Data: It includes the following: ${response.content}`;

    return {
        messages: [new AIMessage({ content: researchData })],
        researchData: response.content,
        nextAgent: 'supervisor',
    };
};

const analysisAgent: GraphNode<typeof State> = async (state) => {
    const researchData = state.researchData;
    const response = await model.invoke([
        new HumanMessage({
            content: `
            You are a analysis agent that is responsible for analyzing the research data: ${researchData}.
            You are not allowed to make up any information and you need to analyze the research data.
            Your analysis should be based on the research data and should be complete and accurate and is not allowed to make up any information.
            `,
        }),
    ]);
    console.log('Analysis: ', response.content);

    return {
        messages: [new AIMessage({ content: response.content })],
        analysisData: response.content,
        nextAgent: 'supervisor',
    };
};

const writerAgent: GraphNode<typeof State> = async (state) => {
    const analysisData = state.analysisData;
    const response = await model.invoke([
        new HumanMessage({
            content: `
            You are a writer agent that is responsible for writing the final report based on the analysis data: ${analysisData}.
            You are not allowed to make up any information and you need to write the final report based on the analysis data.
            `,
        }),
    ]);

    console.log('Writer: ', response.content);
    return {
        messages: [new AIMessage({ content: response.content })],
        finalReport: response.content,
        nextAgent: 'supervisor',
        taskCompleted: true,
    };
};

const nextAgentRouter = (state: typeof State.State): string => {
    if (state.nextAgent === 'planning') {
        return 'planning';
    } else if (state.nextAgent === 'research') {
        return 'research';
    } else if (state.nextAgent === 'analysis') {
        return 'analysis';
    } else if (state.nextAgent === 'writer') {
        return 'writer';
    } else if (state.nextAgent === 'end') {
        return END;
    }
    return 'supervisor';
};

export const wrokflow = new StateGraph<typeof State>(State)
    .addNode('supervisor', supervisorAgent)
    .addNode('planning', planningAgent)
    .addNode('research', researchAgent)
    .addNode('analysis', analysisAgent)
    .addNode('writer', writerAgent)
    .addEdge(START, 'supervisor')
    .addConditionalEdges('supervisor', nextAgentRouter, [
        'planning',
        'research',
        'analysis',
        'writer',
        END,
    ])
    .addEdge('planning', 'research')
    .addEdge('research', 'analysis')
    .addEdge('analysis', 'writer')
    .addEdge('writer', END)
    .compile({ checkpointer, store });
